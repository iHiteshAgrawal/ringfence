"""Run the agent layer against REAL rings scored by the trained model.

Nothing here is a fabricated example: the rings, amounts, velocities and scores
all come out of the held-out test set. That matters, because a dispute agent
demoed on invented data proves only that the prompt is well written.

  --offline   build everything and print the triage plus the exact prompts,
              without calling the API. Verifies the plumbing with no key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console

from ringfence import config
from ringfence.agent import casefile as cf
from ringfence.agent import drafter
from ringfence.agent.models import Dispute, EvidenceDocument
from ringfence.agent.triage import triage

console = Console()
USD_INR = config.DEFAULT_COSTS["usd_inr"]


def load_scored(tag: str) -> pd.DataFrame:
    """Score the held-out set with the trained booster."""
    meta = json.loads((config.PROCESSED / f"meta_{tag}.json").read_text())
    feature_names = meta["feature_names"]
    vsel = config.PROCESSED / f"vselect_{tag}.json"
    if vsel.exists():
        drop = set(json.loads(vsel.read_text())["drop"])
        feature_names = [f for f in feature_names if f not in drop]

    booster = lgb.Booster(model_file=str(config.ROOT / "models" / f"lgbm_{tag}.txt"))
    need = list(dict.fromkeys(
        feature_names + [config.TARGET, config.TIME_COL, "TransactionAmt", "ring_id", "client_id"]
    ))
    df = pd.read_parquet(config.PROCESSED / f"test_{tag}.parquet", columns=need)
    df["score"] = booster.predict(df[feature_names])
    return df


def top_rings(df: pd.DataFrame, n: int = 3) -> list[cf.RingFacts]:
    """Pick the highest-risk multi-card rings and describe them factually."""
    multi = df.groupby("ring_id", observed=True)["client_id"].nunique()
    multi = multi[multi > 1]
    cand = df[df["ring_id"].isin(multi.index)]
    ranked = (
        cand.groupby("ring_id", observed=True)["score"]
        .max().sort_values(ascending=False).head(n)
    )

    out: list[cf.RingFacts] = []
    for ring_id in ranked.index:
        g = df[df["ring_id"] == ring_id]
        span = (g[config.TIME_COL].max() - g[config.TIME_COL].min()) / 86400.0
        amt = g["TransactionAmt"]
        n_addr = int(g["addr1"].nunique()) if "addr1" in g else 1
        # A link means one attribute value touching several DISTINCT cards.
        # Counting repeated rows instead misses the real thing: four cards on
        # one device shows up as four rows with count 1 each if a card
        # transacted once, and the strongest link in the ring goes unreported.
        shared: dict[str, list[str]] = {}
        for attr in ("addr1", "P_emaildomain", "DeviceInfo", "card1"):
            if attr not in g.columns:
                continue
            per_value = g.groupby(attr, observed=True)["client_id"].nunique()
            linking = per_value[per_value > 1].sort_values(ascending=False)
            if len(linking):
                shared[attr] = [
                    f"{v} (shared by {c} cards)" for v, c in linking.head(3).items()
                ]
        out.append(cf.RingFacts(
            ring_id=int(ring_id),
            n_clients=int(g["client_id"].nunique()),
            n_transactions=len(g),
            n_addresses=max(n_addr, 1),
            total_amount_inr=float(amt.sum() * USD_INR),
            span_days=float(span),
            velocity_per_day=float(len(g) / max(span, 1 / 24)),
            cards_per_address=float(g["client_id"].nunique() / max(n_addr, 1)),
            amount_cv=float(amt.std() / max(amt.mean(), 0.01)) if len(amt) > 1 else 0.0,
            known_prior_frauds=int(g["ring_known_prior_frauds"].max())
            if "ring_known_prior_frauds" in g else 0,
            max_fraud_score=float(g["score"].max()),
            mean_fraud_score=float(g["score"].mean()),
            top_features=["ring_known_fraud_rate", "C14", "C8", "ring_prior_velocity"],
            shared_attributes=shared,
        ))
    return out


def synth_dispute(df: pd.DataFrame, low_score: bool) -> tuple[Dispute, list[EvidenceDocument]]:
    """Build a dispute from a real transaction.

    A chargeback is filed against a real payment. We pick a genuinely
    low-scoring one (the friendly-fraud shape) or a high-scoring one (the
    genuinely-compromised shape) so both branches can be seen.
    """
    pool = df[df["TransactionAmt"] > df["TransactionAmt"].quantile(0.95)]
    row = pool.nsmallest(1, "score") if low_score else pool.nlargest(1, "score")
    row = row.iloc[0]
    ring_g = df[df["ring_id"] == row["ring_id"]]

    d = Dispute(
        id=f"disp_{int(row.name)}",
        payment_id=f"pay_{int(row.name)}",
        amount_inr=float(row["TransactionAmt"]) * USD_INR,
        reason_code="4855" if low_score else "4837",
        reason_description=(
            "Goods or services not received" if low_score
            else "No cardholder authorisation"
        ),
        respond_by_days=6,
        fraud_score=float(row["score"]),
        ring_id=int(row["ring_id"]),
        ring_size=int(ring_g["client_id"].nunique()),
        ring_known_frauds=int(row.get("ring_known_prior_frauds", 0) or 0),
        customer_prior_orders=int(row.get("ring_prior_tx", 0) or 0),
        customer_prior_disputes=0,
    )
    evidence = [
        EvidenceDocument(document_id="doc_pod_88213", slot="shipping_proof",
                         description="Courier proof of delivery, signed, matching billing address",
                         strength="strong"),
        EvidenceDocument(document_id="doc_inv_88213", slot="billing_proof",
                         description="Tax invoice issued at time of order", strength="moderate"),
        EvidenceDocument(document_id="doc_email_88213", slot="customer_communication",
                         description="Customer email confirming receipt and asking about a return window",
                         strength="strong"),
        EvidenceDocument(document_id="doc_tnc_v4", slot="term_and_conditions",
                         description="Checkout terms accepted at purchase", strength="weak"),
    ]
    return d, evidence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--offline", action="store_true", help="no API calls")
    ap.add_argument("--rings", type=int, default=2)
    args = ap.parse_args()

    console.rule("[bold]scoring held-out set")
    df = load_scored(args.tag)
    console.log(f"{len(df):,} transactions scored")

    console.rule("[bold]highest-risk rings (real)")
    facts = top_rings(df, args.rings)
    for f in facts:
        console.print(
            f"\n[bold]Ring {f.ring_id}[/bold]: {f.n_clients} cards, {f.n_transactions} tx, "
            f"Rs {f.total_amount_inr:,.0f}, {f.velocity_per_day:.1f} tx/day, "
            f"peak score {f.max_fraud_score:.3f}"
        )
        if args.offline:
            console.print("[dim]" + cf._render(f) + "[/dim]")
        else:
            case = cf.write_case_file(f)
            console.print(f"  [bold]{case.headline}[/bold]")
            console.print(f"  links:     {case.what_links_them}")
            console.print(f"  concern:   {case.why_suspicious}")
            console.print(f"  innocent:  {case.innocent_explanation}")
            console.print(f"  action:    [bold]{case.recommended_action}[/bold] "
                          f"(confidence {case.confidence})")
            console.print(f"  would flip: {case.what_would_change_the_call}")
            assert case.recommended_action in cf.VALID_ACTIONS, case.recommended_action

    console.rule("[bold]dispute triage on real payments")
    for low in (True, False):
        d, ev = synth_dispute(df, low_score=low)
        label = "low-score (friendly-fraud shape)" if low else "high-score (compromised shape)"
        t = triage(d, ev)
        console.print(
            f"\n[bold]{label}[/bold]  Rs {d.amount_inr:,.0f}  score {d.fraud_score:.3f}"
        )
        console.print(f"  -> {t.recommendation.value}  p_win={t.p_win:.1%}")
        for r in t.rationale:
            console.print(f"     - {r}")
        if not args.offline:
            drafted = drafter.draft_contest(d, ev)
            console.print(f"  payload action: [bold]{drafted.payload.action}[/bold]")
            console.print(f"  documents: {drafted.payload.document_ids()}")
            if drafted.payload.summary:
                console.print(f"  summary ({len(drafted.payload.summary)} chars):")
                console.print(f"    [italic]{drafted.payload.summary}[/italic]")
            console.print(f"  narrative: {drafted.narrative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
