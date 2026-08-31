"""Generate the architecture diagram as SVG.

Written as code rather than drawn by hand so the geometry stays consistent and
the diagram can be regenerated when the system changes. Emits
docs/architecture.svg, which index.html inlines at build time.

What the diagram has to show that a linear flow cannot:

  * TWO PLANES. Training runs on a schedule over history; scoring runs per
    transaction at arrival. They are different execution contexts.
  * SHARED STATE. Both planes touch the same entity graph and the same model
    artifact. Drawing them as separate pipelines would be a lie.
  * SHARED CODE. The feature module is literally the same in both planes; that
    is what makes the offline metrics predictive of online behaviour.
  * A FEEDBACK LOOP. A confirmed chargeback re-enters the ring's fraud history,
    but only after the reporting lag. This cycle is the single most important
    structural fact about the system and a top-to-bottom flow hides it.
  * AN EXTERNAL ACTOR. The issuer, not the model, triggers the dispute path.
"""
from __future__ import annotations

from pathlib import Path

W, H = 1120, 790
REPO = "https://github.com/iHiteshAgrawal/ringfence/blob/main"
OUT = Path(__file__).resolve().parent.parent / "docs" / "architecture.svg"

INK = "var(--text-primary)"
SUB = "var(--text-secondary)"
MUT = "var(--text-muted)"
LINE = "var(--axis)"
RULE = "var(--rule)"
S1 = "var(--series-1)"
BAD = "var(--invalid)"
FILL = "var(--surface-2)"
SURF = "var(--surface)"

p: list[str] = []


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def lane(x, y, w, h, label, tone=MUT):
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
             f'fill="none" style="stroke:{RULE}" stroke-dasharray="2 4"/>')
    p.append(f'<text x="{x + 12}" y="{y + 18}" font-size="10" letter-spacing="1.2" '
             f'style="fill:{tone}">{esc(label)}</text>')


def srclabel(x, y, src):
    """A clickable path to the file that implements a node.

    Rendered below the box rather than inside it, so adding source links needed
    no change to the node geometry. SVG <a> works because the diagram is inlined
    into the page rather than loaded through <img>.
    """
    p.append(f'<a href="{REPO}/{src}" target="_blank" rel="noopener">'
             f'<text x="{x}" y="{y}" font-size="9" style="fill:{MUT}" '
             f'text-decoration="underline">{esc(src)} \u2197</text></a>')


def box(x, y, w, h, title, sub=None, note=None, accent=None, src=None):
    stroke = accent or RULE
    p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" '
             f'style="fill:{FILL};stroke:{stroke}"/>')
    p.append(f'<text x="{x + 12}" y="{y + 21}" font-size="12" style="fill:{INK}">{esc(title)}</text>')
    if sub:
        p.append(f'<text x="{x + 12}" y="{y + 37}" font-size="9.5" style="fill:{SUB}">{esc(sub)}</text>')
    if note:
        p.append(f'<text x="{x + 12}" y="{y + 52}" font-size="9.5" style="fill:{S1}">{esc(note)}</text>')
    if src:
        srclabel(x + 2, y + h + 12, src)


def store(x, y, w, h, title, sub=None, accent=S1, src=None):
    """A datastore, drawn as a cylinder so it reads as state, not a step."""
    ry = 9
    p.append(f'<path d="M{x} {y + ry} a {w / 2} {ry} 0 0 1 {w} 0 v {h - 2 * ry} '
             f'a {w / 2} {ry} 0 0 1 {-w} 0 z" style="fill:{FILL};stroke:{accent}"/>')
    p.append(f'<path d="M{x} {y + ry} a {w / 2} {ry} 0 0 0 {w} 0" '
             f'fill="none" style="stroke:{accent}" opacity="0.7"/>')
    p.append(f'<text x="{x + w / 2}" y="{y + 34}" font-size="11.5" text-anchor="middle" '
             f'style="fill:{INK}">{esc(title)}</text>')
    if sub:
        p.append(f'<text x="{x + w / 2}" y="{y + 49}" font-size="9" text-anchor="middle" '
                 f'style="fill:{SUB}">{esc(sub)}</text>')
    if src:
        srclabel(x + 2, y + h + 12, src)


def diamond(cx, cy, rw, rh, title, sub=None):
    p.append(f'<path d="M{cx} {cy - rh} L{cx + rw} {cy} L{cx} {cy + rh} L{cx - rw} {cy} z" '
             f'style="fill:{FILL};stroke:{S1}"/>')
    p.append(f'<text x="{cx}" y="{cy - 1}" font-size="11" text-anchor="middle" style="fill:{INK}">{esc(title)}</text>')
    if sub:
        p.append(f'<text x="{cx}" y="{cy + 13}" font-size="9" text-anchor="middle" style="fill:{SUB}">{esc(sub)}</text>')


def arrow(pts, label=None, dashed=False, color=LINE, lx=None, ly=None, anchor="middle"):
    d = " ".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    p.append(f'<polyline points="{d}" fill="none" style="stroke:{color}" stroke-width="1.4"'
             f'{dash} marker-end="url(#a1)"/>')
    if label:
        x = lx if lx is not None else (pts[0][0] + pts[-1][0]) / 2
        y = ly if ly is not None else (pts[0][1] + pts[-1][1]) / 2 - 6
        p.append(f'<text x="{x}" y="{y}" font-size="9" text-anchor="{anchor}" '
                 f'style="fill:{color}">{esc(label)}</text>')


# ---------------------------------------------------------------- lanes
# Each lane reserves 36px of head-room so its label never sits under a node.
lane(8, 24, W - 16, 176, "OFFLINE  \u00b7  RUNS ON A SCHEDULE OVER HISTORY")
lane(8, 210, W - 16, 116, "SHARED STATE  \u00b7  WRITTEN ONLINE, READ BY BOTH")
lane(8, 336, W - 16, 176, "ONLINE  \u00b7  PER TRANSACTION")
lane(8, 522, W - 16, 244, "HUMAN  &  EXTERNAL PARTIES", MUT)

# ---------------------------------------------------------------- offline
OY = 62
box(24, OY, 148, 68, "historical txns", "IEEE-CIS, labelled", "590,540",
    src="scripts/download_data.py")
box(196, OY, 156, 68, "entity resolution", "fingerprint + D1 anchor", "222,477 clients",
    src="ringfence/entity/resolve.py")
box(376, OY, 164, 68, "feature build", "causal, 30-day label lag", "shared module",
    src="ringfence/features/causal.py")
box(564, OY, 148, 68, "temporal split", "+7-day embargo", "453,779 / 118,108",
    src="ringfence/data/load.py")
box(736, OY, 156, 68, "LightGBM + Platt", "time-aware validation", "PR-AUC 0.5986",
    src="ringfence/model/train.py")
box(916, OY, 164, 68, "cost curve", "both errors priced", "threshold 0.400",
    src="ringfence/eval/metrics.py")
for a, b in ((172, 196), (352, 376), (540, 564), (712, 736), (892, 916)):
    arrow([(a, OY + 34), (b, OY + 34)])

# ---------------------------------------------------------------- state
SY = 244
store(196, SY, 240, 72, "entity graph", "clients \u00b7 rings \u00b7 fraud history")
store(564, SY, 200, 72, "model artifact", "booster + calibrator")
store(880, SY, 200, 72, "evidence store", "merchant's own documents", RULE)

# ---------------------------------------------------------------- online
NY = 374
box(24, NY, 148, 68, "new transaction", "arrives at the gateway")
box(196, NY, 156, 68, "resolve + link", "union-find, at arrival", "same module",
    src="ringfence/entity/streaming.py")
box(376, NY, 164, 68, "feature build", "prior transactions only", "same module",
    src="ringfence/features/causal.py")
box(564, NY, 148, 68, "score", "calibrated probability")
diamond(800, NY + 34, 76, 42, "threshold", "cost-optimal")
box(916, NY - 26, 164, 52, "allow", "97.5% of traffic")
box(916, NY + 42, 164, 52, "review queue", "2.51% of traffic", accent=S1)
# NOTE: these are (from_x, to_x) pairs. An earlier version wrote (x, 383),
# reading the y-coordinate as the destination x, so every connector was drawn
# straight through the box it should have stopped at.
for a, b in ((172, 196), (352, 376), (540, 564), (712, 724)):
    arrow([(a, NY + 34), (b, NY + 34)])
arrow([(876, NY + 34), (896, NY + 34), (896, NY), (916, NY)])
arrow([(876, NY + 34), (896, NY + 34), (896, NY + 68), (916, NY + 68)])

# offline trains the model; online reads it
arrow([(814, OY + 68), (814, 186), (664, 186), (664, SY)], "writes", color=S1, lx=690, ly=182)
arrow([(664, SY + 72), (664, 352), (638, 352), (638, NY)], "reads", color=S1, lx=606, ly=332)
# online maintains the entity graph; offline reads it back
arrow([(274, NY), (274, SY + 72)], "writes", color=S1, lx=282, ly=332, anchor="start")
arrow([(400, SY), (400, 196), (458, 196), (458, OY + 68)], "reads", color=S1, lx=372, ly=192, anchor="end")

# ---------------------------------------------------------------- human / external
HY = 656
box(24, HY, 180, 68, "issuer files chargeback", "external event", "weeks later")
box(228, HY, 168, 68, "dispute triage", "invert the fraud score", "contest / accept",
    src="ringfence/agent/triage.py")
box(420, HY, 168, 68, "draft payload", "evidence assembled", "action = draft",
    src="ringfence/agent/drafter.py")
box(612, HY, 156, 68, "human approves", "always required", accent=BAD)
box(792, HY, 180, 68, "Razorpay contest API", "representment filed")
box(996, HY, 110, 68, "analyst", "works the queue")
for a, b in ((204, 228), (396, 420), (588, 612), (768, 792)):
    arrow([(a, HY + 34), (b, HY + 34)])

box(908, 560, 180, 60, "case file agent", "graded, reversible action", accent=S1,
    src="ringfence/agent/casefile.py")
arrow([(998, NY + 94), (998, 560)], "flagged ring", color=S1, lx=1004, ly=530, anchor="start")
arrow([(1040, 620), (1040, HY)])

# the dispute path needs the score, and the merchant's own documents
arrow([(638, NY + 68), (638, 612), (312, 612), (312, HY)],
      "score, once a dispute exists", color=LINE, lx=470, ly=608)
arrow([(900, SY + 72), (900, 500), (504, 500), (504, HY)], "documents",
      dashed=True, color=LINE, lx=700, ly=496)

# ---------------------------------------------------------------- feedback
# Routed down the far-left gutter so it crosses no node.
arrow([(24, HY + 34), (14, HY + 34), (14, SY + 36), (196, SY + 36)],
      None, dashed=True, color=BAD)
p.append(f'<text x="28" y="474" font-size="9.5" style="fill:{BAD}">confirmed chargeback</text>')
p.append(f'<text x="28" y="488" font-size="9.5" style="fill:{BAD}">re-enters ring history,</text>')
p.append(f'<text x="28" y="502" font-size="9.5" style="fill:{BAD}">after a 30-day lag</text>')

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img"
  aria-label="Ringfence architecture. An offline training plane and an online per-transaction scoring plane share one entity graph and one model artifact, and run the same feature module. Scoring ends at a cost-optimal threshold that either allows a transaction or sends it to a review queue worked by an analyst. Separately, an issuer filing a chargeback triggers dispute triage, which drafts a payload a human must approve. Confirmed chargebacks feed back into the ring fraud history after a thirty-day reporting lag.">
  <defs>
    <marker id="a1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <polygon points="0,1 9,5 0,9" fill="context-stroke"/>
    </marker>
  </defs>
  <g font-family="IBM Plex Mono, ui-monospace, monospace">
    {chr(10).join("    " + x for x in p)}
  </g>
</svg>
'''
OUT.write_text(svg)
print(f"wrote {OUT} ({len(svg)} bytes, {len(p)} elements)")
