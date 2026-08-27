"""Download IEEE-CIS Fraud Detection from Kaggle into data/raw/.

Prerequisites (one time):
  1. Create a Kaggle account.
  2. Visit https://www.kaggle.com/competitions/ieee-fraud-detection/rules
     and click "I Understand and Accept". The API returns 403 until you do.
  3. Kaggle -> Settings -> API -> "Create New Token", which downloads a
     kaggle.json containing a username and a key.

Then supply the credentials either way:
  a) save the file to ~/.kaggle/kaggle.json and `chmod 600` it, or
  b) put KAGGLE_USERNAME=... and KAGGLE_KEY=... in a .env at the repo root.

Note there is no such thing as a single KAGGLE_API_TOKEN variable -- the API
needs the username and the key as two separate values.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv

from ringfence import config

load_dotenv(config.ROOT / ".env")

EXPECTED = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
]


def main() -> int:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    env_ok = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    if not cred.exists() and not env_ok:
        found = [k for k in os.environ if k.startswith("KAGGLE")]
        print("ERROR: no Kaggle credentials found.")
        if found:
            print(f"  (saw {found}, but both KAGGLE_USERNAME and KAGGLE_KEY are required)")
        print(f"\n{__doc__}")
        return 1

    if all((config.RAW / f).exists() for f in EXPECTED):
        print("All files already present in data/raw/. Nothing to do.")
        return 0

    print(f"Downloading {config.KAGGLE_COMPETITION} -> {config.RAW} (~118MB zipped)")
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "competitions", "download",
            "-c", config.KAGGLE_COMPETITION, "-p", str(config.RAW),
        ],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print("Kaggle download failed:\n", result.stdout, result.stderr)
        if "403" in (result.stdout + result.stderr):
            print("\n403 usually means you have not accepted the competition rules yet.")
        return 1

    for zp in config.RAW.glob("*.zip"):
        print(f"Extracting {zp.name}")
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(config.RAW)
        zp.unlink()

    missing = [f for f in EXPECTED if not (config.RAW / f).exists()]
    if missing:
        print("WARNING: missing after extract:", missing)
        return 1

    for f in EXPECTED:
        size = (config.RAW / f).stat().st_size / 1e6
        print(f"  {f:28s} {size:8.1f} MB")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
