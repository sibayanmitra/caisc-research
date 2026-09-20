#!/usr/bin/env python3
"""Pull AnAge build-14 dataset, extract heart rate + lifespan + body mass."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

ANAGE_URL = "https://genomics.senescence.info/species/dataset.zip"
OUT = DATA / "anage_raw.csv"


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(f"fetching {ANAGE_URL}")
    r = requests.get(ANAGE_URL, timeout=120)
    r.raise_for_status()
    log(f"got {len(r.content)} bytes")

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        log(f"zip members: {names}")
        target = next((n for n in names if n.lower().endswith(".txt") and "anage" in n.lower()), None)
        if target is None:
            target = next((n for n in names if n.endswith(".txt")), None)
        if target is None:
            log("ERROR: no .txt file in archive")
            sys.exit(2)
        log(f"reading {target}")
        with zf.open(target) as f:
            df = pd.read_csv(f, sep="\t", low_memory=False)

    log(f"loaded {len(df)} rows, {len(df.columns)} cols")
    log(f"columns sample: {list(df.columns)[:15]}")

    # AnAge column names per build-14:
    # 'Common name', 'Order', 'Family', 'Genus', 'Species',
    # 'Adult weight (g)', 'Maximum longevity (yrs)', 'Metabolic rate (W)',
    # 'Body mass (g)', 'Specific metabolic rate (W/g)'
    # Heart rate column candidates:
    hr_candidates = [c for c in df.columns if "heart" in c.lower() or "hr" in c.lower()[:5]]
    log(f"heart rate candidates: {hr_candidates}")

    keep_cols = []
    for want in ["Class", "Order", "Family", "Genus", "Species", "Common name",
                 "Adult weight (g)", "Body mass (g)", "Maximum longevity (yrs)",
                 "Metabolic rate (W)", "Specific metabolic rate (W/g)"]:
        if want in df.columns:
            keep_cols.append(want)

    keep_cols += hr_candidates
    subset = df[keep_cols].copy()
    subset.to_csv(OUT, index=False)
    log(f"wrote {OUT} with {len(subset)} rows, {len(subset.columns)} cols")

    # Filter for mammals + birds with heart rate
    if "Class" in subset.columns:
        cl = subset["Class"].value_counts()
        log(f"class counts: {dict(cl)}")
    if hr_candidates:
        hr = subset[hr_candidates[0]]
        log(f"non-null heart rate rows: {hr.notna().sum()}")


if __name__ == "__main__":
    main()
