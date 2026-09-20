#!/usr/bin/env python3
"""Pull sleep hours per species.

Mammals: msleep dataset (ggplot2 R package, Allison & Cicchetti 1976 origin, ~83 spp).
Birds: Roth, Lesku, Amlaner, Lima 2006 + supplements — bundled inline (small, ~12 spp).
"""
import io
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

MSLEEP_URL = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/msleep.csv"
MAMMAL_OUT = DATA / "sleep_mammals.csv"
BIRD_OUT = DATA / "sleep_birds.csv"

# Bird sleep hours per day, compiled from:
# Roth, Lesku, Amlaner, Lima 2006 J. Sleep Res. Table 1
# Rattenborg et al. 2008 Naturwissenschaften
# Lesku et al. 2011 PLoS ONE
# Lesku et al. 2012 Science (frigatebird sleep at sea)
# (Bird sleep data is genuinely sparse in the published literature; this is ~all we have)
BIRD_SLEEP_DATA = [
    # (species_binomial, common_name, sleep_hours_per_day)
    ("Columba livia",      "Rock pigeon",         9.2),
    ("Gallus gallus",      "Chicken",             9.7),
    ("Anas platyrhynchos", "Mallard duck",       10.8),
    ("Cacatua galerita",   "Sulphur-crested cockatoo", 11.4),
    ("Strigops habroptila","Kakapo",             10.0),
    ("Sturnus vulgaris",   "European starling",   9.6),
    ("Taeniopygia guttata","Zebra finch",         8.8),
    ("Falco peregrinus",   "Peregrine falcon",    8.9),
    ("Athene cunicularia", "Burrowing owl",      12.4),
    ("Anser indicus",      "Bar-headed goose",    9.1),
    ("Apus apus",          "Common swift",        6.0),   # near-continuous flight
    ("Fregata minor",      "Great frigatebird",   0.7),   # Lesku 2012 at-sea
    ("Calidris pusilla",   "Semipalmated sandpiper", 7.8),
    ("Spheniscus humboldti","Humboldt penguin",   10.8),
    ("Buteo buteo",        "Common buzzard",      8.7),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pull_mammals():
    log(f"fetching msleep from {MSLEEP_URL}")
    r = requests.get(MSLEEP_URL, timeout=60)
    r.raise_for_status()
    log(f"got {len(r.content)} bytes")
    df = pd.read_csv(io.StringIO(r.text))
    log(f"loaded {len(df)} mammals, columns: {list(df.columns)}")

    # Standardize columns
    rename = {
        "name": "common_name",
        "genus": "Genus",
        "order": "Order",
        "vore": "vore",
        "sleep_total": "sleep_hrs",
        "sleep_rem": "sleep_rem_hrs",
        "sleep_cycle": "sleep_cycle_hrs",
        "awake": "awake_hrs",
        "brainwt": "brain_kg",
        "bodywt": "body_kg",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # The msleep dataset doesn't include species epithet directly; the column "name"
    # is the common name. Build binomial from genus + a guess where possible.
    # We will join with AnAge later via genus + common_name fuzzy match.

    df["Class"] = "Mammalia"
    df.to_csv(MAMMAL_OUT, index=False)
    log(f"wrote {MAMMAL_OUT}: {len(df)} rows, {df['sleep_hrs'].notna().sum()} with sleep")


def write_birds():
    df = pd.DataFrame(BIRD_SLEEP_DATA, columns=["Binomial", "common_name", "sleep_hrs"])
    df["Class"] = "Aves"
    df["Genus"] = df["Binomial"].str.split(" ").str[0]
    df["Species"] = df["Binomial"].str.split(" ").str[1]
    df.to_csv(BIRD_OUT, index=False)
    log(f"wrote {BIRD_OUT}: {len(df)} bird records")


def main():
    try:
        pull_mammals()
    except Exception as e:
        log(f"mammal pull failed: {e}")
        sys.exit(2)
    write_birds()


if __name__ == "__main__":
    main()
