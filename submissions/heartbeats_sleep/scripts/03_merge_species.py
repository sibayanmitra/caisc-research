#!/usr/bin/env python3
"""Merge AnAge (heart-rate proxy = metabolic rate, lifespan, mass) with sleep data.

Matching strategy:
- Mammals (msleep): match by genus + closest log(body mass) to disambiguate species
- Birds (curated table): exact binomial match against AnAge Genus + Species
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ANAGE = DATA / "anage_raw.csv"
SLEEP_M = DATA / "sleep_mammals.csv"
SLEEP_B = DATA / "sleep_birds.csv"
OUT = DATA / "merged.csv"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    anage = pd.read_csv(ANAGE)
    sleep_m = pd.read_csv(SLEEP_M)
    sleep_b = pd.read_csv(SLEEP_B)
    log(f"AnAge: {len(anage)} | sleep_mammals: {len(sleep_m)} | sleep_birds: {len(sleep_b)}")

    # Filter AnAge to viable rows
    anage = anage[anage["Class"].isin(["Mammalia", "Aves"])].copy()
    anage = anage.dropna(subset=["Metabolic rate (W)", "Maximum longevity (yrs)", "Body mass (g)"])
    log(f"AnAge after filter: {len(anage)} mammals+birds with MR + lifespan + mass")

    # Standardise Genus capitalisation
    anage["Genus"] = anage["Genus"].str.strip().str.capitalize()
    sleep_m["Genus"] = sleep_m["Genus"].str.strip().str.capitalize()
    sleep_b["Genus"] = sleep_b["Genus"].str.strip().str.capitalize()

    # MATCH MAMMALS BY GENUS + CLOSEST BODY MASS ----------------------------
    sleep_m["body_g"] = sleep_m["body_kg"] * 1000.0
    anage_m = anage[anage["Class"] == "Mammalia"]

    matched_rows = []
    for _, row in sleep_m.iterrows():
        cand = anage_m[anage_m["Genus"] == row["Genus"]]
        if len(cand) == 0:
            continue
        cand = cand.copy()
        cand["mass_log_diff"] = np.abs(np.log(cand["Body mass (g)"]) - np.log(row["body_g"]))
        best = cand.sort_values("mass_log_diff").iloc[0]
        matched_rows.append({
            "Class": "Mammalia",
            "Genus": row["Genus"],
            "Species": best["Species"],
            "common_name_sleep": row["common_name"],
            "common_name_anage": best["Common name"],
            "sleep_hrs": row["sleep_hrs"],
            "vore": row.get("vore", None),
            "body_g": best["Body mass (g)"],
            "body_g_msleep": row["body_g"],
            "max_lifespan_yrs": best["Maximum longevity (yrs)"],
            "metabolic_rate_W": best["Metabolic rate (W)"],
            "Order": best["Order"],
        })
    log(f"matched mammals: {len(matched_rows)}")

    # MATCH BIRDS BY EXACT BINOMIAL ----------------------------------------
    anage_b = anage[anage["Class"] == "Aves"]
    for _, row in sleep_b.iterrows():
        cand = anage_b[(anage_b["Genus"] == row["Genus"]) & (anage_b["Species"] == row["Species"])]
        if len(cand) == 0:
            # fall back to genus alone
            cand = anage_b[anage_b["Genus"] == row["Genus"]]
            if len(cand) == 0:
                continue
        best = cand.iloc[0]
        matched_rows.append({
            "Class": "Aves",
            "Genus": row["Genus"],
            "Species": best["Species"],
            "common_name_sleep": row["common_name"],
            "common_name_anage": best["Common name"],
            "sleep_hrs": row["sleep_hrs"],
            "vore": None,
            "body_g": best["Body mass (g)"],
            "body_g_msleep": None,
            "max_lifespan_yrs": best["Maximum longevity (yrs)"],
            "metabolic_rate_W": best["Metabolic rate (W)"],
            "Order": best["Order"],
        })

    merged = pd.DataFrame(matched_rows)
    # Derived columns
    SECONDS_PER_YEAR = 365.25 * 86400.0
    merged["lifetime_energy_J"] = merged["metabolic_rate_W"] * SECONDS_PER_YEAR * merged["max_lifespan_yrs"]
    merged["log_lifetime_energy"] = np.log(merged["lifetime_energy_J"])
    merged["log_body_kg"] = np.log(merged["body_g"] / 1000.0)
    merged["log_sleep_hrs"] = np.log(merged["sleep_hrs"])
    merged["log_activity_hrs"] = np.log(24.0 - merged["sleep_hrs"])
    merged["log_lifespan"] = np.log(merged["max_lifespan_yrs"])
    merged["log_MR"] = np.log(merged["metabolic_rate_W"])

    merged.to_csv(OUT, index=False)
    log(f"wrote {OUT}: {len(merged)} rows")
    log(f"  by class: {dict(merged['Class'].value_counts())}")
    log(f"  sleep range: {merged['sleep_hrs'].min():.1f} - {merged['sleep_hrs'].max():.1f} hrs")
    log(f"  log_lifetime_energy range: {merged['log_lifetime_energy'].min():.1f} - {merged['log_lifetime_energy'].max():.1f}")


if __name__ == "__main__":
    main()
