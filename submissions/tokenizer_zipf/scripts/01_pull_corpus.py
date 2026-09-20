#!/usr/bin/env python3
"""Pull text samples for English, French, German, Hindi.

Uses HuggingFace `datasets` library. The default 'wikipedia' configs require
apache_beam which is large; we use a streaming-friendly alternative:
`wikimedia/wikipedia` (single-file Parquet streaming).
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# How much text per language (in characters, very approximately)
EN_TARGET = 200_000_000   # 200 MB
NON_EN_TARGET = 50_000_000   # 50 MB

LANGS = [
    ("en", "20231101.en", EN_TARGET, DATA / "corpus_en/text.txt"),
    ("fr", "20231101.fr", NON_EN_TARGET, DATA / "corpus_fr/text.txt"),
    ("de", "20231101.de", NON_EN_TARGET, DATA / "corpus_de/text.txt"),
    ("hi", "20231101.hi", NON_EN_TARGET, DATA / "corpus_hi/text.txt"),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def pull_streaming(lang, config, target_chars, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size >= target_chars * 0.9:
        log(f"  {lang}: cached at {out_path.stat().st_size / 1e6:.1f} MB, skipping")
        return
    from datasets import load_dataset

    log(f"  {lang}: streaming wikimedia/wikipedia config={config}")
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    total = 0
    n_articles = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for ex in ds:
            text = ex.get("text", "")
            if not text:
                continue
            fh.write(text)
            fh.write("\n\n")
            total += len(text) + 2
            n_articles += 1
            if total >= target_chars:
                break
            if n_articles % 5000 == 0:
                log(f"    {lang}: {n_articles} articles, {total / 1e6:.0f} MB")
    log(f"  {lang}: done, {n_articles} articles, {total / 1e6:.1f} MB at {out_path}")


def main():
    for lang, config, target, out_path in LANGS:
        try:
            pull_streaming(lang, config, target, out_path)
        except Exception as e:
            log(f"  {lang}: FAILED — {e}")
            # If en fails we cannot continue; the others are optional
            if lang == "en":
                sys.exit(2)


if __name__ == "__main__":
    main()
