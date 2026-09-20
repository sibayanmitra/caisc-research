#!/usr/bin/env python3
"""Expand downstream n=4 → n=7 by adding Spanish, Italian, Finnish.
All three have clean whitespace word boundaries (like en/de/fr/tr).
Computes Δα_mid (qwen2.5 vs word) + bpb from Qwen2.5-1.5B.
Saves downstream_bpb_expanded.csv and downstream_expanded_result.json.
Run with: CUDA_VISIBLE_DEVICES=2 python scripts/10_expand_downstream.py
"""
import pickle, re, csv, math, json
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTES = ROOT / "notes"
TARGET_CHARS = 30_000_000
RANK_MIN, RANK_MAX = 10, 10_000

NEW_LANGS = [("es", "20231101.es"), ("it", "20231101.it"), ("fi", "20231101.fi")]


def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def pull(lang, config, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size >= TARGET_CHARS * 0.5:
        log(f"{lang}: corpus cached"); return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for ex in ds:
            t = ex["text"]; fh.write(t); fh.write("\n"); total += len(t)
            if total >= TARGET_CHARS: break
    log(f"{lang}: pulled {total/1e6:.1f} MB")


def tok_word(t):
    return re.findall(r"\w+", t.lower(), flags=re.UNICODE)


def tok_hf(t, model_name):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(model_name)
    ids = []
    for i in range(0, len(t), 200_000):
        ids += tk.encode(t[i:i+200_000], add_special_tokens=False)
    return ids


def alpha_loglog(counts):
    counts = np.sort(np.asarray(counts, float))[::-1]
    ranks = np.arange(1, len(counts) + 1)
    lo, hi = RANK_MIN, min(RANK_MAX, len(counts))
    if hi - lo < 50:
        return float("nan")
    A = np.column_stack([np.log(ranks[lo:hi]), np.ones(hi - lo)])
    return -float(np.linalg.lstsq(A, np.log(counts[lo:hi]), rcond=None)[0][0])


# ── Phase 1: tokenize + fit alpha ─────────────────────────────────────────────
log("=== Phase 1: tokenize + fit alpha ===")
new_alpha = {}  # (lang, tok) -> alpha_loglog

for lang, cfg in NEW_LANGS:
    out = DATA / f"corpus_{lang}/text.txt"
    pull(lang, cfg, out)
    text = open(out, encoding="utf-8").read(TARGET_CHARS)
    log(f"{lang}: {len(text):,} chars loaded")

    for tname, tfn in [("word", tok_word),
                       ("qwen2.5", lambda t: tok_hf(t, "Qwen/Qwen2.5-1.5B"))]:
        pkl = DATA / "token_counts" / f"{tname}_{lang}.pkl"
        if pkl.exists():
            ctr = pickle.load(open(pkl, "rb"))
            log(f"  {tname:10s} {lang}: loaded from cache")
        else:
            ctr = Counter(tfn(text))
            (DATA / "token_counts").mkdir(parents=True, exist_ok=True)
            pickle.dump(ctr, open(pkl, "wb"))
        counts = np.fromiter(ctr.values(), int)
        al = alpha_loglog(counts)
        new_alpha[(lang, tname)] = al
        log(f"  {tname:10s} {lang}: n_types={len(counts):,}  alpha_loglog={al:.4f}")

# ── Phase 2: bits-per-byte ────────────────────────────────────────────────────
log("\n=== Phase 2: bits-per-byte with Qwen2.5-1.5B ===")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer as HFTok

MODEL = "Qwen/Qwen2.5-1.5B"
BYTES_CAP = 2_000_000
WIN = 1024

log("Loading model...")
htok = HFTok.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()

new_bpb = {}
for lang, _ in NEW_LANGS:
    p = DATA / f"corpus_{lang}/text.txt"
    text = open(p, encoding="utf-8").read()[: BYTES_CAP * 2]
    nbytes = len(text.encode("utf-8"))
    ids = htok(text, return_tensors="pt").input_ids[0]
    total_nll = 0.0; total_tok = 0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, WIN):
            chunk = ids[i: i + WIN + 1].unsqueeze(0).to("cuda:0")
            if chunk.shape[1] < 2: break
            out = model(chunk[:, :-1])
            logits = out.logits[0].float()
            tgt = chunk[0, 1:]
            nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
            total_nll += float(nll); total_tok += tgt.numel()
    bpb = (total_nll / math.log(2)) / nbytes
    new_bpb[lang] = round(bpb, 4)
    log(f"  {lang}: bpb={bpb:.4f}  (nbytes={nbytes:,}, ntok={total_tok:,})")

# ── Phase 3: combined correlation ─────────────────────────────────────────────
log("\n=== Phase 3: combined correlation ===")
import pandas as pd

# Load existing alpha table
df = pd.read_csv(NOTES / "alpha_table.csv")

def get_alpha(lang, tok):
    rows = df[(df.lang == lang) & (df.tokenizer == tok)]
    if len(rows):
        return float(rows.iloc[0]["alpha_loglog"])
    return new_alpha.get((lang, tok), float("nan"))

# Load existing bpb
existing_bpb = {}
with open(NOTES / "downstream_bpb.csv") as f:
    for row in csv.DictReader(f):
        existing_bpb[row["lang"]] = float(row["bits_per_byte"])

# Main languages (clean word baselines)
main_langs = ["en", "de", "fr", "tr"] + [l for l, _ in NEW_LANGS]
points = []
for lang in main_langs:
    a_word = get_alpha(lang, "word")
    a_qwen = get_alpha(lang, "qwen2.5")
    bpb = existing_bpb.get(lang, new_bpb.get(lang, float("nan")))
    if any(np.isnan(x) for x in [a_word, a_qwen, bpb]):
        log(f"  {lang}: SKIP (nan: word={a_word:.3f}, qwen={a_qwen:.3f}, bpb={bpb})")
        continue
    da = a_qwen - a_word
    points.append({"lang": lang, "da_mid": round(da, 4), "bpb": bpb,
                   "alpha_word": round(a_word, 4), "alpha_qwen": round(a_qwen, 4)})
    log(f"  {lang}: Δα={da:.4f}  bpb={bpb:.4f}")

da_arr = np.array([p["da_mid"] for p in points])
bpb_arr = np.array([p["bpb"] for p in points])
r_val = float(np.corrcoef(da_arr, bpb_arr)[0, 1])
log(f"\nr(Δα_mid, bpb) = {r_val:.4f}  (n={len(points)} languages)")

# ── Save outputs ──────────────────────────────────────────────────────────────
# Expanded bpb CSV
expanded = []
with open(NOTES / "downstream_bpb.csv") as f:
    expanded = list(csv.DictReader(f))
for lang, _ in NEW_LANGS:
    expanded.append({
        "lang": lang, "bytes": "N/A", "tokens": "N/A",
        "bits_per_byte": new_bpb.get(lang, "N/A"), "bits_per_token": "N/A"
    })
with open(NOTES / "downstream_bpb_expanded.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=expanded[0].keys())
    w.writeheader(); w.writerows(expanded)
log(f"Wrote downstream_bpb_expanded.csv")

result = {
    "n_langs": len(points),
    "langs": [p["lang"] for p in points],
    "r_da_mid_bpb": round(r_val, 4),
    "per_lang": points,
}
with open(NOTES / "downstream_expanded_result.json", "w") as f:
    json.dump(result, f, indent=2)
log(f"Wrote downstream_expanded_result.json")
log("=== DONE ===")
