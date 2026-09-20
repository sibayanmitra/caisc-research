#!/usr/bin/env python3
"""Downstream connection (#2): does the Zipf shift predict a real modelling cost?

Compute bits-per-byte of a multilingual LM (Qwen2.5-1.5B) on each language corpus,
then (offline) correlate with that tokenizer's Δα. bits-per-byte is tokenizer/length-
normalised, so it is comparable across languages and is the natural 'token tax' cost.
"""
import math
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; NOTES = ROOT / "notes"
MODEL = "Qwen/Qwen2.5-1.5B"
LANGS = ["en", "de", "fr", "tr", "zh", "hi"]
BYTES_CAP = 2_000_000   # ~2 MB of text per language
WIN = 1024

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

rows = []
for lang in LANGS:
    p = DATA / f"corpus_{lang}/text.txt"
    if not p.exists():
        print(f"{lang}: no corpus, skip", flush=True); continue
    text = open(p, encoding="utf-8").read()[: BYTES_CAP * 2]
    nbytes = len(text.encode("utf-8"))
    ids = tok(text, return_tensors="pt").input_ids[0]
    total_nll = 0.0; total_tok = 0
    with torch.no_grad():
        for i in range(0, len(ids) - 1, WIN):
            chunk = ids[i : i + WIN + 1].unsqueeze(0).to("cuda:0")
            if chunk.shape[1] < 2: break
            out = model(chunk[:, :-1])
            logits = out.logits[0].float()
            tgt = chunk[0, 1:]
            nll = torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
            total_nll += float(nll); total_tok += tgt.numel()
    bpb = (total_nll / math.log(2)) / nbytes
    bpt = (total_nll / math.log(2)) / total_tok
    rows.append({"lang": lang, "bytes": nbytes, "tokens": total_tok,
                 "bits_per_byte": round(bpb, 4), "bits_per_token": round(bpt, 4)})
    print(f"{lang}: bits/byte={bpb:.4f}  bits/token={bpt:.4f}  (n_tok={total_tok})", flush=True)

import csv
with open(NOTES / "downstream_bpb.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(f"wrote {NOTES/'downstream_bpb.csv'}")
