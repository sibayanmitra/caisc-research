#!/usr/bin/env python3
"""Add typologically diverse languages: Turkish (agglutinative) and Chinese (isolating).
Pull -> tokenize (all 6) -> fit log-log + CSN. Appends to notes/alpha_table_extra.csv.

Turkish: Latin script, whitespace words -> clean word baseline, predicts HIGH fertility
         and strong steepening (tests the mechanism hardest).
Chinese: no whitespace word units -> word baseline is ill-defined (reported as a
         stress case; tokenizer alphas still computable).
"""
import pickle, re, csv
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTES = ROOT / "notes"
TARGET_CHARS = 30_000_000

LANGS = [("tr", "20231101.tr"), ("zh", "20231101.zh")]
RANK_MIN, RANK_MAX = 10, 10_000


def log(m): print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)


def pull(lang, config, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size >= TARGET_CHARS * 0.9:
        log(f"{lang}: cached"); return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", config, split="train", streaming=True)
    total = 0
    with open(out, "w", encoding="utf-8") as fh:
        for ex in ds:
            t = ex["text"]; fh.write(t); fh.write("\n"); total += len(t)
            if total >= TARGET_CHARS: break
    log(f"{lang}: pulled {total/1e6:.1f} MB")


def tok_word(t): return re.findall(r"\w+", t.lower(), flags=re.UNICODE)
def tok_byte(t): return list(t.encode("utf-8"))
def tok_tik(t, enc):
    import tiktoken; return tiktoken.get_encoding(enc).encode(t, disallowed_special=())
def tok_hf(t, m):
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(m); ids = []
    for i in range(0, len(t), 200_000): ids += tk.encode(t[i:i+200_000], add_special_tokens=False)
    return ids

TOKS = [("word", tok_word), ("gpt2_bpe", lambda t: tok_tik(t,"gpt2")),
        ("cl100k", lambda t: tok_tik(t,"cl100k_base")), ("o200k", lambda t: tok_tik(t,"o200k_base")),
        ("qwen2.5", lambda t: tok_hf(t,"Qwen/Qwen2.5-1.5B")), ("byte", tok_byte)]


def alpha_loglog(counts):
    counts = np.sort(np.asarray(counts, float))[::-1]
    ranks = np.arange(1, len(counts)+1); lo, hi = RANK_MIN, min(RANK_MAX, len(counts))
    if hi-lo < 50: return float("nan")
    A = np.column_stack([np.log(ranks[lo:hi]), np.ones(hi-lo)])
    return -float(np.linalg.lstsq(A, np.log(counts[lo:hi]), rcond=None)[0][0])


def alpha_csn(counts):
    try:
        import powerlaw
        arr = np.asarray(counts, float); arr = arr[arr>0]
        f = powerlaw.Fit(arr, discrete=True, verbose=False)
        return float(f.alpha), float(f.xmin)
    except Exception: return float("nan"), float("nan")


rows = []
for lang, cfg in LANGS:
    out = DATA / f"corpus_{lang}/text.txt"
    pull(lang, cfg, out)
    text = open(out, encoding="utf-8").read(TARGET_CHARS)
    log(f"{lang}: {len(text):,} chars")
    for tname, tfn in TOKS:
        pkl = DATA / "token_counts" / f"{tname}_{lang}.pkl"
        if pkl.exists():
            ctr = pickle.load(open(pkl, "rb"))
        else:
            ctr = Counter(tfn(text)); pickle.dump(ctr, open(pkl, "wb"))
        counts = np.fromiter(ctr.values(), int)
        al = alpha_loglog(counts); ac, xm = alpha_csn(counts)
        rows.append({"tokenizer": tname, "lang": lang, "total_tokens": int(counts.sum()),
                     "vocab": len(counts), "alpha_loglog": round(al,4),
                     "alpha_csn": round(ac,4), "xmin_csn": round(xm,1)})
        log(f"  {tname:10s} {lang}: n={int(counts.sum()):,} a_ll={al:.3f} a_csn={ac:.3f} xmin={xm:.0f}")

with open(NOTES/"alpha_table_extra.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
log(f"wrote {NOTES/'alpha_table_extra.csv'}")
