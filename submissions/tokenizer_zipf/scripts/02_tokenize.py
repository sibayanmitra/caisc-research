#!/usr/bin/env python3
"""Tokenize each language corpus under each tokenizer; save Counter per pair."""
import os
import pickle
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COUNTS = DATA / "token_counts"
COUNTS.mkdir(parents=True, exist_ok=True)

LANG_FILES = {
    "en": DATA / "corpus_en/text.txt",
    "fr": DATA / "corpus_fr/text.txt",
    "de": DATA / "corpus_de/text.txt",
    "hi": DATA / "corpus_hi/text.txt",
}

# How many characters to read per language (cap to bound runtime)
CHAR_CAP_EN = 150_000_000  # 150 MB
CHAR_CAP_OTHER = 30_000_000  # 30 MB


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def read_corpus(path, cap):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read(cap)
    return text


def tokenize_word_level(text):
    """Lowercase + split on whitespace and word boundaries."""
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def tokenize_byte(text):
    return list(text.encode("utf-8"))


def tokenize_tiktoken(text, encoding_name):
    import tiktoken
    enc = tiktoken.get_encoding(encoding_name)
    return enc.encode(text, disallowed_special=())


def tokenize_hf(text, model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    # Avoid streaming; just slice into chunks of 200k chars
    chunk = 200_000
    all_ids = []
    for i in range(0, len(text), chunk):
        ids = tok.encode(text[i:i + chunk], add_special_tokens=False)
        all_ids.extend(ids)
    return all_ids


TOKENIZERS = [
    ("word",       lambda t: tokenize_word_level(t)),
    ("gpt2_bpe",   lambda t: tokenize_tiktoken(t, "gpt2")),
    ("cl100k",     lambda t: tokenize_tiktoken(t, "cl100k_base")),
    ("o200k",      lambda t: tokenize_tiktoken(t, "o200k_base")),
    ("qwen2.5",    lambda t: tokenize_hf(t, "Qwen/Qwen2.5-1.5B")),
    ("byte",       lambda t: tokenize_byte(t)),
]


def main():
    do_langs = sys.argv[1:] if len(sys.argv) > 1 else list(LANG_FILES.keys())
    for lang in do_langs:
        path = LANG_FILES[lang]
        if not path.exists():
            log(f"  {lang}: SKIP — corpus not found at {path}")
            continue
        cap = CHAR_CAP_EN if lang == "en" else CHAR_CAP_OTHER
        log(f"reading {lang} from {path} (cap {cap // 1_000_000} MB)")
        text = read_corpus(path, cap)
        log(f"  {lang}: {len(text):,} chars in memory")
        for tname, tfn in TOKENIZERS:
            out = COUNTS / f"{tname}_{lang}.pkl"
            if out.exists():
                log(f"  [{tname}/{lang}] cached")
                continue
            try:
                t0 = datetime.now()
                tokens = tfn(text)
                dt = (datetime.now() - t0).total_seconds()
                log(f"  [{tname}/{lang}] {len(tokens):,} tokens in {dt:.1f}s")
                ctr = Counter(tokens)
                with open(out, "wb") as fh:
                    pickle.dump(ctr, fh)
                log(f"  [{tname}/{lang}] vocab seen: {len(ctr):,}; saved {out}")
            except Exception as e:
                log(f"  [{tname}/{lang}] FAILED — {e}")


if __name__ == "__main__":
    main()
