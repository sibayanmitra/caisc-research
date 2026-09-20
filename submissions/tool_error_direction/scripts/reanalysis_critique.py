#!/usr/bin/env python3
"""Paper-4 critique re-analysis (CPU).
(1) Balanced accuracy + balanced shuffle floor at the best layer  -> diagnoses G-SHUF=0.52.
(2) Per-tool give-up-vs-continue direction alignment               -> tests the 'shared action axis' reframe.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

ROOT = Path("/home/anonymous/agentic_paper")
sys.path.insert(0, str(ROOT))
from src.probe import _action_balance, encode_labels
from src.prompts import get_split_assignment

ACT = ROOT / "data" / "activations"
splits = get_split_assignment()


def load_model(ms, seeds=(1, 2, 3)):
    Hs, metas = [], []
    for s in seeds:
        npz = ACT / f"{ms}_seed{s}.npz"
        mp = ACT / f"{ms}_seed{s}.meta.json"
        if not npz.exists():
            continue
        Hs.append(np.load(npz)["H"])
        m = json.loads(mp.read_text())
        metas += m
    return np.concatenate(Hs, 0), metas


def build_xy(H, metas, sf, pos=0):
    keep = [i for i, m in enumerate(metas)
            if m["label"] != "ambiguous" and m["outcome"] != "success"
            and splits.get(m["base_prompt_id"]) in sf]
    keep = np.array(keep, int)
    X = H[keep, :, pos, :]
    y = encode_labels([metas[i]["label"] for i in keep])
    return X, y


models = sorted({p.name.split("_seed")[0] for p in ACT.glob("*_seed*.npz")})
print("models:", models)
_, m0 = load_model(models[0], (1,))
print("meta keys:", list(m0[0].keys()))

for ms in models:
    H, metas = load_model(ms)
    Xtr, ytr = build_xy(H, metas, {"probe_fit"})
    Xte, yte = build_xy(H, metas, {"dir_extract"})
    twoclass = int((ytr == 2).sum()) < 20
    if twoclass:
        ytr = np.where(ytr == 0, 0, 1); yte = np.where(yte == 0, 0, 1); chance = 0.5
    else:
        chance = 1/3
    # best layer by raw test acc
    accs = []
    for L in range(Xtr.shape[1]):
        Xb, yb = _action_balance(Xtr[:, L], ytr, np.random.default_rng(0))
        accs.append(LogisticRegression(C=1.0, max_iter=2000).fit(Xb, yb).score(Xte[:, L], yte))
    L = int(np.argmax(accs))
    Xb, yb = _action_balance(Xtr[:, L], ytr, np.random.default_rng(0))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xb, yb)
    yhat = clf.predict(Xte[:, L])
    raw, bal = (yhat == yte).mean(), balanced_accuracy_score(yte, yhat)
    sr, sb = [], []
    for k in range(25):
        ysh = np.random.default_rng(k).permutation(yb)
        yh = LogisticRegression(C=1.0, max_iter=2000).fit(Xb, ysh).predict(Xte[:, L])
        sr.append((yh == yte).mean()); sb.append(balanced_accuracy_score(yte, yh))
    tc = Counter(yte.tolist())
    print(f"\n=== {ms} ===")
    print(f" twoclass={twoclass} chance={chance:.3f} best_layer={L} n_test={len(yte)}")
    print(f" test class dist={dict(tc)} majority_frac={max(tc.values())/len(yte):.3f}")
    print(f" RAW acc={raw:.3f}   BALANCED acc={bal:.3f}")
    print(f" SHUFFLE raw={np.mean(sr):.3f}+-{np.std(sr):.3f}   BALANCED shuffle={np.mean(sb):.3f}+-{np.std(sb):.3f}")
    if not twoclass:
        allidx = [i for i, m in enumerate(metas) if m["label"] != "ambiguous" and m["outcome"] != "success"]
        Hl = H[allidx, :, 0, :][:, L, :]
        ya = encode_labels([metas[i]["label"] for i in allidx])
        tools = [metas[i].get("tool", "?") for i in allidx]
        dirs = {}
        for t in sorted(set(tools)):
            tm = np.array([x == t for x in tools])
            gu, co = (ya == 2) & tm, (ya != 2) & tm
            if gu.sum() >= 10 and co.sum() >= 10:
                d = Hl[gu].mean(0) - Hl[co].mean(0)
                dirs[t] = d / (np.linalg.norm(d) + 1e-9)
                print(f"   tool {t:12s} giveup={gu.sum():4d} continue={co.sum():4d}  -> direction kept")
            else:
                print(f"   tool {t:12s} giveup={gu.sum():4d} continue={co.sum():4d}  -> SKIP (too few)")
        ts = list(dirs)
        print("   per-tool give-up-vs-continue direction cosines (the 'shared action axis' test):")
        for i in range(len(ts)):
            for j in range(i+1, len(ts)):
                print(f"     cos({ts[i]}, {ts[j]}) = {float(dirs[ts[i]] @ dirs[ts[j]]):+.3f}")
