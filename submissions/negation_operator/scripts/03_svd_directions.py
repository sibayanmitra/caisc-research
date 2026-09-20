#!/usr/bin/env python3
"""
Phase 3 — SVD on activation deltas.

KEY DESIGN: Train d_neg on TRAIN_MORPHOLOGIES (which all use 'not' or n't),
then test cross-morphology transfer to ANTONYM pairs (which contain NO 'not' token).

If d_neg captures logical negation → cosine(d_neg_train, d_neg_antonym) is high.
If d_neg captures "not" token presence → low cosine, no antonym transfer.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from negation_lib import (
    log, get_paper_dir, get_all_morphologies,
    TRAIN_MORPHOLOGIES, TEST_MORPHOLOGIES,
)


def svd_top1(delta):
    """SVD on (n, d) deltas. Returns (top_dir, top_singular_values, ev_ratio)."""
    if delta.shape[0] < 2:
        return np.zeros(delta.shape[1], dtype=np.float32), np.zeros(10, dtype=np.float32), 0.0
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    top_dir = Vt[0]
    top_s = S[:min(10, len(S))]
    ev = float(S[0]**2 / (S**2).sum()) if S.sum() > 0 else 0.0
    return top_dir.astype(np.float32), top_s.astype(np.float32), ev


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    log("=== Paper 2 Phase 3: SVD directions (train vs held-out morphology) ===")
    paper_dir = get_paper_dir()
    act_dir = paper_dir / "activations"
    dir_dir = paper_dir / "directions"
    dir_dir.mkdir(parents=True, exist_ok=True)

    pos = torch.load(act_dir / "pos.pt", weights_only=False)
    neg = torch.load(act_dir / "neg.pt", weights_only=False)
    pos_r = pos["resid"].float().numpy()
    neg_r = neg["resid"].float().numpy()
    meta  = pos["meta"]
    morphs_per_pair = meta["morphologies_in_order"]
    n_pairs, n_layers, d_model = pos_r.shape
    log(f"Loaded: {n_pairs} pairs, {n_layers} layers, d_model={d_model}")

    # Index pairs by morphology
    idx_by_morph = {}
    for i, m in enumerate(morphs_per_pair):
        idx_by_morph.setdefault(m, []).append(i)

    # ── Per-morphology SVD at every layer ────────────────────────────────
    morphs_all = get_all_morphologies() + ["not_nonneg"]
    morphs_present = [m for m in morphs_all if m in idx_by_morph and len(idx_by_morph[m]) >= 2]
    log(f"Morphologies with ≥2 pairs: {morphs_present}")

    # d_per_morph_per_layer: shape (n_morph, n_layers, d_model)
    d_per_morph_per_layer = np.zeros((len(morphs_present), n_layers, d_model), dtype=np.float32)
    ev_per_morph_per_layer = np.zeros((len(morphs_present), n_layers), dtype=np.float32)

    for mi, morph in enumerate(morphs_present):
        idxs = idx_by_morph[morph]
        for L in range(n_layers):
            delta = neg_r[idxs, L, :] - pos_r[idxs, L, :]
            delta = delta - delta.mean(axis=0, keepdims=True)  # center
            top_dir, _, ev = svd_top1(delta)
            d_per_morph_per_layer[mi, L] = top_dir
            ev_per_morph_per_layer[mi, L] = ev

    # ── Train direction: pooled across TRAIN_MORPHOLOGIES ────────────────
    train_idx = []
    for m in TRAIN_MORPHOLOGIES:
        train_idx.extend(idx_by_morph.get(m, []))
    log(f"Train morphologies: {TRAIN_MORPHOLOGIES} → {len(train_idx)} pairs")

    d_neg_train = np.zeros((n_layers, d_model), dtype=np.float32)
    ev_train    = np.zeros(n_layers, dtype=np.float32)
    s_train     = np.zeros((n_layers, 10), dtype=np.float32)
    for L in range(n_layers):
        delta = neg_r[train_idx, L, :] - pos_r[train_idx, L, :]
        # Mean-center within each training morphology to remove per-morph mean
        delta_centered = delta.copy()
        for m in TRAIN_MORPHOLOGIES:
            m_idxs_pos_in_train = [k for k, ti in enumerate(train_idx) if morphs_per_pair[ti] == m]
            if m_idxs_pos_in_train:
                delta_centered[m_idxs_pos_in_train] -= delta_centered[m_idxs_pos_in_train].mean(axis=0)
        top_dir, top_s, ev = svd_top1(delta_centered)
        d_neg_train[L] = top_dir
        ev_train[L]    = ev
        s_len = min(10, len(top_s))
        s_train[L, :s_len] = top_s[:s_len]

    peak_L = int(np.argmax(ev_train))
    log(f"Peak layer (train): {peak_L}, EV: {ev_train[peak_L]:.4f}")

    # ── HELD-OUT TEST: cosine(d_neg_train, d_neg_antonym) per layer ──────
    # The key test of whether d_neg is logical or just "not-token-detector"
    transfer_cosines = {}
    for morph in TEST_MORPHOLOGIES:
        if morph not in morphs_present:
            log(f"  ⚠ Test morph '{morph}' not present in data — skipping")
            continue
        mi = morphs_present.index(morph)
        cos_per_layer = np.array([
            cosine(d_neg_train[L], d_per_morph_per_layer[mi, L])
            for L in range(n_layers)
        ])
        transfer_cosines[morph] = cos_per_layer.tolist()
        log(f"\n  TRANSFER TEST: cos(d_train, d_{morph})")
        log(f"    peak-layer cosine: {cos_per_layer[peak_L]:+.4f}")
        log(f"    max cosine across layers: {cos_per_layer.max():+.4f} at L={cos_per_layer.argmax()}")
        log(f"    {'PASS ✓ logical-negation hypothesis supported' if cos_per_layer[peak_L] > 0.3 else 'WARN ✗ low transfer — direction may be lexical not logical'}")

    # ── CONFOUND CONTROL: d_neg vs d_not_nonneg ──────────────────────────
    # If d_neg is real logical-NOT, it should be ORTHOGONAL or weak on not_nonneg
    if "not_nonneg" in morphs_present:
        mi = morphs_present.index("not_nonneg")
        cos_nonneg = np.array([
            cosine(d_neg_train[L], d_per_morph_per_layer[mi, L])
            for L in range(n_layers)
        ])
        transfer_cosines["not_nonneg_should_be_LOW"] = cos_nonneg.tolist()
        log(f"\n  CONFOUND TEST: cos(d_train, d_not_nonneg) at peak L: {cos_nonneg[peak_L]:+.4f}")
        verdict = "PASS - low cosine = d_neg ignores non-negation 'not'" if abs(cos_nonneg[peak_L]) < 0.3 else "WARN - d_neg may track lexical 'not'"
        log(f"    {verdict}")

    # ── Per-train-morphology cross-cosines (within train set) ────────────
    train_morph_present = [m for m in TRAIN_MORPHOLOGIES if m in morphs_present]
    train_cross = np.zeros((len(train_morph_present), len(train_morph_present)), dtype=np.float32)
    for i, mi_name in enumerate(train_morph_present):
        for j, mj_name in enumerate(train_morph_present):
            mi = morphs_present.index(mi_name)
            mj = morphs_present.index(mj_name)
            train_cross[i, j] = cosine(d_per_morph_per_layer[mi, peak_L],
                                        d_per_morph_per_layer[mj, peak_L])

    log("\nCross-morphology cosines at peak layer (training morphs only):")
    log("                " + " ".join(f"{m[:6]:>7s}" for m in train_morph_present))
    for i, m in enumerate(train_morph_present):
        log(f"  {m[:14]:14s} " + " ".join(f"{train_cross[i,j]:+7.3f}" for j in range(len(train_morph_present))))

    # ── Save ─────────────────────────────────────────────────────────────
    np.save(dir_dir / "d_neg_train.npy", d_neg_train)
    np.save(dir_dir / "d_per_morph_per_layer.npy", d_per_morph_per_layer)
    np.save(dir_dir / "ev_train.npy", ev_train)
    np.save(dir_dir / "singular_values_train.npy", s_train)
    np.save(dir_dir / "train_morph_cross_cosines.npy", train_cross)

    # also expose under the legacy filenames so 04/05/08 keep working
    np.save(dir_dir / "d_neg_per_layer.npy", d_neg_train)
    np.save(dir_dir / "explained_variance.npy", ev_train)
    np.save(dir_dir / "singular_values_per_layer.npy", s_train)
    # per-category direction at peak layer only, shape (n_morphs, d_model) — for 05_cross_category
    d_neg_per_cat_peak = d_per_morph_per_layer[:, peak_L, :]  # (n_morphs, d_model)
    np.save(dir_dir / "d_neg_per_category.npy", d_neg_per_cat_peak)
    # Save mapping of category index → morphology name so 05 can label rows correctly
    np.save(dir_dir / "morph_index_to_name.npy", np.array(morphs_present, dtype=object))

    probe_eval = {
        "peak_layer": peak_L,
        "peak_ev": float(ev_train[peak_L]),
        "ev_per_layer": ev_train.tolist(),
        "train_morphologies": TRAIN_MORPHOLOGIES,
        "test_morphologies":  TEST_MORPHOLOGIES,
        "morphs_present": morphs_present,
        "transfer_cosines_per_layer": transfer_cosines,
        "train_cross_cosines_at_peak": train_cross.tolist(),
        "design_principle": (
            "d_neg trained on TRAIN morphologies (uses 'not'/n't). "
            "Transfer to TEST morphologies (antonyms, NO 'not' token) tests "
            "logical-negation vs literal-'not'-token interpretation."
        ),
    }
    with open(dir_dir / "probe_eval.json", "w") as f:
        json.dump(probe_eval, f, indent=2)

    log(f"\n✓ Phase 3 done. Peak layer: {peak_L}. Transfer-to-antonym is the critical test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
