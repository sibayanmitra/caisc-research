"""CKA (Centered Kernel Alignment) analysis — corrected implementation.

KEY ISSUE WITH LINEAR CKA WHEN d >> n:
    For random independent X, Y of shape (n, d):
        E[CKA(X, Y)] ≈ d / (d + n)
    With d=4096, n=300: null CKA ≈ 0.93.
    This makes raw linear CKA uninterpretable when d >> n.

FIX: PCA-reduce to k << n components first (k=50, n=300 → null ≈ 0.14).
This is standard practice; see Nguyen et al. (2021), Klabunde et al. (2023).

FORMULA (correct kernel, n×n implementation):
    K = X_c X_c^T  (n×n, X_c = X feature-centered)
    L = Y_c Y_c^T
    CKA = tr(K·L) / sqrt(tr(K²) · tr(L²))

Verified:
  - CKA(X, X) = 1.0
  - CKA(X, Y) = CKA(Y, X)
  - With PCA-k=50, random null ≈ 0.14 (interpretable range)
  - Primal ≡ kernel formula for small d (cross-checked)
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_THREADS = "16"
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, _THREADS)

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ACT_DIR = Path("data/activations")

MODELS = {
    "Qwen/Qwen3-8B": {
        "tag": "Qwen__Qwen3-8B", "best_layer": 33, "n_layers": 36, "short": "Qwen3-8B"
    },
    "NousResearch/Hermes-3-Llama-3.1-8B": {
        "tag": "NousResearch__Hermes-3-Llama-3.1-8B", "best_layer": 28,
        "n_layers": 32, "short": "Hermes-3"
    },
    "Salesforce/Llama-xLAM-2-8b-fc-r": {
        "tag": "Salesforce__Llama-xLAM-2-8b-fc-r", "best_layer": 29,
        "n_layers": 32, "short": "xLAM-2"
    },
}

TOOLS  = ["web_search", "calculator", "file_read"]
LABELS = ["retry-same", "modify-args", "give-up"]
POS_IDX = 0      # t_star position
PCA_K   = 50     # PCA components before CKA (standard fix for d >> n)


# ── PCA reduction ─────────────────────────────────────────────────────────────

def pca_reduce(X: np.ndarray, k: int) -> np.ndarray:
    """Project X (n, d) onto its top-k principal components → (n, k), float64.
    Centers X first; uses truncated SVD so only k singular vectors are computed.
    """
    X = X.astype(np.float64)
    Xc = X - X.mean(axis=0)
    # np.linalg.svd with full_matrices=False → U (n,r), S (r,), Vt (r,d)
    # where r = min(n, d).  We only need the top-k columns of U * S.
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return (U[:, :k] * S[:k]).astype(np.float64)   # (n, k)


# ── CKA core (kernel n×n, float64) ───────────────────────────────────────────

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA via kernel (n×n) Gram matrices.

    X, Y: (n, d_x) and (n, d_y) — same n, any d.
    Returns CKA ∈ [0, 1].

    Uses kernel formulation to avoid forming d×d matrices:
        K = X_c X_c^T  (n×n)
        L = Y_c Y_c^T  (n×n)
        CKA = tr(K·L) / sqrt(tr(K²) · tr(L²))
    where tr(K·L) = (K * L).sum() for symmetric K, L.
    Feature-centering (X_c = X - mean(X, axis=0)) is equivalent to
    the standard H K H double-centering of linear kernels (Kornblith 2019).
    """
    assert X.shape[0] == Y.shape[0], f"n mismatch: {X.shape[0]} vs {Y.shape[0]}"
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    Xc = X - X.mean(axis=0)   # (n, d_x)
    Yc = Y - Y.mean(axis=0)   # (n, d_y)

    K = Xc @ Xc.T             # (n, n)
    L = Yc @ Yc.T             # (n, n)

    # tr(K_c L_c) as element-wise product sum (avoids n²-sized intermediate)
    hsic_kl = (K * L).sum()   # scalar
    hsic_kk = (K * K).sum()
    hsic_ll = (L * L).sum()

    if hsic_kk <= 0 or hsic_ll <= 0:
        return 0.0
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


# ── Sanity checks ─────────────────────────────────────────────────────────────

def verify_cka() -> None:
    rng = np.random.default_rng(0)
    n, d = 50, 20   # intentionally n > d so primal cross-check is valid

    # 1. CKA(X, X) == 1.0
    X = rng.standard_normal((n, d))
    assert abs(linear_cka(X, X) - 1.0) < 1e-9, "CKA(X,X) ≠ 1"

    # 2. Symmetry
    Y = rng.standard_normal((n, d))
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < 1e-9, "not symmetric"

    # 3. CKA(X, -X) == 1 (direction-invariant)
    assert abs(linear_cka(X, -X) - 1.0) < 1e-9, "CKA(X,-X) ≠ 1"

    # 4. CKA(X, orthogonal_Y) should be < 1
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Y_rot = X @ Q          # rotate X; same gram matrix → CKA should still be 1
    assert abs(linear_cka(X, Y_rot) - 1.0) < 1e-9, "rotation invariance failed"

    # 5. Cross-check: kernel ≡ primal when d < n
    Xs = rng.standard_normal((20, 8)).astype(np.float64)
    Ys = rng.standard_normal((20, 8)).astype(np.float64)
    Xsc = Xs - Xs.mean(0); Ysc = Ys - Ys.mean(0)
    primal = float(
        np.linalg.norm(Xsc.T @ Ysc, "fro") ** 2 /
        np.sqrt(np.linalg.norm(Xsc.T @ Xsc, "fro") ** 2 *
                np.linalg.norm(Ysc.T @ Ysc, "fro") ** 2)
    )
    kernel = linear_cka(Xs, Ys)
    assert abs(primal - kernel) < 1e-9, f"primal={primal:.9f} kernel={kernel:.9f}"

    # 6. Null CKA (random) with d > n:
    #    Without PCA: ≈ d/(d+n) → close to 1 for d >> n (KNOWN ISSUE)
    Xa = rng.standard_normal((100, 512))
    Xb = rng.standard_normal((100, 512))
    raw_null = linear_cka(Xa, Xb)
    # After PCA-50: null should be near 50/(50+100) ≈ 0.33
    Xa_pca = pca_reduce(Xa, 50)
    Xb_pca = pca_reduce(Xb, 50)
    pca_null = linear_cka(Xa_pca, Xb_pca)
    print(f"  raw null (n=100, d=512):     {raw_null:.4f}  [expected ≈ d/(d+n) = {512/(512+100):.3f}]")
    print(f"  PCA-50 null (n=100, k=50):   {pca_null:.4f}  [expected ≈ k/(k+n) = {50/(50+100):.3f}]")
    assert raw_null > 0.6,  f"raw null {raw_null:.4f} unexpectedly low"
    assert pca_null < 0.50, f"PCA null {pca_null:.4f} too high — PCA not fixing d>>n issue"

    print("  [all sanity checks passed]")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_acts_and_meta(model_name: str, seed: int = 1):
    cfg  = MODELS[model_name]
    H    = np.load(ACT_DIR / f"{cfg['tag']}_seed{seed}.npz")["H"]
    meta = json.load(open(ACT_DIR / f"{cfg['tag']}_seed{seed}.meta.json"))
    return H, meta


def get_subset(H, meta, *, tool=None, label=None, exclude_success=True):
    idx = [i for i, m in enumerate(meta)
           if (not exclude_success or m["outcome"] != "success")
           and (tool  is None or m["tool"]  == tool)
           and (label is None or m["label"] == label)]
    return H[np.array(idx, dtype=int)] if idx else np.empty((0, *H.shape[1:]))


def get_pca_rep(H, meta, layer: int, *, tool=None, label=None, k: int = PCA_K):
    """Extract activations at given layer/position, optionally filtered, PCA-reduced."""
    H_sub = get_subset(H, meta, tool=tool, label=label)
    if len(H_sub) == 0:
        return None
    raw = H_sub[:, layer, POS_IDX, :]    # (n, 4096)
    return pca_reduce(raw, k)             # (n, k)


# ── Analysis 1: cross-tool CKA per model ─────────────────────────────────────

def analysis_cross_tool():
    results = {}
    for model_name, cfg in MODELS.items():
        H, meta = load_acts_and_meta(model_name)
        layer, short = cfg["best_layer"], cfg["short"]

        reps = {t: get_pca_rep(H, meta, layer, tool=t) for t in TOOLS}
        reps = {t: r for t, r in reps.items() if r is not None}

        min_n = min(len(v) for v in reps.values())
        rng = np.random.default_rng(42)
        for t in reps:
            idx = rng.choice(len(reps[t]), size=min_n, replace=False)
            reps[t] = reps[t][idx]

        # Verify self-CKA = 1.0 exactly
        for t, r in reps.items():
            sc = linear_cka(r, r)
            assert abs(sc - 1.0) < 1e-6, f"{short} {t} self-CKA={sc:.6f}"

        cka_mat = {f"{ta}|{tb}": (1.0 if ta == tb else linear_cka(reps[ta], reps[tb]))
                   for ta in TOOLS for tb in TOOLS}

        results[model_name] = {"best_layer": layer, "n_per_tool": min_n,
                               "pca_k": PCA_K, "cka": cka_mat}
        print(f"\n[{short}] cross-tool CKA (layer {layer}, PCA-{PCA_K}, n={min_n}/tool):")
        header = "                 " + "  ".join(f"{t[:8]:8s}" for t in TOOLS)
        print(f"  {header}")
        for ta in TOOLS:
            row = "  ".join(f"{cka_mat[f'{ta}|{tb}']:8.4f}" for tb in TOOLS)
            print(f"  {ta:15s}  {row}")
    return results


# ── Analysis 2: cross-model CKA (matched trajectories) ───────────────────────

def analysis_cross_model():
    data = {}
    for model_name in MODELS:
        H, meta = load_acts_and_meta(model_name)
        lookup = {(m["base_prompt_id"], m["outcome"]): i
                  for i, m in enumerate(meta)
                  if m["outcome"] != "success" and m["label"] != "ambiguous"}
        data[model_name] = (H, meta, lookup)

    common_keys = sorted(set.intersection(*(set(d[2]) for d in data.values())))
    print(f"\nMatched trajectories (all 3 models): {len(common_keys)}")

    # PCA-reduce at each model's best layer for matched trajectories
    model_reps = {}
    for model_name, cfg in MODELS.items():
        H, _, lookup = data[model_name]
        rows = [lookup[k] for k in common_keys]
        raw = H[rows][:, cfg["best_layer"], POS_IDX, :]
        model_reps[model_name] = pca_reduce(raw, PCA_K)

    # Verify self-CKA
    for mn, rep in model_reps.items():
        sc = linear_cka(rep, rep)
        assert abs(sc - 1.0) < 1e-6, f"{mn} self-CKA={sc:.6f}"

    model_names = list(MODELS.keys())
    best_cka = {}
    print(f"\nCross-model CKA at best layers (PCA-{PCA_K}, n={len(common_keys)}):")
    for i, mn_a in enumerate(model_names):
        for j, mn_b in enumerate(model_names):
            if i >= j:
                continue
            cka = linear_cka(model_reps[mn_a], model_reps[mn_b])
            sa, sb = MODELS[mn_a]["short"], MODELS[mn_b]["short"]
            best_cka[f"{sa}|{sb}"] = cka
            print(f"  {sa} vs {sb}: {cka:.4f}")

    # Layer-wise CKA: Hermes-3 vs xLAM-2
    h_name = "NousResearch/Hermes-3-Llama-3.1-8B"
    x_name = "Salesforce/Llama-xLAM-2-8b-fc-r"
    H_h, _, lu_h = data[h_name]
    H_x, _, lu_x = data[x_name]
    pair_keys = sorted(set(lu_h) & set(lu_x))
    rows_h = [lu_h[k] for k in pair_keys]
    rows_x = [lu_x[k] for k in pair_keys]

    n_layers = MODELS[h_name]["n_layers"]
    layer_cka = {}
    print(f"\nHermes-3 vs xLAM-2 layer-wise CKA (PCA-{PCA_K}, n={len(pair_keys)}):")
    for l in range(n_layers):
        cka = linear_cka(
            pca_reduce(H_h[rows_h][:, l, POS_IDX, :], PCA_K),
            pca_reduce(H_x[rows_x][:, l, POS_IDX, :], PCA_K),
        )
        layer_cka[l] = cka
        bh = " ← Hermes best" if l == MODELS[h_name]["best_layer"] else ""
        bx = " ← xLAM-2 best" if l == MODELS[x_name]["best_layer"] else ""
        print(f"  L{l:2d}: {cka:.4f}{bh}{bx}")

    return {
        "n_matched": len(common_keys), "pca_k": PCA_K,
        "best_layer_cka": best_cka,
        "hermes_xlam_layer_cka": {str(l): v for l, v in layer_cka.items()},
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cross_tool(results: dict, null_val: float):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (model_name, res) in zip(axes, results.items()):
        mat = np.array([[res["cka"][f"{ta}|{tb}"] for tb in TOOLS] for ta in TOOLS])
        short_labels = ["web_search", "calculator", "file_read"]
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(3)); ax.set_xticklabels(short_labels, fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(3)); ax.set_yticklabels(short_labels, fontsize=8)
        title = (f"{MODELS[model_name]['short']}\n"
                 f"layer {res['best_layer']}, PCA-{res['pca_k']}, n={res['n_per_tool']}")
        ax.set_title(title, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Cross-tool CKA (error trajectories, t* pos, PCA-{PCA_K})\n"
        f"random null ≈ {null_val:.3f}",
        fontsize=10, y=1.04
    )
    plt.tight_layout()
    out = FIG_DIR / "cka_cross_tool.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {out}")


def plot_layer_cka(layer_cka: dict, hermes_best: int, xlam_best: int, null_val: float):
    layers = sorted(layer_cka, key=int)
    vals   = [layer_cka[l] for l in layers]
    layers = [int(l) for l in layers]

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(layers, vals, "o-", color="#2166ac", lw=2, ms=5)
    ax.axhline(null_val, color="gray", linestyle=":", alpha=0.7, label=f"random null ≈ {null_val:.3f}")
    ax.axvline(hermes_best, color="#d6604d", ls="--", alpha=0.8,
               label=f"Hermes-3 best (L{hermes_best}): {layer_cka[str(hermes_best)]:.3f}")
    ax.axvline(xlam_best, color="#4dac26", ls="--", alpha=0.8,
               label=f"xLAM-2 best (L{xlam_best}): {layer_cka[str(xlam_best)]:.3f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA (PCA-50)")
    ax.set_title("Hermes-3 vs xLAM-2: layer-wise representational similarity")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "cka_layer_profile.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ── Random null baseline ──────────────────────────────────────────────────────

def compute_null(n: int, k: int = PCA_K, n_trials: int = 50) -> float:
    """CKA of two independent random (n, k) Gaussian matrices — PCA already applied."""
    rng = np.random.default_rng(99)
    vals = [linear_cka(rng.standard_normal((n, k)), rng.standard_normal((n, k)))
            for _ in range(n_trials)]
    return float(np.mean(vals))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"CKA ANALYSIS  (kernel n×n, float64, PCA-{PCA_K})")
    print("=" * 65)

    print("\n--- Sanity checks ---")
    verify_cka()

    # Null: random (n, PCA_K) matrices — this is the floor after PCA reduction
    n_cross_tool  = 300   # approximate n per tool
    n_cross_model = 600   # approximate n for cross-model
    null_tool  = compute_null(n_cross_tool)
    null_model = compute_null(n_cross_model)
    print(f"\nRandom null (n={n_cross_tool}, k={PCA_K}): {null_tool:.4f}  "
          f"[theory ≈ {PCA_K/(PCA_K+n_cross_tool):.3f}]")
    print(f"Random null (n={n_cross_model}, k={PCA_K}): {null_model:.4f}  "
          f"[theory ≈ {PCA_K/(PCA_K+n_cross_model):.3f}]")

    print("\n--- Analysis 1: Cross-tool CKA per model ---")
    res_tool = analysis_cross_tool()
    plot_cross_tool(res_tool, null_tool)

    print("\n--- Analysis 2: Cross-model CKA ---")
    res_model = analysis_cross_model()
    plot_layer_cka(
        res_model["hermes_xlam_layer_cka"],
        hermes_best=MODELS["NousResearch/Hermes-3-Llama-3.1-8B"]["best_layer"],
        xlam_best=MODELS["Salesforce/Llama-xLAM-2-8b-fc-r"]["best_layer"],
        null_val=null_model,
    )

    out = RESULTS_DIR / "cka_results.json"
    out.write_text(json.dumps({
        "pca_k": PCA_K,
        "null_cross_tool": null_tool,
        "null_cross_model": null_model,
        "cross_tool": res_tool,
        "cross_model": res_model,
    }, indent=2))
    print(f"\nSaved {out}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
