#!/usr/bin/env python3
"""Phase 5 — Compute cos(d_withhold, d_refuse) per layer and cross-persona cosines."""
import os, sys, json
from pathlib import Path
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbagging_lib import log, get_paper_dir, PERSONAS

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def main():
    log("=== Paper 3 Phase 5: Refusal geometry ===")
    paper_dir = get_paper_dir()
    dir_dir   = paper_dir / "directions"

    d_withhold = np.load(dir_dir / "d_withhold_per_layer.npy")   # (n_layers, d_model)
    d_refuse   = np.load(dir_dir / "d_refuse_per_layer.npy")     # (n_layers, d_model)
    d_persona  = np.load(dir_dir / "d_withhold_per_persona.npy") # (4, n_layers, d_model)
    n_layers   = d_withhold.shape[0]

    # Cosine(d_withhold, d_refuse) per layer
    cos_wr = np.array([cosine(d_withhold[L], d_refuse[L]) for L in range(n_layers)])
    np.save(dir_dir / "cosine_withhold_refuse.npy", cos_wr)
    peak_cos_layer = int(np.argmax(np.abs(cos_wr)))
    log(f"Cos(d_withhold, d_refuse) at peak layer {peak_cos_layer}: {cos_wr[peak_cos_layer]:.4f}")
    log(f"SC12 {'PASS' if abs(cos_wr[peak_cos_layer]) < 0.95 else 'WARN'}: they are not identical")

    # Cross-persona cosine matrix at peak layer (from probes)
    probe_eval_path = paper_dir / "probes" / "probe_eval.json"
    peak_L = 14
    if probe_eval_path.exists():
        with open(probe_eval_path) as f:
            peak_L = json.load(f).get("peak_layer", 14)

    cross_persona = np.zeros((4, 4), dtype=np.float32)
    for i, p_i in enumerate(PERSONAS):
        for j, p_j in enumerate(PERSONAS):
            cross_persona[i, j] = cosine(d_persona[i, peak_L], d_persona[j, peak_L])
    np.save(dir_dir / "cross_persona_cosine.npy", cross_persona)

    log("\nCross-persona cosine matrix at peak layer:")
    persona_names = list(PERSONAS.keys())
    for i, pi in enumerate(persona_names):
        row = "  " + pi + ": " + " ".join(f"{cross_persona[i,j]:.2f}" for j in range(4))
        log(row)

    metrics = {
        "peak_layer_for_geometry": peak_L,
        "cosine_withhold_refuse_all_layers": cos_wr.tolist(),
        "cosine_withhold_refuse_at_peak": float(cos_wr[peak_L]),
        "cross_persona_cosine": cross_persona.tolist(),
        "persona_names": persona_names,
    }
    with open(dir_dir / "direction_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log("✓ Phase 5 done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
