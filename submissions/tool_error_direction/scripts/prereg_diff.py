"""Produce a pre-registration-vs-final diff table for the paper appendix."""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path("/home/anonymous/agentic_paper/results")


def main() -> None:
    decision = json.loads((RESULTS_DIR / "decision.json").read_text())
    rows = []

    # H1 thresholds vs observed
    for m, s in decision["summaries"].items():
        if "best_layer_acc" in s:
            use2 = s.get("use_twoclass", False)
            thr = 0.75 if use2 else 0.60
            probe_note = " [2-class fallback; Qwen3 rarely gives up on errors]" if use2 else ""
            rows.append({
                "claim": f"H1(a) probe ≥ {thr:.0%}{probe_note} [{m}]",
                "prereg": f"≥ {thr:.0%} with permutation p < 0.001",
                "observed": f"{s['best_layer_acc']:.3f} (p={s['permutation_p']:.4f})",
                "passed": decision["h1a_per_model"].get(m, False),
            })
    for m, c in decision["causal_eval"].items():
        if c.get("available"):
            rows.append({
                "claim": f"H1(b) causal flip ≥ 25pp [{m}]",
                "prereg": "retry − random ≥ 0.25 absolute; h ≥ 0.5; non-overlapping CIs",
                "observed": (
                    f"retry={c.get('retry_flip_at_alpha_star', float('nan')):.2%}, "
                    f"random={c.get('random_flip_at_alpha_star', float('nan')):.2%}, "
                    f"h={c.get('cohens_h', float('nan')):.2f}, "
                    f"disjoint_CI={c.get('ci_non_overlapping', False)}"
                ),
                "passed": decision["h1b_per_model"].get(m, False),
            })
    rows.append({
        "claim": "H1(c) cross-family",
        "prereg": "both families independently clear H1(a) and H1(b)",
        "observed": ", ".join(f"{m}={decision['h1_cleared_per_model'].get(m, False)}" for m in decision["summaries"]),
        "passed": all(decision["h1_cleared_per_model"].values()),
    })

    lines = ["# Pre-Registration vs. Final\n", "| Claim | Pre-registered | Observed | Passed |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['claim']} | {r['prereg']} | {r['observed']} | {r['passed']} |")
    (RESULTS_DIR / "prereg_vs_final.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS_DIR / 'prereg_vs_final.md'}")


if __name__ == "__main__":
    main()
