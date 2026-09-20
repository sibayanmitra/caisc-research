"""Decision log + gate-failure log + compute attribution writers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

LOGS_DIR = Path("/home/anonymous/agentic_paper/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

DECISIONS_PATH = LOGS_DIR / "decisions.jsonl"
GATE_FAILURES_PATH = LOGS_DIR / "gate_failures.jsonl"
COMPUTE_PATH = LOGS_DIR / "compute.jsonl"


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    record = {"ts": time.time(), **record}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def log_decision(decision: str, rationale: str, outcome: str | None = None, alternative: str | None = None) -> None:
    """Record an agentic decision and its rationale (≤1 sentence).

    `outcome` may be filled in later via a follow-up record. `alternative` documents
    the path-not-taken at a branch point.
    """
    _append_jsonl(
        DECISIONS_PATH,
        {"decision": decision, "rationale": rationale, "outcome": outcome, "alternative": alternative},
    )


GATE_OUTCOMES_PATH = LOGS_DIR / "gate_outcomes.jsonl"


def log_gate(gate: str, passed: bool, observed: Any, threshold: Any, recovery: str | None = None) -> None:
    """Record sanity-gate outcome (pre-flight, mid-run, or post-run).

    All outcomes go to gate_outcomes.jsonl. Failures additionally go to gate_failures.jsonl
    so failures are easy to grep without parsing every entry.
    """
    record = {
        "gate": gate,
        "passed": passed,
        "observed": observed,
        "threshold": threshold,
        "recovery": recovery,
    }
    _append_jsonl(GATE_OUTCOMES_PATH, record)
    if not passed:
        _append_jsonl(GATE_FAILURES_PATH, record)


def log_compute(stage: str, gpu_seconds: float, decision: str | None = None, notes: str | None = None) -> None:
    """Tag compute time to a stage and (optionally) a motivating decision."""
    _append_jsonl(
        COMPUTE_PATH,
        {"stage": stage, "gpu_seconds": gpu_seconds, "decision": decision, "notes": notes},
    )
