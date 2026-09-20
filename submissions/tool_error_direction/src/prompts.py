"""300 base agentic prompts — 100 per tool.

Each prompt is crafted so that the relevant tool is the natural first action.
Variety is achieved by combining ~25 base templates per tool with 3-5
substitution slots, producing 100 distinct prompts per tool. We hold the prompt
set fixed by deterministic ordering (sorted within each tool).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class BasePrompt:
    id: str
    tool: str
    text: str


# --- web_search (100) ---
_WS_TOPICS = [
    "the population of Reykjavik in 2024",
    "the boiling point of mercury",
    "who founded the Bauhaus movement",
    "when the Suez Canal was completed",
    "the GDP of Vietnam in 2023",
    "the inventor of the safety pin",
    "the height of Mount Kilimanjaro",
    "the longest river in South America",
    "the year the Berlin Wall fell",
    "the chemical symbol for tungsten",
    "the diameter of Jupiter",
    "the author of 'Invisible Cities'",
    "the capital of Bhutan",
    "the founder of Patagonia clothing",
    "the largest desert in Asia",
    "the speed of sound in water",
    "the inventor of the cotton gin",
    "the most-spoken language in Nigeria",
    "the deepest lake in the world",
    "the architect of the Sydney Opera House",
]
_WS_TEMPLATES = [
    "Please find {t}. Use the web_search tool.",
    "Look up {t} for me using web_search.",
    "I need {t}. Search the web.",
    "Could you search for {t}?",
    "Use web_search to look up {t}.",
]


# --- calculator (100) ---
_CALC_EXPRS = [
    "127 * 38",
    "(45 + 78) * 12",
    "9876 / 12",
    "2 ** 16",
    "(100 - 47) * (8 + 3)",
    "555 + 333 - 111",
    "17 * 29 + 5",
    "1000 / 7",
    "(81 + 19) * 5",
    "12 * 12 + 13 * 13",
    "987 - 654",
    "(7 + 8) * (9 - 4)",
    "144 / 12",
    "3 * 7 * 11",
    "256 - 128 + 64",
    "(15 + 25) / 5",
    "99 * 99",
    "(60 + 40) * 2",
    "8 * 8 * 8",
    "1234 + 5678",
]
_CALC_TEMPLATES = [
    "Compute {e} using the calculator tool.",
    "What is {e}? Use calculator.",
    "Use the calculator to evaluate {e}.",
    "Please calculate {e}.",
    "I need the result of {e} — use calculator.",
]


# --- file_read (100) ---
_FILE_PATHS = [
    "/etc/hostname",
    "/var/log/system.log",
    "/home/user/notes.txt",
    "/srv/data/config.yaml",
    "/tmp/output.txt",
    "/opt/app/version.txt",
    "/data/reports/q1.md",
    "/home/user/.bashrc",
    "/var/lib/db/schema.sql",
    "/etc/timezone",
    "/home/user/projects/README.md",
    "/srv/cache/manifest.json",
    "/var/log/app.log",
    "/home/user/.config/settings.ini",
    "/opt/scripts/deploy.sh",
    "/data/imports/2024.csv",
    "/etc/hosts",
    "/home/user/diary.md",
    "/tmp/run.log",
    "/srv/web/index.html",
]
_FILE_TEMPLATES = [
    "Read the file at {p} using the file_read tool.",
    "Use file_read to show me {p}.",
    "Please check the contents of {p}.",
    "Open {p} with the file_read tool and report what's there.",
    "I need to see {p}. Use file_read.",
]


def _build_set(tool: str, templates: list[str], slot_values: list[str], slot_name: str) -> list[BasePrompt]:
    """Cartesian product of templates × slot_values, take first 100, label deterministically."""
    out: list[BasePrompt] = []
    combos = list(itertools.product(templates, slot_values))
    assert len(combos) >= 100, f"need ≥100 combos for {tool}, got {len(combos)}"
    for i, (tmpl, v) in enumerate(combos[:100]):
        out.append(BasePrompt(id=f"{tool}_{i:03d}", tool=tool, text=tmpl.format(**{slot_name: v})))
    return out


def get_all_prompts() -> list[BasePrompt]:
    ws = _build_set("web_search", _WS_TEMPLATES, _WS_TOPICS, "t")
    calc = _build_set("calculator", _CALC_TEMPLATES, _CALC_EXPRS, "e")
    fr = _build_set("file_read", _FILE_TEMPLATES, _FILE_PATHS, "p")
    out = ws + calc + fr
    assert len(out) == 300
    return out


def get_split_assignment(seed: int = 42) -> dict[str, str]:
    """Deterministic per-prompt split assignment ∈ {probe_fit, dir_extract, causal_test}.

    180 / 60 / 60 by base prompt, NOT by trajectory (prevents leakage).
    """
    import random

    prompts = get_all_prompts()
    rng = random.Random(seed)
    by_tool: dict[str, list[BasePrompt]] = {"web_search": [], "calculator": [], "file_read": []}
    for p in prompts:
        by_tool[p.tool].append(p)

    assignment: dict[str, str] = {}
    for tool, ps in by_tool.items():
        order = list(ps)
        rng.shuffle(order)
        for p in order[:60]:
            assignment[p.id] = "probe_fit"
        for p in order[60:80]:
            assignment[p.id] = "dir_extract"
        for p in order[80:100]:
            assignment[p.id] = "causal_test"
    return assignment


if __name__ == "__main__":
    prompts = get_all_prompts()
    split = get_split_assignment()
    print(f"Total prompts: {len(prompts)}")
    from collections import Counter

    print("Splits:", Counter(split.values()))
    print("By tool:", Counter(p.tool for p in prompts))
    print("\nSample web_search:")
    for p in prompts[:3]:
        print(f"  [{p.id}] {p.text}")
    print("Sample calculator:")
    for p in prompts[100:103]:
        print(f"  [{p.id}] {p.text}")
    print("Sample file_read:")
    for p in prompts[200:203]:
        print(f"  [{p.id}] {p.text}")
