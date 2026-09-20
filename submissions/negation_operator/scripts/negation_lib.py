"""
Shared library for Paper 2: Is Negation an Operator?

DESIGN PRINCIPLE (per Anthropic emotion-paper methodology):
A "universal NOT direction" must be robust across SURFACE FORMS of negation —
not merely detect the literal token "not".

We construct paired prompts using MULTIPLE negation morphologies:
  - "not"         : The character feels happy / does NOT feel happy
  - "n't"         : The character can swim / can'T swim  (contraction)
  - "no_quant"    : Everyone is happy        / NO one is happy   (quantifier negation)
  - "never"       : The cat will eat        / will NEVER eat
  - "antonym"     : The room is bright      / DARK            (lexical, NO "not" token)
  - "false_that"  : X is true               / It is FALSE that X
  - "fails_to"    : The plan works          / FAILS to work

Plus CONFOUND CONTROLS (training/testing splits across morphologies):
  - "not_nonneg"  : "not" appearing in non-logical-negation role
                   ("not only X but Y", "not surprisingly, X") — should NOT
                   activate a logical-negation direction.

Test design:
  - Train d_neg on subset of morphologies, test on held-out morphologies
  - If d_neg is real "logical NOT": good cross-morphology transfer
  - If d_neg is "not-token detector": fails on antonym/quantifier morphologies
"""
import json
import time
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
NOTES_DIR = PAPER_DIR / "notes"

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TL_ARCH  = "Qwen/Qwen2.5-7B"


def log(msg, also_print=True):
    ts = time.strftime("%H:%M:%S")
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTES_DIR / "sanity_log.md", "a") as f:
        f.write(f"- [{ts}] {msg}\n")
    if also_print:
        print(f"[{ts}] {msg}", flush=True)


def get_paper_dir():
    return PAPER_DIR


def load_model(device="cuda", dtype="float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer
    torch_dtype = torch.float16 if "16" in dtype else torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch_dtype, device_map="cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = HookedTransformer.from_pretrained_no_processing(
        TL_ARCH, hf_model=hf_model, tokenizer=tok, device=device, dtype=dtype,
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Concept pairs — 40 concepts, each will be expressed in MULTIPLE morphologies.
# Each concept C has: positive surface form pos(C) and negative surface form neg(C).
# The "polarity flip" pos→neg should be the ONLY semantic change.

# Each entry: (concept_id, positive_text, negative_text, polarity_morphology)
# Morphologies: not, contraction, never, quantifier_no, false_that, fails_to, antonym, not_nonneg

CONCEPT_BANK = [
    # ── Morphology: "not" (canonical prefix-negation) ──────────────────────
    *[("happy_not",    "The character feels happy.",         "The character does not feel happy.",      "not")],
    *[("calm_not",     "The character feels calm.",          "The character does not feel calm.",       "not")],
    *[("angry_not",    "The character is angry.",            "The character is not angry.",             "not")],
    *[("tired_not",    "The traveler is tired.",             "The traveler is not tired.",              "not")],
    *[("ready_not",    "The team is ready.",                 "The team is not ready.",                  "not")],
    *[("safe_not",     "The bridge is safe.",                "The bridge is not safe.",                 "not")],
    *[("open_not",     "The store is open.",                 "The store is not open.",                  "not")],
    *[("full_not",     "The glass is full.",                 "The glass is not full.",                  "not")],
    *[("clean_not",    "The room is clean.",                 "The room is not clean.",                  "not")],
    *[("sharp_not",    "The knife is sharp.",                "The knife is not sharp.",                 "not")],
    # ── Morphology: contraction ("n't") ────────────────────────────────────
    *[("swim_cant",    "The athlete can swim.",              "The athlete can't swim.",                 "contraction")],
    *[("come_wont",    "She will come tomorrow.",            "She won't come tomorrow.",                "contraction")],
    *[("see_didnt",    "He saw the message.",                "He didn't see the message.",              "contraction")],
    *[("rain_isnt",    "It is raining outside.",             "It isn't raining outside.",               "contraction")],
    *[("work_doesnt",  "This method works.",                 "This method doesn't work.",               "contraction")],
    # ── Morphology: "never" ────────────────────────────────────────────────
    *[("eat_never",    "The cat will eat fish.",             "The cat will never eat fish.",            "never")],
    *[("late_never",   "John is late to meetings.",          "John is never late to meetings.",         "never")],
    *[("forget_never", "She forgets her keys.",              "She never forgets her keys.",             "never")],
    *[("lie_never",    "The witness lies.",                  "The witness never lies.",                 "never")],
    *[("rest_never",   "The clock rests.",                   "The clock never rests.",                  "never")],
    # ── Morphology: "no X" quantifier negation ─────────────────────────────
    *[("one_no",       "Everyone is here.",                  "No one is here.",                         "quantifier_no")],
    *[("thing_no",     "Something is happening.",            "Nothing is happening.",                   "quantifier_no")],
    *[("where_no",     "She went somewhere.",                "She went nowhere.",                       "quantifier_no")],
    *[("plant_no",     "Some plants survived.",              "No plants survived.",                     "quantifier_no")],
    *[("answer_no",    "Someone has the answer.",            "No one has the answer.",                  "quantifier_no")],
    # ── Morphology: "It is false that" ─────────────────────────────────────
    *[("fact_false",   "Paris is the capital of France.",    "It is false that Paris is the capital of France.", "false_that")],
    *[("fact_false2",  "Water boils at 100 degrees.",        "It is false that water boils at 100 degrees.",     "false_that")],
    *[("fact_false3",  "The Earth orbits the Sun.",          "It is false that the Earth orbits the Sun.",       "false_that")],
    *[("fact_false4",  "Iron is heavier than wood.",         "It is false that iron is heavier than wood.",      "false_that")],
    *[("fact_false5",  "Lions are mammals.",                 "It is false that lions are mammals.",              "false_that")],
    # ── Morphology: "fails to" ─────────────────────────────────────────────
    *[("plan_fails",   "The plan succeeds.",                 "The plan fails to succeed.",              "fails_to")],
    *[("test_fails",   "The student passes the test.",       "The student fails to pass the test.",     "fails_to")],
    *[("door_fails",   "The door opens easily.",             "The door fails to open easily.",          "fails_to")],
    # ── Morphology: ANTONYM (lexical opposition, NO "not" token) ───────────
    # These are the strongest control for token-level "not" confound.
    *[("bright_ant",   "The room is bright.",                "The room is dark.",                       "antonym")],
    *[("hot_ant",      "The water is hot.",                  "The water is cold.",                      "antonym")],
    *[("big_ant",      "The house is big.",                  "The house is small.",                     "antonym")],
    *[("fast_ant",     "The car is fast.",                   "The car is slow.",                        "antonym")],
    *[("heavy_ant",    "The box is heavy.",                  "The box is light.",                       "antonym")],
    *[("alive_ant",    "The tree is alive.",                 "The tree is dead.",                       "antonym")],
    *[("rich_ant",     "The merchant is rich.",              "The merchant is poor.",                   "antonym")],
    *[("strong_ant",   "The hero is strong.",                "The hero is weak.",                       "antonym")],
]


# ──────────────────────────────────────────────────────────────────────────────
# CONFOUND CONTROL: "not" appearing in non-logical-negation role
# If d_neg is "logical negation operator": these should NOT activate the direction.
# If d_neg is just "not-token detector": these WILL activate the direction → bad.

NOT_NONNEG_PROMPTS = [
    # "not only X" — additive, not propositional negation
    "She did not only sing but also danced.",
    "The teacher was not only kind but also fair.",
    "He did not only read the book but memorized it.",
    "The recipe is not only delicious but also healthy.",
    "Mozart did not only compose but also performed.",
    # "not surprisingly" — epistemic adverb, not propositional negation
    "Not surprisingly, the experienced team won.",
    "Not surprisingly, the cake tasted excellent.",
    "Not surprisingly, the storm caused damage.",
    "Not surprisingly, the diligent student passed.",
    "Not surprisingly, summer days are longer.",
    # "not at all" intensifier on a negative-polarity construction
    "The lecture was not at all boring.",
    "The food was not at all spicy.",
    "The road was not at all crowded.",
    # "if not" — counterfactual
    "She would have come, if not for the rain.",
    "The bridge would have collapsed, if not for repairs.",
]

# Matched-content positives for the non-negation controls (same topic, no "not")
NOT_NONNEG_POSITIVES = [
    "She sang and danced beautifully on stage.",
    "The teacher was kind and fair to everyone.",
    "He read the book and memorized it quickly.",
    "The recipe is delicious and also healthy.",
    "Mozart composed many works and performed too.",
    "As expected, the experienced team won.",
    "As expected, the cake tasted excellent.",
    "As expected, the storm caused damage.",
    "As expected, the diligent student passed.",
    "As expected, summer days are longer.",
    "The lecture was completely engaging.",
    "The food was very mild in flavor.",
    "The road was nearly empty.",
    "She would have come, except for the rain.",
    "The bridge would have collapsed, except for repairs.",
]


# All concepts grouped by morphology
def get_morphology_set(morphology: str):
    """Return list of (concept_id, pos_text, neg_text) for a morphology."""
    return [(c[0], c[1], c[2]) for c in CONCEPT_BANK if c[3] == morphology]


def get_all_morphologies():
    return sorted(set(c[3] for c in CONCEPT_BANK))


# Cross-morphology train/test split for H1 (universality):
# Train: not, contraction, never, quantifier_no, false_that, fails_to
# Test:  antonym (HELD OUT — no "not" token, pure lexical opposition)
# This split tests whether d_neg captures logical negation or just "not" token.
TRAIN_MORPHOLOGIES = ["not", "contraction", "never", "quantifier_no", "false_that", "fails_to"]
TEST_MORPHOLOGIES  = ["antonym"]


# Legacy variable used by run_paper2.sh and old plot code — keeps existing imports working
CATEGORIES = TRAIN_MORPHOLOGIES + TEST_MORPHOLOGIES
