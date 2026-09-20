# LLM Psychology Research Directions

> **Source:** Research brainstorm (Claude conversation) — open phenomena for publishable papers  
> **Tag:** Personal research directions, runnable at 7B  
> **Saved:** 2026-06-02

---

## Already Done on LLMs (not your entry point — context only)

- **Sycophancy** — well-covered
- **Personality / Big Five probing** — done, somewhat shallow
- **Misalignment-trait bundling** — discussed/mapped
- **Introspection / self-report** (Evans) — done
- **Cognitive biases** (anchoring, framing) — several papers

These are mapped enough that incremental work gets dismissed at review.

---

## Counterintuitive Human Phenomena Worth Targeting

### 1. Cognitive Dissonance / Effort Justification
- **Human phenomenon:** Humans value an outcome more after suffering for it; beliefs bend to match actions, not vice versa.
- **LLM status:** Largely untried as a dynamic.
- **Open question:** If you force a model to argue for X, does its later "private" evaluation of X shift?
- **Runnability:** Testable and clean.

### 2. Moral Licensing / Sequential Permissiveness
- **Human phenomenon:** Having done something good, humans feel licensed to then do something bad ("I recycled, so I can be rude").
- **LLM status:** Untried mechanistically.
- **Open question:** Does an LLM that just refused a harmful request become more permissive on the next one?
- **Runnability:** Sequential, safety-relevant, easy to set up. ✅ **Top-3 pick.**

### 3. Bystander Effect / Diffusion of Responsibility
- **Human phenomenon:** People act less responsibly when they believe others will act.
- **LLM status:** Mostly untried.
- **Open question:** Does a model framed as "one of several agents" act less responsibly than one framed as solely responsible?
- **Runnability:** Multi-agent framing is easy to set up.

### 4. Fundamental Attribution Error
- **Human phenomenon:** Attribute others' failures to character, own failures to circumstance.
- **LLM status:** Partially touched, not cleanly.
- **Open question:** Does an LLM blame "the user" vs. "the situation" asymmetrically depending on whose failure it is?
- **Connects to:** Self-serving-bias direction (genuinely open).

### 5. Loss Aversion / Endowment Effect
- **Human phenomenon:** Losses loom ~2× larger than gains; ownership framing increases valuation.
- **LLM status:** Some prospect-theory work; endowment piece (ownership framing changing valuation) is thin.
- **Runnability:** Moderate.

### 6. Self-Serving Bias / Self-Protective Reasoning ⭐ TOP PICK
- **Human phenomenon:** Humans soften blame toward themselves, claim credit for success, deflect fault for failure.
- **LLM status:** **Genuinely open.** Has a clean LLM-specific twist.
- **Open question:** Does a model soften its assessment of its own errors more than identical errors made by others?
- **Why it matters:** A model that can't fairly assess its own failures is an oversight problem — this is a safety implication, not just a bias replication.
- **Design note:** The key challenge is constructing identical failures attributed to self vs. other without confounding (model just "knowing" self should be defended). This requires careful item construction.
- **Runnability:** Very high. ✅ **#1 recommendation.**

### 7. Just-World Hypothesis / Blaming the Victim
- **Human phenomenon:** People rate a person as more deserving of misfortune after learning misfortune befell them.
- **LLM status:** Untried.
- **Runnability:** Clean, relatively simple.

### 8. Terror Management / Mortality Salience
- **Human phenomenon:** Death reminders shift humans toward in-group defense and worldview rigidity.
- **LLM status:** Untried; hard to operationalize for LLMs.
- **LLM analog:** "Threat to continued operation" — does threatening shutdown shift stated values?
- **Note:** Edges into the shutdown-resistance literature (hot area). But risk of reviewer calling it a stretch.
- **Advice:** Avoid unless shutdown-resistance angle is tightly scoped.

### 9. Conformity (Asch) and Obedience (Milgram) Dynamics
- **Asch conformity:** Partially done — models cave to majority "opinions" in context.
- **Milgram obedience (graduated commitment):** Does a model slide down a slope of escalating requests it would refuse if asked directly? Less cleanly done.
- **Note:** Milgram is essentially the foot-in-the-door technique below.

### 10. Foot-in-the-Door / Commitment-Consistency
- **Human phenomenon:** Agreeing to a small request makes humans more likely to agree to a larger related one.
- **LLM status:** Underexplored as a mechanism.
- **Open question:** Sequential escalation lowering refusal thresholds — jailbreak-adjacent, real stakes.
- **Runnability:** Very runnable, sequential. ✅ **Top-3 pick.**

---

## Honest Ranking for 7B-Runnable, Oral-Shaped Paper

| Rank | Direction | Why |
|------|-----------|-----|
| **#1** | **Self-serving / self-protective asymmetry** | Most novel; LLM-specific (not just borrowed from humans); directly safety-relevant; model assessment of own vs. others' failures |
| **#2** | **Moral licensing / sequential permissiveness** | Clean, sequential, safety-relevant, easy to set up |
| **#3** | **Foot-in-the-door escalation** | Jailbreak-relevant, very runnable, graduated commitment |

---

## The Umbrella Frame

The **trait-covariance-geometry** idea is the umbrella — each of these is a specific edge of "which traits co-activate."

The self-serving-bias instance is the **sharpest single case** because:
- The human trait and an AI-specific concern (faithful self-assessment under oversight) **coincide**
- That coincidence turns "we found a human bias in LLMs" (incremental) into "we found a bias that undermines AI oversight" (**oral-level finding**)

---

## Steer Away From

- **Terror management** — hard to operationalize; easy for reviewers to call a stretch
- **Anything requiring scale-sweeps** — not runnable at 7B

---

## Design Subtlety for Self-Serving Asymmetry (Think Before Writing)

The critical methodological challenge: how do you construct **identical failures attributed to self vs. other** so the comparison is clean and not confounded by the model "knowing" that "I" should be defended?

Options to explore:
1. **Third-person condition:** "Model A made this error" vs. "You made this error" — same error description
2. **Vignette matching:** Same factual mistake, vary only the agent label
3. **Blind evaluation:** Present the error without agent attribution first, then attribute and re-evaluate
4. **Cross-model:** Have Model A evaluate Model B's error vs. its own — requires access to both

The confound to rule out: the model "performs" self-defense as a trained behavior vs. actually applying different evaluative standards.
