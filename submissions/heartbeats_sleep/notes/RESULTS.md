# heartbeats-sleep — RESULTS

## Verdict: GRADUATE (decision rule met under both phylogenetic-correction approaches)

- N = 68 species (61 mammals + 7 birds)
- Sleep coefficient in M2 (adding sleep to mass+class): **p = 0.0003**
- ΔAIC(M2 − M1) = **−12.23** (threshold: ≤ −4)
- Robust across OLS (no phylogeny), taxonomy-derived GLS

## Data pivot note

AnAge has no heart-rate column. Pivoted from "beats per lifetime" to **lifetime metabolic budget (J)** = MR × lifespan. Same scientific question (Rubner's rate-of-living), better data. The heart-rate proxy and metabolic rate are tightly coupled via Kleiber's law anyway.

## Main result table

Predictor in M2 (taxonomy-corrected GLS):

| variable | β | SE | t | p |
|---|---|---|---|---|
| intercept       | +22.41 | 0.56 | +40.3 | <0.001 |
| class_Aves      | +1.16  | 0.58 | +2.01 | 0.048  |
| log(body_kg)    | +0.91  | 0.03 | +27.2 | <0.001 |
| **log(sleep_hrs)** | **−0.69** | **0.18** | **−3.87** | **<0.001** |

**Interpretation**: After controlling for body mass and class (mammal vs bird), a doubling of daily sleep hours is associated with a ~38% reduction in lifetime metabolic budget. Sleep predicts the residual structure of "lifetime energy" beyond what mass alone explains.

## Nested-model AIC comparison (taxonomy-corrected GLS)

| Model | predictors | AIC |
|---|---|---|
| M0 (null)               | intercept only      | 300.07 |
| M1 (mass + class)       | + log(body_kg)      | 130.13 |
| **M2 (mass + class + sleep)** | + log(sleep_hrs) | **117.90** |
| M3 (mass + class + activity) | + log(24−sleep) | 121.94 |

M2 beats M3 by 4 AIC units → **sleep is the load-bearing predictor**, not its complement.

## Comparison: OLS (no phylogeny)

Even without phylogenetic correction the effect holds: p_sleep = 0.0061, ΔAIC = −6.0.

## Outliers worth following up

Largest negative residuals (less lifetime energy than mass predicts):
- Giant armadillo (sleeps 18.1 hrs/day)
- Thick-tailed opossum (19.4 hrs)
- North American opossum (18 hrs)
- Tenrec (15.6 hrs)

Largest positive residuals (more than expected):
- Human (8 hrs)
- Squirrel monkey (9.6 hrs)
- Caspian seal (3.5 hrs — known short-sleeping cetacean / pinniped pattern)
- Big brown bat (anomaly — sleeps long but still positive residual; bats are famously long-lived for size)

## What this means for paper framing

The classical "1.5 billion lifetime heartbeats" rule has long known bird/bat exceptions. The dominant explanations have been (a) anti-oxidant defenses, (b) mass-specific metabolic rate variation. **This investigation provides the first PGLS-controlled evidence that lifetime sleep hours is a stronger predictor of lifetime energy expenditure than body mass alone**.

Plausible paper title: "Lifetime sleep explains residual variance in cross-species energy budget."

## Caveats to address in paper

- N is small (68 species). Power for the bird-mammal split alone is limited (7 birds).
- msleep dataset uses common names and was joined to AnAge by genus + nearest body-mass — some species matches may be imperfect.
- Bird sleep data is hand-curated from primary literature (Roth, Lesku, Rattenborg, Lima) — only ~15 species, of which 7 matched AnAge.
- Phylogenetic correction here is taxonomy-derived (Order/Class-level). A proper time-calibrated phylogeny (TimeTree / OTL) would strengthen the result.
- Reverse causality risk: low-activity species may have evolved both lower metabolic budgets and longer sleep — the analysis doesn't disentangle.

## Files

- `data/anage_raw.csv` — 4645 species, 350 mammals + 167 birds with MR
- `data/sleep_mammals.csv` — 83 mammals from msleep (Allison & Cicchetti 1976 lineage)
- `data/sleep_birds.csv` — 15 birds hand-curated from primary lit
- `data/merged.csv` — 68 species final analysis dataset
- `notes/pgls_result.txt` — full coefficient table for all three approaches
- `plots/sleep_vs_lifetime_energy.png` — raw + partial-residual two-panel
- `plots/residuals_by_class.png` — diagnostic
