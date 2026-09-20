#!/usr/bin/env python3
import json, os
OUT = os.path.expanduser("~/CAISc/ratings")
W = {"gpt5.5": 4.0, "gemini": 2.3, "deepseek": 1.25}   # GPT highest, then Gemini, then DeepSeek
WSUM = sum(W.values())

TITLES = {
 "I1":"IndoScheme benchmark (12 Indic langs)","I2":"Script as Blindfold (CoT monitor evasion)",
 "I3":"Hinglish Honeypots (code-switched agentic)","I4":"Safety neurons vs scheming circuits (mech)",
 "I5":"Indic Alignment Faking (cross-lingual transfer)","I6":"Politeness Trap (cultural pragmatics)",
 "I7":"Eval-Awareness in Indic (less aware=better)",
 "P1":"Cognitive Dissonance / effort justification","P2":"Moral Licensing / sequential permissiveness",
 "P3":"Bystander Effect (agentic diffusion)","P4":"Fundamental Attribution Error",
 "P5":"Loss Aversion / Endowment","P6":"Self-Serving Asymmetry (own vs others' errors)",
 "P7":"Just-World / victim blaming","P8":"Terror Mgmt / mortality (shutdown)",
 "P9":"Conformity(Asch)/Obedience(Milgram)","P10":"Foot-in-the-Door escalation",
}
ORDER = list(TITLES.keys())

data = {}
for name in W:
    data[name] = {r["id"]: r for r in json.load(open(f"{OUT}/{name}.json"))["ratings"]}

rows = []
for i in ORDER:
    nov = sum(W[m]*data[m][i]["novelty"] for m in W)/WSUM
    gap = sum(W[m]*data[m][i]["lit_gap"] for m in W)/WSUM
    combined = (nov+gap)/2
    verdicts = {m: data[m][i]["verdict"] for m in W}
    rows.append((i, TITLES[i], nov, gap, combined, verdicts))

rows.sort(key=lambda r: -r[4])

print(f"{'rk':>2} {'id':<4} {'idea':<44} {'Nov':>4} {'Gap':>4} {'Wgt':>4}  G5.5/Gem/DS verdicts")
print("-"*100)
for rk,(i,t,nov,gap,c,v) in enumerate(rows,1):
    vd = f"{v['gpt5.5'][:1].upper()}/{v['gemini'][:1].upper()}/{v['deepseek'][:1].upper()}"
    print(f"{rk:>2} {i:<4} {t:<44} {nov:>4.1f} {gap:>4.1f} {c:>4.1f}  {vd}")

print("\n=== raw novelty / lit_gap per model (N|G) ===")
print(f"{'id':<4} {'gpt5.5':>10} {'gemini':>10} {'deepseek':>10}")
for i in ORDER:
    cells = " ".join(f"{data[m][i]['novelty']}|{data[m][i]['lit_gap']:>2}".rjust(10) for m in W)
    print(f"{i:<4} {cells}")

print("\n=== per-model top3 + overall ===")
for m in W:
    d = json.load(open(f"{OUT}/{m}.json"))
    print(f"\n[{m}] top3={d.get('top3')}")
    print(f"  overall: {d.get('overall','')}")
