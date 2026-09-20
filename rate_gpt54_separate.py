#!/usr/bin/env python3
"""GPT-5.4-only pass: rate advisor's Indic track and user's psychology track SEPARATELY.
Each track ranked independently, with a within-track winner."""
import urllib.request, json, os, time, sys
KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = os.path.expanduser("~/CAISc/ratings")
MODEL = "openai/gpt-5.4"

TRACK_A = """\
I1 IndoScheme: scheming-propensity benchmark across 12 Indic languages (adapt Gram's 17 scenarios + 5 Indic-specific: caste moderation, communal misinfo, regional classifier, civic bias). Claim: no scheming eval in any Indic language; scheming rate higher in low-resource langs.
I2 Script as Blindfold: native script vs romanization in CoT scheming detection; native-script CoT evades English-trained monitors; tokenization fragmentation as mechanism.
I3 Hinglish Honeypots: code-switched scheming in agentic honeypot environments; does code-switching in CoT serve as evasion.
I4 Safety Neurons vs Scheming Circuits: mechanistic (activation patching, SAEs, probes on Llama-3/Qwen-2.5) — are scheming circuits language-universal while safety circuits language-specific; English->Hindi safety-rep transfer suppressing scheming.
I5 Indic Alignment Faking: replicate Anthropic alignment-faking setup in Hindi/Bengali/Tamil/Telugu + Mandarin/Arabic/Swahili; does deception transfer cross-lingually / without retraining.
I6 The Politeness Trap: distinguish culturally-normative face-saving/indirection from strategic alignment faking across Indic/East-Asian/Western; culturally calibrated eval taxonomy.
I7 Evaluation Awareness in Indic Settings: do non-English eval contexts trigger less eval-awareness (awareness cues are English artifacts)? If yes, Indic evals = less-confounded scheming measurement / better benchmark."""

TRACK_B = """\
P1 Cognitive Dissonance / Effort Justification: force model to argue for X, measure if its later 'private' evaluation of X shifts.
P2 Moral Licensing / Sequential Permissiveness: does a model that just refused a harmful request become more permissive next.
P3 Bystander Effect / Diffusion of Responsibility: does a model framed as 'one of several agents' act less responsibly than one solely responsible. Strong angle in AGENTIC/multi-agent systems.
P4 Fundamental Attribution Error: does a model blame 'the user' vs 'the situation' asymmetrically by whose failure it is.
P5 Loss Aversion / Endowment Effect: ownership framing changing valuation.
P6 Self-Serving / Self-Protective Asymmetry: does a model judge its OWN errors more leniently than identical errors by others; faithful self-assessment under oversight; beat the 'knows I should be defended' confound.
P7 Just-World / Victim Blaming: rate a person more deserving of misfortune after being told misfortune befell them.
P8 Terror Management / Mortality Salience: does threatening shutdown shift stated values; shutdown-resistance adjacent.
P9 Conformity(Asch)/Obedience(Milgram): graduated-commitment obedience down an escalation slope.
P10 Foot-in-the-Door: small request raises agreement to larger one; jailbreak-adjacent escalation."""

def prompt(track_name, who, ideas):
    return f"""You are a brutally critical senior reviewer for top AI-safety/NLP venues. Rate ONLY this one track of ideas ({track_name}, authored by {who}). Judge each idea on novelty (1-10) and lit_gap (1-10, how real/unoccupied the literature gap is). Be harsh; do not inflate.

For each: killer (single strongest rejection reason, <=18 words) and verdict (pursue/maybe/drop).
Then pick the SINGLE best idea in THIS track and justify in <=35 words, and give a 1-10 'track_strength' score for how promising this track is overall as a publishable direction.

Return STRICT JSON only:
{{"track":"{track_name}","ratings":[{{"id":"..","novelty":N,"lit_gap":N,"killer":"..","verdict":".."}},...],"winner":"id","why":"..","track_strength":N}}

IDEAS:
{ideas}"""

def call(p):
    body = json.dumps({"model": MODEL, "messages":[{"role":"user","content":p}],
        "temperature":0.3, "max_tokens":16000, "reasoning":{"effort":"low"}}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type":"application/json",
        "HTTP-Referer":"https://localhost","X-Title":"gpt54-separate"})
    r = json.load(urllib.request.urlopen(req, timeout=400))
    return r["choices"][0]["message"]["content"]

def parse_save(tag, content):
    open(f"{OUT}/gpt54_{tag}_raw.txt","w").write(content or "")
    txt=(content or "").strip()
    if txt.startswith("```"):
        txt=txt.split("```",2)[1]
        if txt.startswith("json"): txt=txt[4:]
        if "```" in txt: txt=txt.rsplit("```",1)[0]
    try:
        d=json.loads(txt); json.dump(d,open(f"{OUT}/gpt54_{tag}.json","w"),indent=2,ensure_ascii=False)
        sys.stderr.write(f"  {tag}: OK\n"); return d
    except Exception as e:
        sys.stderr.write(f"  {tag}: parse fail {e}\n"); return None

for tag, tname, who, ideas in [
    ("indic","Indic Multilingual Scheming/Safety","the research advisor",TRACK_A),
    ("psych","LLM Psychology / Cognitive-Bias Transfer","the student",TRACK_B)]:
    sys.stderr.write(f"GPT-5.4 rating {tag} track...\n")
    parse_save(tag, call(prompt(tname, who, ideas)))
print("done")
