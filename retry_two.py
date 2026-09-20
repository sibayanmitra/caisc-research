#!/usr/bin/env python3
import urllib.request, json, os, time, sys
KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
OUT = os.path.expanduser("~/CAISc/ratings")
PROMPT = open(f"{OUT}/_prompt.txt").read()
MODELS = {"gemini": "google/gemini-3.1-pro-preview", "deepseek": "deepseek/deepseek-v4-pro"}

def call(model):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0.3,
        "max_tokens": 16000,
        "reasoning": {"effort": "low"},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost", "X-Title": "idea-rating"})
    r = json.load(urllib.request.urlopen(req, timeout=400))
    ch = r["choices"][0]
    sys.stderr.write(f"  finish={ch.get('finish_reason')} usage={r.get('usage')}\n")
    return ch["message"]["content"]

for name, model in MODELS.items():
    sys.stderr.write(f"Calling {name}...\n")
    try:
        content = call(model)
    except Exception as e:
        sys.stderr.write(f"  FAIL {e}\n"); continue
    open(f"{OUT}/{name}_raw.txt", "w").write(content or "")
    txt = (content or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1]
        if txt.startswith("json"): txt = txt[4:]
        if "```" in txt: txt = txt.rsplit("```", 1)[0]
    try:
        parsed = json.loads(txt)
        json.dump(parsed, open(f"{OUT}/{name}.json", "w"), indent=2, ensure_ascii=False)
        sys.stderr.write(f"  {name}: OK ({len(parsed.get('ratings',[]))})\n")
    except Exception as e:
        sys.stderr.write(f"  {name}: parse fail {e}\n")
print("done")
