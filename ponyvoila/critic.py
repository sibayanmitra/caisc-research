#!/usr/bin/env python3
"""
critic.py — Cross-model critique utility for autovoila.

Usage:
    python3 critic.py "Your research idea or draft text here"
    echo "text" | python3 critic.py

Primary:  OpenRouter -> openai/gpt-5.4  (needs OPENROUTER_API_KEY)
Fallback: Anthropic  -> claude-sonnet-4-6 (needs ANTHROPIC_API_KEY)
If both fail: prints a clear message so the agent can critique manually.
"""

import sys
import os
import json
import urllib.request
import urllib.error

OPENROUTER_MODEL = "openai/gpt-5.4"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

CRITIC_SYSTEM_PROMPT = """You are a rigorous ML/AI conference reviewer providing constructive criticism.
Evaluate the research based on:
1. Novelty — is this surprising to experts? Is it already known?
2. Rigor — does the evidence support the claims? Are confounds addressed?
3. Fruitfulness — does this open new research directions? Does anyone care?
4. Feasibility — can this realistically be done with available resources?

Be direct, specific, and critical. Point out weaknesses first, then strengths.
Suggest concrete improvements. Keep response under 400 words."""


def get_input():
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print("Usage: python3 critic.py 'your research text'")
    sys.exit(1)


def call_openrouter(text):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        "max_tokens": 600
    }).encode()

    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/lossfunk/autovoila",
            "X-Title": "autovoila-critic"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def call_anthropic(text):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": 600,
        "system": CRITIC_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": text}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    return data["content"][0]["text"]


def main():
    text = get_input()

    # Try OpenRouter first
    try:
        result = call_openrouter(text)
        print(f"[CRITIC via {OPENROUTER_MODEL}]\n")
        print(result)
        return
    except ValueError as e:
        print(f"[OpenRouter skipped: {e}]", file=sys.stderr)
    except Exception as e:
        print(f"[OpenRouter failed: {e}]", file=sys.stderr)

    # Fallback to Anthropic/Claude
    try:
        result = call_anthropic(text)
        print(f"[CRITIC via {ANTHROPIC_MODEL} (fallback)]\n")
        print(result)
        return
    except ValueError as e:
        print(f"[Anthropic skipped: {e}]", file=sys.stderr)
    except Exception as e:
        print(f"[Anthropic failed: {e}]", file=sys.stderr)

    # Both failed — signal the agent to critique manually
    print("[CRITIC UNAVAILABLE] No API keys found or calls failed.")
    print("ACTION REQUIRED: Claude agent should perform the critique inline.")
    sys.exit(2)


if __name__ == "__main__":
    main()
