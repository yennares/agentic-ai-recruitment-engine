"""Shared Groq plumbing: one client, two model tiers, and two call shapes
(free-text and structured JSON) that every agent and legacy route builds on."""

import os
import re
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Fast tier: cheap/low-latency, used for extraction & generation tasks.
# Deep tier: stronger reasoning, used for evaluative/judgment tasks that
# directly influence the hire/no-hire decision. Every call tries its primary
# model first and automatically fails over to the other tier on error.
MODEL_FAST = "llama-3.1-8b-instant"
MODEL_DEEP = "openai/gpt-oss-120b"

CODE_FENCE_RE = re.compile(r"```[a-zA-Z0-9]*\n?(.*)```", re.DOTALL)


def strip_code_fences(text):
    """Models sometimes wrap output in ```html / ```json fences, occasionally
    with prose before it (e.g. "HTML Email Template: ```html..."). Extract the
    fenced block's content - greedy match so inline ``` inside the content
    (e.g. a code sample mentioned in an email) doesn't truncate it early; if
    there's no fence, return the text as-is."""
    text = text.strip()
    match = CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


def generate_completion(messages, max_tokens=3000, models=(MODEL_DEEP, MODEL_FAST)):
    """Free-text chat completion with automatic failover across models."""
    last_error = None
    for model in models:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens
            )
            return strip_code_fences(completion.choices[0].message.content)
        except Exception as e:
            last_error = e
            continue
    raise last_error


def _extract_json_block(text):
    """Best-effort recovery when a model doesn't return pure JSON despite instructions."""
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError(f"Could not parse JSON from model output: {text[:200]}")


def generate_json(system_prompt, user_prompt, models=(MODEL_FAST, MODEL_DEEP), max_tokens=2000, temperature=0.2):
    """Structured-output chat completion used by every pipeline agent.

    Tries each model in order; for each, first attempts Groq's native JSON
    mode, falling back to prompt-enforced JSON with manual extraction if the
    model/provider rejects the response_format parameter. Only moves to the
    next model tier once both attempts on the current one fail.
    """
    messages = [
        {"role": "system", "content": system_prompt + "\n\nRespond with ONLY a single valid JSON object. No markdown, no commentary, no code fences."},
        {"role": "user", "content": user_prompt}
    ]
    last_error = None
    for model in models:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return _extract_json_block(completion.choices[0].message.content)
        except Exception as e:
            last_error = e
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return _extract_json_block(completion.choices[0].message.content)
        except Exception as e:
            last_error = e
            continue
    raise last_error
