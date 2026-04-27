# python clean_data/gold_api_llm_classify.py
# requires: OPENROUTER_API_KEY environment variable

"""
Classify the gold 500-row Hebrew RE dataset using cloud LLMs via OpenRouter
(single API key, OpenAI-compatible REST API).

Models:
  - google/gemini-3.1-pro-preview   ("Gemini 3 Pro")
  - openai/gpt-5                    ("ChatGPT 5")

Each model is evaluated with 2 prompt languages (Hebrew + English), 3-shot
examples (2 positive, 1 negative), producing 4 columns per model:
  confidence_<tag>_he / relation_present_<tag>_he
  confidence_<tag>_en / relation_present_<tag>_en

Confidence: derived from logprobs of the first generated token ("0" / "1").
            Falls back to 1.0/0.0 (binary) when logprobs are unavailable.

Output:
  data/crocodile_heb25_gold_500_api_llm_classified.csv
"""

import os
import json
import math
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR   = "/home/nlp/ronke21/hebrew_RE"
INPUT_CSV  = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500_api_llm_classified.csv")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY        = os.environ.get("OPENROUTER_API_KEY", "")

MAX_WORKERS  = 8    # parallel API calls per (model, language) pass
MAX_RETRIES  = 4    # retries with exponential back-off on transient errors

MODELS = [
    {"id": "google/gemini-3.1-pro-preview", "tag": "gemini3_pro"},
    {"id": "openai/gpt-5",                  "tag": "gpt5"},
]

# USD per 1M tokens — update from openrouter.ai/models as prices change
PRICES_PER_1M = {
    "google/gemini-3.1-pro-preview": {"input": 2.50,  "output": 10.00},
    "openai/gpt-5":                  {"input": 30.00, "output": 60.00},
}

# ── Few-shot examples ──────────────────────────────────────────────────────────
# 3 examples: 2 positive (label=1), 1 negative (label=0).

FEW_SHOT_HE = [
    {
        "text":      "אלברט איינשטיין נולד ב-14 במרץ 1879 באולם שבגרמניה ונפטר ב-18 באפריל 1955 בפרינסטון, ניו ג'רזי.",
        "subject":   "אלברט איינשטיין",
        "predicate": "מקום לידה",
        "object":    "אולם",
        "label":     "1",
    },
    {
        "text":      "פריז היא בירת צרפת ועיר הגדולה ביותר במדינה.",
        "subject":   "פריז",
        "predicate": "בירת",
        "object":    "צרפת",
        "label":     "1",
    },
    {
        "text":      "לאונרדו דה וינצ'י היה אמן, מדען וממציא איטלקי מן הרנסנס.",
        "subject":   "לאונרדו דה וינצ'י",
        "predicate": "מקום לידה",
        "object":    "רומא",
        "label":     "0",
    },
]

FEW_SHOT_EN = [
    {
        "text":      "Albert Einstein was born on 14 March 1879 in Ulm, Germany, and died on 18 April 1955 in Princeton, New Jersey.",
        "subject":   "Albert Einstein",
        "predicate": "place of birth",
        "object":    "Ulm",
        "label":     "1",
    },
    {
        "text":      "Paris is the capital and largest city of France.",
        "subject":   "Paris",
        "predicate": "capital of",
        "object":    "France",
        "label":     "1",
    },
    {
        "text":      "Leonardo da Vinci was an Italian Renaissance artist, scientist, and inventor.",
        "subject":   "Leonardo da Vinci",
        "predicate": "place of birth",
        "object":    "Rome",
        "label":     "0",
    },
]

# ── Prompt builders ────────────────────────────────────────────────────────────

SYSTEM_HE = (
    "אתה מסווג חילוץ יחסים בעברית. "
    "בהינתן טקסט עברי ויחס בין ישויות (נושא, יחס, מושא), "
    "ענה בדיוק '1' אם היחס מופיע במפורש בטקסט, או '0' אם לא. "
    "פלט את הספרה בלבד, ללא הסבר."
)

SYSTEM_EN = (
    "You are a Hebrew relation extraction classifier. "
    "Given a Hebrew text and a relation triplet (subject, predicate, object), "
    "answer with exactly '1' if the relation is explicitly stated in the text, "
    "or '0' if it is not. Output only the digit, no explanation."
)


def _fmt_he(text, subject, predicate, obj):
    return f"טקסט: {text}\nיחס: {subject} {predicate} {obj}\nתשובה (1 או 0):"


def _fmt_en(text, subject, predicate, obj):
    return f"Text: {text}\nRelation: {subject} {predicate} {obj}\nAnswer (1 or 0):"


def build_messages(text, subject, predicate, obj, lang):
    """Return an OpenAI-format messages list with few-shot examples."""
    system   = SYSTEM_HE   if lang == "he" else SYSTEM_EN
    examples = FEW_SHOT_HE if lang == "he" else FEW_SHOT_EN
    fmt      = _fmt_he     if lang == "he" else _fmt_en

    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user",      "content": fmt(ex["text"], ex["subject"], ex["predicate"], ex["object"])})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": fmt(text, subject, predicate, obj)})
    return messages


# ── API call ───────────────────────────────────────────────────────────────────

def call_openrouter(model_id, messages, max_retries=MAX_RETRIES):
    """
    Call OpenRouter with retries. Returns the raw response dict or raises.
    Requests logprobs=True + top_logprobs=5 for confidence extraction.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      model_id,
        "messages":   messages,
        "max_tokens": 1,
        "logprobs":   True,
        "top_logprobs": 5,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                print(f"  [HTTP {resp.status_code}] retrying in {wait}s …")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")

    raise RuntimeError("Exhausted retries")


# ── Response parsing ───────────────────────────────────────────────────────────

def parse_response(response):
    """
    Extract (confidence, prediction) from an OpenRouter response.

    Confidence = P("1") / (P("0") + P("1")) derived from top_logprobs.
    If logprobs are absent, confidence is 1.0 or 0.0 based on the text token.
    """
    choice = response["choices"][0]
    text_token = choice["message"]["content"].strip()

    # Try logprobs path
    logprobs_data = choice.get("logprobs")
    if logprobs_data and logprobs_data.get("content"):
        top = logprobs_data["content"][0].get("top_logprobs", [])
        lp = {entry["token"].strip(): entry["logprob"] for entry in top}

        lp0 = lp.get("0")
        lp1 = lp.get("1")

        if lp0 is not None and lp1 is not None:
            p0 = math.exp(lp0)
            p1 = math.exp(lp1)
            confidence = p1 / (p0 + p1)
            prediction = 1 if confidence >= 0.5 else 0
            return confidence, prediction

        if lp1 is not None:
            return math.exp(lp1), 1
        if lp0 is not None:
            return 1.0 - math.exp(lp0), 1 if text_token == "1" else 0

    # Fallback: parse text response
    if "1" in text_token:
        return 1.0, 1
    if "0" in text_token:
        return 0.0, 0

    print(f"  [WARN] Unexpected response token: '{text_token}' — defaulting to 0")
    return 0.0, 0


# ── Per-row worker ─────────────────────────────────────────────────────────────

def classify_row(args):
    """Worker function for ThreadPoolExecutor. Returns (idx, confidence, prediction, usage)."""
    idx, model_id, messages = args
    try:
        response = call_openrouter(model_id, messages)
        conf, pred = parse_response(response)
        usage = response.get("usage", {})
    except Exception as e:
        print(f"  [ERROR] row {idx}: {e}")
        conf, pred, usage = None, None, {}
    return idx, conf, pred, usage


# ── Main ───────────────────────────────────────────────────────────────────────

def run_one_pass(df, model_id, tag, lang):
    """Submit all rows to the thread pool and collect results + token usage."""
    tasks = []
    for idx, row in df.iterrows():
        messages = build_messages(
            str(row["text"]), str(row["subject"]),
            str(row["predicate"]), str(row["object"]),
            lang,
        )
        tasks.append((idx, model_id, messages))

    results = {}
    total_prompt_tokens, total_completion_tokens = 0, 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(classify_row, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  [{tag}|{lang}]"):
            idx, conf, pred, usage = future.result()
            results[idx] = (conf, pred)
            total_prompt_tokens     += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)

    conf_col = [results[i][0] for i in df.index]
    pred_col = [results[i][1] for i in df.index]
    return conf_col, pred_col, total_prompt_tokens, total_completion_tokens


def main():
    import time, json as _json
    if not API_KEY:
        raise EnvironmentError("Set OPENROUTER_API_KEY environment variable before running.")

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded gold dataset: {len(df)} rows")

    t_start   = time.time()
    usage_log = []   # one entry per (model, lang) pass

    for cfg in MODELS:
        model_id, tag = cfg["id"], cfg["tag"]
        prices = PRICES_PER_1M.get(model_id, {"input": 0.0, "output": 0.0})
        print(f"\n{'='*60}\nModel: {tag}  ({model_id})\n{'='*60}")

        for lang in ("he", "en"):
            print(f"\n  -- Prompt language: {lang} --")
            t0 = time.time()
            conf, pred, pt, ct = run_one_pass(df, model_id, tag, lang)
            elapsed = time.time() - t0
            cost_usd = (pt * prices["input"] + ct * prices["output"]) / 1_000_000
            usage_log.append({
                "model_id": model_id, "tag": tag, "lang": lang,
                "prompt_tokens": pt, "completion_tokens": ct,
                "cost_usd": round(cost_usd, 6), "runtime_sec": round(elapsed, 2),
            })
            print(f"  tokens: {pt} prompt + {ct} completion = ${cost_usd:.4f}")
            df[f"confidence_{tag}_{lang}"]       = conf
            df[f"relation_present_{tag}_{lang}"] = pred

    total_runtime = time.time() - t_start
    total_cost    = sum(e["cost_usd"] for e in usage_log)

    metadata = {
        "script":           "gold_api_llm_classify.py",
        "runtime_sec":      round(total_runtime, 2),
        "cost_usd":         round(total_cost, 6),
        "usage_by_pass":    usage_log,
    }
    meta_path = OUTPUT_CSV.replace(".csv", "_metadata.json")
    with open(meta_path, "w") as f:
        _json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved to {OUTPUT_CSV}")

    rp_cols = [c for c in df.columns if c.startswith("relation_present_")]
    print("\nPrediction summary:")
    print(df[["relation_present"] + rp_cols].describe())


if __name__ == "__main__":
    main()
