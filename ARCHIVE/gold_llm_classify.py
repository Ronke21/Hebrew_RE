# CUDA_VISIBLE_DEVICES=0,1 python clean_data/gold_llm_classify.py

"""
Run open-source LLMs on the gold 500-row dataset to classify whether each
relation triplet is present in the text.

Each model is evaluated with TWO prompt languages (Hebrew + English) using
3-shot examples (2 positive, 1 negative), producing 4 columns per model:
  confidence_<tag>_he / relation_present_<tag>_he
  confidence_<tag>_en / relation_present_<tag>_en

Models:
  - google/gemma-4-31B-it          (instruct → chat-template few-shot)
  - dicta-il/DictaLM-3.0-24B-Base  (base     → completion few-shot)

Strategy: single forward pass — read logits at the last prompt token position,
compare P("1") vs P("0") for confidence score and binary prediction (threshold 0.5).

Output:
  data/crocodile_heb25_gold_500_llm_classified.csv
"""

import os
import torch
import pandas as pd
import transformers
from tqdm import tqdm

BASE_DIR   = "/home/nlp/ronke21/hebrew_RE"
INPUT_CSV  = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500_llm_classified.csv")

BATCH_SIZE = 4   # reduce to 1 if OOM

MODELS = [
    {"hf_id": "google/gemma-4-31B-it",         "tag": "gemma4_31b",   "style": "instruct"},
    {"hf_id": "dicta-il/DictaLM-3.0-24B-Base", "tag": "dictaLM3_24b","style": "base"},
]

# ── Few-shot examples ──────────────────────────────────────────────────────────
# 3 examples: 2 positive (label=1), 1 negative (label=0).
# Hebrew examples use Hebrew text + Hebrew subject/predicate/object.
# English examples use English text + English subject/predicate/object.
# (Test rows always contain Hebrew text regardless of prompt language.)

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
    "פלט את הספרה בלבד."
)

SYSTEM_EN = (
    "You are a Hebrew relation extraction classifier. "
    "Given a Hebrew text and a relation triplet (subject, predicate, object), "
    "answer with exactly '1' if the relation is explicitly stated in the text, "
    "or '0' if it is not. Output only the digit."
)


def _fmt_he(text, subject, predicate, obj):
    return f"טקסט: {text}\nיחס: {subject} {predicate} {obj}\nתשובה (1 או 0):"


def _fmt_en(text, subject, predicate, obj):
    return f"Text: {text}\nRelation: {subject} {predicate} {obj}\nAnswer (1 or 0):"


def build_instruct_prompt(tokenizer, text, subject, predicate, obj, lang):
    """Chat-template few-shot for instruction-tuned models (e.g. Gemma-it)."""
    system   = SYSTEM_HE   if lang == "he" else SYSTEM_EN
    examples = FEW_SHOT_HE if lang == "he" else FEW_SHOT_EN
    fmt      = _fmt_he     if lang == "he" else _fmt_en

    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user",      "content": fmt(ex["text"], ex["subject"], ex["predicate"], ex["object"])})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": fmt(text, subject, predicate, obj)})

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_base_prompt(text, subject, predicate, obj, lang):
    """Completion-style few-shot for base models (e.g. DictaLM-Base)."""
    if lang == "he":
        header   = "להלן טקסט עברי ויחס בין ישויות. ענה 1 אם היחס מופיע ישירות בטקסט, או 0 אם לא.\n\n"
        examples = FEW_SHOT_HE
        fmt      = _fmt_he
    else:
        header   = "Below is a Hebrew text and a relation between entities. Answer 1 if the relation is explicitly present in the text, or 0 if not.\n\n"
        examples = FEW_SHOT_EN
        fmt      = _fmt_en

    prompt = header
    for ex in examples:
        prompt += fmt(ex["text"], ex["subject"], ex["predicate"], ex["object"])
        prompt += f" {ex['label']}\n\n"
    prompt += fmt(text, subject, predicate, obj)
    return prompt


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(hf_id):
    print(f"  Loading tokenizer: {hf_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {hf_id}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


# ── Inference ──────────────────────────────────────────────────────────────────

def get_token_ids_for_digit(tokenizer, digit: str):
    """All single-token IDs that decode to the digit (with/without leading space)."""
    ids = set()
    for prefix in ("", " ", "\n"):
        enc = tokenizer.encode(prefix + digit, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return list(ids)


def classify_batch(model, tokenizer, prompts, tok0_ids, tok1_ids):
    """
    Single forward pass. Returns (confidences, predictions).
    confidence = P("1") / (P("0") + P("1"))   prediction = 1 if confidence >= 0.5
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    input_ids      = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    # logits at the last non-padding token for each sequence
    seq_lens    = attention_mask.sum(dim=1)
    last_logits = out.logits[torch.arange(len(prompts)), seq_lens - 1]  # (B, vocab)
    probs       = last_logits.float().softmax(dim=-1)

    p0 = probs[:, tok0_ids].sum(dim=-1)
    p1 = probs[:, tok1_ids].sum(dim=-1)

    confidence = (p1 / (p0 + p1).clamp(min=1e-9)).cpu().tolist()
    prediction = [1 if c >= 0.5 else 0 for c in confidence]
    return confidence, prediction


def run_one_pass(model, tokenizer, tok0_ids, tok1_ids, df, style, lang, tag):
    """Build prompts for all rows in the given language and run inference."""
    fmt = build_instruct_prompt if style == "instruct" else build_base_prompt

    prompts = []
    for _, row in df.iterrows():
        text = str(row["text"])
        subj = str(row["subject"])
        pred = str(row["predicate"])
        obj  = str(row["object"])
        if style == "instruct":
            prompts.append(build_instruct_prompt(tokenizer, text, subj, pred, obj, lang))
        else:
            prompts.append(build_base_prompt(text, subj, pred, obj, lang))

    all_conf, all_pred = [], []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc=f"  [{tag}|{lang}] batches"):
        conf, pred = classify_batch(model, tokenizer, prompts[i:i+BATCH_SIZE], tok0_ids, tok1_ids)
        all_conf.extend(conf)
        all_pred.extend(pred)
    return all_conf, all_pred


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded gold dataset: {len(df)} rows, columns: {df.columns.tolist()}")

    for cfg in MODELS:
        hf_id, tag, style = cfg["hf_id"], cfg["tag"], cfg["style"]
        print(f"\n{'='*60}\nModel: {tag}  ({hf_id})\n{'='*60}")

        model, tokenizer = load_model(hf_id)
        tok0_ids = get_token_ids_for_digit(tokenizer, "0")
        tok1_ids = get_token_ids_for_digit(tokenizer, "1")
        print(f"  Token IDs  '0': {tok0_ids}  |  '1': {tok1_ids}")

        for lang in ("he", "en"):
            print(f"\n  -- Prompt language: {lang} --")
            conf, pred = run_one_pass(model, tokenizer, tok0_ids, tok1_ids, df, style, lang, tag)
            df[f"confidence_{tag}_{lang}"]       = conf
            df[f"relation_present_{tag}_{lang}"] = pred

        del model
        torch.cuda.empty_cache()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")

    rp_cols = [c for c in df.columns if c.startswith("relation_present_")]
    print("\nPrediction summary:")
    print(df[["relation_present"] + rp_cols].describe())


if __name__ == "__main__":
    main()
