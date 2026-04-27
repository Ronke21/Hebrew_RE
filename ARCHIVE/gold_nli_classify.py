# CUDA_VISIBLE_DEVICES=0 python clean_data/gold_nli_classify.py

"""
Run all finetuned Hebrew NLI encoders on the gold 500-row dataset and
add a relation_present_<model> and confidence_<model> column for each model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python gold_nli_classify.py

Output:
    data/crocodile_heb25_gold_500_nli_classified.csv
"""

import os
import re
import torch
import transformers
import pandas as pd
from tqdm import tqdm

ENTAILMENT_THRESHOLD = 0.75
BATCH_SIZE = 32
BASE_DIR = "/home/nlp/ronke21/hebrew_RE"
MODELS_DIR = os.path.join(BASE_DIR, "finetuned_Heb_NLI_encoders")
INPUT_CSV = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500_nli_classified.csv")

# (checkpoint_relative_path, short_tag)
MODELS = [
    ("mmBERT-base_hebnli/checkpoint-1000",          "mmBERT"),
    ("mt5-xl_hebnli/checkpoint-8000",               "mt5_xl"),
    ("multilingual-e5-large_hebnli/checkpoint-4000","me5large"),
    ("neodictabert_hebnli/checkpoint-4500",          "neodictabert"),
    ("xlm-roberta-large_hebnli/checkpoint-2000",    "xlmrobertalarge"),
]


def load_model(checkpoint_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    config = transformers.AutoConfig.from_pretrained(
        checkpoint_path,
        output_hidden_states=False,
        output_attentions=False,
        trust_remote_code=True,
    )
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, config=config, trust_remote_code=True, use_safetensors=True
        )
    except Exception as e:
        print(f"[WARN] safetensors load failed ({e}), retrying without flag.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, config=config, trust_remote_code=True
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.half()
    return model, tokenizer, device


def predict_entailment(model, tokenizer, device, premises, hypotheses):
    """Return entailment confidence scores (index 1) for each (premise, hypothesis) pair."""
    scores = []
    pairs = list(zip(premises, hypotheses))

    for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc="  batches"):
        batch = pairs[i : i + BATCH_SIZE]
        enc = tokenizer(
            [p for p, _ in batch],
            [h for _, h in batch],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="longest",
            truncation=True,
            return_token_type_ids=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, return_dict=True).logits
        probs = logits.softmax(dim=1)[:, 1].cpu().float().tolist()
        scores.extend(probs)

    return scores


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded gold dataset: {len(df)} rows, columns: {df.columns.tolist()}")

    # Build hypothesis strings: "subject predicate object"
    hypotheses = (
        df["subject"].astype(str) + " "
        + df["predicate"].astype(str) + " "
        + df["object"].astype(str)
    ).tolist()
    premises = df["text"].astype(str).tolist()

    for rel_path, tag in MODELS:
        checkpoint_path = os.path.join(MODELS_DIR, rel_path)
        print(f"\n=== Model: {tag}  ({checkpoint_path}) ===")

        model, tokenizer, device = load_model(checkpoint_path)
        scores = predict_entailment(model, tokenizer, device, premises, hypotheses)

        df[f"confidence_{tag}"] = scores
        df[f"relation_present_{tag}"] = (pd.Series(scores) >= ENTAILMENT_THRESHOLD).astype(int).values

        # Free GPU memory before loading the next model
        del model
        torch.cuda.empty_cache()

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved classified dataset to {OUTPUT_CSV}")
    print(df[[c for c in df.columns if c.startswith("relation_present")]].describe())


if __name__ == "__main__":
    main()
