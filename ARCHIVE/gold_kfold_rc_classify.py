# CUDA_VISIBLE_DEVICES=0 python clean_data/gold_kfold_rc_classify.py

"""
K-Fold Relation Classification (RC) approach for Hebrew RE gold dataset.

Pipeline:
  1. Sample 100K rows from the full dataset (noisy distant-supervision labels).
  2. Filter to predicates with >= MIN_PRED_SAMPLES occurrences (multi-class RC).
  3. K-fold CV for each encoder:
       - For each fold k: train on the other K-1 folds, predict on fold k.
       - This gives every row one out-of-fold (OOF) prediction from a model
         that never saw it during training — different folds → different noise.
  4. Evaluate on the 500-row gold dataset:
       - Average the K fold-models' predicted P(correct predicate).
       - confidence = avg_P(gold_predicate)
       - relation_present = 1 if confidence >= THRESHOLD

Encoders (base models from HuggingFace, no prior NLI fine-tuning):
  - xlm-roberta-base               → tag: xlmroberta
  - bert-base-multilingual-cased   → tag: mmbert
  - dicta-il/neodictabert          → tag: neodictabert

Output:
  data/crocodile_heb25_gold_500_kfold_rc_classified.csv
  outputs/kfold_rc/<encoder_tag>/fold_<k>/  (saved checkpoints)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import KFold
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR       = "/home/nlp/ronke21/hebrew_RE"
FULL_CSV       = os.path.join(BASE_DIR, "data", "crocodile_heb25_full_dataset_3124k.csv")
GOLD_CSV       = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500.csv")
OUTPUT_CSV     = os.path.join(BASE_DIR, "data", "crocodile_heb25_gold_500_kfold_rc_classified.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "outputs", "kfold_rc")

SAMPLE_SIZE       = 100_000
K_FOLDS           = 5
MIN_PRED_SAMPLES  = 50    # discard predicates with fewer occurrences in the 100K sample
MAX_LENGTH        = 256
TRAIN_BATCH_SIZE  = 16
EVAL_BATCH_SIZE   = 32
EPOCHS            = 3
LR                = 2e-5
WARMUP_RATIO      = 0.06
GRAD_CLIP         = 1.0
SEED              = 42
THRESHOLD         = 0.3   # P(correct predicate) threshold for relation_present=1

ENCODERS = [
    {"hf_id": "xlm-roberta-base",               "tag": "xlmroberta",   "trust_remote_code": False},
    {"hf_id": "bert-base-multilingual-cased",    "tag": "mmbert",       "trust_remote_code": False},
    {"hf_id": "dicta-il/neodictabert",           "tag": "neodictabert", "trust_remote_code": True},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ────────────────────────────────────────────────────────────────────

class RCDataset(Dataset):
    """
    Input:  text (truncated) + entity pair "subject [SEP] object"
    Label:  predicate class id  (None for inference)
    """
    def __init__(self, texts, subjects, objects, labels, tokenizer, max_length):
        self.texts     = texts
        self.subjects  = subjects
        self.objects   = objects
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        entity_pair = str(self.subjects[idx]) + " " + str(self.objects[idx])
        enc = self.tokenizer(
            str(self.texts[idx]),
            entity_pair,
            max_length=self.max_len,
            truncation="only_first",   # keep entity pair intact, truncate text
            padding="max_length",
            return_tensors="pt",
            return_token_type_ids=False,
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Training and evaluation ────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="    train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_probs(model, loader, device, num_classes):
    """Return (N, num_classes) probability matrix."""
    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="    predict", leave=False):
        labels = batch.pop("labels", None)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs  = logits.float().softmax(dim=-1).cpu()
        all_probs.append(probs)
    return torch.cat(all_probs, dim=0).numpy()


def build_model(hf_id, num_labels, trust_remote_code):
    return AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=num_labels,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=True,
    )


def build_optimizer_scheduler(model, num_training_steps):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


# ── K-Fold runner ──────────────────────────────────────────────────────────────

def run_kfold_encoder(df_train, gold_df, label2id, hf_id, tag, trust_remote_code, device):
    """
    For one encoder:
      - Run K-fold CV on df_train, collect out-of-fold probability matrices.
      - Accumulate probability matrices on gold_df across all K folds.
    Returns:
      oof_probs  : (N_train, num_classes) — averaged OOF predictions
      gold_probs : (N_gold,  num_classes) — averaged predictions across K fold-models
    """
    num_classes = len(label2id)
    texts   = df_train["text"].tolist()
    subjs   = df_train["subject"].tolist()
    objs    = df_train["object"].tolist()
    labels  = df_train["label_id"].tolist()
    indices = np.arange(len(df_train))

    gold_texts = gold_df["text"].tolist()
    gold_subjs = gold_df["subject"].tolist()
    gold_objs  = gold_df["object"].tolist()

    oof_probs  = np.zeros((len(df_train), num_classes), dtype=np.float32)
    gold_probs = np.zeros((len(gold_df),  num_classes), dtype=np.float32)

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    print(f"  Loading tokenizer: {hf_id}")
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trust_remote_code)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n  -- Fold {fold_idx+1}/{K_FOLDS} --  train={len(train_idx)}  val={len(val_idx)}")
        ckpt_path = os.path.join(CHECKPOINT_DIR, tag, f"fold_{fold_idx+1}")
        os.makedirs(ckpt_path, exist_ok=True)

        train_ds = RCDataset(
            [texts[i]  for i in train_idx], [subjs[i] for i in train_idx],
            [objs[i]   for i in train_idx], [labels[i] for i in train_idx],
            tokenizer, MAX_LENGTH,
        )
        val_ds = RCDataset(
            [texts[i]  for i in val_idx], [subjs[i] for i in val_idx],
            [objs[i]   for i in val_idx], [labels[i] for i in val_idx],
            tokenizer, MAX_LENGTH,
        )
        gold_ds = RCDataset(
            gold_texts, gold_subjs, gold_objs, [0]*len(gold_df),
            tokenizer, MAX_LENGTH,
        )

        train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=EVAL_BATCH_SIZE,  shuffle=False, num_workers=4, pin_memory=True)
        gold_loader  = DataLoader(gold_ds,  batch_size=EVAL_BATCH_SIZE,  shuffle=False, num_workers=4, pin_memory=True)

        model = build_model(hf_id, num_classes, trust_remote_code).to(device)
        num_steps = len(train_loader) * EPOCHS
        optimizer, scheduler = build_optimizer_scheduler(model, num_steps)

        best_loss = float("inf")
        for epoch in range(EPOCHS):
            loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            print(f"    Epoch {epoch+1}/{EPOCHS}  loss={loss:.4f}")
            if loss < best_loss:
                best_loss = loss
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

        # Out-of-fold predictions
        fold_val_probs = predict_probs(model, val_loader, device, num_classes)
        oof_probs[val_idx] = fold_val_probs

        # Gold predictions — accumulate across folds (will average later)
        fold_gold_probs = predict_probs(model, gold_loader, device, num_classes)
        gold_probs += fold_gold_probs

        del model
        torch.cuda.empty_cache()

    gold_probs /= K_FOLDS  # average across K fold-models
    return oof_probs, gold_probs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load and sample full dataset ──────────────────────────────────────────
    print(f"\nLoading full dataset (sampling {SAMPLE_SIZE:,} rows)…")
    df_full = pd.read_csv(FULL_CSV, usecols=["text", "subject", "predicate", "object"])
    df_sample = df_full.sample(n=SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)
    del df_full

    # Filter rare predicates
    pred_counts = df_sample["predicate"].value_counts()
    valid_preds = pred_counts[pred_counts >= MIN_PRED_SAMPLES].index.tolist()
    df_sample = df_sample[df_sample["predicate"].isin(valid_preds)].reset_index(drop=True)
    print(f"  After predicate filter (>={MIN_PRED_SAMPLES} samples): "
          f"{len(df_sample):,} rows, {len(valid_preds)} predicate classes")

    # Encode predicate labels
    label2id = {p: i for i, p in enumerate(sorted(valid_preds))}
    id2label = {i: p for p, i in label2id.items()}
    df_sample["label_id"] = df_sample["predicate"].map(label2id)

    # Save label mapping for reference
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(os.path.join(CHECKPOINT_DIR, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print(f"  Label map saved to {CHECKPOINT_DIR}/label2id.json")

    # ── Load gold dataset ─────────────────────────────────────────────────────
    gold_df = pd.read_csv(GOLD_CSV)
    print(f"\nGold dataset: {len(gold_df)} rows")

    # Pre-compute gold predicate IDs (None if unseen)
    gold_pred_ids = [label2id.get(p) for p in gold_df["predicate"].tolist()]
    n_unseen = sum(1 for x in gold_pred_ids if x is None)
    print(f"  Gold predicates covered by label set: {len(gold_df)-n_unseen}/{len(gold_df)}")

    # ── Run K-fold for each encoder ───────────────────────────────────────────
    for enc in ENCODERS:
        hf_id, tag, trust = enc["hf_id"], enc["tag"], enc["trust_remote_code"]
        print(f"\n{'='*60}\nEncoder: {tag}  ({hf_id})\n{'='*60}")

        oof_probs, gold_probs = run_kfold_encoder(
            df_sample, gold_df, label2id, hf_id, tag, trust, device
        )

        # Save OOF predictions
        oof_path = os.path.join(CHECKPOINT_DIR, tag, "oof_probs.npy")
        np.save(oof_path, oof_probs)
        print(f"  OOF probs saved to {oof_path}")

        # Gold: confidence = avg P(correct predicate class) across K fold-models
        confidences = []
        for i, pred_id in enumerate(gold_pred_ids):
            if pred_id is None:
                confidences.append(0.0)
            else:
                confidences.append(float(gold_probs[i, pred_id]))

        gold_df[f"confidence_kfold_{tag}"]       = confidences
        gold_df[f"relation_present_kfold_{tag}"] = [
            1 if c >= THRESHOLD else 0 for c in confidences
        ]

        # OOF accuracy on training data (sanity check)
        oof_preds = oof_probs.argmax(axis=1)
        oof_acc   = (oof_preds == df_sample["label_id"].values).mean()
        print(f"  OOF accuracy: {oof_acc:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    gold_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved gold classifications to {OUTPUT_CSV}")

    rp_cols = [c for c in gold_df.columns if c.startswith("relation_present_kfold_")]
    print("\nPrediction summary:")
    print(gold_df[["relation_present"] + rp_cols].describe())


if __name__ == "__main__":
    main()
