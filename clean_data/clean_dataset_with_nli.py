"""
Run multiple NLI encoder models on a CSV file with multiple premise-hypothesis column pairs.
For each (model, hypothesis_col) combination adds:
  confidence_{tag}_{hyp_col}  — entailment probability

Evaluates each combination at multiple fixed thresholds AND with 5-fold CV
threshold optimisation against the gold label column, then writes:
  - per-combo metrics table
  - k-fold CV threshold search results (honest OOF F1 estimate)
  - ensemble columns
  - summary file / error analysis file

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m clean_data.clean_dataset_with_nli
    CUDA_VISIBLE_DEVICES=2 python -m clean_data.clean_dataset_with_nli --input outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv
    CUDA_VISIBLE_DEVICES=4 python -m clean_data.clean_dataset_with_nli 2>&1 | tee nli_run_temp.txt
"""

import os
import csv
import time
import logging
import argparse

import numpy as np
import torch
import transformers
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE          = "outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_FILE         = "outputs/NLI_clean/nli_classified.csv"
LOG_FILE            = "outputs/NLI_clean/nli_classify.log"
SUMMARY_FILE        = "outputs/NLI_clean/nli_summary.txt"
ERROR_ANALYSIS_FILE = "outputs/NLI_clean/nli_error_analysis.txt"

ANALYSIS_THRESHOLD = 0.7   # threshold for per-row decisions in error analysis

LABEL_COL = "relation_present"   # gold label column: "1" = present, "0" = absent

# K-fold CV threshold optimisation
CV_N_SPLITS    = 5
THRESHOLD_GRID = [round(t / 100, 2) for t in range(5, 96, 5)]   # 0.05 … 0.95

# Base directory for all local NLI encoder checkpoints
NLI_ENCODERS_BASE = "finetuned_Heb_NLI_encoders"

# List of (relative_checkpoint_path, short_tag) to run.
# Tag is used in output column names: confidence_{tag}_{hyp_col}
NLI_MODELS = [
    ("mmBERT-base_hebnli/checkpoint-1000",           "mmbert"),
    ("multilingual-e5-large_hebnli/checkpoint-4000", "me5large"),
    ("neodictabert_hebnli/checkpoint-4500",          "neodictabert"),
    ("xlm-roberta-large_hebnli/checkpoint-2000",     "xlmroberta"),
]

# List of (premise_col, hypothesis_col) pairs to evaluate per model
PREMISE_HYPOTHESIS_PAIRS = [
    ("text", "basic_relation"),
    ("text", "template_relation"),
    ("text", "llm_relation"),
]

# Fixed thresholds at which to also evaluate each (model, hypothesis) combination
EVAL_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

NLI_MAX_LENGTH        = 256
DEFAULT_BATCH_SIZE    = 64
LARGE_TEXT_THRESHOLD  = 256    # combined char len above which smaller batch is used
LARGE_TEXT_BATCH_SIZE = 12
LOG_PROGRESS_EVERY    = 25     # log inference progress every N% of batches


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TqdmToLogger:
    def __init__(self, logger):
        self._logger = logger

    def write(self, msg):
        msg = msg.strip()
        if msg:
            self._logger.info(msg)

    def flush(self):
        pass


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("clean_dataset_with_nli")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:  return f"{h}h {m}m {s}s"
    if m:  return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scores: list[float], labels: list[str], threshold: float) -> dict:
    TP = FP = FN = TN = 0
    for score, label in zip(scores, labels):
        pred = score >= threshold
        gold = label == "1"
        if   pred and gold:      TP += 1
        elif pred and not gold:  FP += 1
        elif not pred and gold:  FN += 1
        else:                    TN += 1

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP)    if (TP + FP) else 0
    recall    = TP / (TP + FN)    if (TP + FN) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    sv     = sorted(scores)
    n      = len(sv)
    mean   = sum(scores) / n if n else 0
    median = sv[n // 2]      if n else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "score_mean": mean, "score_median": median,
        "score_min": sv[0] if n else 0, "score_max": sv[-1] if n else 0,
    }


# ---------------------------------------------------------------------------
# K-fold CV threshold optimisation
# ---------------------------------------------------------------------------

def kfold_threshold_cv(
    scores: list[float],
    labels: list[str],
    n_splits: int,
    threshold_grid: list[float],
) -> tuple[float, float, list[float], list[float]]:
    """
    Stratified k-fold CV: for each fold find the threshold maximising F1 on the
    training split, then evaluate that threshold on the held-out fold.
    Returns (mean_cv_f1, std_cv_f1, fold_thresholds, fold_f1s).
    This gives an honest OOF estimate — the threshold is never fit on the eval data.
    """
    scores_arr = np.array(scores)
    labels_int = np.array([int(l) for l in labels])

    kf        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s: list[float]       = []
    fold_thresholds: list[float] = []

    for train_idx, val_idx in kf.split(scores_arr.reshape(-1, 1), labels_int):
        tr_s, tr_l = scores_arr[train_idx], labels_int[train_idx]
        va_s, va_l = scores_arr[val_idx],   labels_int[val_idx]

        # ── find best threshold on train split ────────────────────────────────
        best_t, best_train_f1 = threshold_grid[0], -1.0
        for t in threshold_grid:
            preds = (tr_s >= t).astype(int)
            tp = int(((preds == 1) & (tr_l == 1)).sum())
            fp = int(((preds == 1) & (tr_l == 0)).sum())
            fn = int(((preds == 0) & (tr_l == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_train_f1:
                best_train_f1, best_t = f1, t

        fold_thresholds.append(best_t)

        # ── evaluate on val split ─────────────────────────────────────────────
        val_preds = (va_s >= best_t).astype(int)
        tp = int(((val_preds == 1) & (va_l == 1)).sum())
        fp = int(((val_preds == 1) & (va_l == 0)).sum())
        fn = int(((val_preds == 0) & (va_l == 1)).sum())
        prec    = tp / (tp + fp) if (tp + fp) else 0.0
        rec     = tp / (tp + fn) if (tp + fn) else 0.0
        val_f1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fold_f1s.append(val_f1)

    arr = np.array(fold_f1s)
    return float(arr.mean()), float(arr.std()), fold_thresholds, fold_f1s


# ---------------------------------------------------------------------------
# Ensemble (soft majority voting)
# ---------------------------------------------------------------------------

def compute_ensemble(rows: list[dict], conf_col_names: list[str]) -> list[float]:
    results = []
    for row in rows:
        vals = [float(row[col]) for col in conf_col_names if col in row]
        results.append(sum(vals) / len(vals) if vals else 0.0)
    return results


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(model_path: str, log: logging.Logger):
    log.info(f"    path: {model_path}")
    log.info(f"    torch {torch.__version__}  CUDA build={torch.version.cuda}  "
             f"available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Make sure PyTorch is installed with CUDA support "
            "and CUDA_VISIBLE_DEVICES is set correctly."
        )

    log.info(f"    visible GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config    = transformers.AutoConfig.from_pretrained(
        model_path, output_hidden_states=False, output_attentions=False, trust_remote_code=True
    )

    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config, trust_remote_code=True, use_safetensors=True
        )
    except Exception as e:
        log.warning(f"    safetensors load failed ({e}), trying normal load.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config, trust_remote_code=True
        )

    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"    ready on {device} ({n_params:.0f}M params)")
    return model, tokenizer, device


def unload_model(model, log: logging.Logger):
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


# ---------------------------------------------------------------------------
# NLI inference
# ---------------------------------------------------------------------------

def _adaptive_batch_size(texts: list[tuple[str, str]]) -> int:
    if not texts:
        return DEFAULT_BATCH_SIZE
    max_len = max(len(p) + len(h) for p, h in texts)
    return LARGE_TEXT_BATCH_SIZE if max_len > LARGE_TEXT_THRESHOLD else DEFAULT_BATCH_SIZE


def run_nli(
    model, tokenizer, device,
    texts: list[tuple[str, str]],
    log: logging.Logger,
    desc: str = "",
) -> list[float]:
    if not texts:
        return []

    batch_size = _adaptive_batch_size(texts)
    n_batches  = (len(texts) + batch_size - 1) // batch_size
    log_every  = max(1, n_batches * LOG_PROGRESS_EVERY // 100)
    log.info(f"    rows={len(texts)}, batch_size={batch_size}, n_batches={n_batches}")

    result      = []
    t_inf_start = time.time()

    for batch_idx, start in enumerate(
        tqdm(range(0, len(texts), batch_size), desc=f"    {desc}",
             leave=False, file=TqdmToLogger(log)), 1
    ):
        batch   = texts[start : start + batch_size]
        encoded = tokenizer(
            [p for p, _ in batch], [h for _, h in batch],
            return_tensors="pt", add_special_tokens=True,
            max_length=NLI_MAX_LENGTH, padding="longest",
            return_token_type_ids=False, truncation=True,
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)

        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)

        result.append(outputs["logits"].softmax(dim=1))
        del outputs

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed   = time.time() - t_inf_start
            rows_done = min(start + batch_size, len(texts))
            log.info(
                f"    progress: {rows_done}/{len(texts)} rows "
                f"({100 * rows_done / len(texts):.0f}%)  {_fmt_duration(elapsed)}"
            )

    logits = torch.cat(result)
    scores = logits[:, 1] if logits.dim() > 1 else logits.unsqueeze(0)
    return scores.cpu().tolist()


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    run_stats: list[dict],
    summary_path: str,
    log: logging.Logger,
    total_time: float,
    ensemble_stats: list[dict] | None = None,
) -> None:
    W = 110
    lines: list[str] = []

    lines.append("=" * W)
    lines.append("NLI CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time : {_fmt_duration(total_time)}")
    lines.append(f"Fixed thresholds: {EVAL_THRESHOLDS}")
    lines.append(f"CV folds        : {CV_N_SPLITS}  |  "
                 f"thresh grid: {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} "
                 f"({len(THRESHOLD_GRID)} values, step 0.05)")
    lines.append("=" * W)

    # ── Per-model timing ──────────────────────────────────────────────────────
    model_times: dict[str, float] = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0)
        model_times[s["model"]] += s["time"]
    lines.append("")
    lines.append("Per-model total inference time:")
    for model, t in model_times.items():
        lines.append(f"  {model:<16}  {_fmt_duration(t)}")

    # ── Fixed-threshold metrics table ─────────────────────────────────────────
    lines.append("")
    lines.append("FIXED-THRESHOLD METRICS")
    hdr = (
        f"  {'model':<14} {'hypothesis':<22} {'thresh':>6}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'mean_conf':>9}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in run_stats:
        first = True
        for thresh, m in s["metrics_by_threshold"].items():
            time_str = _fmt_duration(s["time"]) if first else ""
            lines.append(
                f"  {s['model']:<14} {s['hypothesis']:<22} {thresh:>6.1f}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                f"{m['score_mean']:>9.4f}  {time_str:>8}"
            )
            first = False
        lines.append("  " + "-" * (len(hdr) - 2))

    # ── K-fold CV threshold optimisation ─────────────────────────────────────
    lines.append("")
    lines.append(f"K-FOLD CV THRESHOLD OPTIMISATION  ({CV_N_SPLITS} folds, stratified, random_state=42)")
    lines.append("  NOTE: threshold found on train folds, evaluated on held-out fold — honest OOF F1.")
    hdr_cv = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  {'mean_t':>6}  fold_thresholds → fold_F1s"
    )
    lines.append(hdr_cv)
    lines.append("  " + "-" * 90)
    for s in sorted(run_stats, key=lambda x: -x["cv_mean_f1"]):
        mean_t     = sum(s["cv_thresholds"]) / len(s["cv_thresholds"])
        thresh_str = "  ".join(f"{t:.2f}" for t in s["cv_thresholds"])
        f1_str     = "  ".join(f"{f:.3f}" for f in s["cv_fold_f1s"])
        lines.append(
            f"  {s['model']:<14} {s['hypothesis']:<22}  "
            f"{s['cv_mean_f1']:>8.4f} {s['cv_std_f1']:>7.4f}  {mean_t:>6.2f}  "
            f"[{thresh_str}] → [{f1_str}]"
        )

    # ── Ensemble metrics ──────────────────────────────────────────────────────
    if ensemble_stats:
        lines.append("")
        lines.append("ENSEMBLE COLUMNS  (averaged confidence scores)")
        hdr2 = (
            f"  {'column':<36} {'thresh':>6}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
            f"{'mean_conf':>9}  {'voters':>6}"
        )
        lines.append(hdr2)
        lines.append("  " + "-" * (len(hdr2) - 2))
        for s in ensemble_stats:
            first = True
            for thresh, m in s["metrics_by_threshold"].items():
                voters_str = str(s["n_voters"]) if first else ""
                lines.append(
                    f"  {s['col']:<36} {thresh:>6.1f}  "
                    f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                    f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                    f"{m['score_mean']:>9.4f}  {voters_str:>6}"
                )
                first = False
            lines.append("  " + "-" * (len(hdr2) - 2))

    lines.append("=" * W)
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    log.info("=" * W)
    for line in lines:
        log.info(line)
    log.info("=" * W)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def _get_metrics_at(s: dict, thresh: float) -> dict:
    if thresh in s["metrics_by_threshold"]:
        return s["metrics_by_threshold"][thresh]
    closest = min(s["metrics_by_threshold"], key=lambda t: abs(t - thresh))
    return s["metrics_by_threshold"][closest]


def write_error_analysis(
    rows: list[dict],
    labels: list[str],
    run_stats: list[dict],
    analysis_path: str,
    log: logging.Logger,
) -> None:
    if not run_stats or not rows:
        log.warning("[error_analysis] No data to analyze.")
        return

    thresh      = ANALYSIS_THRESHOLD
    n_rows      = len(rows)
    hyp_cols    = list(dict.fromkeys(s["hypothesis"] for s in run_stats))
    premise_col = run_stats[0]["premise"] if run_stats else "text"

    all_combos    = [(s["model"], s["hypothesis"]) for s in run_stats]
    combo_scores: dict[tuple, list[float]] = {}
    combo_preds:  dict[tuple, list[bool]]  = {}
    for s in run_stats:
        key            = (s["model"], s["hypothesis"])
        conf_col       = f"confidence_{s['model']}_{s['hypothesis']}"
        sc             = [float(r[conf_col]) for r in rows]
        combo_scores[key] = sc
        combo_preds[key]  = [v >= thresh for v in sc]

    gold_bool = [lbl == "1" for lbl in labels]
    n_pos     = sum(gold_bool)

    lines: list[str] = []

    def sep(title: str = "") -> None:
        lines.append("")
        lines.append("=" * 100)
        if title:
            lines.append(f"  {title}")
            lines.append("=" * 100)

    sep()
    lines.append("  NLI ERROR ANALYSIS")
    lines.append(f"  Analysis threshold : {thresh}")
    lines.append(f"  Total rows         : {n_rows}")
    lines.append(f"  Positive / Negative: {n_pos} / {n_rows - n_pos}  "
                 f"({100 * n_pos / n_rows:.1f}% / {100 * (n_rows - n_pos) / n_rows:.1f}%)")
    lines.append("=" * 100)

    # ── Section 1: Best configuration per metric ──────────────────────────────
    sep("SECTION 1 — BEST CONFIGURATION PER METRIC")
    for metric in ("f1", "accuracy", "precision", "recall"):
        best_val, best_key = -1.0, None
        for s in run_stats:
            for t, m in s["metrics_by_threshold"].items():
                if m[metric] > best_val:
                    best_val = m[metric]
                    best_key = (s["model"], s["hypothesis"], t)
        if best_key:
            lines.append(
                f"  {metric:<10}  {best_val:.4f}  →  "
                f"model={best_key[0]}  hyp={best_key[1]}  thresh={best_key[2]}"
            )

    # ── Section 2: CV best thresholds ─────────────────────────────────────────
    sep(f"SECTION 2 — CV BEST THRESHOLDS  ({CV_N_SPLITS}-fold, metrics at mean CV threshold)")
    hdr2 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  {'mean_t':>6}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1@t':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for s in sorted(run_stats, key=lambda x: -x["cv_mean_f1"]):
        mean_t = sum(s["cv_thresholds"]) / len(s["cv_thresholds"])
        key    = (s["model"], s["hypothesis"])
        m      = compute_metrics(combo_scores[key], labels, mean_t)
        lines.append(
            f"  {s['model']:<14} {s['hypothesis']:<22}  "
            f"{s['cv_mean_f1']:>8.4f} {s['cv_std_f1']:>7.4f}  {mean_t:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    # ── Section 3: Per-hypothesis ranking ────────────────────────────────────
    sep(f"SECTION 3 — PER-HYPOTHESIS RANKING  (thresh={thresh})")
    col_hdr = (
        f"  {'rank':<5} {'model':<14} "
        f"{'f1':>6} {'acc':>6} {'prec':>6} {'rec':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    for hyp_col in hyp_cols:
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append(col_hdr)
        lines.append("  " + "-" * (len(col_hdr) - 2))
        ranked = sorted(
            [s for s in run_stats if s["hypothesis"] == hyp_col],
            key=lambda s: _get_metrics_at(s, thresh)["f1"], reverse=True
        )
        for rank, s in enumerate(ranked, 1):
            m = _get_metrics_at(s, thresh)
            lines.append(
                f"  {rank:<5} {s['model']:<14} "
                f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
            )

    # ── Section 4: Model agreement matrix ────────────────────────────────────
    sep(f"SECTION 4 — MODEL AGREEMENT MATRIX  (thresh={thresh}, % same binary prediction)")
    for hyp_col in hyp_cols:
        avail_tags = [s["model"] for s in run_stats if s["hypothesis"] == hyp_col]
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append("  " + " " * 14 + "".join(f"{t:>14}" for t in avail_tags))
        for t1 in avail_tags:
            p1      = combo_preds[(t1, hyp_col)]
            row_str = f"  {t1:<14}"
            for t2 in avail_tags:
                p2     = combo_preds[(t2, hyp_col)]
                agree  = sum(a == b for a, b in zip(p1, p2)) / n_rows
                row_str += f"  {agree:>10.3f}  "
            lines.append(row_str)

    # ── Section 5: Hard examples — every combo wrong ──────────────────────────
    sep(f"SECTION 5 — HARD EXAMPLES: ALL COMBOS WRONG  (thresh={thresh})")
    row_n_correct = [
        sum(1 for key in all_combos if combo_preds[key][i] == gold_bool[i])
        for i in range(n_rows)
    ]
    hard_idx = [i for i, nc in enumerate(row_n_correct) if nc == 0]
    lines.append(
        f"\n  {len(hard_idx)} rows ({len(hard_idx) / n_rows * 100:.1f}%) "
        "where every model+hypothesis prediction was wrong.  Showing up to 20."
    )
    for i in hard_idx[:20]:
        lines.append("")
        lines.append(f"  [row {i}]  gold={labels[i]}")
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        for key in all_combos:
            tag, hyp_col = key
            lines.append(
                f"    [{tag}/{hyp_col}]  "
                f"conf={combo_scores[key][i]:.4f}  pred={'1' if combo_preds[key][i] else '0'}"
            )

    # ── Section 6: Confident mistakes ────────────────────────────────────────
    sep("SECTION 6 — CONFIDENT MISTAKES")
    CONF_FP_MIN = 0.85
    CONF_FN_MAX = 0.15
    fp_rows, fn_rows = [], []
    for i in range(n_rows):
        confs = {key: combo_scores[key][i] for key in all_combos}
        if not gold_bool[i]:
            worst_key = max(confs, key=confs.get)
            if confs[worst_key] >= CONF_FP_MIN:
                fp_rows.append((i, worst_key, confs[worst_key], confs))
        else:
            worst_key = min(confs, key=confs.get)
            if confs[worst_key] <= CONF_FN_MAX:
                fn_rows.append((i, worst_key, confs[worst_key], confs))
    fp_rows.sort(key=lambda x: -x[2])
    fn_rows.sort(key=lambda x:  x[2])

    lines.append(
        f"\n  False Positives (gold=0, max_conf >= {CONF_FP_MIN}): "
        f"{len(fp_rows)} — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fp_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=0  "
            f"most-confident={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    lines.append(
        f"\n  False Negatives (gold=1, min_conf <= {CONF_FN_MAX}): "
        f"{len(fn_rows)} — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fn_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=1  "
            f"least-confident={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    # ── Section 7: Confidence distribution by outcome ─────────────────────────
    sep(f"SECTION 7 — MEAN CONFIDENCE BY OUTCOME CLASS  (thresh={thresh})")
    hdr7 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'outcome':<6}  {'n':>5}  {'mean':>7}  {'median':>7}  {'min':>7}  {'max':>7}"
    )
    lines.append(hdr7)
    lines.append("  " + "-" * (len(hdr7) - 2))
    for s in run_stats:
        key = (s["model"], s["hypothesis"])
        outcome_confs: dict[str, list[float]] = {"TP": [], "FP": [], "FN": [], "TN": []}
        for i in range(n_rows):
            sc   = combo_scores[key][i]
            pred = combo_preds[key][i]
            gold = gold_bool[i]
            if   pred and gold:     outcome_confs["TP"].append(sc)
            elif pred and not gold: outcome_confs["FP"].append(sc)
            elif not pred and gold: outcome_confs["FN"].append(sc)
            else:                   outcome_confs["TN"].append(sc)
        first = True
        for outcome, vals in outcome_confs.items():
            if not vals:
                continue
            sv = sorted(vals)
            n  = len(sv)
            lines.append(
                f"  {s['model'] if first else '':<14} "
                f"{s['hypothesis'] if first else '':<22}  "
                f"{outcome:<6}  {n:>5}  "
                f"{sum(vals)/n:>7.4f}  {sv[n//2]:>7.4f}  {sv[0]:>7.4f}  {sv[-1]:>7.4f}"
            )
            first = False
        lines.append("  " + "-" * (len(hdr7) - 2))

    # ── Section 8: Text length vs accuracy ───────────────────────────────────
    sep(f"SECTION 8 — TEXT LENGTH VS ACCURACY  (thresh={thresh})")
    length_bins = [(0, 100, "<100"), (100, 200, "100-200"), (200, 400, "200-400"), (400, 10**9, "400+")]
    lengths     = [len(rows[i].get(premise_col, "")) for i in range(n_rows)]
    bin_counts  = [sum(1 for l in lengths if lo <= l < hi) for lo, hi, _ in length_bins]
    lines.append("\n  Bins: " + "  ".join(
        f"{lb}(n={cnt})" for (_, _, lb), cnt in zip(length_bins, bin_counts)
    ))
    lines.append("")
    hdr8 = f"  {'model':<14} {'hypothesis':<22}  " + "  ".join(f"{lb:>10}" for _, _, lb in length_bins)
    lines.append(hdr8)
    lines.append("  " + "-" * (len(hdr8) - 2))
    for s in run_stats:
        key   = (s["model"], s["hypothesis"])
        cells = []
        for lo, hi, _ in length_bins:
            idx = [i for i, l in enumerate(lengths) if lo <= l < hi]
            if not idx:
                cells.append("       N/A")
                continue
            m = compute_metrics([combo_scores[key][i] for i in idx],
                                [labels[i] for i in idx], thresh)
            cells.append(f"{m['accuracy']:>10.3f}")
        lines.append(f"  {s['model']:<14} {s['hypothesis']:<22}  " + "  ".join(cells))

    # ── Write report and TSV ──────────────────────────────────────────────────
    sep()
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"[error_analysis]  report → {analysis_path}")

    tsv_path = os.path.splitext(analysis_path)[0] + "_predictions.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["row_idx", "gold_label", "premise_preview"]
        for hyp_col in hyp_cols:
            header.append(f"hyp_{hyp_col}")
        for tag, hyp_col in all_combos:
            header += [f"conf_{tag}_{hyp_col}", f"pred_{tag}_{hyp_col}"]
        writer.writerow(header)
        for i, row in enumerate(rows):
            record = [i, labels[i], row.get(premise_col, "")[:120]]
            for hyp_col in hyp_cols:
                record.append(row.get(hyp_col, ""))
            for key in all_combos:
                record += [f"{combo_scores[key][i]:.4f}", "1" if combo_preds[key][i] else "0"]
            writer.writerow(record)
    log.info(f"[error_analysis]  predictions TSV → {tsv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="NLI classification with k-fold CV threshold optimisation"
    )
    parser.add_argument("--input",          default=INPUT_FILE,          help=f"Input CSV (default: {INPUT_FILE})")
    parser.add_argument("--output",         default=OUTPUT_FILE,         help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--log",            default=LOG_FILE,            help=f"Log file (default: {LOG_FILE})")
    parser.add_argument("--summary",        default=SUMMARY_FILE,        help=f"Summary file (default: {SUMMARY_FILE})")
    parser.add_argument("--error-analysis", default=ERROR_ANALYSIS_FILE, help=f"Error analysis (default: {ERROR_ANALYSIS_FILE})")
    parser.add_argument("--label-col",      default=LABEL_COL,           help=f"Gold label column (default: {LABEL_COL})")
    args = parser.parse_args()

    base          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path    = os.path.join(base, args.input)
    output_path   = os.path.join(base, args.output)
    log_path      = os.path.join(base, args.log)
    summary_path  = os.path.join(base, args.summary)
    error_path    = os.path.join(base, args.error_analysis)
    encoders_base = os.path.join(base, NLI_ENCODERS_BASE)
    for p in (output_path, log_path, summary_path, error_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    log.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    log.info(f"torch {torch.__version__}  CUDA={torch.version.cuda}  "
             f"available={torch.cuda.is_available()}")

    wall_start = time.time()
    log.info("=" * 70)
    log.info("clean_dataset_with_nli.py  started")
    log.info(f"  input      : {input_path}")
    log.info(f"  output     : {output_path}")
    log.info(f"  log        : {log_path}")
    log.info(f"  summary    : {summary_path}")
    log.info(f"  error      : {error_path}")
    log.info(f"  label col  : {args.label_col}")
    log.info(f"  thresholds : {EVAL_THRESHOLDS}  (fixed)")
    log.info(f"  CV folds   : {CV_N_SPLITS}  |  thresh grid: "
             f"{THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} ({len(THRESHOLD_GRID)} values)")
    log.info(f"  models ({len(NLI_MODELS)}):")
    for ckpt, tag in NLI_MODELS:
        log.info(f"    [{tag}]  {ckpt}")
    log.info(f"  pairs ({len(PREMISE_HYPOTHESIS_PAIRS)}):")
    for p, h in PREMISE_HYPOTHESIS_PAIRS:
        log.info(f"    {p!r} -> {h!r}")
    log.info("=" * 70)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_rows = len(rows)
    log.info(f"[load]  {n_rows} rows")

    available_cols = set(rows[0].keys())
    if args.label_col not in available_cols:
        raise ValueError(f"Label column '{args.label_col}' not found in {sorted(available_cols)}")
    for premise_col, hyp_col in PREMISE_HYPOTHESIS_PAIRS:
        for col in (premise_col, hyp_col):
            if col not in available_cols:
                raise ValueError(f"Column '{col}' not found in {sorted(available_cols)}")

    n_pos = sum(1 for r in rows if r[args.label_col] == "1")
    log.info(f"[load]  pos={n_pos}  neg={n_rows - n_pos}  "
             f"({100 * n_pos / n_rows:.1f}% / {100 * (n_rows - n_pos) / n_rows:.1f}%)")
    labels = [r[args.label_col] for r in rows]

    # ── Main inference loop ───────────────────────────────────────────────────
    run_stats: list[dict] = []

    for model_idx, (ckpt_rel, tag) in enumerate(NLI_MODELS, 1):
        model_path = os.path.join(encoders_base, ckpt_rel)
        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(NLI_MODELS)}]  {tag}")

        t_model          = time.time()
        model, tokenizer, device = load_model(model_path, log)
        log.info(f"    model loaded in {_fmt_duration(time.time() - t_model)}")

        for pair_idx, (premise_col, hyp_col) in enumerate(PREMISE_HYPOTHESIS_PAIRS, 1):
            conf_col = f"confidence_{tag}_{hyp_col}"
            desc     = f"{tag}/{hyp_col}"
            log.info(f"  [{pair_idx}/{len(PREMISE_HYPOTHESIS_PAIRS)}]  "
                     f"premise={premise_col!r}  hypothesis={hyp_col!r}  → {conf_col}")

            texts = [(r[premise_col], r[hyp_col]) for r in rows]
            t0    = time.time()
            scores = run_nli(model, tokenizer, device, texts, log, desc=desc)
            elapsed = time.time() - t0

            for r, score in zip(rows, scores):
                r[conf_col] = f"{score:.4f}"

            log.info(f"    done  ({_fmt_duration(elapsed)}, "
                     f"{n_rows / elapsed:.1f} rows/s)")

            # Score distribution (threshold-independent)
            m0 = compute_metrics(scores, labels, EVAL_THRESHOLDS[0])
            log.info(f"    score stats: mean={m0['score_mean']:.4f}  "
                     f"median={m0['score_median']:.4f}  "
                     f"min={m0['score_min']:.4f}  max={m0['score_max']:.4f}")

            # Fixed-threshold evaluation
            metrics_by_threshold: dict[float, dict] = {}
            for thresh in EVAL_THRESHOLDS:
                m = compute_metrics(scores, labels, thresh)
                metrics_by_threshold[thresh] = m
                log.info(
                    f"    thresh={thresh:.1f}  "
                    f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  "
                    f"acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
                    f"rec={m['recall']:.3f} F1={m['f1']:.3f}"
                )

            # K-fold CV threshold optimisation
            log.info(f"    [cv{CV_N_SPLITS}]  running threshold search …")
            cv_mean, cv_std, cv_thresholds, cv_fold_f1s = kfold_threshold_cv(
                scores, labels, CV_N_SPLITS, THRESHOLD_GRID
            )
            mean_t = sum(cv_thresholds) / len(cv_thresholds)
            log.info(
                f"    [cv{CV_N_SPLITS}]  mean_F1={cv_mean:.4f} ±{cv_std:.4f}  "
                f"mean_thresh={mean_t:.2f}  "
                f"fold_thresholds={[f'{t:.2f}' for t in cv_thresholds]}  "
                f"fold_F1s={[f'{f:.3f}' for f in cv_fold_f1s]}"
            )

            run_stats.append({
                "model":               tag,
                "premise":             premise_col,
                "hypothesis":          hyp_col,
                "time":                elapsed,
                "metrics_by_threshold": metrics_by_threshold,
                "cv_mean_f1":          cv_mean,
                "cv_std_f1":           cv_std,
                "cv_thresholds":       cv_thresholds,
                "cv_fold_f1s":         cv_fold_f1s,
            })

        unload_model(model, log)
        log.info(f"  [{tag}] total: {_fmt_duration(time.time() - t_model)}")

    # ── Ensemble columns ──────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("[ensemble]  building ensemble columns")
    ensemble_stats: list[dict] = []
    hyp_cols_all = list(dict.fromkeys(h for _, h in PREMISE_HYPOTHESIS_PAIRS))

    # 1. Per-hypothesis: average across all models
    for hyp_col in hyp_cols_all:
        source_cols = [f"confidence_{tag}_{hyp_col}" for _, tag in NLI_MODELS]
        ens_col     = f"nli_ensemble_{hyp_col}"
        scores      = compute_ensemble(rows, source_cols)
        for r, sc in zip(rows, scores):
            r[ens_col] = f"{sc:.4f}"
        mbt = {t: compute_metrics(scores, labels, t) for t in EVAL_THRESHOLDS}
        ensemble_stats.append({"col": ens_col, "n_voters": len(NLI_MODELS),
                                "metrics_by_threshold": mbt})
        log.info(f"  {ens_col}: best F1={max(m['f1'] for m in mbt.values()):.3f}")

    # 2. Best-per-model: pick best hypothesis for each model, then average
    best_cols = []
    for _, tag in NLI_MODELS:
        model_runs = [s for s in run_stats if s["model"] == tag]
        best = max(model_runs, key=lambda s: _get_metrics_at(s, ANALYSIS_THRESHOLD)["f1"])
        best_cols.append(f"confidence_{tag}_{best['hypothesis']}")
        log.info(f"  best hyp for {tag}: {best['hypothesis']!r}  "
                 f"(F1={_get_metrics_at(best, ANALYSIS_THRESHOLD)['f1']:.3f})")
    ens_col = "nli_ensemble_best_per_model"
    scores  = compute_ensemble(rows, best_cols)
    for r, sc in zip(rows, scores):
        r[ens_col] = f"{sc:.4f}"
    mbt = {t: compute_metrics(scores, labels, t) for t in EVAL_THRESHOLDS}
    ensemble_stats.append({"col": ens_col, "n_voters": len(best_cols),
                            "metrics_by_threshold": mbt})
    log.info(f"  {ens_col}: best F1={max(m['f1'] for m in mbt.values()):.3f}")

    # 3. All combos: average everything
    all_conf_cols = [f"confidence_{tag}_{h}" for _, tag in NLI_MODELS for h in hyp_cols_all]
    ens_col = "nli_ensemble_all"
    scores  = compute_ensemble(rows, all_conf_cols)
    for r, sc in zip(rows, scores):
        r[ens_col] = f"{sc:.4f}"
    mbt = {t: compute_metrics(scores, labels, t) for t in EVAL_THRESHOLDS}
    ensemble_stats.append({"col": ens_col, "n_voters": len(all_conf_cols),
                            "metrics_by_threshold": mbt})
    log.info(f"  {ens_col}: best F1={max(m['f1'] for m in mbt.values()):.3f}")

    # ── Save enriched CSV ─────────────────────────────────────────────────────
    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows → {output_path}")

    # ── Summary + error analysis ──────────────────────────────────────────────
    total = time.time() - wall_start
    write_summary(run_stats, summary_path, log, total, ensemble_stats=ensemble_stats)
    log.info(f"Summary written to {summary_path}")

    write_error_analysis(rows, labels, run_stats, error_path, log)
    log.info(f"Error analysis written to {error_path}")

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  {output_path}")
    log.info(f"  {summary_path}")
    log.info(f"  {error_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()