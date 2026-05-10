"""
Run multiple NLI encoder models on a CSV file with multiple premise-hypothesis column pairs.
For each (model, hypothesis_col) combination adds:
  confidence_{tag}_{hyp_col}  — entailment probability

Evaluates each combination at multiple thresholds against the gold label column
and writes accuracy / precision / recall / F1 to the log and to a summary file.

Usage:
    CUDA_VISIBLE_DEVICES=0 python clean_dataset_with_nli.py
    CUDA_VISIBLE_DEVICES=2 python clean_dataset_with_nli.py --input outputs/prepared_gold_dataset.csv
    CUDA_VISIBLE_DEVICES=4 python clean_dataset_with_nli.py 2>&1 | tee nli_run_temp.txt
"""

import os
import csv
import time
import logging
import argparse

import torch
import transformers
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE    = "outputs/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_FILE   = "outputs/nli_classified.csv"
LOG_FILE      = "outputs/nli_classify.log"
SUMMARY_FILE          = "outputs/nli_summary.txt"
ERROR_ANALYSIS_FILE   = "outputs/nli_error_analysis.txt"

ANALYSIS_THRESHOLD = 0.7   # threshold used for per-row decisions in error analysis

LABEL_COL = "relation_present"   # gold label column: "1" = present, "0" = absent

# Base directory for all local NLI encoder checkpoints
NLI_ENCODERS_BASE = "finetuned_Heb_NLI_encoders"

# List of (relative_checkpoint_path, short_tag) to run.
# Tag is used in output column names: confidence_{tag}_{hyp_col}
NLI_MODELS = [
    ("mmBERT-base_hebnli/checkpoint-1000",          "mmbert"),
    ("multilingual-e5-large_hebnli/checkpoint-4000","me5large"),
    ("neodictabert_hebnli/checkpoint-4500",         "neodictabert"),
    ("xlm-roberta-large_hebnli/checkpoint-2000",    "xlmroberta"),
]

# List of (premise_col, hypothesis_col) pairs to evaluate per model
PREMISE_HYPOTHESIS_PAIRS = [
    ("text", "basic_relation"),
    ("text", "template_relation"),
    ("text", "llm_relation"),
]

# Thresholds at which to evaluate each (model, hypothesis) combination
EVAL_THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

NLI_MAX_LENGTH        = 256    # tokenizer max length per (premise, hypothesis) pair
DEFAULT_BATCH_SIZE    = 64
LARGE_TEXT_THRESHOLD  = 256    # combined char length above which smaller batch is used
LARGE_TEXT_BATCH_SIZE = 12
LOG_PROGRESS_EVERY    = 25     # log inference progress every N% of batches

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TqdmToLogger:
    """Redirect tqdm output to a logger at INFO level."""
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
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def _fmt_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(scores: list[float], labels: list[str], threshold: float) -> dict:
    TP = FP = FN = TN = 0
    for score, label in zip(scores, labels):
        pred = score >= threshold
        gold = label == "1"
        if pred and gold:       TP += 1
        elif pred and not gold: FP += 1
        elif not pred and gold: FN += 1
        else:                   TN += 1

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall    = TP / (TP + FN) if (TP + FN) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    mean   = sum(scores) / n if n else 0
    median = sorted_scores[n // 2] if n else 0
    s_min  = sorted_scores[0]  if n else 0
    s_max  = sorted_scores[-1] if n else 0

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "score_mean": mean, "score_median": median,
        "score_min": s_min, "score_max": s_max,
    }


# ---------------------------------------------------------------------------
# Ensemble (majority) voting
# ---------------------------------------------------------------------------

def compute_ensemble(rows: list[dict], conf_col_names: list[str]) -> list[float]:
    """
    Average confidence scores from `conf_col_names` for each row.
    NLI-adapted equivalent of majority voting: soft ensemble of continuous scores.
    """
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

    # ---- CUDA diagnostics ----
    log.info("    checking CUDA availability...")
    log.info(f"    torch version: {torch.__version__}")
    log.info(f"    torch CUDA build: {torch.version.cuda}")
    log.info(f"    torch.cuda.is_available(): {torch.cuda.is_available()}")
    log.info(f"    torch.cuda.device_count(): {torch.cuda.device_count()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this environment.\n"
            f"torch version: {torch.__version__}\n"
            f"torch CUDA build: {torch.version.cuda}\n"
            f"device count: {torch.cuda.device_count()}\n"
            "This usually means PyTorch was installed without CUDA support "
            "or the current environment cannot access GPUs."
        )

    log.info(f"    visible GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    config = transformers.AutoConfig.from_pretrained(
        model_path,
        output_hidden_states=False,
        output_attentions=False,
        trust_remote_code=True,
    )

    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            use_safetensors=True
        )
    except Exception as e:
        log.warning(f"    safetensors load failed ({e}), trying normal load.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True
        )

    # ---- Force GPU usage ----
    device = torch.device("cuda:0")

    log.info(f"    moving model to {device}...")
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
# NLI inference  (same logic as DRAFT_clean_with_encoder_NLI.py)
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
    """
    texts: list of (premise, hypothesis) pairs.
    Returns a list of entailment probabilities, one per pair.
    Logs progress every LOG_PROGRESS_EVERY % of batches to the log file.
    """
    if not texts:
        return []

    batch_size = _adaptive_batch_size(texts)
    n_batches  = (len(texts) + batch_size - 1) // batch_size
    log_every  = max(1, n_batches * LOG_PROGRESS_EVERY // 100)
    log.info(f"    rows={len(texts)}, batch_size={batch_size}, n_batches={n_batches}")

    result = []
    t_inf_start = time.time()

    for batch_idx, start in enumerate(
        tqdm(range(0, len(texts), batch_size), desc=f"    {desc}",
             leave=False, file=TqdmToLogger(log)), 1
    ):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            [p for p, _ in batch],
            [h for _, h in batch],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=NLI_MAX_LENGTH,
            padding="longest",
            return_token_type_ids=False,
            truncation=True,
        )
        for key in encoded:
            encoded[key] = encoded[key].to(device)

        with torch.no_grad():
            outputs = model(**encoded, return_dict=True)

        result.append(outputs["logits"].softmax(dim=1))
        del outputs

        if batch_idx % log_every == 0 or batch_idx == n_batches:
            elapsed = time.time() - t_inf_start
            rows_done = min(start + batch_size, len(texts))
            log.info(
                f"    progress: {rows_done}/{len(texts)} rows "
                f"({100 * rows_done / len(texts):.0f}%)  {_fmt_duration(elapsed)}"
            )

    logits = torch.cat(result)
    scores = logits[:, 1] if logits.dim() > 1 else logits.unsqueeze(0)
    return scores.cpu().tolist()


# ---------------------------------------------------------------------------
# Summary file writer
# ---------------------------------------------------------------------------

def write_summary(run_stats: list[dict], summary_path: str, log: logging.Logger,
                  total_time: float, ensemble_stats: list[dict] | None = None):
    lines = []
    lines.append("=" * 100)
    lines.append("NLI CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total_time)}")
    lines.append(f"Thresholds evaluated: {EVAL_THRESHOLDS}")
    lines.append("=" * 100)

    # Per-model timing block
    model_times: dict[str, float] = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0)
        model_times[s["model"]] += s["time"]

    lines.append("")
    lines.append("Per-model total inference time:")
    for model, t in model_times.items():
        lines.append(f"  {model:<16}  {_fmt_duration(t)}")

    # Per-model metrics table — one row per (model, hypothesis, threshold)
    lines.append("")
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

    # Ensemble table — one row per (ensemble_col, threshold)
    if ensemble_stats:
        lines.append("")
        lines.append("Ensemble (averaged confidence) columns:")
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

    lines.append("=" * 100)
    text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    log.info("=" * 100)
    for line in lines:
        log.info(line)
    log.info("=" * 100)


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
    label_col: str,
    analysis_path: str,
    log: logging.Logger,
):
    if not run_stats or not rows:
        log.warning("[error_analysis] No data to analyze.")
        return

    thresh = ANALYSIS_THRESHOLD
    n_rows = len(rows)
    tags     = list(dict.fromkeys(s["model"]     for s in run_stats))
    hyp_cols = list(dict.fromkeys(s["hypothesis"] for s in run_stats))
    premise_col = run_stats[0]["premise"] if run_stats else "text"

    # Precompute scores and binary predictions for every (model, hyp_col) combo
    all_combos = [(s["model"], s["hypothesis"]) for s in run_stats]
    combo_scores: dict[tuple, list[float]] = {}
    combo_preds:  dict[tuple, list[bool]]  = {}
    for s in run_stats:
        key      = (s["model"], s["hypothesis"])
        conf_col = f"confidence_{s['model']}_{s['hypothesis']}"
        scores   = [float(r[conf_col]) for r in rows]
        combo_scores[key] = scores
        combo_preds[key]  = [sc >= thresh for sc in scores]

    gold_bool = [lbl == "1" for lbl in labels]
    n_pos = sum(gold_bool)

    lines = []

    def sep(title=""):
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

    # ── Section 1: Best configuration per metric ───────────────────────────────
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

    # ── Section 2: Per-hypothesis ranking ──────────────────────────────────────
    sep(f"SECTION 2 — PER-HYPOTHESIS RANKING  (thresh={thresh})")
    col_hdr = f"  {'rank':<5} {'model':<14} {'f1':>6} {'acc':>6} {'prec':>6} {'rec':>6}  {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    for hyp_col in hyp_cols:
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append(col_hdr)
        lines.append("  " + "-" * (len(col_hdr) - 2))
        ranked = sorted(
            [s for s in run_stats if s["hypothesis"] == hyp_col],
            key=lambda s: _get_metrics_at(s, thresh)["f1"],
            reverse=True,
        )
        for rank, s in enumerate(ranked, 1):
            m = _get_metrics_at(s, thresh)
            lines.append(
                f"  {rank:<5} {s['model']:<14} "
                f"{m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
            )

    # ── Section 3: Model agreement matrix ─────────────────────────────────────
    sep(f"SECTION 3 — MODEL AGREEMENT MATRIX  (thresh={thresh}, % same binary prediction)")
    for hyp_col in hyp_cols:
        avail_tags = [s["model"] for s in run_stats if s["hypothesis"] == hyp_col]
        lines.append(f"\n  Hypothesis: {hyp_col!r}")
        lines.append("  " + " " * 14 + "".join(f"{t:>14}" for t in avail_tags))
        for t1 in avail_tags:
            p1 = combo_preds[(t1, hyp_col)]
            row_str = f"  {t1:<14}"
            for t2 in avail_tags:
                p2 = combo_preds[(t2, hyp_col)]
                agree = sum(a == b for a, b in zip(p1, p2)) / n_rows
                row_str += f"  {agree:>10.3f}  "
            lines.append(row_str)

    # ── Section 4: Hard examples — every combo wrong ───────────────────────────
    sep(f"SECTION 4 — HARD EXAMPLES: ALL COMBOS WRONG  (thresh={thresh})")
    row_n_correct = [
        sum(1 for key in all_combos if combo_preds[key][i] == gold_bool[i])
        for i in range(n_rows)
    ]
    hard_idx = [i for i, nc in enumerate(row_n_correct) if nc == 0]
    lines.append(
        f"\n  {len(hard_idx)} rows ({len(hard_idx) / n_rows * 100:.1f}%) where "
        "every model+hypothesis prediction was wrong.  Showing up to 20."
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

    # ── Section 5: Confident mistakes ─────────────────────────────────────────
    sep("SECTION 5 — CONFIDENT MISTAKES")
    CONF_FP_MIN = 0.85   # gold=0, but model very confident it's positive
    CONF_FN_MAX = 0.15   # gold=1, but model very confident it's negative

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
        f"\n  False Positives (gold=0, max confidence >= {CONF_FP_MIN}): "
        f"{len(fp_rows)} found — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fp_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=0  "
            f"most-confident combo={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    lines.append(
        f"\n  False Negatives (gold=1, min confidence <= {CONF_FN_MAX}): "
        f"{len(fn_rows)} found — showing up to 15"
    )
    for i, worst_key, worst_conf, confs in fn_rows[:15]:
        lines.append("")
        lines.append(
            f"  [row {i}]  gold=1  "
            f"least-confident combo={worst_key[0]}/{worst_key[1]}  conf={worst_conf:.4f}"
        )
        lines.append(f"    premise  : {rows[i].get(premise_col, '')[:120]!r}")
        for hyp_col in hyp_cols:
            lines.append(f"    {hyp_col:<24}: {rows[i].get(hyp_col, '')!r}")
        lines.append("    scores   : " + "  ".join(
            f"{k[0]}/{k[1]}={v:.3f}" for k, v in confs.items()
        ))

    # ── Section 6: Confidence distribution by outcome ──────────────────────────
    sep(f"SECTION 6 — MEAN CONFIDENCE BY OUTCOME CLASS  (thresh={thresh})")
    hdr6 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'outcome':<6}  {'n':>5}  {'mean':>7}  {'median':>7}  {'min':>7}  {'max':>7}"
    )
    lines.append(hdr6)
    lines.append("  " + "-" * (len(hdr6) - 2))
    for s in run_stats:
        key = (s["model"], s["hypothesis"])
        outcome_confs: dict[str, list[float]] = {"TP": [], "FP": [], "FN": [], "TN": []}
        for i in range(n_rows):
            sc   = combo_scores[key][i]
            pred = combo_preds[key][i]
            gold = gold_bool[i]
            if   pred and gold:      outcome_confs["TP"].append(sc)
            elif pred and not gold:  outcome_confs["FP"].append(sc)
            elif not pred and gold:  outcome_confs["FN"].append(sc)
            else:                    outcome_confs["TN"].append(sc)
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
        lines.append("  " + "-" * (len(hdr6) - 2))

    # ── Section 7: Text length vs accuracy ────────────────────────────────────
    sep(f"SECTION 7 — TEXT LENGTH VS ACCURACY  (thresh={thresh})")
    length_bins = [(0, 100, "<100"), (100, 200, "100-200"), (200, 400, "200-400"), (400, 10**9, "400+")]
    lengths = [len(rows[i].get(premise_col, "")) for i in range(n_rows)]
    bin_counts = [sum(1 for l in lengths if lo <= l < hi) for lo, hi, _ in length_bins]
    bin_labels_str = "  ".join(f"{lb}(n={cnt})" for (_, _, lb), cnt in zip(length_bins, bin_counts))
    lines.append(f"\n  Bins: {bin_labels_str}")
    lines.append("")
    hdr7 = f"  {'model':<14} {'hypothesis':<22}  " + "  ".join(f"{lb:>10}" for _, _, lb in length_bins)
    lines.append(hdr7)
    lines.append("  " + "-" * (len(hdr7) - 2))
    for s in run_stats:
        key = (s["model"], s["hypothesis"])
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

    # ── Write text report ─────────────────────────────────────────────────────
    sep()
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"[error_analysis]  report  → {analysis_path}")

    # ── Section 8: Full predictions TSV ───────────────────────────────────────
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
        description="NLI classification on CSV with multiple models and hypothesis columns"
    )
    parser.add_argument("--input",     default=INPUT_FILE,   help=f"Input CSV (default: {INPUT_FILE})")
    parser.add_argument("--output",    default=OUTPUT_FILE,  help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--log",       default=LOG_FILE,     help=f"Log file (default: {LOG_FILE})")
    parser.add_argument("--summary",   default=SUMMARY_FILE, help=f"Summary file (default: {SUMMARY_FILE})")
    parser.add_argument("--label-col",      default=LABEL_COL,            help=f"Gold label column (default: {LABEL_COL})")
    parser.add_argument("--error-analysis", default=ERROR_ANALYSIS_FILE,  help=f"Error analysis file (default: {ERROR_ANALYSIS_FILE})")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path    = os.path.join(base, args.input)
    output_path   = os.path.join(base, args.output)
    log_path      = os.path.join(base, args.log)
    summary_path  = os.path.join(base, args.summary)
    error_path    = os.path.join(base, args.error_analysis)
    encoders_base = os.path.join(base, NLI_ENCODERS_BASE)
    for p in (output_path, log_path, summary_path, error_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    log.info("CUDA DEBUG INFO")
    log.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    log.info(f"torch version = {torch.__version__}")
    log.info(f"torch CUDA build = {torch.version.cuda}")
    log.info(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    log.info(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

    wall_start = time.time()

    log.info("=" * 70)
    log.info("clean_dataset_with_nli.py  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output:     {output_path}")
    log.info(f"  log:        {log_path}")
    log.info(f"  summary:    {summary_path}")
    log.info(f"  label col:  {args.label_col}")
    log.info(f"  thresholds: {EVAL_THRESHOLDS}")
    log.info(f"  models ({len(NLI_MODELS)}):")
    for ckpt, tag in NLI_MODELS:
        log.info(f"    [{tag}]  {ckpt}")
    log.info(f"  premise-hypothesis pairs ({len(PREMISE_HYPOTHESIS_PAIRS)}):")
    for p, h in PREMISE_HYPOTHESIS_PAIRS:
        log.info(f"    {p!r} -> {h!r}")
    log.info(f"  batch config: default={DEFAULT_BATCH_SIZE}, large={LARGE_TEXT_BATCH_SIZE} "
             f"(threshold={LARGE_TEXT_THRESHOLD} chars)")
    log.info("=" * 70)

    # --- Load CSV ---
    t0 = time.time()
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_rows = len(rows)
    log.info(f"[load]  {n_rows} rows loaded  ({_fmt_duration(time.time() - t0)})")

    # Validate columns
    available_cols = set(rows[0].keys())
    if args.label_col not in available_cols:
        raise ValueError(f"Label column '{args.label_col}' not found. Available: {sorted(available_cols)}")
    for premise_col, hyp_col in PREMISE_HYPOTHESIS_PAIRS:
        for col in (premise_col, hyp_col):
            if col not in available_cols:
                raise ValueError(f"Column '{col}' not found. Available: {sorted(available_cols)}")

    n_positive = sum(1 for r in rows if r[args.label_col] == "1")
    n_negative = n_rows - n_positive
    log.info(f"[load]  label distribution: positive={n_positive}, negative={n_negative} "
             f"({100 * n_positive / n_rows:.1f}% / {100 * n_negative / n_rows:.1f}%)")

    labels = [r[args.label_col] for r in rows]

    # --- Main loop: one model load per model, all pairs inside ---
    run_stats: list[dict] = []

    for model_idx, (ckpt_rel, tag) in enumerate(NLI_MODELS, 1):
        model_path = os.path.join(encoders_base, ckpt_rel)
        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(NLI_MODELS)}]  {tag}")

        t_model = time.time()
        model, tokenizer, device = load_model(model_path, log)
        load_time = time.time() - t_model
        log.info(f"    model loaded in {_fmt_duration(load_time)}")

        for pair_idx, (premise_col, hyp_col) in enumerate(PREMISE_HYPOTHESIS_PAIRS, 1):
            conf_col = f"confidence_{tag}_{hyp_col}"
            desc = f"{tag}/{hyp_col}"
            log.info(f"  [{pair_idx}/{len(PREMISE_HYPOTHESIS_PAIRS)}]  "
                     f"premise={premise_col!r}  hypothesis={hyp_col!r}")
            log.info(f"    output col: {conf_col}")

            texts = [(r[premise_col], r[hyp_col]) for r in rows]

            t0 = time.time()
            scores = run_nli(model, tokenizer, device, texts, log, desc=desc)
            elapsed = time.time() - t0

            for r, score in zip(rows, scores):
                r[conf_col] = f"{score:.4f}"

            rows_per_sec = n_rows / elapsed if elapsed > 0 else float("inf")
            log.info(f"    inference done  ({_fmt_duration(elapsed)}, {rows_per_sec:.1f} rows/s)")

            # Score distribution stats (threshold-independent)
            first_metrics = compute_metrics(scores, labels, EVAL_THRESHOLDS[0])
            log.info(f"    score stats: mean={first_metrics['score_mean']:.4f}  "
                     f"median={first_metrics['score_median']:.4f}  "
                     f"min={first_metrics['score_min']:.4f}  max={first_metrics['score_max']:.4f}")

            # Evaluate at each threshold
            metrics_by_threshold: dict[float, dict] = {}
            for thresh in EVAL_THRESHOLDS:
                m = compute_metrics(scores, labels, thresh)
                metrics_by_threshold[thresh] = m
                log.info(
                    f"    thresh={thresh:.1f}  predicted={m['TP'] + m['FP']}/{n_rows}  "
                    f"TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  "
                    f"acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
                    f"rec={m['recall']:.4f} F1={m['f1']:.4f}"
                )

            log.info(f"    examples (first 3 rows):")
            for r in rows[:3]:
                log.info(f"      premise   : {r[premise_col][:120]!r}")
                log.info(f"      hypothesis: {r[hyp_col]!r}")
                log.info(f"      score={float(r[conf_col]):.4f}  gold={r[args.label_col]}")

            run_stats.append({
                "model":               tag,
                "premise":             premise_col,
                "hypothesis":          hyp_col,
                "time":                elapsed,
                "metrics_by_threshold": metrics_by_threshold,
            })

        model_total = time.time() - t_model
        unload_model(model, log)
        log.info(f"  [{tag}] total time (load + inference): {_fmt_duration(model_total)}")

    # --- Ensemble (majority) voting ---
    log.info("=" * 70)
    log.info("[ensemble]  computing ensemble columns (averaged confidence scores)")
    ensemble_stats: list[dict] = []
    hyp_cols = list(dict.fromkeys(h for _, h in PREMISE_HYPOTHESIS_PAIRS))

    # 1. Per hypothesis: average confidence across all models
    for hyp_col in hyp_cols:
        source_cols = [f"confidence_{tag}_{hyp_col}" for _, tag in NLI_MODELS]
        ens_col = f"nli_ensemble_{hyp_col}"
        scores = compute_ensemble(rows, source_cols)
        for r, score in zip(rows, scores):
            r[ens_col] = f"{score:.4f}"
        metrics_by_threshold = {
            thresh: compute_metrics(scores, labels, thresh) for thresh in EVAL_THRESHOLDS
        }
        ensemble_stats.append({"col": ens_col, "n_voters": len(NLI_MODELS),
                                "metrics_by_threshold": metrics_by_threshold})
        best_f1 = max(m["f1"] for m in metrics_by_threshold.values())
        log.info(f"  {ens_col}: best F1={best_f1:.3f}  (voters: {len(source_cols)})")

    # 2. Best-per-model: for each model pick its best hyp_col by F1 at ANALYSIS_THRESHOLD, then average
    best_cols = []
    for _, tag in NLI_MODELS:
        model_runs = [s for s in run_stats if s["model"] == tag]
        best = max(model_runs,
                   key=lambda s: _get_metrics_at(s, ANALYSIS_THRESHOLD)["f1"])
        best_cols.append(f"confidence_{tag}_{best['hypothesis']}")
        log.info(f"  best hyp for {tag}: {best['hypothesis']!r}  "
                 f"(F1={_get_metrics_at(best, ANALYSIS_THRESHOLD)['f1']:.3f} at thresh={ANALYSIS_THRESHOLD})")
    ens_col = "nli_ensemble_best_per_model"
    scores = compute_ensemble(rows, best_cols)
    for r, score in zip(rows, scores):
        r[ens_col] = f"{score:.4f}"
    metrics_by_threshold = {
        thresh: compute_metrics(scores, labels, thresh) for thresh in EVAL_THRESHOLDS
    }
    ensemble_stats.append({"col": ens_col, "n_voters": len(best_cols),
                            "metrics_by_threshold": metrics_by_threshold})
    best_f1 = max(m["f1"] for m in metrics_by_threshold.values())
    log.info(f"  {ens_col}: best F1={best_f1:.3f}  (voters: {len(best_cols)})")

    # 3. Overall: average all (model × hyp_col) confidence scores
    all_cols = [f"confidence_{tag}_{hyp_col}"
                for _, tag in NLI_MODELS for hyp_col in hyp_cols]
    ens_col = "nli_ensemble_all"
    scores = compute_ensemble(rows, all_cols)
    for r, score in zip(rows, scores):
        r[ens_col] = f"{score:.4f}"
    metrics_by_threshold = {
        thresh: compute_metrics(scores, labels, thresh) for thresh in EVAL_THRESHOLDS
    }
    ensemble_stats.append({"col": ens_col, "n_voters": len(all_cols),
                            "metrics_by_threshold": metrics_by_threshold})
    best_f1 = max(m["f1"] for m in metrics_by_threshold.values())
    log.info(f"  {ens_col}: best F1={best_f1:.3f}  (voters: {len(all_cols)})")

    # --- Save CSV (includes both raw confidence and ensemble columns) ---
    t0 = time.time()
    fieldnames = list(dict.fromkeys(rows[0].keys()))  # deduplicate, preserve order
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows written to {output_path}  ({_fmt_duration(time.time() - t0)})")

    # --- Summary ---
    total = time.time() - wall_start
    write_summary(run_stats, summary_path, log, total, ensemble_stats=ensemble_stats)
    log.info(f"Summary written to {summary_path}")

    # --- Error analysis ---
    write_error_analysis(rows, labels, run_stats, args.label_col, error_path, log)
    log.info(f"Error analysis written to {error_path}")


if __name__ == "__main__":
    main()
