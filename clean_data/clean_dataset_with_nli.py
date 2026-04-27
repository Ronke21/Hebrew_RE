"""
Run multiple NLI encoder models on a CSV file with multiple premise-hypothesis column pairs.
For each (model, hypothesis_col) combination adds:
  confidence_{tag}_{hyp_col}  — entailment probability
  clean_{tag}_{hyp_col}       — "yes" / "no" based on threshold

Evaluates each combination against the gold label column (relation_present)
and writes accuracy / precision / recall / F1 to the log and to a summary file.

Usage:
    CUDA_VISIBLE_DEVICES=2 python clean_dataset_with_nli.py
    CUDA_VISIBLE_DEVICES=2 python clean_dataset_with_nli.py --input outputs/prepared_gold_dataset.csv
    CUDA_VISIBLE_DEVICES=4,5,6 python clean_dataset_with_nli.py 2>&1 | tee nli_run.txt
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
SUMMARY_FILE  = "outputs/nli_summary.txt"

LABEL_COL = "relation_present"   # gold label column: "1" = present, "0" = absent

# Base directory for all local NLI encoder checkpoints
NLI_ENCODERS_BASE = "finetuned_Heb_NLI_encoders"

# List of (relative_checkpoint_path, short_tag) to run.
# Tag is used in output column names: confidence_{tag}_{hyp_col}, clean_{tag}_{hyp_col}
NLI_MODELS = [
    ("mmBERT-base_hebnli/checkpoint-1000",          "mmbert"),
    ("mt5-xl_hebnli/checkpoint-8000",               "mt5xl"),
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

ENTAILMENT_THRESHOLD  = 0.75
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

def compute_metrics(rows: list[dict], clean_col: str, label_col: str) -> dict:
    TP = FP = FN = TN = 0
    scores = []
    for r in rows:
        pred     = r[clean_col] == "yes"
        gold     = r[label_col] == "1"
        conf     = float(r[clean_col.replace("clean_", "confidence_", 1)])
        scores.append(conf)
        if pred and gold:      TP += 1
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
    mean  = sum(scores) / n if n else 0
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
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(model_path: str, log: logging.Logger):
    log.info(f"    path: {model_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_path,
        output_hidden_states=False,
        output_attentions=False,
        trust_remote_code=True,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.half()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"    ready on {device}  ({n_params:.0f}M params)")
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
                  total_time: float):
    lines = []
    lines.append("=" * 90)
    lines.append("NLI CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total_time)}")
    lines.append("=" * 90)

    # Per-model timing block
    model_times: dict[str, float] = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0)
        model_times[s["model"]] += s["time"]

    lines.append("")
    lines.append("Per-model total inference time:")
    for model, t in model_times.items():
        lines.append(f"  {model:<16}  {_fmt_duration(t)}")

    # Metrics table
    lines.append("")
    hdr = (
        f"  {'model':<14} {'hypothesis':<22} "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'mean_conf':>9}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for s in run_stats:
        m = s["metrics"]
        lines.append(
            f"  {s['model']:<14} {s['hypothesis']:<22} "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['score_mean']:>9.4f}  {_fmt_duration(s['time']):>8}"
        )

    lines.append("=" * 90)
    text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    log.info("=" * 90)
    for line in lines:
        log.info(line)
    log.info("=" * 90)


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
    parser.add_argument("--label-col", default=LABEL_COL,    help=f"Gold label column (default: {LABEL_COL})")
    parser.add_argument("--threshold", type=float, default=ENTAILMENT_THRESHOLD,
                        help=f"Entailment threshold for 'clean' (default: {ENTAILMENT_THRESHOLD})")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path    = os.path.join(base, args.input)
    output_path   = os.path.join(base, args.output)
    log_path      = os.path.join(base, args.log)
    summary_path  = os.path.join(base, args.summary)
    encoders_base = os.path.join(base, NLI_ENCODERS_BASE)
    for p in (output_path, log_path, summary_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    log.info("=" * 70)
    log.info("clean_dataset_with_nli.py  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output:     {output_path}")
    log.info(f"  log:        {log_path}")
    log.info(f"  summary:    {summary_path}")
    log.info(f"  label col:  {args.label_col}")
    log.info(f"  threshold:  {args.threshold}")
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
            conf_col  = f"confidence_{tag}_{hyp_col}"
            clean_col = f"clean_{tag}_{hyp_col}"
            desc = f"{tag}/{hyp_col}"
            log.info(f"  [{pair_idx}/{len(PREMISE_HYPOTHESIS_PAIRS)}]  "
                     f"premise={premise_col!r}  hypothesis={hyp_col!r}")
            log.info(f"    output cols: {conf_col}, {clean_col}")

            texts = [(r[premise_col], r[hyp_col]) for r in rows]

            t0 = time.time()
            scores = run_nli(model, tokenizer, device, texts, log, desc=desc)
            elapsed = time.time() - t0

            for r, score in zip(rows, scores):
                r[conf_col]  = f"{score:.4f}"
                r[clean_col] = "yes" if score >= args.threshold else "no"

            metrics = compute_metrics(rows, clean_col, args.label_col)
            rows_per_sec = n_rows / elapsed if elapsed > 0 else float("inf")

            log.info(f"    inference done  ({_fmt_duration(elapsed)}, {rows_per_sec:.1f} rows/s)")
            log.info(f"    predicted clean : {metrics['TP'] + metrics['FP']} / {n_rows}")
            log.info(f"    score stats     : mean={metrics['score_mean']:.4f}  "
                     f"median={metrics['score_median']:.4f}  "
                     f"min={metrics['score_min']:.4f}  max={metrics['score_max']:.4f}")
            log.info(f"    confusion matrix: TP={metrics['TP']}  FP={metrics['FP']}  "
                     f"FN={metrics['FN']}  TN={metrics['TN']}")
            log.info(f"    accuracy={metrics['accuracy']:.4f}  precision={metrics['precision']:.4f}  "
                     f"recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

            log.info(f"    examples (first 3 rows):")
            for r in rows[:3]:
                log.info(f"      premise   : {r[premise_col][:120]!r}")
                log.info(f"      hypothesis: {r[hyp_col]!r}")
                log.info(f"      score={float(r[conf_col]):.4f}  prediction={r[clean_col]}  gold={r[args.label_col]}")

            run_stats.append({
                "model":      tag,
                "premise":    premise_col,
                "hypothesis": hyp_col,
                "time":       elapsed,
                "metrics":    metrics,
            })

        model_total = time.time() - t_model
        unload_model(model, log)
        log.info(f"  [{tag}] total time (load + inference): {_fmt_duration(model_total)}")

    # --- Save CSV ---
    t0 = time.time()
    fieldnames = list(dict.fromkeys(rows[0].keys()))  # deduplicate, preserve order
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows written to {output_path}  ({_fmt_duration(time.time() - t0)})")

    # --- Summary ---
    total = time.time() - wall_start
    write_summary(run_stats, summary_path, log, total)
    log.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
