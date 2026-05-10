"""
Advanced post-processing and evaluation for the NLI classification output.

Reads the NLI-classified CSV (produced by clean_dataset_with_nli.py) and performs:
  1. K-fold CV threshold optimisation — per (model × hypothesis) combo
  2. Score calibration — Platt scaling and isotonic regression (OOF)
  3. Weighted ensemble — weighted average by per-combo CV F1
  4. Stacking — OOF logistic regression on all confidence scores
  5. Predicate-stratified evaluation — metrics by predicate + in-sample best threshold

Outputs:
  - nli_advanced_classified.csv   — input CSV enriched with new score columns
  - nli_advanced_eval.log         — full run log
  - nli_advanced_summary.txt      — concise metrics table (all methods)
  - nli_advanced_error_analysis.txt — FP/FN breakdown + hard-row examples
  - nli_advanced_analysis.txt     — detailed per-section report

Usage:
    conda activate heb_relation_extraction
    python nli_advanced_eval.py
    python nli_advanced_eval.py --input outputs/nli_classified.csv
"""

import os
import csv
import logging
import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE              = "outputs/ARCHIVE/nli_classified.csv"
OUTPUT_FILE             = "outputs/NLI_clean/nli_advanced_classified.csv"
LOG_FILE                = "outputs/NLI_clean/nli_advanced_eval.log"
SUMMARY_FILE            = "outputs/NLI_clean/nli_advanced_summary.txt"
ERROR_ANALYSIS_FILE     = "outputs/NLI_clean/nli_advanced_error_analysis.txt"
ANALYSIS_FILE           = "outputs/NLI_clean/nli_advanced_analysis.txt"

LABEL_COL               = "relation_present"   # "1" = present, "0" = absent
PREDICATE_COL           = "predicate"

CV_N_SPLITS             = 5
THRESHOLD_GRID          = [round(t / 100, 2) for t in range(5, 96, 5)]  # 0.05 … 0.95
EVAL_THRESHOLDS         = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ANALYSIS_THRESHOLD      = 0.5    # natural decision boundary for calibrated scores
MIN_PREDICATE_EXAMPLES  = 5      # skip predicates with fewer rows in per-pred analysis
MAX_ERROR_EXAMPLES      = 10     # max FP/FN rows shown per method in error analysis

# Hypothesis columns used when parsing confidence column names
KNOWN_HYP_COLS = {"basic_relation", "template_relation", "llm_relation"}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("nli_advanced_eval")
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
    m, s = divmod(rem, 60)
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
        if pred and gold:        TP += 1
        elif pred and not gold:  FP += 1
        elif not pred and gold:  FN += 1
        else:                    TN += 1
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total if total else 0
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall    = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 1. K-fold CV threshold search
# ---------------------------------------------------------------------------

def kfold_threshold_search(
    scores: np.ndarray,
    labels: np.ndarray,
    n_splits: int,
    threshold_grid: list[float],
) -> tuple[float, float, list[float]]:
    """
    Per fold: find the threshold maximising F1 on the train split, evaluate on val split.
    Returns (mean_cv_f1, std_cv_f1, best_threshold_per_fold).
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1s: list[float] = []
    fold_thresholds: list[float] = []

    for train_idx, val_idx in kf.split(scores.reshape(-1, 1), labels):
        tr_s, tr_l = scores[train_idx], labels[train_idx]
        va_s, va_l = scores[val_idx],   labels[val_idx]

        best_t, best_f1 = threshold_grid[0], -1.0
        for t in threshold_grid:
            preds = (tr_s >= t).astype(int)
            tp = int(((preds == 1) & (tr_l == 1)).sum())
            fp = int(((preds == 1) & (tr_l == 0)).sum())
            fn = int(((preds == 0) & (tr_l == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t

        fold_thresholds.append(best_t)
        val_preds = (va_s >= best_t).astype(int)
        tp = int(((val_preds == 1) & (va_l == 1)).sum())
        fp = int(((val_preds == 1) & (va_l == 0)).sum())
        fn = int(((val_preds == 0) & (va_l == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        fold_f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    arr = np.array(fold_f1s)
    return float(arr.mean()), float(arr.std()), fold_thresholds


# ---------------------------------------------------------------------------
# 2. Calibration (OOF)
# ---------------------------------------------------------------------------

def kfold_platt_calibrate(
    scores: np.ndarray, labels: np.ndarray, n_splits: int
) -> np.ndarray:
    """OOF Platt scaling: fit logistic regression on raw scores per train fold."""
    kf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros_like(scores, dtype=float)
    X   = scores.reshape(-1, 1)
    for train_idx, val_idx in kf.split(X, labels):
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X[train_idx], labels[train_idx])
        oof[val_idx] = lr.predict_proba(X[val_idx])[:, 1]
    return oof


def kfold_isotonic_calibrate(
    scores: np.ndarray, labels: np.ndarray, n_splits: int
) -> np.ndarray:
    """OOF isotonic regression calibration."""
    kf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros_like(scores, dtype=float)
    for train_idx, val_idx in kf.split(scores.reshape(-1, 1), labels):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(scores[train_idx], labels[train_idx])
        oof[val_idx] = ir.transform(scores[val_idx])
    return oof


# ---------------------------------------------------------------------------
# 3. Weighted ensemble
# ---------------------------------------------------------------------------

def compute_weighted_ensemble(
    score_arrays: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted average of score arrays. Weights need not sum to 1."""
    total_w = sum(weights.values()) or 1.0
    return sum(arr * (weights[col] / total_w) for col, arr in score_arrays.items())


# ---------------------------------------------------------------------------
# 4. Stacking
# ---------------------------------------------------------------------------

def kfold_stacking(X: np.ndarray, labels: np.ndarray, n_splits: int) -> np.ndarray:
    """OOF logistic regression stacking across all (model × hyp) confidence scores."""
    kf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(labels), dtype=float)
    for train_idx, val_idx in kf.split(X, labels):
        lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        lr.fit(X[train_idx], labels[train_idx])
        oof[val_idx] = lr.predict_proba(X[val_idx])[:, 1]
    return oof


# ---------------------------------------------------------------------------
# 5. Predicate-stratified helpers
# ---------------------------------------------------------------------------

def compute_predicate_metrics(
    rows: list[dict],
    scores: list[float],
    labels: list[str],
    threshold: float,
) -> dict[str, dict]:
    """Per-predicate metrics at a fixed threshold."""
    pred_idx: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        pred_idx.setdefault(row.get(PREDICATE_COL, "unknown"), []).append(i)
    result = {}
    for pred, idx in pred_idx.items():
        m = compute_metrics([scores[i] for i in idx], [labels[i] for i in idx], threshold)
        m["n"]     = len(idx)
        m["n_pos"] = sum(1 for i in idx if labels[i] == "1")
        result[pred] = m
    return result


def best_threshold_per_predicate(
    rows: list[dict],
    scores: list[float],
    labels: list[str],
    threshold_grid: list[float],
    min_examples: int,
) -> dict[str, dict]:
    """
    In-sample: for each predicate with >= min_examples rows find the threshold
    maximising F1, return those metrics.
    Note: in-sample — shows the ceiling, not a generalisation estimate.
    """
    pred_idx: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        pred_idx.setdefault(row.get(PREDICATE_COL, "unknown"), []).append(i)

    result = {}
    for pred, idx in pred_idx.items():
        if len(idx) < min_examples:
            continue
        ps     = [scores[i] for i in idx]
        ls_str = [labels[i] for i in idx]
        best_t, best_f1 = threshold_grid[0], -1.0
        for t in threshold_grid:
            f1 = compute_metrics(ps, ls_str, t)["f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        m = compute_metrics(ps, ls_str, best_t)
        m["n"]           = len(idx)
        m["n_pos"]       = sum(1 for i in idx if labels[i] == "1")
        m["best_thresh"] = best_t
        result[pred] = m
    return result


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(
    cv_results: dict,
    calib_results: dict,
    wt_metrics_by_thresh: dict,
    stack_metrics_by_thresh: dict,
    summary_path: str,
    log: logging.Logger,
    total_time: float,
) -> None:
    W = 110
    lines: list[str] = []

    lines.append("=" * W)
    lines.append("  NLI ADVANCED EVALUATION — SUMMARY")
    lines.append(f"  Total wall time : {_fmt_duration(total_time)}")
    lines.append(f"  CV folds        : {CV_N_SPLITS}")
    lines.append(
        f"  Threshold grid  : {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} "
        f"({len(THRESHOLD_GRID)} values)"
    )
    lines.append("=" * W)

    # ── CV threshold results ──────────────────────────────────────────────────
    lines.append("")
    lines.append("K-FOLD CV THRESHOLD OPTIMISATION")
    hdr1 = (
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  mean_thresh"
    )
    lines.append(hdr1)
    lines.append("  " + "-" * 60)
    for conf_col, cv in sorted(cv_results.items(), key=lambda x: -x[1]["mean_f1"]):
        mean_t = sum(cv["fold_thresholds"]) / len(cv["fold_thresholds"])
        lines.append(
            f"  {cv['model']:<14} {cv['hypothesis']:<22}  "
            f"{cv['mean_f1']:>8.4f} {cv['std_f1']:>7.4f}  {mean_t:.2f}"
        )

    # ── Calibration ───────────────────────────────────────────────────────────
    lines.append("")
    lines.append(f"SCORE CALIBRATION  (OOF, threshold={ANALYSIS_THRESHOLD})")
    hdr2 = (
        f"  {'model':<14} {'hypothesis':<22} {'method':<10}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for conf_col in sorted(calib_results, key=lambda c: -calib_results[c]["platt_metrics"]["f1"]):
        cr  = calib_results[conf_col]
        tag = cv_results[conf_col]["model"]
        hyp = cv_results[conf_col]["hypothesis"]
        for method_name, m in [("platt", cr["platt_metrics"]), ("isotonic", cr["iso_metrics"])]:
            lines.append(
                f"  {tag:<14} {hyp:<22} {method_name:<10}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
            )

    # ── Weighted ensemble ─────────────────────────────────────────────────────
    lines.append("")
    lines.append("WEIGHTED ENSEMBLE  (weights = per-combo CV F1)")
    hdr3 = (
        f"  {'thresh':>6}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr3)
    lines.append("  " + "-" * (len(hdr3) - 2))
    for thresh in sorted(wt_metrics_by_thresh):
        m = wt_metrics_by_thresh[thresh]
        lines.append(
            f"  {thresh:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    # ── Stacking ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("STACKING  (OOF logistic regression on all confidence scores)")
    lines.append(hdr3)
    lines.append("  " + "-" * (len(hdr3) - 2))
    for thresh in sorted(stack_metrics_by_thresh):
        m = stack_metrics_by_thresh[thresh]
        lines.append(
            f"  {thresh:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    # ── Cross-method best F1 comparison ──────────────────────────────────────
    lines.append("")
    lines.append("CROSS-METHOD BEST F1 COMPARISON")
    hdr4 = (
        f"  {'method':<40} {'best_thresh':>11}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}"
    )
    lines.append(hdr4)
    lines.append("  " + "-" * (len(hdr4) - 2))

    # CV (reported as mean CV F1, no single-threshold metrics)
    for conf_col, cv in sorted(cv_results.items(), key=lambda x: -x[1]["mean_f1"])[:3]:
        mean_t = sum(cv["fold_thresholds"]) / len(cv["fold_thresholds"])
        lines.append(
            f"  cv/{cv['model']}/{cv['hypothesis']:<22}  "
            f"{mean_t:>11.2f}  "
            f"{'N/A':>6} {'N/A':>6} {'N/A':>6} {cv['mean_f1']:>6.3f}*"
        )
    lines.append(f"  {'  * CV F1 is the honest OOF estimate':<60}")

    # Calibration best
    for conf_col, cr in calib_results.items():
        tag = cv_results[conf_col]["model"]
        hyp = cv_results[conf_col]["hypothesis"]
        for method_name, m in [("platt", cr["platt_metrics"]), ("isotonic", cr["iso_metrics"])]:
            label = f"{method_name}/{tag}/{hyp}"
            lines.append(
                f"  {label:<40} {ANALYSIS_THRESHOLD:>11.2f}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['f1']:>6.3f}"
            )
        break  # show only the first combo to keep the table concise

    # Weighted ensemble best
    best_wt    = max(wt_metrics_by_thresh.items(), key=lambda x: x[1]["f1"])
    best_stack = max(stack_metrics_by_thresh.items(), key=lambda x: x[1]["f1"])
    for label, thresh, m in [
        ("weighted_ensemble", best_wt[0],    best_wt[1]),
        ("stacking_oof",      best_stack[0], best_stack[1]),
    ]:
        lines.append(
            f"  {label:<40} {thresh:>11.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}"
        )

    lines.append("=" * W)
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    log.info("=" * W)
    for line in lines:
        log.info(line)
    log.info("=" * W)


# ---------------------------------------------------------------------------
# Error analysis writer
# ---------------------------------------------------------------------------

def write_error_analysis(
    rows: list[dict],
    labels: list[str],
    calib_results: dict,
    combo_info: dict,
    wt_scores: np.ndarray,
    stack_oof: np.ndarray,
    error_path: str,
    log: logging.Logger,
    max_examples: int = MAX_ERROR_EXAMPLES,
) -> None:
    W = 110
    n_rows = len(rows)
    n_pos  = labels.count("1")
    thresh = ANALYSIS_THRESHOLD

    # Build dict of method_name → score array for all calibrated combos + ensemble + stacking
    all_methods: dict[str, np.ndarray] = {}
    for conf_col, cr in calib_results.items():
        tag = combo_info[conf_col][0]
        hyp = combo_info[conf_col][1]
        all_methods[f"platt/{tag}/{hyp}"]    = cr["platt_oof"]
        all_methods[f"isotonic/{tag}/{hyp}"] = cr["iso_oof"]
    all_methods["weighted_ensemble"] = wt_scores
    all_methods["stacking_oof"]      = stack_oof

    gold_bool = [l == "1" for l in labels]
    lines: list[str] = []

    def sep(title: str = "") -> None:
        lines.append("")
        lines.append("=" * W)
        if title:
            lines.append(f"  {title}")
            lines.append("=" * W)

    sep()
    lines.append("  NLI ADVANCED EVALUATION — ERROR ANALYSIS")
    lines.append(f"  Threshold        : {thresh}")
    lines.append(f"  Total rows       : {n_rows}")
    lines.append(f"  Pos / Neg        : {n_pos} / {n_rows - n_pos}  "
                 f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")
    lines.append("=" * W)

    # ── Section 1: FP / FN count summary for every method ────────────────────
    sep("SECTION 1 — FP / FN COUNT SUMMARY  (all methods, threshold=0.5)")
    hdr1 = (
        f"  {'method':<42}  "
        f"{'f1':>6} {'prec':>6} {'rec':>6}  "
        f"{'FP':>5} {'FN':>5}  fp_rate  fn_rate"
    )
    lines.append(hdr1)
    lines.append("  " + "-" * (len(hdr1) - 2))
    for method_name, scores_arr in sorted(all_methods.items(),
                                          key=lambda x: -compute_metrics(
                                              x[1].tolist(), labels, thresh)["f1"]):
        m       = compute_metrics(scores_arr.tolist(), labels, thresh)
        fp_rate = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) else 0
        fn_rate = m["FN"] / (m["FN"] + m["TP"]) if (m["FN"] + m["TP"]) else 0
        lines.append(
            f"  {method_name:<42}  "
            f"{m['f1']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f}  "
            f"{m['FP']:>5} {m['FN']:>5}  {fp_rate:.3f}    {fn_rate:.3f}"
        )

    # ── Sections 2 & 3: FP and FN examples for the two key new methods ────────
    focus_methods = {
        k: v for k, v in all_methods.items()
        if k in ("weighted_ensemble", "stacking_oof")
    }

    for method_name, scores_arr in focus_methods.items():
        score_list = scores_arr.tolist()
        fp_rows = [(i, rows[i]) for i in range(n_rows)
                   if score_list[i] >= thresh and not gold_bool[i]]
        fn_rows = [(i, rows[i]) for i in range(n_rows)
                   if score_list[i] < thresh and gold_bool[i]]

        for section_label, examples in [
            (f"FALSE POSITIVES  [{method_name}]  (predicted=1, gold=0) — {len(fp_rows)} total",
             fp_rows),
            (f"FALSE NEGATIVES  [{method_name}]  (predicted=0, gold=1) — {len(fn_rows)} total",
             fn_rows),
        ]:
            sep(section_label)
            for _, (row_idx, row) in enumerate(examples[:max_examples]):
                text_snip = row.get("text", "").replace("\n", " ")[:150]
                lines.append(f"  [row {row_idx}]  predicate={row.get(PREDICATE_COL, '')}  "
                             f"score={score_list[row_idx]:.4f}")
                lines.append(f"    text      : {text_snip!r}")
                lines.append(f"    subject   : {row.get('subject', '')!r}")
                lines.append(f"    object    : {row.get('object', '')!r}")
                lines.append(f"    relation  : {row.get('basic_relation', '')!r}")
            if len(examples) > max_examples:
                lines.append(f"  ... and {len(examples) - max_examples} more")

    # ── Section 4: Hard rows — wrong across ALL calibrated methods ─────────────
    sep("SECTION 4 — HARD ROWS: ALL METHODS WRONG")
    preds_bool = {
        name: [s >= thresh for s in arr.tolist()]
        for name, arr in all_methods.items()
    }
    n_correct_per_row = [
        sum(1 for name in all_methods if preds_bool[name][i] == gold_bool[i])
        for i in range(n_rows)
    ]
    hard_idx = [i for i, nc in enumerate(n_correct_per_row) if nc == 0]
    lines.append(
        f"\n  {len(hard_idx)} rows ({100*len(hard_idx)/n_rows:.1f}%) "
        f"where every method is wrong.  Showing up to {max_examples}."
    )
    for i in hard_idx[:max_examples]:
        text_snip = rows[i].get("text", "").replace("\n", " ")[:150]
        lines.append(f"\n  [row {i}]  gold={labels[i]}  "
                     f"predicate={rows[i].get(PREDICATE_COL, '')}")
        lines.append(f"    text    : {text_snip!r}")
        lines.append(f"    subject : {rows[i].get('subject', '')!r}")
        lines.append(f"    object  : {rows[i].get('object', '')!r}")
        for name, arr in all_methods.items():
            pred = "1" if arr[i] >= thresh else "0"
            lines.append(f"    {name:<42} score={arr[i]:.4f}  pred={pred}")
    if len(hard_idx) > max_examples:
        lines.append(f"  ... and {len(hard_idx) - max_examples} more hard rows")

    # ── Section 5: Confident mistakes (highest-confidence wrong predictions) ───
    sep("SECTION 5 — CONFIDENT MISTAKES")
    CONF_FP_MIN = 0.85
    CONF_FN_MAX = 0.15

    for method_name, scores_arr in focus_methods.items():
        score_list = scores_arr.tolist()
        conf_fp = sorted(
            [(i, score_list[i]) for i in range(n_rows)
             if score_list[i] >= CONF_FP_MIN and not gold_bool[i]],
            key=lambda x: -x[1],
        )
        conf_fn = sorted(
            [(i, score_list[i]) for i in range(n_rows)
             if score_list[i] <= CONF_FN_MAX and gold_bool[i]],
            key=lambda x: x[1],
        )
        lines.append(
            f"\n  [{method_name}]  "
            f"confident FP (score>={CONF_FP_MIN}): {len(conf_fp)}  |  "
            f"confident FN (score<={CONF_FN_MAX}): {len(conf_fn)}"
        )
        for i, score in conf_fp[:5]:
            lines.append(
                f"    FP [row {i}] score={score:.4f}  "
                f"pred={rows[i].get(PREDICATE_COL,'')}  "
                f"{rows[i].get('basic_relation','')!r}"
            )
        for i, score in conf_fn[:5]:
            lines.append(
                f"    FN [row {i}] score={score:.4f}  "
                f"pred={rows[i].get(PREDICATE_COL,'')}  "
                f"{rows[i].get('basic_relation','')!r}"
            )

    sep()
    with open(error_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Error analysis written to {error_path}")

    # Log section 1 counts so they appear in the log file
    log.info("[error_analysis]  FP / FN summary:")
    for method_name, scores_arr in sorted(all_methods.items(),
                                          key=lambda x: -compute_metrics(
                                              x[1].tolist(), labels, thresh)["f1"]):
        m = compute_metrics(scores_arr.tolist(), labels, thresh)
        log.info(
            f"  {method_name:<42}  F1={m['f1']:.3f}  "
            f"FP={m['FP']}  FN={m['FN']}"
        )


# ---------------------------------------------------------------------------
# Detailed analysis writer  (same as before — unchanged)
# ---------------------------------------------------------------------------

def write_advanced_analysis(
    cv_results: dict,
    calib_results: dict,
    wt_metrics_by_thresh: dict,
    stack_metrics_by_thresh: dict,
    pred_stats_by_combo: dict,
    pred_thresh_by_combo: dict,
    analysis_path: str,
    log: logging.Logger,
) -> None:
    W = 110
    lines: list[str] = []

    def sep(title: str = "") -> None:
        lines.append("")
        lines.append("=" * W)
        if title:
            lines.append(f"  {title}")
            lines.append("=" * W)

    sep()
    lines.append("  NLI ADVANCED ANALYSIS")
    lines.append(
        f"  CV folds: {CV_N_SPLITS}  |  "
        f"threshold grid: {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f} "
        f"({len(THRESHOLD_GRID)} values, step 0.05)"
    )
    lines.append("=" * W)

    sep("SECTION 1 — K-FOLD CV THRESHOLD OPTIMISATION  (honest F1 estimate)")
    lines.append(
        f"  {'model':<14} {'hypothesis':<22}  "
        f"{'cv_f1':>8} {'±std':>7}  fold thresholds"
    )
    lines.append("  " + "-" * 90)
    for conf_col, cv in sorted(cv_results.items(), key=lambda x: -x[1]["mean_f1"]):
        fold_t_str = "  ".join(f"{t:.2f}" for t in cv["fold_thresholds"])
        lines.append(
            f"  {cv['model']:<14} {cv['hypothesis']:<22}  "
            f"{cv['mean_f1']:>8.4f} {cv['std_f1']:>7.4f}  [{fold_t_str}]"
        )

    sep("SECTION 2 — SCORE CALIBRATION  (OOF, evaluated at threshold=0.5)")
    hdr2 = (
        f"  {'model':<14} {'hypothesis':<22} {'method':<10}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for conf_col in sorted(calib_results, key=lambda c: -calib_results[c]["platt_metrics"]["f1"]):
        cr  = calib_results[conf_col]
        tag = cv_results[conf_col]["model"]
        hyp = cv_results[conf_col]["hypothesis"]
        for method_name, m in [("platt", cr["platt_metrics"]), ("isotonic", cr["iso_metrics"])]:
            lines.append(
                f"  {tag:<14} {hyp:<22} {method_name:<10}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
                f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
            )
        lines.append("  " + "-" * (len(hdr2) - 2))

    sep("SECTION 3 — WEIGHTED ENSEMBLE  (weights = per-combo CV F1)")
    hdr3 = (
        f"  {'thresh':>6}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr3)
    lines.append("  " + "-" * (len(hdr3) - 2))
    for thresh in sorted(wt_metrics_by_thresh):
        m = wt_metrics_by_thresh[thresh]
        lines.append(
            f"  {thresh:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    sep("SECTION 4 — STACKING  (OOF logistic regression on all confidence scores)")
    lines.append(hdr3)
    lines.append("  " + "-" * (len(hdr3) - 2))
    for thresh in sorted(stack_metrics_by_thresh):
        m = stack_metrics_by_thresh[thresh]
        lines.append(
            f"  {thresh:>6.2f}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}"
        )

    sep(
        f"SECTION 5 — PREDICATE ANALYSIS  "
        f"(min_examples={MIN_PREDICATE_EXAMPLES})"
    )

    # Short display names for combo columns
    _HYP_SHORT = {"basic_relation": "bas", "template_relation": "tpl", "llm_relation": "llm"}

    def _short(combo_key: str) -> str:
        if combo_key == "weighted_ensemble": return "wt_ens"
        if combo_key == "stacking_oof":      return "stack"
        if "/" in combo_key:
            model, hyp = combo_key.split("/", 1)
            # shorten model: take first 4 chars after any hyphens stripped
            m_short = model.replace("-", "").replace("_", "")[:4]
            h_short = _HYP_SHORT.get(hyp, hyp[:3])
            return f"{m_short}:{h_short}"
        return combo_key[:8]

    # Column order: ensembles first, then base combos sorted by mean predicate F1
    priority = ["weighted_ensemble", "stacking_oof"]
    base_ck  = [k for k in pred_stats_by_combo if k not in priority]
    we_stats = pred_stats_by_combo.get("weighted_ensemble", {})

    # All predicates with enough examples (use weighted_ensemble as reference)
    all_preds = sorted(
        {p for ps in pred_stats_by_combo.values()
         for p, m in ps.items() if m["n"] >= MIN_PREDICATE_EXAMPLES},
        key=lambda p: -we_stats.get(p, {}).get("f1", 0),
    )

    ordered_ck = priority + sorted(
        base_ck,
        key=lambda k: -sum(
            pred_stats_by_combo[k].get(p, {}).get("f1", 0) for p in all_preds
        ) / max(len(all_preds), 1),
    )

    # ── 5a: F1 matrix ────────────────────────────────────────────────────────
    lines.append(
        f"\n  5a. F1 matrix — predicates × combos  "
        f"(threshold={ANALYSIS_THRESHOLD}, sorted by weighted_ensemble F1)"
    )
    PRED_W  = 28
    COL_W   = 7   # enough for " 0.821"
    short_hdrs = [_short(k) for k in ordered_ck]
    # header row
    hdr_mat = f"  {'predicate':<{PRED_W}} {'n':>4} {'pos%':>5} "
    hdr_mat += " ".join(f"{h:>{COL_W}}" for h in short_hdrs)
    lines.append(hdr_mat)
    lines.append("  " + "-" * (PRED_W + 12 + len(ordered_ck) * (COL_W + 1)))
    for pred in all_preds:
        m0    = we_stats.get(pred, {})
        n     = m0.get("n", 0)
        n_pos = m0.get("n_pos", 0)
        pos_p = 100 * n_pos / n if n else 0
        row   = f"  {pred:<{PRED_W}} {n:>4} {pos_p:>4.0f}% "
        row  += " ".join(
            f"{pred_stats_by_combo[k].get(pred, {}).get('f1', float('nan')):>{COL_W}.3f}"
            if pred in pred_stats_by_combo.get(k, {}) else f"{'—':>{COL_W}}"
            for k in ordered_ck
        )
        lines.append(row)
    skipped_preds = {
        p for ps in pred_stats_by_combo.values()
        for p, m in ps.items() if m["n"] < MIN_PREDICATE_EXAMPLES
    }
    if skipped_preds:
        lines.append(
            f"  (omitted {len(skipped_preds)} predicate(s) with <{MIN_PREDICATE_EXAMPLES} examples: "
            + ", ".join(sorted(skipped_preds)) + ")"
        )

    # ── 5b: per-predicate best threshold (wt_ens + stacking_oof only) ─────────
    lines.append(
        f"\n  5b. Per-predicate best threshold  "
        f"(in-sample ceiling — NOT a generalisation estimate)"
    )
    hdr_bt = (
        f"    {'predicate':<{PRED_W}} {'n':>4}  "
        f"{'F1@0.5':>7}  {'best_t':>6}  {'best_F1':>7}  {'delta':>6}"
    )
    for focus_key in ["weighted_ensemble", "stacking_oof"]:
        lines.append(f"\n  [{focus_key}]")
        lines.append(hdr_bt)
        lines.append("    " + "-" * (len(hdr_bt) - 4))
        pt = pred_thresh_by_combo.get(focus_key, {})
        ps = pred_stats_by_combo.get(focus_key, {})
        for pred in sorted(pt, key=lambda p: -pt[p]["f1"]):
            m_best     = pt[pred]
            f1_default = ps.get(pred, {}).get("f1", 0.0)
            delta      = m_best["f1"] - f1_default
            lines.append(
                f"    {pred:<{PRED_W}} {m_best['n']:>4}  "
                f"{f1_default:>7.3f}  {m_best['best_thresh']:>6.2f}  "
                f"{m_best['f1']:>7.3f}  {delta:>+6.3f}"
            )

    sep()
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Detailed analysis written to {analysis_path}")


# ---------------------------------------------------------------------------
# Column name parser
# ---------------------------------------------------------------------------

def parse_conf_col(col: str) -> tuple[str, str]:
    """
    Parse 'confidence_{model}_{hyp_col}' → (model_tag, hyp_col).
    Matches hyp_col by known suffix so model tags with underscores work correctly.
    """
    rest = col[len("confidence_"):]
    for hyp_col in sorted(KNOWN_HYP_COLS, key=len, reverse=True):
        if rest.endswith("_" + hyp_col):
            return rest[: -(len(hyp_col) + 1)], hyp_col
    parts = rest.split("_", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args, base: str, log: logging.Logger | None = None) -> None:
    input_path        = os.path.join(base, args.input)
    output_path       = os.path.join(base, args.output)
    log_path          = os.path.join(base, args.log)
    summary_path      = os.path.join(base, args.summary)
    error_path        = os.path.join(base, args.error_analysis)
    analysis_path     = os.path.join(base, args.analysis)
    for p in (output_path, log_path, summary_path, error_path, analysis_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    if log is None:
        log = setup_logger(log_path)
    wall_start = time.time()

    log.info("=" * 70)
    log.info("nli_advanced_eval.py  started")
    log.info(f"  input:          {input_path}")
    log.info(f"  output:         {output_path}")
    log.info(f"  log:            {log_path}")
    log.info(f"  summary:        {summary_path}")
    log.info(f"  error analysis: {error_path}")
    log.info(f"  full analysis:  {analysis_path}")
    log.info(f"  cv folds:       {CV_N_SPLITS}")
    log.info(
        f"  thresh grid:    {THRESHOLD_GRID[0]:.2f}–{THRESHOLD_GRID[-1]:.2f}  "
        f"({len(THRESHOLD_GRID)} values)"
    )
    log.info("=" * 70)

    # --- Load CSV ---
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_rows = len(rows)
    log.info(f"[load]  {n_rows} rows")

    labels     = [r[args.label_col] for r in rows]
    labels_int = np.array([int(l) for l in labels])
    n_pos      = int(labels_int.sum())
    log.info(
        f"[load]  label dist: pos={n_pos}, neg={n_rows - n_pos} "
        f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)"
    )

    conf_cols = sorted(col for col in rows[0].keys() if col.startswith("confidence_"))
    if not conf_cols:
        raise ValueError(
            "No 'confidence_*' columns found. Run clean_dataset_with_nli.py first."
        )
    log.info(f"[load]  {len(conf_cols)} confidence columns detected")

    combo_info: dict[str, tuple[str, str]] = {col: parse_conf_col(col) for col in conf_cols}
    all_score_arrays: dict[str, np.ndarray] = {
        col: np.array([float(r[col]) for r in rows]) for col in conf_cols
    }

    # ── 1. K-fold CV threshold search ─────────────────────────────────────────
    log.info("-" * 70)
    log.info("[cv_thresh]  K-fold CV threshold optimisation")
    cv_results: dict[str, dict] = {}
    for col in conf_cols:
        tag, hyp_col = combo_info[col]
        mean_f1, std_f1, fold_thresholds = kfold_threshold_search(
            all_score_arrays[col], labels_int, CV_N_SPLITS, THRESHOLD_GRID
        )
        cv_results[col] = {
            "model": tag, "hypothesis": hyp_col,
            "mean_f1": mean_f1, "std_f1": std_f1,
            "fold_thresholds": fold_thresholds,
        }
        log.info(
            f"  {tag}/{hyp_col}: CV F1={mean_f1:.4f}±{std_f1:.4f}  "
            f"fold thresholds={[f'{t:.2f}' for t in fold_thresholds]}"
        )

    # ── 2. Calibration ────────────────────────────────────────────────────────
    log.info("-" * 70)
    log.info("[calibrate]  Platt scaling + isotonic regression (OOF)")
    calib_results: dict[str, dict] = {}
    for col in conf_cols:
        tag, hyp_col = combo_info[col]
        raw = all_score_arrays[col]

        platt_oof = kfold_platt_calibrate(raw, labels_int, CV_N_SPLITS)
        iso_oof   = kfold_isotonic_calibrate(raw, labels_int, CV_N_SPLITS)

        for i, r in enumerate(rows):
            r[f"platt_{col}"]    = f"{platt_oof[i]:.4f}"
            r[f"iso_{col}"]      = f"{iso_oof[i]:.4f}"

        platt_m = compute_metrics(platt_oof.tolist(), labels, ANALYSIS_THRESHOLD)
        iso_m   = compute_metrics(iso_oof.tolist(),   labels, ANALYSIS_THRESHOLD)
        calib_results[col] = {
            "platt_metrics": platt_m, "iso_metrics": iso_m,
            "platt_oof": platt_oof,   "iso_oof": iso_oof,
        }
        log.info(
            f"  [platt]    {tag}/{hyp_col}: "
            f"F1={platt_m['f1']:.3f}  prec={platt_m['precision']:.3f}  "
            f"rec={platt_m['recall']:.3f}"
        )
        log.info(
            f"  [isotonic] {tag}/{hyp_col}: "
            f"F1={iso_m['f1']:.3f}  prec={iso_m['precision']:.3f}  "
            f"rec={iso_m['recall']:.3f}"
        )

    # ── 3. Weighted ensemble ──────────────────────────────────────────────────
    log.info("-" * 70)
    log.info("[weighted_ensemble]  weights = per-combo CV F1")
    cv_f1_weights = {col: cv_results[col]["mean_f1"] for col in conf_cols}
    wt_scores     = compute_weighted_ensemble(all_score_arrays, cv_f1_weights)
    for i, r in enumerate(rows):
        r["nli_ensemble_weighted"] = f"{wt_scores[i]:.4f}"
    wt_metrics_by_thresh = {
        t: compute_metrics(wt_scores.tolist(), labels, t) for t in EVAL_THRESHOLDS
    }
    best_wt = max(wt_metrics_by_thresh.values(), key=lambda m: m["f1"])
    log.info(
        f"  best F1={best_wt['f1']:.3f}  "
        f"prec={best_wt['precision']:.3f}  rec={best_wt['recall']:.3f}"
    )

    # ── 4. Stacking ────────────────────────────────────────────────────────────
    log.info("-" * 70)
    log.info("[stacking]  OOF logistic regression on all confidence scores")
    X_stack   = np.column_stack([all_score_arrays[col] for col in conf_cols])
    stack_oof = kfold_stacking(X_stack, labels_int, CV_N_SPLITS)
    for i, r in enumerate(rows):
        r["nli_stacking_oof"] = f"{stack_oof[i]:.4f}"
    stack_metrics_by_thresh = {
        t: compute_metrics(stack_oof.tolist(), labels, t) for t in EVAL_THRESHOLDS
    }
    best_stack = max(stack_metrics_by_thresh.values(), key=lambda m: m["f1"])
    log.info(
        f"  best F1={best_stack['f1']:.3f}  "
        f"prec={best_stack['precision']:.3f}  rec={best_stack['recall']:.3f}"
    )

    # ── 5. Predicate-stratified analysis ──────────────────────────────────────
    log.info("-" * 70)
    log.info("[predicate]  stratified metrics + per-predicate threshold search")
    pred_stats_by_combo:  dict[str, dict] = {}
    pred_thresh_by_combo: dict[str, dict] = {}

    for col in conf_cols:
        tag, hyp_col  = combo_info[col]
        combo_key     = f"{tag}/{hyp_col}"
        score_list    = all_score_arrays[col].tolist()
        pred_stats_by_combo[combo_key]  = compute_predicate_metrics(
            rows, score_list, labels, ANALYSIS_THRESHOLD
        )
        pred_thresh_by_combo[combo_key] = best_threshold_per_predicate(
            rows, score_list, labels, THRESHOLD_GRID, MIN_PREDICATE_EXAMPLES
        )

    for combo_key, score_arr in [
        ("weighted_ensemble", wt_scores),
        ("stacking_oof",      stack_oof),
    ]:
        score_list = score_arr.tolist()
        pred_stats_by_combo[combo_key]  = compute_predicate_metrics(
            rows, score_list, labels, ANALYSIS_THRESHOLD
        )
        pred_thresh_by_combo[combo_key] = best_threshold_per_predicate(
            rows, score_list, labels, THRESHOLD_GRID, MIN_PREDICATE_EXAMPLES
        )

    # ── Save enriched CSV ─────────────────────────────────────────────────────
    log.info("-" * 70)
    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows → {output_path}")

    # ── Write all output files ────────────────────────────────────────────────
    total = time.time() - wall_start

    log.info("-" * 70)
    log.info("[summary]  writing summary")
    write_summary(
        cv_results, calib_results,
        wt_metrics_by_thresh, stack_metrics_by_thresh,
        summary_path, log, total,
    )
    log.info(f"Summary written to {summary_path}")

    log.info("-" * 70)
    log.info("[error_analysis]  writing error analysis")
    write_error_analysis(
        rows, labels, calib_results, combo_info,
        wt_scores, stack_oof, error_path, log,
    )

    log.info("-" * 70)
    log.info("[analysis]  writing detailed analysis")
    write_advanced_analysis(
        cv_results, calib_results,
        wt_metrics_by_thresh, stack_metrics_by_thresh,
        pred_stats_by_combo, pred_thresh_by_combo,
        analysis_path, log,
    )

    log.info("=" * 70)
    log.info(f"Done.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  {output_path}")
    log.info(f"  {summary_path}")
    log.info(f"  {error_path}")
    log.info(f"  {analysis_path}")
    log.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced NLI post-processing: calibration, stacking, per-predicate analysis"
    )
    parser.add_argument("--input",          default=INPUT_FILE,          help=f"Input CSV  (default: {INPUT_FILE})")
    parser.add_argument("--output",         default=OUTPUT_FILE,         help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--log",            default=LOG_FILE,            help=f"Log file   (default: {LOG_FILE})")
    parser.add_argument("--summary",        default=SUMMARY_FILE,        help=f"Summary    (default: {SUMMARY_FILE})")
    parser.add_argument("--error-analysis", default=ERROR_ANALYSIS_FILE, help=f"Error file (default: {ERROR_ANALYSIS_FILE})")
    parser.add_argument("--analysis",       default=ANALYSIS_FILE,       help=f"Full report(default: {ANALYSIS_FILE})")
    parser.add_argument("--label-col",      default=LABEL_COL,           help=f"Gold label column (default: {LABEL_COL})")
    args = parser.parse_args()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(args, base)


if __name__ == "__main__":
    main()