# python clean_data/run_all_and_evaluate.py [--force] [--skip NLI,LLM_local,...]

"""
Orchestrator: run all 4 gold-dataset classification methods, merge results
into one CSV, and produce an accuracy / precision / recall / F1 / runtime /
cost table.

Methods
-------
  NLI        gold_nli_classify.py          5 NLI encoder models
  LLM_local  gold_llm_classify.py          2 open-source LLMs × 2 prompt langs
  LLM_api    gold_api_llm_classify.py      2 cloud LLMs × 2 prompt langs (OpenRouter)
  KFold_RC   gold_kfold_rc_classify.py     3 encoders trained with K-fold RC

Each method is run as a subprocess so GPU memory is fully released between them.
If an output CSV already exists the method is skipped (override with --force).

Outputs
-------
  data/crocodile_heb25_gold_500_ALL_classified.csv   — merged predictions
  results/metrics_table.csv                          — per-variant metrics + cost
  results/metrics_table.txt                          — human-readable table

Runtime/cost
------------
  GPU methods : estimated cost = wall_time_hours × GPU_COST_PER_HOUR
  LLM_api     : actual cost from token usage saved in *_metadata.json by the script

Usage
-----
  # Run everything (GPU=0 for all except LLM_local which needs 2)
  CUDA_VISIBLE_DEVICES=0   python clean_data/run_all_and_evaluate.py
  OPENROUTER_API_KEY=sk-or-... python clean_data/run_all_and_evaluate.py

  # Re-run one method even if output exists
  python clean_data/run_all_and_evaluate.py --force --skip LLM_local,KFold_RC

  # Skip running (just merge + evaluate existing outputs)
  python clean_data/run_all_and_evaluate.py --skip NLI,LLM_local,LLM_api,KFold_RC
"""

import os
import sys
import json
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR    = "/home/nlp/ronke21/hebrew_RE"
SCRIPTS_DIR = os.path.join(BASE_DIR, "clean_data")
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

GOLD_CSV   = os.path.join(DATA_DIR, "crocodile_heb25_gold_500.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "crocodile_heb25_gold_500_ALL_classified.csv")

GPU_COST_PER_HOUR = 2.0   # USD — adjust to your GPU rental/cloud rate

# CUDA_VISIBLE_DEVICES used when launching each subprocess
METHODS = [
    {
        "name":        "NLI",
        "script":      "gold_nli_classify.py",
        "output_csv":  os.path.join(DATA_DIR, "crocodile_heb25_gold_500_nli_classified.csv"),
        "type":        "gpu",
        "cuda":        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    },
    {
        "name":        "LLM_local",
        "script":      "gold_llm_classify.py",
        "output_csv":  os.path.join(DATA_DIR, "crocodile_heb25_gold_500_llm_classified.csv"),
        "type":        "gpu",
        "cuda":        os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
    },
    {
        "name":        "LLM_api",
        "script":      "gold_api_llm_classify.py",
        "output_csv":  os.path.join(DATA_DIR, "crocodile_heb25_gold_500_api_llm_classified.csv"),
        "type":        "api",
        "cuda":        None,
    },
    {
        "name":        "KFold_RC",
        "script":      "gold_kfold_rc_classify.py",
        "output_csv":  os.path.join(DATA_DIR, "crocodile_heb25_gold_500_kfold_rc_classified.csv"),
        "type":        "gpu",
        "cuda":        os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    },
]

# ── Runner ─────────────────────────────────────────────────────────────────────

def run_method(method, force=False):
    """
    Run a classification script as a subprocess.
    Returns runtime_sec (float) or loads it from saved metadata.
    """
    name       = method["name"]
    script     = os.path.join(SCRIPTS_DIR, method["script"])
    output_csv = method["output_csv"]

    if not force and os.path.exists(output_csv):
        print(f"[SKIP] {name} — output already exists: {output_csv}")
        return _load_runtime(method)

    print(f"\n{'='*70}")
    print(f"[RUN]  {name}  ({method['script']})")
    print(f"{'='*70}")

    env = os.environ.copy()
    if method["cuda"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = method["cuda"]

    t_start = time.time()
    result  = subprocess.run(
        [sys.executable, script],
        env=env,
        check=False,   # don't raise on non-zero exit — we capture it
    )
    runtime = time.time() - t_start

    if result.returncode != 0:
        print(f"[WARN] {name} exited with code {result.returncode}")

    # Save timing alongside the output CSV for later reference
    meta_path = output_csv.replace(".csv", "_metadata.json")
    if method["type"] == "gpu":
        _save_gpu_metadata(meta_path, method["script"], runtime)

    print(f"[DONE] {name}  — {_fmt_time(runtime)}")
    return runtime


def _save_gpu_metadata(path, script, runtime):
    existing = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing.update({"script": script, "runtime_sec": round(runtime, 2)})
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _load_runtime(method):
    meta_path = method["output_csv"].replace(".csv", "_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return json.load(f).get("runtime_sec", 0.0)
        except Exception:
            pass
    return 0.0


def _load_cost(method):
    """Return USD cost for this method (from metadata JSON)."""
    meta_path = method["output_csv"].replace(".csv", "_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if method["type"] == "api":
                return meta.get("cost_usd", 0.0)
        except Exception:
            pass
    return None   # GPU cost computed from runtime in caller


def _fmt_time(sec):
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}min"
    return f"{sec/3600:.2f}h"


# ── Merge ──────────────────────────────────────────────────────────────────────

def merge_outputs(methods):
    """
    Load gold CSV and each method's output CSV.
    Collect all relation_present_* and confidence_* columns into one DataFrame.
    """
    base = pd.read_csv(GOLD_CSV)

    for method in methods:
        csv_path = method["output_csv"]
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing output for {method['name']}: {csv_path}")
            continue
        pred_df = pd.read_csv(csv_path)
        new_cols = [c for c in pred_df.columns
                    if c.startswith("relation_present_") or c.startswith("confidence_")]
        for col in new_cols:
            base[col] = pred_df[col].values

    base.to_csv(OUTPUT_CSV, index=False)
    print(f"\nMerged dataset saved to {OUTPUT_CSV}  ({len(base)} rows, {len(base.columns)} columns)")
    return base


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(df):
    """
    For every relation_present_* column, compute classification metrics
    against the gold relation_present column.
    Returns a list of dicts (one per variant).
    """
    y_true = df["relation_present"].values.astype(int)
    rows   = []

    for col in sorted(df.columns):
        if not col.startswith("relation_present_"):
            continue
        y_pred = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).values
        rows.append({
            "column":   col,
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "pred_pos":  int(y_pred.sum()),   # how many rows predicted positive
        })
    return rows


def _method_for_col(col, methods):
    """Match a relation_present_* column back to its method config."""
    for m in methods:
        if not os.path.exists(m["output_csv"]):
            continue
        pred_df = pd.read_csv(m["output_csv"], nrows=0)
        if col in pred_df.columns:
            return m
    return None


# ── Table ──────────────────────────────────────────────────────────────────────

_METHOD_PREFIXES = {
    "NLI":       ["mmBERT", "mt5_xl", "me5large", "neodictabert", "xlmrobertalarge"],
    "LLM_local": ["gemma4_31b", "dictaLM3_24b"],
    "LLM_api":   ["gemini3_pro", "gpt5"],
    "KFold_RC":  ["kfold_xlmroberta", "kfold_mmbert", "kfold_neodictabert"],
}


def _assign_method(col):
    variant = col.replace("relation_present_", "")
    for method_name, prefixes in _METHOD_PREFIXES.items():
        if any(variant.startswith(p) for p in prefixes):
            return method_name
    return "unknown"


def build_table(metrics_rows, methods, runtimes):
    """Combine metrics with runtime/cost into one display table."""
    # Pre-load cost and runtime per method name
    method_meta = {}
    for m in methods:
        name    = m["name"]
        runtime = runtimes.get(name, 0.0)
        cost    = _load_cost(m)
        if cost is None:  # GPU method: estimate from runtime
            cost = (runtime / 3600) * GPU_COST_PER_HOUR
        method_meta[name] = {"runtime_sec": runtime, "cost_usd": cost}

    table_rows = []
    for r in metrics_rows:
        method_name = _assign_method(r["column"])
        meta        = method_meta.get(method_name, {"runtime_sec": 0.0, "cost_usd": 0.0})
        variant     = r["column"].replace("relation_present_", "")
        table_rows.append([
            method_name,
            variant,
            f"{r['accuracy']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            r["pred_pos"],
            _fmt_time(meta["runtime_sec"]),
            f"${meta['cost_usd']:.4f}",
        ])

    headers = ["Method", "Variant", "Accuracy", "Precision", "Recall", "F1",
               "Pred+", "Runtime", "Cost (USD)"]
    return headers, table_rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-run methods even if output CSV already exists")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated method names to skip, e.g. LLM_local,KFold_RC")
    args = parser.parse_args()

    skip_set = set(s.strip() for s in args.skip.split(",") if s.strip())
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Step 1: Run each method ───────────────────────────────────────────────
    runtimes = {}
    for method in METHODS:
        if method["name"] in skip_set:
            print(f"[SKIP] {method['name']} (--skip)")
            runtimes[method["name"]] = _load_runtime(method)
            continue
        runtimes[method["name"]] = run_method(method, force=args.force)

    # ── Step 2: Merge all outputs ─────────────────────────────────────────────
    print("\n── Merging outputs ──────────────────────────────────────────────────")
    merged_df = merge_outputs(METHODS)

    # ── Step 3: Compute metrics ───────────────────────────────────────────────
    print("\n── Computing metrics ────────────────────────────────────────────────")
    metrics_rows = compute_metrics(merged_df)

    # ── Step 4: Build and display table ──────────────────────────────────────
    headers, table_rows = build_table(metrics_rows, METHODS, runtimes)

    table_str = tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".4f")
    print("\n" + table_str)

    # Save as CSV
    metrics_df = pd.DataFrame(table_rows, columns=headers)
    csv_path   = os.path.join(RESULTS_DIR, "metrics_table.csv")
    txt_path   = os.path.join(RESULTS_DIR, "metrics_table.txt")
    metrics_df.to_csv(csv_path, index=False)
    with open(txt_path, "w") as f:
        f.write(table_str + "\n")

    print(f"\nMetrics table saved to {csv_path}")
    print(f"Metrics text  saved to {txt_path}")

    # Summary: best F1 per method group
    print("\n── Best F1 per method ───────────────────────────────────────────────")
    best = {}
    for r in metrics_rows:
        m = _assign_method(r["column"])
        if m not in best or r["f1"] > best[m]["f1"]:
            best[m] = r
    for m, r in best.items():
        print(f"  {m:12s}  best variant: {r['column'].replace('relation_present_',''):35s}  F1={r['f1']:.4f}")


if __name__ == "__main__":
    main()
