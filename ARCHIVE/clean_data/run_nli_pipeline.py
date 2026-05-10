"""
Full NLI pipeline: classification → advanced evaluation in a single run.

Phase 1 (GPU): load each NLI encoder model and score all (model × hypothesis) pairs,
               write enriched CSV + NLI-level summary / error analysis.
Phase 2 (CPU): k-fold CV threshold optimisation, Platt / isotonic calibration,
               weighted ensemble, stacking, predicate-stratified evaluation.

Both phases share a single log file.  Pass --skip-classify to re-run only Phase 2
against an existing nli_classified.csv (e.g. to iterate on the analysis).

Usage:
    cd hebrew_RE
    CUDA_VISIBLE_DEVICES=2 python -m clean_data.run_nli_pipeline
    CUDA_VISIBLE_DEVICES=2 python -m clean_data.run_nli_pipeline --input outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv
    python -m clean_data.run_nli_pipeline --skip-classify
"""

import argparse
import logging
import os
import time

from .clean_dataset_with_nli import run as _run_classify
from .nli_advanced_eval      import run as _run_advanced, setup_logger, _fmt_duration

# ---------------------------------------------------------------------------
# Default paths  (all relative to project root)
# ---------------------------------------------------------------------------

NLI_INPUT    = "outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv"
NLI_OUTPUT   = "outputs/NLI_clean/nli_classified.csv"
ADV_OUTPUT   = "outputs/NLI_clean/nli_advanced_classified.csv"
LOG_FILE     = "outputs/NLI_clean/nli_pipeline.log"
NLI_SUMMARY  = "outputs/NLI_clean/nli_summary.txt"
NLI_ERROR    = "outputs/NLI_clean/nli_error_analysis.txt"
ADV_SUMMARY  = "outputs/NLI_clean/nli_advanced_summary.txt"
ADV_ERROR    = "outputs/NLI_clean/nli_advanced_error_analysis.txt"
ADV_ANALYSIS = "outputs/NLI_clean/nli_advanced_analysis.txt"
LABEL_COL    = "relation_present"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLI pipeline: classification (GPU) + advanced evaluation (CPU)"
    )
    parser.add_argument("--input",         default=NLI_INPUT,   help=f"Input CSV for NLI classification (default: {NLI_INPUT})")
    parser.add_argument("--nli-output",    default=NLI_OUTPUT,  help=f"NLI classified CSV (default: {NLI_OUTPUT})")
    parser.add_argument("--adv-output",    default=ADV_OUTPUT,  help=f"Advanced eval output CSV (default: {ADV_OUTPUT})")
    parser.add_argument("--log",           default=LOG_FILE,    help=f"Shared log file (default: {LOG_FILE})")
    parser.add_argument("--nli-summary",   default=NLI_SUMMARY, help=f"NLI summary file (default: {NLI_SUMMARY})")
    parser.add_argument("--nli-error",     default=NLI_ERROR,   help=f"NLI error analysis (default: {NLI_ERROR})")
    parser.add_argument("--adv-summary",   default=ADV_SUMMARY, help=f"Advanced summary (default: {ADV_SUMMARY})")
    parser.add_argument("--adv-error",     default=ADV_ERROR,   help=f"Advanced error analysis (default: {ADV_ERROR})")
    parser.add_argument("--adv-analysis",  default=ADV_ANALYSIS,help=f"Advanced full report (default: {ADV_ANALYSIS})")
    parser.add_argument("--label-col",     default=LABEL_COL,   help=f"Gold label column (default: {LABEL_COL})")
    parser.add_argument("--skip-classify", action="store_true",
                        help="Skip Phase 1 (NLI classification); use existing --nli-output CSV")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base, os.path.dirname(args.log)), exist_ok=True)
    log = setup_logger(os.path.join(base, args.log))

    wall_start = time.time()
    log.info("=" * 70)
    log.info("NLI PIPELINE  started")
    log.info(f"  log        : {os.path.join(base, args.log)}")
    log.info(f"  label col  : {args.label_col}")
    log.info(f"  skip phase1: {args.skip_classify}")
    log.info("=" * 70)

    # ── Phase 1: NLI classification (GPU) ────────────────────────────────────
    if not args.skip_classify:
        log.info("")
        log.info("━" * 70)
        log.info("PHASE 1 — NLI CLASSIFICATION  (GPU inference)")
        log.info("━" * 70)
        nli_args = argparse.Namespace(
            input         = args.input,
            output        = args.nli_output,
            log           = args.log,
            summary       = args.nli_summary,
            error_analysis= args.nli_error,
            label_col     = args.label_col,
        )
        _run_classify(nli_args, base, log=log)
    else:
        log.info("Skipping Phase 1 — using existing file: "
                 + os.path.join(base, args.nli_output))

    # ── Phase 2: Advanced evaluation (CPU) ───────────────────────────────────
    log.info("")
    log.info("━" * 70)
    log.info("PHASE 2 — ADVANCED EVALUATION  (CV / calibration / stacking)")
    log.info("━" * 70)
    adv_args = argparse.Namespace(
        input         = args.nli_output,
        output        = args.adv_output,
        log           = args.log,
        summary       = args.adv_summary,
        error_analysis= args.adv_error,
        analysis      = args.adv_analysis,
        label_col     = args.label_col,
    )
    _run_advanced(adv_args, base, log=log)

    total = time.time() - wall_start
    log.info("")
    log.info("=" * 70)
    log.info(f"PIPELINE COMPLETE.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  classified CSV  : {os.path.join(base, args.nli_output)}")
    log.info(f"  advanced CSV    : {os.path.join(base, args.adv_output)}")
    log.info(f"  NLI summary     : {os.path.join(base, args.nli_summary)}")
    log.info(f"  advanced summary: {os.path.join(base, args.adv_summary)}")
    log.info(f"  advanced analysis:{os.path.join(base, args.adv_analysis)}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()