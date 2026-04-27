"""
Classify relation presence in text using API LLMs via OpenRouter.
For each (model × prompt_lang × relation_type) combination adds:
  llm_clean_{tag}_{lang}_{rel}  — "yes" / "no" / "unknown"
  llm_raw_{tag}_{lang}_{rel}    — raw model response (for inspection)

Evaluates each combination against the gold label column (relation_present)
and writes accuracy / precision / recall / F1 + cost to the log and summary file.

Requires:  pip install openai
           export OPENROUTER_API_KEY=sk-or-...

Usage:
    python clean_dataset_with_api_llm.py
    python clean_dataset_with_api_llm.py --input outputs/prepared_gold_dataset.csv
    python clean_dataset_with_api_llm.py --workers 16
"""

import os
import re
import csv
import time
import logging
import argparse
import concurrent.futures

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE   = "outputs/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_FILE  = "outputs/api_llm_classified.csv"
LOG_FILE     = "outputs/api_llm_classify.log"
SUMMARY_FILE = "outputs/api_llm_summary.txt"

LABEL_COL = "relation_present"    # gold label: "1" = present, "0" = absent

OPENROUTER_BASE_URL    = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_SITE_URL    = "https://github.com/nlp/hebrew_RE"
OPENROUTER_APP_NAME    = "Hebrew-RE"

# Columns that carry the relation string for each relation type
RELATION_COLS = {
    "triplet":  "basic_relation",     # subject predicate object (used for formatting only)
    "template": "template_relation",  # natural-language template sentence
}

# API models: id on OpenRouter, short tag, and price per 1M tokens (USD, from openrouter.ai)
API_MODELS = [
    {
        "id":  "openai/gpt-5.4",
        "tag": "gpt54",
        "prices_per_1m": {"input": 2.50, "output": 15.0},
    },
    {
        "id":  "google/gemini-3.1-pro-preview",
        "tag": "gemini31_pro",
        "prices_per_1m": {"input": 2.0, "output": 12.0},
    },
]

PROMPT_LANGS   = ["he", "en"]
RELATION_TYPES = ["triplet", "template"]

LLM_MAX_TOKENS = 10      # we only need yes/no
MAX_WORKERS    = 8       # concurrent API requests per combo
MAX_RETRIES    = 3       # retries on API error
RETRY_BASE_DELAY = 2.0   # seconds; doubles each retry

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
    logger = logging.getLogger("clean_dataset_with_api_llm")
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
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Prompt builders  (all API models are chat-based)
# ---------------------------------------------------------------------------

def build_messages(lang: str, rel_type: str, row: dict) -> list[dict]:
    """Return a list of chat messages (system + user) for the given combo."""
    text     = row["text"]
    subject  = row["subject"]
    pred     = row["predicate"]
    obj      = row["object"]
    relation = row[RELATION_COLS[rel_type]]

    if lang == "he":
        if rel_type == "triplet":
            system = (
                "אתה מסייע לזיהוי יחסים בטקסטים בעברית. "
                "יחס הוא קשר סמנטי בין שתי ישויות: נושא (subject), סוג היחס (predicate) ואובייקט (object). "
                "תפקידך לקרוא טקסט ולקבוע האם היחס הנתון מופיע או נובע מהטקסט, "
                "כלומר האם ניתן להסיק מהטקסט שהנושא קשור לאובייקט באמצעות היחס המצוין. "
                'ענה "כן" אם היחס מופיע בטקסט, או "לא" אם אינו מופיע. ענה במילה אחת בלבד.'
            )
            user = (
                f"טקסט:\n{text}\n\n"
                f"יחס לבדיקה:\n"
                f"  נושא (subject):   {subject}\n"
                f"  יחס (predicate):  {pred}\n"
                f"  אובייקט (object): {obj}\n\n"
                f"חפש בטקסט האם הקשר בין \"{subject}\" ל-\"{obj}\" דרך היחס \"{pred}\" מוזכר או נובע ממנו.\n"
                'ענה "כן" או "לא" בלבד.'
            )
        else:
            system = (
                "אתה מסייע לזיהוי יחסים בטקסטים בעברית. "
                "יחס הוא קשר סמנטי בין שתי ישויות, המנוסח כמשפט. "
                "תפקידך לקרוא טקסט ולקבוע האם המשפט הנתון נובע מהטקסט או מופיע בו. "
                'ענה "כן" אם המשפט נובע מהטקסט, או "לא" אחרת. ענה במילה אחת בלבד.'
            )
            user = (
                f"טקסט:\n{text}\n\n"
                f"משפט לבדיקה: {relation}\n\n"
                f"חפש בטקסט האם המשפט הנ\"ל נובע ממנו או מופיע בו.\n"
                'ענה "כן" או "לא" בלבד.'
            )
    else:  # en
        if rel_type == "triplet":
            system = (
                "You are a relation extraction assistant. "
                "A relation is a semantic connection between two entities, described by a subject, a predicate (relation type), and an object. "
                "Your task is to read a text and determine whether the given relation is expressed or can be inferred from it — "
                "i.e., whether the text indicates that the subject is connected to the object via the given predicate. "
                'Answer "yes" if the relation is expressed in the text, "no" if it is not. One word only.'
            )
            user = (
                f"Text:\n{text}\n\n"
                f"Relation to check:\n"
                f"  Subject:   {subject}\n"
                f"  Predicate: {pred}\n"
                f"  Object:    {obj}\n\n"
                f"Look in the text for whether the connection between \"{subject}\" and \"{obj}\" "
                f"via the relation \"{pred}\" is mentioned or can be inferred.\n"
                'Answer "yes" or "no" only.'
            )
        else:
            system = (
                "You are a textual entailment assistant. "
                "A relation between two entities is expressed here as a natural-language statement. "
                "Your task is to read a text and determine whether the given statement is entailed by or expressed in it. "
                'Answer "yes" if entailed, "no" if not. One word only.'
            )
            user = (
                f"Text:\n{text}\n\n"
                f"Statement to check: {relation}\n\n"
                f"Look in the text for whether the above statement is expressed or can be inferred from it.\n"
                'Answer "yes" or "no" only.'
            )

    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_YES_TOKENS = {"yes", "כן"}
_NO_TOKENS  = {"no",  "לא"}

def parse_yes_no(raw: str) -> str:
    cleaned = raw.strip()
    tokens = [t.lower() for t in re.split(r"[\s.,!?:;()\[\]\"']+", cleaned) if t]
    if tokens:
        if tokens[0] in _YES_TOKENS:
            return "yes"
        if tokens[0] in _NO_TOKENS:
            return "no"
    lower = cleaned.lower()
    if re.search(r"\byes\b", lower) or "כן" in lower:
        return "yes"
    if re.search(r"\bno\b",  lower) or "לא" in lower:
        return "no"
    return "unknown"


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api(client, model_id: str, messages: list[dict]) -> tuple[str, int, int]:
    """
    Call OpenRouter and return (raw_text, input_tokens, output_tokens).
    Retries up to MAX_RETRIES times with exponential backoff.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=LLM_MAX_TOKENS,
                temperature=0,
            )
            raw   = response.choices[0].message.content or ""
            usage = response.usage
            return raw, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
            else:
                return f"ERROR: {e}", 0, 0


# ---------------------------------------------------------------------------
# Classify all rows for one (model, lang, rel_type) combo
# ---------------------------------------------------------------------------

def classify_rows(
    rows: list[dict],
    client,
    model_id: str,
    lang: str,
    rel_type: str,
    n_workers: int,
    log: logging.Logger,
    desc: str,
) -> tuple[list[str], list[str], int, int]:
    """
    Returns (parsed_labels, raw_outputs, total_input_tokens, total_output_tokens).
    Uses a thread pool for concurrent API calls.
    """
    parsed        = [None] * len(rows)
    raws          = [None] * len(rows)
    total_in_tok  = 0
    total_out_tok = 0

    def _process(idx_row):
        idx, row = idx_row
        messages = build_messages(lang, rel_type, row)
        raw, in_tok, out_tok = call_api(client, model_id, messages)
        return idx, raw, in_tok, out_tok

    n = len(rows)
    log_every = max(1, n // 4)
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process, (i, r)): i for i, r in enumerate(rows)}
        for done_count, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=n,
                 desc=f"    {desc}", leave=False, file=TqdmToLogger(log)), 1
        ):
            idx, raw, in_tok, out_tok = future.result()
            parsed[idx]   = parse_yes_no(raw)
            raws[idx]     = raw.strip()
            total_in_tok  += in_tok
            total_out_tok += out_tok

            if done_count % log_every == 0 or done_count == n:
                elapsed = time.time() - t_start
                rps = done_count / elapsed if elapsed > 0 else 0
                eta = (n - done_count) / rps if rps > 0 else 0
                log.info(
                    f"    progress: {done_count}/{n} ({100*done_count/n:.0f}%)  "
                    f"elapsed={_fmt_duration(elapsed)}  speed={rps:.1f} rows/s  "
                    f"ETA={_fmt_duration(eta)}  "
                    f"tokens so far: in={total_in_tok:,}, out={total_out_tok:,}"
                )

    return parsed, raws, total_in_tok, total_out_tok


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: list[str], gold: list[str]) -> dict:
    TP = FP = FN = TN = unknown = 0
    for pred, g in zip(labels, gold):
        if pred == "unknown":
            unknown += 1
        pos      = pred == "yes"
        gold_pos = g == "1"
        if pos and gold_pos:        TP += 1
        elif pos and not gold_pos:  FP += 1
        elif not pos and gold_pos:  FN += 1
        else:                       TN += 1

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total               if total              else 0
    precision = TP / (TP + FP)                  if (TP + FP)          else 0
    recall    = TP / (TP + FN)                  if (TP + FN)          else 0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "unknown": unknown}


def _estimate_cost(in_tok: int, out_tok: int, prices: dict) -> float:
    return (in_tok * prices["input"] + out_tok * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(run_stats: list[dict], summary_path: str, log: logging.Logger, total: float):
    lines = []
    lines.append("=" * 110)
    lines.append("API LLM CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total)}")
    lines.append("=" * 110)

    # Per-model totals
    model_agg: dict[str, dict] = {}
    for s in run_stats:
        agg = model_agg.setdefault(s["model"], {"time": 0.0, "rows": 0, "cost": 0.0,
                                                  "in_tok": 0, "out_tok": 0})
        agg["time"]    += s["time"]
        agg["rows"]    += s["n_rows"]
        agg["cost"]    += s["cost"]
        agg["in_tok"]  += s["in_tok"]
        agg["out_tok"] += s["out_tok"]

    lines.append("")
    lines.append("Per-model totals:")
    for model, agg in model_agg.items():
        rps = agg["rows"] / agg["time"] if agg["time"] > 0 else float("inf")
        lines.append(
            f"  {model:<14}  time={_fmt_duration(agg['time'])}  "
            f"rows/s={rps:.1f}  cost=${agg['cost']:.4f}  "
            f"tokens: in={agg['in_tok']:,}, out={agg['out_tok']:,}"
        )

    # Metrics table
    lines.append("")
    hdr = (
        f"  {'model':<14} {'lang':<4} {'rel_type':<10} "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'unk':>4}  {'rows/s':>7}  {'cost':>8}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for s in run_stats:
        m   = s["metrics"]
        rps = s["n_rows"] / s["time"] if s["time"] > 0 else float("inf")
        lines.append(
            f"  {s['model']:<14} {s['lang']:<4} {s['rel_type']:<10} "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['unknown']:>4}  {rps:>7.1f}  ${s['cost']:>7.4f}  {_fmt_duration(s['time']):>8}"
        )

    lines.append("=" * 110)
    text = "\n".join(lines)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="API LLM relation classification via OpenRouter"
    )
    parser.add_argument("--input",     default=INPUT_FILE,   help=f"Input CSV (default: {INPUT_FILE})")
    parser.add_argument("--output",    default=OUTPUT_FILE,  help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--log",       default=LOG_FILE,     help=f"Log file (default: {LOG_FILE})")
    parser.add_argument("--summary",   default=SUMMARY_FILE, help=f"Summary file (default: {SUMMARY_FILE})")
    parser.add_argument("--label-col", default=LABEL_COL,    help=f"Gold label column (default: {LABEL_COL})")
    parser.add_argument("--workers",   type=int, default=MAX_WORKERS,
                        help=f"Concurrent API requests (default: {MAX_WORKERS})")
    args = parser.parse_args()

    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {OPENROUTER_API_KEY_ENV} is not set. "
            f"Run: export {OPENROUTER_API_KEY_ENV}=sk-or-..."
        )

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path   = os.path.join(base, args.input)
    output_path  = os.path.join(base, args.output)
    log_path     = os.path.join(base, args.log)
    summary_path = os.path.join(base, args.summary)
    for p in (output_path, log_path, summary_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    combos = [(lang, rel) for lang in PROMPT_LANGS for rel in RELATION_TYPES]

    log.info("=" * 70)
    log.info("clean_dataset_with_api_llm.py  started")
    log.info(f"  input:        {input_path}")
    log.info(f"  output:       {output_path}")
    log.info(f"  log:          {log_path}")
    log.info(f"  summary:      {summary_path}")
    log.info(f"  label col:    {args.label_col}")
    log.info(f"  workers:      {args.workers}")
    log.info(f"  max new tok:  {LLM_MAX_TOKENS}")
    log.info(f"  max retries:  {MAX_RETRIES}")
    log.info(f"  models ({len(API_MODELS)}):")
    for m in API_MODELS:
        log.info(f"    [{m['tag']}]  {m['id']}  "
                 f"(in=${m['prices_per_1m']['input']}/1M, out=${m['prices_per_1m']['output']}/1M)")
    log.info(f"  prompt combos per model ({len(combos)}): {combos}")
    log.info("=" * 70)

    # --- Load CSV ---
    t0 = time.time()
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n_rows = len(rows)
    log.info(f"[load]  {n_rows} rows  ({_fmt_duration(time.time() - t0)})")

    available_cols = set(rows[0].keys())
    if args.label_col not in available_cols:
        raise ValueError(f"Label column '{args.label_col}' not found. Available: {sorted(available_cols)}")
    for rel_col in RELATION_COLS.values():
        if rel_col not in available_cols:
            raise ValueError(f"Relation column '{rel_col}' not found. Available: {sorted(available_cols)}")

    gold = [r[args.label_col] for r in rows]
    n_pos = gold.count("1")
    log.info(f"[load]  label distribution: positive={n_pos}, negative={n_rows-n_pos} "
             f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")

    # --- OpenRouter client ---
    from openai import OpenAI
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title":      OPENROUTER_APP_NAME,
        },
    )

    run_stats: list[dict] = []

    # --- Main loop ---
    for model_idx, model_cfg in enumerate(API_MODELS, 1):
        tag      = model_cfg["tag"]
        model_id = model_cfg["id"]
        prices   = model_cfg["prices_per_1m"]

        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(API_MODELS)}]  {tag}  ({model_id})")
        t_model = time.time()

        for combo_idx, (lang, rel_type) in enumerate(combos, 1):
            clean_col = f"llm_clean_{tag}_{lang}_{rel_type}"
            raw_col   = f"llm_raw_{tag}_{lang}_{rel_type}"
            desc      = f"{tag}/{lang}/{rel_type}"

            log.info(f"  [{combo_idx}/{len(combos)}]  model={tag}  lang={lang}  "
                     f"rel_type={rel_type}  hypothesis_col={RELATION_COLS[rel_type]}")
            log.info(f"    output cols: {clean_col}, {raw_col}")

            # Log one example prompt
            example_msgs = build_messages(lang, rel_type, rows[0])
            log.info(f"    example prompt (row 0):")
            log.info(f"    {'-'*40}")
            for msg in example_msgs:
                log.info(f"    [{msg['role']}] {msg['content'][:400]}")
            log.info(f"    {'-'*40}")

            t0 = time.time()
            parsed, raws, in_tok, out_tok = classify_rows(
                rows, client, model_id, lang, rel_type,
                n_workers=args.workers, log=log, desc=desc,
            )
            elapsed = time.time() - t0

            for r, label, raw in zip(rows, parsed, raws):
                r[clean_col] = label
                r[raw_col]   = raw

            metrics = compute_metrics(parsed, gold)
            cost    = _estimate_cost(in_tok, out_tok, prices)
            rps     = n_rows / elapsed if elapsed > 0 else float("inf")
            counts  = {v: parsed.count(v) for v in ("yes", "no", "unknown")}

            log.info(f"    done  ({_fmt_duration(elapsed)}, {rps:.1f} rows/s)")
            log.info(f"    tokens:      in={in_tok:,}, out={out_tok:,}, cost=${cost:.4f}")
            log.info(f"    predictions: yes={counts['yes']}, no={counts['no']}, unknown={counts['unknown']}")
            log.info(f"    confusion:   TP={metrics['TP']}  FP={metrics['FP']}  "
                     f"FN={metrics['FN']}  TN={metrics['TN']}")
            log.info(f"    metrics:     accuracy={metrics['accuracy']:.4f}  "
                     f"precision={metrics['precision']:.4f}  "
                     f"recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

            log.info(f"    examples (first 3 rows):")
            for r, label, raw in zip(rows[:3], parsed[:3], raws[:3]):
                log.info(f"      hypothesis: {r[RELATION_COLS[rel_type]]!r}")
                log.info(f"      raw={raw!r}  parsed={label}  gold={r[args.label_col]}")

            run_stats.append({
                "model": tag, "lang": lang, "rel_type": rel_type,
                "time": elapsed, "n_rows": n_rows,
                "in_tok": in_tok, "out_tok": out_tok, "cost": cost,
                "metrics": metrics,
            })

        log.info(f"  [{tag}] total time: {_fmt_duration(time.time() - t_model)}")

    # --- Save CSV ---
    t0 = time.time()
    fieldnames = list(dict.fromkeys(rows[0].keys()))
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
