"""
Classify relation presence in text using open-source LLMs.
For each (model × prompt_lang × relation_type) combination adds:
  llm_clean_{tag}_{lang}_{rel}  — "yes" / "no" / "unknown"
  llm_raw_{tag}_{lang}_{rel}    — raw first tokens from the model (for inspection)

Evaluates each combination against the gold label column (relation_present)
and writes accuracy / precision / recall / F1 to the log and to a summary file.

Usage:
    CUDA_VISIBLE_DEVICES=0 python clean_dataset_with_opensource_llm.py
    CUDA_VISIBLE_DEVICES=4,5,6 python clean_dataset_with_opensource_llm.py --input outputs/prepared_gold_dataset.csv
    CUDA_VISIBLE_DEVICES=4,5,6 python clean_dataset_with_opensource_llm.py 2>&1 | tee llm_run.txt
"""

import os
import re
import csv
import time
import logging
import argparse

import torch
import transformers
from tqdm import tqdm

try:
    from vllm import LLM as vLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE   = "outputs/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_FILE  = "outputs/opensource_llm_classified.csv"
LOG_FILE     = "outputs/opensource_llm_classify.log"
SUMMARY_FILE = "outputs/opensource_llm_summary.txt"

LABEL_COL = "relation_present"      # gold label: "1" = present, "0" = absent

# Columns that carry the relation string for each relation type
RELATION_COLS = {
    "triplet":  "basic_relation",    # "subject predicate object"
    "template": "template_relation", # natural-language template sentence
}

# Models: id, short tag, model type ("instruct" uses chat template, "base" uses completion),
#         max_new_tokens overrides the default (reasoning models need room for the thinking block)
LLM_MODELS = [
    {"id": "dicta-il/DictaLM-3.0-24B-Base",  "tag": "dictalm3", "type": "base"},
    {"id": "CohereLabs/aya-expanse-32b",           "tag": "aya32b",   "type": "instruct"},
    {"id": "google/gemma-3-27b-it",               "tag": "gemma3",   "type": "instruct"},
]

PROMPT_LANGS   = ["he", "en"]          # Hebrew and English prompts
RELATION_TYPES = ["triplet", "template"]

LLM_BATCH_SIZE    = 16     # vLLM handles batching internally; used as HF fallback batch size
LLM_MAX_NEW_TOKENS = 32    # we only need yes/no, a few tokens suffice
LLM_MAX_INPUT_LEN  = 2048  # tokenizer truncation length

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
    logger = logging.getLogger("clean_dataset_with_opensource_llm")
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
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt_instruct(lang: str, rel_type: str, row: dict, tokenizer) -> str:
    """Build a chat-template prompt for instruct models (Gemma etc.)."""
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
                f"  נושא (subject):  {subject}\n"
                f"  יחס (predicate): {pred}\n"
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
                f"Look in the text for whether the connection between \"{subject}\" and \"{obj}\" via the relation \"{pred}\" is mentioned or can be inferred.\n"
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

    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_prompt_base(lang: str, rel_type: str, row: dict) -> str:
    """Build a completion-style prompt for base models (DictaLM etc.)."""
    text     = row["text"]
    subject  = row["subject"]
    pred     = row["predicate"]
    obj      = row["object"]
    relation = row[RELATION_COLS[rel_type]]

    if lang == "he":
        if rel_type == "triplet":
            return (
                f"קרא את הטקסט הבא וענה על השאלה.\n\n"
                f"הסבר: יחס הוא קשר סמנטי בין שתי ישויות המתואר על ידי נושא, סוג יחס ואובייקט. "
                f"עליך לקבוע האם יחס זה מופיע או נובע מהטקסט.\n\n"
                f"טקסט:\n{text}\n\n"
                f"יחס לבדיקה:\n"
                f"  נושא (subject):   {subject}\n"
                f"  יחס (predicate):  {pred}\n"
                f"  אובייקט (object): {obj}\n\n"
                f"חפש בטקסט האם הקשר בין \"{subject}\" ל-\"{obj}\" דרך היחס \"{pred}\" מוזכר או נובע ממנו.\n"
                f"תשובה (כן/לא):"
            )
        else:
            return (
                f"קרא את הטקסט הבא וענה על השאלה.\n\n"
                f"הסבר: עליך לקבוע האם המשפט הנתון, המבטא קשר בין שתי ישויות, נובע מהטקסט או מופיע בו.\n\n"
                f"טקסט:\n{text}\n\n"
                f"משפט לבדיקה: {relation}\n\n"
                f"חפש בטקסט האם המשפט הנ\"ל נובע ממנו או מופיע בו.\n"
                f"תשובה (כן/לא):"
            )
    else:  # en
        if rel_type == "triplet":
            return (
                f"Read the following text and answer the question.\n\n"
                f"Note: A relation is a semantic connection between two entities described by a subject, a predicate (relation type), and an object. "
                f"Determine whether this relation is expressed or can be inferred from the text.\n\n"
                f"Text:\n{text}\n\n"
                f"Relation to check:\n"
                f"  Subject:   {subject}\n"
                f"  Predicate: {pred}\n"
                f"  Object:    {obj}\n\n"
                f"Look in the text for whether the connection between \"{subject}\" and \"{obj}\" via the relation \"{pred}\" is mentioned or can be inferred.\n"
                f"Answer (yes/no):"
            )
        else:
            return (
                f"Read the following text and answer the question.\n\n"
                f"Note: Determine whether the statement below, which expresses a relation between two entities, is entailed by or expressed in the text.\n\n"
                f"Text:\n{text}\n\n"
                f"Statement to check: {relation}\n\n"
                f"Look in the text for whether the above statement is expressed or can be inferred from it.\n"
                f"Answer (yes/no):"
            )


def build_prompt(model_type: str, lang: str, rel_type: str, row: dict, tokenizer) -> str:
    if model_type == "instruct":
        return _build_prompt_instruct(lang, rel_type, row, tokenizer)
    return _build_prompt_base(lang, rel_type, row)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Tokens that map to yes / no
_YES_TOKENS = {"yes", "כן"}
_NO_TOKENS  = {"no",  "לא"}

def parse_yes_no(raw: str) -> str:
    """
    Extract yes/no from model generation. Returns '1', '0', or 'unknown'.
    Strips <think>...</think> reasoning blocks first (reasoning models).
    Prioritises the first non-empty token, then falls back to substring search.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    tokens = re.split(r"[\s.,!?:;()\[\]\"']+", cleaned)
    tokens = [t.lower() for t in tokens if t]

    if tokens:
        if tokens[0] in _YES_TOKENS:
            return "1"
        if tokens[0] in _NO_TOKENS:
            return "0"

    # Fallback: substring search (whole word)
    lower = cleaned.lower()
    if re.search(r"\byes\b", lower) or "כן" in lower:
        return "1"
    if re.search(r"\bno\b",  lower) or "לא" in lower:
        return "0"

    return "unknown"


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_llm(model_id: str, log: logging.Logger, n_gpus: int = 1):
    log.info(f"    path: {model_id}")

    if VLLM_AVAILABLE:
        log.info(f"    backend: vLLM  (tensor_parallel_size={n_gpus})")
        llm = vLLM(
            model=model_id,
            tensor_parallel_size=n_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=LLM_MAX_INPUT_LEN + LLM_MAX_NEW_TOKENS,
            gpu_memory_utilization=0.90,
        )
        tokenizer = llm.get_tokenizer()
        log.info(f"    vLLM model ready")
        return llm, tokenizer

    # HuggingFace fallback with flash attention and larger batch
    log.info(f"    backend: HuggingFace + flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left" # Better to lose the system prompt than the strict output constraint, but 2048 prevents both.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        log.info("    flash_attention_2 enabled")
    except Exception as e:
        log.warning(f"    flash_attention_2 failed ({e}), falling back to default attention")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"    ready  ({n_params:.1f}B params, device_map=auto, bfloat16)")
    return model, tokenizer


def unload_llm(model, log: logging.Logger):
    if VLLM_AVAILABLE and isinstance(model, vLLM):
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

_BASE_STOP_STRINGS = ["\n", "כן", "לא", "yes", "no"]


def _stop_token_ids(tokenizer, stop_strings: list[str]) -> list[int]:
    """Return single-token IDs for each stop string (with and without leading space)."""
    ids: set[int] = set()
    for s in stop_strings:
        for variant in [s, " " + s]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return list(ids)


def classify_rows(
    rows: list[dict],
    model,
    tokenizer,
    model_type: str,
    lang: str,
    rel_type: str,
    batch_size: int,
    log: logging.Logger,
    desc: str,
    max_new_tokens: int = LLM_MAX_NEW_TOKENS,
) -> tuple[list[str], list[str]]:
    """
    Returns (parsed_labels, raw_outputs) — one entry per row.
    Uses vLLM if available, otherwise HuggingFace generate.
    """
    prompts = [build_prompt(model_type, lang, rel_type, r, tokenizer) for r in rows]
    n = len(prompts)

    # --- vLLM path ---
    if VLLM_AVAILABLE and isinstance(model, vLLM):
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            stop=_BASE_STOP_STRINGS if model_type == "base" else [],
        )
        log.info(f"    vLLM generating {n} prompts (continuous batching)...")
        outputs = model.generate(prompts, sampling_params)
        parsed, raws = [], []
        for out in outputs:
            raw = out.outputs[0].text.strip()
            raws.append(raw)
            parsed.append(parse_yes_no(raw))
        return parsed, raws

    # --- HuggingFace path ---
    parsed, raws = [], []
    log_every = max(1, (n // batch_size) // 4)

    hf_stop_kwargs = {}
    if model_type == "base":
        stop_ids = _stop_token_ids(tokenizer, _BASE_STOP_STRINGS)
        if stop_ids:
            hf_stop_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + stop_ids

    for batch_idx, start in enumerate(
        tqdm(range(0, n, batch_size), desc=f"    {desc}", leave=False,
             file=TqdmToLogger(log)), 1
    ):
        batch_prompts = prompts[start : start + batch_size]
        encodings = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=LLM_MAX_INPUT_LEN,
        )
        input_len = encodings["input_ids"].shape[1]
        for key in encodings:
            encodings[key] = encodings[key].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                **hf_stop_kwargs,
            )

        for out in output_ids:
            raw = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            raws.append(raw)
            parsed.append(parse_yes_no(raw))

        if batch_idx % log_every == 0 or (start + batch_size) >= n:
            done = min(start + batch_size, n)
            log.info(f"    progress: {done}/{n} ({100*done/n:.0f}%)")

    return parsed, raws


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels: list[str], gold: list[str]) -> dict:
    """
    labels: "1"/"0"/"unknown"  (unknown treated as "0")
    gold:   "1" / "0"
    """
    TP = FP = FN = TN = 0
    unknown = 0
    for pred, g in zip(labels, gold):
        if pred == "unknown":
            unknown += 1
        pos = pred == "1"
        gold_pos = g == "1"
        if pos and gold_pos:       TP += 1
        elif pos and not gold_pos: FP += 1
        elif not pos and gold_pos: FN += 1
        else:                      TN += 1

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total      if total               else 0
    precision = TP / (TP + FP)         if (TP + FP)           else 0
    recall    = TP / (TP + FN)         if (TP + FN)           else 0
    f1        = (2*precision*recall /
                 (precision + recall)) if (precision+recall)  else 0

    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "unknown": unknown}


# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------

def write_summary(run_stats: list[dict], summary_path: str, log: logging.Logger, total: float):
    lines = []
    lines.append("=" * 100)
    lines.append("LLM CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total)}")
    lines.append("=" * 100)

    # Per-model timing
    model_times: dict[str, float] = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0.0)
        model_times[s["model"]] += s["time"]

    lines.append("")
    lines.append("Per-model total inference time:")
    for m, t in model_times.items():
        rows_total = sum(s["n_rows"] for s in run_stats if s["model"] == m)
        rps = rows_total / t if t > 0 else float("inf")
        lines.append(f"  {m:<12}  {_fmt_duration(t)}  ({rps:.1f} rows/s across all combos)")

    # Metrics table
    lines.append("")
    hdr = (
        f"  {'model':<10} {'lang':<4} {'rel_type':<10} "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'unk':>4}  {'rows/s':>7}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for s in run_stats:
        m = s["metrics"]
        rps = s["n_rows"] / s["time"] if s["time"] > 0 else float("inf")
        lines.append(
            f"  {s['model']:<10} {s['lang']:<4} {s['rel_type']:<10} "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['unknown']:>4}  {rps:>7.1f}  {_fmt_duration(s['time']):>8}"
        )

    lines.append("=" * 100)
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
        description="LLM-based relation classification on a CSV with multiple models/prompts"
    )
    parser.add_argument("--input",      default=INPUT_FILE,    help=f"Input CSV (default: {INPUT_FILE})")
    parser.add_argument("--output",     default=OUTPUT_FILE,   help=f"Output CSV (default: {OUTPUT_FILE})")
    parser.add_argument("--log",        default=LOG_FILE,      help=f"Log file (default: {LOG_FILE})")
    parser.add_argument("--summary",    default=SUMMARY_FILE,  help=f"Summary file (default: {SUMMARY_FILE})")
    parser.add_argument("--label-col",  default=LABEL_COL,     help=f"Gold label column (default: {LABEL_COL})")
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE,
                        help=f"Generation batch size (default: {LLM_BATCH_SIZE})")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path   = os.path.join(base, args.input)
    output_path  = os.path.join(base, args.output)
    log_path     = os.path.join(base, args.log)
    summary_path = os.path.join(base, args.summary)
    for p in (output_path, log_path, summary_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    log.info(f"  backend: {'vLLM' if VLLM_AVAILABLE else 'HuggingFace'}  "
             f"({n_gpus} GPU{'s' if n_gpus != 1 else ''} visible)")

    combos = [(lang, rel) for lang in PROMPT_LANGS for rel in RELATION_TYPES]

    log.info("=" * 70)
    log.info("clean_dataset_with_opensource_llm.py  started")
    log.info(f"  input:       {input_path}")
    log.info(f"  output:      {output_path}")
    log.info(f"  log:         {log_path}")
    log.info(f"  summary:     {summary_path}")
    log.info(f"  label col:   {args.label_col}")
    log.info(f"  batch size:  {args.batch_size}")
    log.info(f"  max new tok: {LLM_MAX_NEW_TOKENS}")
    log.info(f"  models ({len(LLM_MODELS)}):")
    for m in LLM_MODELS:
        log.info(f"    [{m['tag']}]  {m['id']}  (type={m['type']})")
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
    log.info(f"[load]  label distribution: positive={n_pos}, negative={n_rows - n_pos} "
             f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")

    run_stats: list[dict] = []

    # --- Main loop: load each model once, run all combos inside ---
    for model_idx, model_cfg in enumerate(LLM_MODELS, 1):
        tag        = model_cfg["tag"]
        model_id   = model_cfg["id"]
        model_type = model_cfg["type"]

        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(LLM_MODELS)}]  {tag}  ({model_type})")
        t_model = time.time()
        model, tokenizer = load_llm(model_id, log, n_gpus=n_gpus)
        log.info(f"    loaded in {_fmt_duration(time.time() - t_model)}")

        max_new_tokens = model_cfg.get("max_new_tokens", LLM_MAX_NEW_TOKENS)

        for combo_idx, (lang, rel_type) in enumerate(combos, 1):
            clean_col = f"llm_clean_{tag}_{lang}_{rel_type}"
            raw_col   = f"llm_raw_{tag}_{lang}_{rel_type}"
            desc      = f"{tag}/{lang}/{rel_type}"

            log.info("*" * 70)
            log.info(f"  STARTING COMBINATION {combo_idx}/{len(combos)}")
            log.info(f"    model:     {tag}  ({model_id})")
            log.info(f"    lang:      {lang}")
            log.info(f"    rel_type:  {rel_type}")
            log.info("*" * 70)
            log.info(f"    output cols: {clean_col}, {raw_col}")
            log.info(f"    relation col: {RELATION_COLS[rel_type]}")

            # Log one example prompt for transparency
            example_prompt = build_prompt(model_type, lang, rel_type, rows[0], tokenizer)
            log.info(f"    example prompt (row 0):\n{'-'*40}\n{example_prompt}\n{'-'*40}")
            t0 = time.time()
            parsed, raws = classify_rows(
                rows, model, tokenizer, model_type, lang, rel_type,
                batch_size=args.batch_size, log=log, desc=desc,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.time() - t0

            for r, label, raw in zip(rows, parsed, raws):
                r[clean_col] = label
                r[raw_col]   = raw

            metrics = compute_metrics(parsed, gold)
            rps     = n_rows / elapsed if elapsed > 0 else float("inf")

            # Count each label
            counts = {v: parsed.count(v) for v in ("1", "0", "unknown")}
            m = metrics
            log.info(f"  {'=' * 66}")
            log.info(f"  RESULT  {tag} | {lang} | {rel_type}")
            log.info(f"  {'─' * 66}")
            log.info(
                f"  {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}   "
                f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}   "
                f"{'unk':>4}  {'rows/s':>7}  {'time':>8}"
            )
            log.info(
                f"  {m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}   "
                f"  {m['TP']:>4}   {m['FP']:>4}   {m['FN']:>4}   {m['TN']:>4}   "
                f"{m['unknown']:>4}  {rps:>7.1f}  {_fmt_duration(elapsed):>8}"
            )
            log.info(f"  {'=' * 66}")

            log.info(f"    examples (first 3 rows):")
            for r, label, raw in zip(rows[:3], parsed[:3], raws[:3]):
                log.info(f"      hypothesis: {r[RELATION_COLS[rel_type]]!r}")
                log.info(f"      raw={raw!r}  parsed={label}  gold={r[args.label_col]}")

            run_stats.append({
                "model": tag, "lang": lang, "rel_type": rel_type,
                "time": elapsed, "n_rows": n_rows, "metrics": metrics,
            })

        model_total = time.time() - t_model
        unload_llm(model, log)
        log.info(f"  [{tag}] total time: {_fmt_duration(model_total)}")

        # --- Save CSV after each model ---
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
