"""
Prepare a dataset (gold or full) with cleaned text, basic_relation, template_relation, and llm_relation.

Usage:
    conda activate heb_relation_extraction
    python prepare_dataset.py
    python prepare_dataset.py --skip-llm
    python prepare_dataset.py --input data/crocodile_heb25_full_dataset_3124k.csv
    CUDA_VISIBLE_DEVICES=4,5,6 python prepare_dataset.py
"""

import re
import csv
import argparse
import os
import time
import logging
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Hyperparameters / macros
# ---------------------------------------------------------------------------

INPUT_FILE  = "data/crocodile_heb25_gold_500.csv"
OUTPUT_DIR  = "outputs"

LLM_MODEL_ID    = "google/gemma-3-27b-it"
LLM_BATCH_SIZE  = 4
LLM_MAX_TOKENS  = 80        # max new tokens to generate per row
LLM_MAX_INPUT_LEN = 512     # tokenizer truncation length

# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------

_SECTION_HEADERS = ["==קישורים חיצוניים==", "==ראו גם==", "==הערות שוליים=="]

def preprocess_text(text: str) -> str:
    # Cut at the earliest wiki-section header we want to strip
    cut_pos = len(text)
    for header in _SECTION_HEADERS:
        idx = text.find(header)
        if idx != -1 and idx < cut_pos:
            cut_pos = idx
    text = text[:cut_pos]

    # Remove empty lines
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Basic relation
# ---------------------------------------------------------------------------

def make_basic_relation(subject: str, predicate: str, obj: str) -> str:
    return f"{subject} {predicate} {obj}"


# ---------------------------------------------------------------------------
# Template relation
# ---------------------------------------------------------------------------

PREDICATE_TEMPLATES = {
    "אב":                                  "האב של {subject} הוא {object}",
    "אזרחות":                              "האזרחות של {subject} היא {object}",
    "אחים ואחיות":                         "יש יחסי אחים בין {subject} ל-{object}",
    "אל של":                               "{subject} הוא האל של {object}",
    "אמן מבצע":                            "האמן המבצע של {subject} הוא {object}",
    "ארץ מקור":                            "ארץ המקור של {subject} היא {object}",
    "בירה של":                             "{subject} היא הבירה של {object}",
    "בסיס פעולה מרכזי":                    "בסיס הפעולה המרכזי של {subject} הוא {object}",
    "בעלים":                               "הבעלים של {subject} הוא {object}",
    "גובל עם":                             "{subject} גובל ב-{object}",
    "גוף תקינה":                           "גוף התקינה של {subject} הוא {object}",
    "דת":                                  "הדת של {subject} היא {object}",
    "ההפך מ־":                             "ההפך מ-{subject} הוא {object}",
    "המקום שמרכז התחבורה משרת":            "מרכז התחבורה {subject} משרת את {object}",
    "הנקודה הגבוהה ביותר":                 "הנקודה הגבוהה ביותר של {subject} היא {object}",
    "הקודם":                               "הקודם ל-{subject} הוא {object}",
    "השפה של היצירה או של השם":            "השפה של {subject} היא {object}",
    "השתתף ב־":                            "הייתה השתתפות של {subject} ב-{object}",
    "זוכה":                                "הזוכה ב-{subject} הוא {object}",
    "זרם אמנותי":                          "הזרם האמנותי של {subject} הוא {object}",
    "חבר בקבוצת ספורט":                    "{subject} משתייך לקבוצת הספורט {object}",
    "חברת תקליטים":                        "חברת התקליטים של {subject} היא {object}",
    "חלוקה משנית":                         "{object} הוא חלוקה משנית של {subject}",
    "חלק מהסדרה":                          "{subject} הוא חלק מהסדרה {object}",
    "חלק מתוך":                            "{subject} הוא חלק מתוך {object}",
    "יבשת":                                "המיקום של {subject} הוא ביבשת {object}",
    "יחידה מנהלית":                        "{subject} היא יחידה מנהלית של {object}",
    "יחסים דיפלומטיים":                    "יש יחסים דיפלומטיים בין {subject} ל-{object}",
    "יצרן":                                "היצרן של {subject} הוא {object}",
    "ליגה":                                "הליגה של {subject} היא {object}",
    "ליגה נמוכה יותר":                     "הליגה הנמוכה יותר של {subject} היא {object}",
    "מארגן":                               "המארגן של {subject} הוא {object}",
    "מדינה":                               "המיקום של {subject} הוא במדינת {object}",
    "מדינה בתחום של ספורט":                "הייצוג של {object} בספורט נעשה על ידי {subject}",
    "מדינות אגן הניקוז":                   "הזרימה של {subject} עוברת דרך {object}",
    "מועמד שנבחר":                         "המועמד שנבחר ב-{subject} הוא {object}",
    "מוענק על ידי":                        "{subject} מוענק על ידי {object}",
    "מופע של":                             "{subject} הוא סוג של {object}",
    "מוצג":                                "{subject} מציג את {object}",
    "מוצר":                                "{object} הוא מוצר של {subject}",
    "מוקד פעילות":                         "מוקד הפעילות של {subject} הוא {object}",
    "מותג":                                "{subject} שייך למותג {object}",
    "מחבר":                                "המחבר של {subject} הוא {object}",
    "מחבר המילים":                         "מחבר המילים של {subject} הוא {object}",
    "מטבח":                                "{subject} הוא מאכל ממטבח {object}",
    "מייסד":                               "המייסד של {subject} הוא {object}",
    "מיקום":                               "המיקום של {subject} הוא ב-{object}",
    "מיקום מטה הארגון":                    "מטה הארגון של {subject} ממוקם ב-{object}",
    "מכיל את החלק":                        "{subject} מכיל את {object}",
    "מכיל חלקים מסוג":                     "{subject} מכיל חלקים מסוג {object}",
    "ממוקם בגוף השמיימי":                  "{subject} ממוקם על {object}",
    "מעסיק":                               "המעסיק של {subject} הוא {object}",
    "מערכת תחבורה":                        "{subject} היא חלק ממערכת התחבורה {object}",
    "מפלגה":                               "{subject} משתייך למפלגת {object}",
    "מפעיל":                               "המפעיל של {subject} הוא {object}",
    "מקום לידה":                           "מקום הלידה של {subject} הוא {object}",
    "מקום לימודים":                        "מוסד הלימודים של {subject} הוא {object}",
    "מקום מוצא":                           "מקום המוצא של {subject} הוא {object}",
    "מקום פטירה":                          "מקום הפטירה של {subject} הוא {object}",
    "משמש לטיפול ב־":                      "{subject} משמש לטיפול ב-{object}",
    "נהרות יוצאים מהאגם":                  "{object} יוצא מתוך {subject}",
    "נושא היצירה":                         "הנושא של {subject} הוא {object}",
    "נושא המשרה":                          "נושא המשרה של {subject} הוא {object}",
    "נמצא בשימוש של":                      "{subject} נמצא בשימוש של {object}",
    "נמצא על שפת גוף מים":                 "{subject} נמצא על שפת {object}",
    "נקרא על שם":                          "{subject} נקרא על שם {object}",
    "נשפך ל":                              "{subject} נשפך אל {object}",
    "סוג יצירה":                           "סוג היצירה של {subject} הוא {object}",
    "סוגה":                                "הסוגה של {subject} היא {object}",
    "סמל מייצג":                           "הסמל המייצג של {subject} הוא {object}",
    "עונת ספורט של":                       "{subject} היא עונת ספורט של {object}",
    "עיסוק":                               "העיסוק של {subject} הוא {object}",
    "עיר בירה":                            "עיר הבירה של {subject} היא {object}",
    "ענף ספורט":                           "ענף הספורט של {subject} הוא {object}",
    "ערוץ שידור מקורי":                    "ערוץ השידור המקורי של {subject} הוא {object}",
    "פורסם ב־":                            "הפרסום של {subject} היה ב-{object}",
    "צאצא":                                "{object} הוא צאצא של {subject}",
    "צבע":                                 "הצבע של {subject} הוא {object}",
    "קבוצות משתתפות":                      "ישנה השתתפות של {object} ב-{subject}",
    "קבוצת כוכבים":                        "{subject} ממוקם בקבוצת הכוכבים {object}",
    "קו רכבת":                             "{subject} הוא חלק מקו הרכבת {object}",
    "קטגוריית כוכבי לכת מינוריים":         "{subject} משויך לקטגוריית {object}",
    "קיבל השראה מ־":                       "ההשראה עבור {subject} התקבלה מ-{object}",
    "רמה טקסונומית":                       "הרמה הטקסונומית של {subject} היא {object}",
    "שחקנים":                              "תפקיד המשחק ב-{subject} מבוצע על ידי {object}",
    "שטח שיפוט":                           "שטח השיפוט של {subject} הוא {object}",
    "שיטת כתב":                            "שיטת הכתב של {subject} היא {object}",
    "שימוש":                               "השימוש של {subject} הוא עבור {object}",
    "שפה מדוברת או נכתבת":                 "השפה בשימוש על ידי {subject} היא {object}",
    "שפה רשמית":                           "השפה הרשמית של {subject} היא {object}",
    "שפה שבשימוש":                         "השפה שבשימוש ב-{subject} היא {object}",
    "שפת אם":                              "שפת האם של {subject} היא {object}",
    "שפת הכתיבה":                          "שפת הכתיבה של {subject} היא {object}",
    "תעשייה":                              "התעשייה בה פועל {subject} היא {object}",
    "תפקיד":                               "התפקיד של {subject} הוא {object}",
    "תקופה":                               "התקופה אליה משויך {subject} היא {object}",
    "תת-קבוצה של":                         "{subject} הוא תת-קבוצה של {object}"
}

def make_template_relation(subject: str, predicate: str, obj: str) -> str:
    template = PREDICATE_TEMPLATES.get(predicate)
    if template is None:
        return f"{subject} {predicate} {obj}"
    return template.format(subject=subject, object=obj)


# ---------------------------------------------------------------------------
# LLM relation (Gemma 4 31B)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "אתה עוזר בלשני מומחה בעברית. "
    "תפקידך לנסח משפט אחד קצר ובעברית תקינה שמבטא קשר בין שני ישויות."
)

def _build_user_message(subject: str, predicate: str, obj: str) -> str:
    return (
        f"נתון שלשה של ידע:\n"
        f"- נושא: {subject}\n"
        f"- יחס: {predicate}\n"
        f"- אובייקט: {obj}\n\n"
        f"כתוב משפט אחד בעברית שמבטא את הקשר הזה בצורה טבעית וקצרה. "
        f"כתוב את המשפט בלבד, ללא הסברים נוספים."
    )

def load_llm(model_id: str, log):
    import torch
    import transformers

    log.info(f"Loading LLM: {model_id}")
    t0 = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    log.info(f"  tokenizer loaded ({_fmt_duration(time.time()-t0)})")

    t1 = time.time()
    from tqdm.contrib.logging import logging_redirect_tqdm
    with logging_redirect_tqdm(loggers=[log]):
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    log.info(f"  model loaded ({_fmt_duration(time.time()-t1)})")
    model.eval()
    return model, tokenizer


LOG_EVERY_N_BATCHES = 5   # log progress every N batches

def generate_llm_relations(rows, model, tokenizer, batch_size: int, log) -> list[str]:
    import torch

    results = []
    n_batches = (len(rows) + batch_size - 1) // batch_size
    t_start = time.time()

    log.info(f"    total rows={len(rows)}, batch_size={batch_size}, n_batches={n_batches}")

    for batch_idx, i in enumerate(
        tqdm(range(0, len(rows), batch_size), desc="LLM generation", file=TqdmToLogger(log)), 1
    ):
        batch = rows[i : i + batch_size]

        prompts = []
        for r in batch:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(r["subject"], r["predicate"], r["object"])},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encodings = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=LLM_MAX_INPUT_LEN,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **encodings,
                max_new_tokens=LLM_MAX_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = encodings["input_ids"].shape[1]
        batch_results = []
        for out in output_ids:
            generated = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            first_sentence = re.split(r"(?<=[.!?])\s", generated)[0]
            batch_results.append(first_sentence)
        results.extend(batch_results)

        if batch_idx % LOG_EVERY_N_BATCHES == 0 or batch_idx == n_batches:
            elapsed  = time.time() - t_start
            rows_done = min(i + batch_size, len(rows))
            rps      = rows_done / elapsed if elapsed > 0 else 0
            eta      = (len(rows) - rows_done) / rps if rps > 0 else 0
            log.info(
                f"    batch {batch_idx}/{n_batches}  "
                f"rows {rows_done}/{len(rows)} ({100*rows_done/len(rows):.0f}%)  "
                f"elapsed={_fmt_duration(elapsed)}  "
                f"speed={rps:.1f} rows/s  "
                f"ETA={_fmt_duration(eta)}"
            )
            # Sample output from the last item in this batch
            log.info(f"    sample output: {batch_results[-1]!r}")

    return results


# ---------------------------------------------------------------------------
# Logging setup
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
    logger = logging.getLogger("prepare_dataset")
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare a dataset with enriched relation columns")
    parser.add_argument("--input",      default=INPUT_FILE,  help=f"Input CSV (default: {INPUT_FILE})")
    parser.add_argument("--output",     default=None,        help="Output CSV (default: auto-named from model)")
    parser.add_argument("--log",        default=None,        help="Log file (default: auto-named from model)")
    parser.add_argument("--skip-llm",   action="store_true", help="Skip LLM generation (no GPU needed)")
    parser.add_argument("--llm-model",  default=LLM_MODEL_ID, help="HuggingFace model ID for LLM column")
    parser.add_argument("--batch-size", type=int, default=LLM_BATCH_SIZE, help="Batch size for LLM inference")
    args = parser.parse_args()

    model_tag = args.llm_model.split("/")[-1].replace("-", "_").lower()
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path  = os.path.join(base, args.input)
    output_path = os.path.join(base, args.output or f"{OUTPUT_DIR}/prepared_gold_dataset_{model_tag}.csv")
    log_path    = os.path.join(base, args.log    or f"{OUTPUT_DIR}/prepared_gold_dataset_{model_tag}.log")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    log.info("=" * 60)
    log.info("prepare_dataset.py  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output:     {output_path}")
    log.info(f"  log:        {log_path}")
    log.info(f"  llm model:  {'SKIPPED' if args.skip_llm else args.llm_model}")
    log.info(f"  batch size: {args.batch_size}")
    log.info("=" * 60)

    step_times: dict[str, float] = {}

    # --- Load ---
    t0 = time.time()
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    step_times["load"] = time.time() - t0
    log.info(f"[load]              {len(rows)} rows loaded  ({_fmt_duration(step_times['load'])})")

    # --- Step 1: preprocess text ---
    t0 = time.time()
    for r in rows:
        r["text"] = preprocess_text(r["text"])
    step_times["preprocess"] = time.time() - t0
    log.info(f"[step 1 preprocess] done  ({_fmt_duration(step_times['preprocess'])})")

    # --- Step 2: basic_relation ---
    t0 = time.time()
    for r in rows:
        r["basic_relation"] = make_basic_relation(r["subject"], r["predicate"], r["object"])
    step_times["basic_relation"] = time.time() - t0
    log.info(f"[step 2 basic_rel]  done  ({_fmt_duration(step_times['basic_relation'])})")

    # --- Step 3: template_relation ---
    t0 = time.time()
    unseen = set()
    for r in rows:
        if r["predicate"] not in PREDICATE_TEMPLATES:
            unseen.add(r["predicate"])
        r["template_relation"] = make_template_relation(r["subject"], r["predicate"], r["object"])
    step_times["template_relation"] = time.time() - t0
    if unseen:
        log.warning(f"[step 3 tmpl_rel]   {len(unseen)} predicates had no template, used fallback: {unseen}")
    log.info(f"[step 3 tmpl_rel]   done  ({_fmt_duration(step_times['template_relation'])})")

    # --- Step 4: llm_relation ---
    if args.skip_llm:
        log.info("[step 4 llm_rel]    SKIPPED (--skip-llm)")
        for r in rows:
            r["llm_relation"] = ""
        step_times["llm"] = 0.0
    else:
        t0 = time.time()
        model, tokenizer = load_llm(args.llm_model, log)
        log.info(f"[step 4 llm_rel]    model loaded, generating for {len(rows)} rows ...")
        llm_outputs = generate_llm_relations(rows, model, tokenizer, batch_size=args.batch_size, log=log)
        for r, llm_out in zip(rows, llm_outputs):
            r["llm_relation"] = llm_out
        step_times["llm"] = time.time() - t0
        rows_per_sec = len(rows) / step_times["llm"] if step_times["llm"] > 0 else float("inf")
        log.info(
            f"[step 4 llm_rel]    done  ({_fmt_duration(step_times['llm'])}, "
            f"{rows_per_sec:.1f} rows/s)"
        )

    # --- Step 5: save ---
    t0 = time.time()
    fieldnames = [
        "docid", "title", "text",
        "subject", "predicate", "object", "relation_present",
        "basic_relation", "template_relation", "llm_relation",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    step_times["save"] = time.time() - t0
    log.info(f"[save]              {len(rows)} rows written  ({_fmt_duration(step_times['save'])})")

    # --- Final summary ---
    total = time.time() - wall_start
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(f"  rows processed     : {len(rows)}")
    for step, t in step_times.items():
        log.info(f"  {step:<20} : {_fmt_duration(t)}")
    log.info(f"  total wall time    : {_fmt_duration(total)}")
    log.info(f"  overall throughput : {len(rows) / total:.1f} rows/s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
