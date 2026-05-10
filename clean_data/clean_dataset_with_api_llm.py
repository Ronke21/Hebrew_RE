"""
API LLM relation classifier — v2.

Dimensions compared:
  * model  : gpt-5.1, gemini-3-pro, claude-opus-4-6
  * lang   : he / en
  * shot   : 0 / 5-shot in-context examples
  * rel    : triplet only
  + weighted majority vote  (F1-weighted across combos)
  + predicate-stratified metrics

Output  : outputs/API_LLM_clean/
Archive : existing outputs/ files moved to outputs/ARCHIVE/ (skipped if already done)

Requires:
  pip install openai
  export OPENROUTER_API_KEY=sk-or-...

Usage
-----
  python clean_data/clean_dataset_with_api_llm.py
  python clean_data/clean_dataset_with_api_llm.py --shots 0 2 --cot-modes nocot
  python clean_data/clean_dataset_with_api_llm.py --debug 10
  python clean_data/clean_dataset_with_api_llm.py --workers 16 --no-archive
"""

import os
import re
import csv
import json
import shutil
import time
import logging
import argparse
import concurrent.futures
from collections import defaultdict

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / macros
# ---------------------------------------------------------------------------

INPUT_FILE    = "outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_SUBDIR = "outputs/API_LLM_clean"
OUTPUT_FILE   = f"{OUTPUT_SUBDIR}/classified.csv"
LOG_FILE      = f"{OUTPUT_SUBDIR}/classify.log"
SUMMARY_FILE  = f"{OUTPUT_SUBDIR}/summary.txt"
ERROR_FILE    = f"{OUTPUT_SUBDIR}/error_analysis.txt"
PRED_FILE     = f"{OUTPUT_SUBDIR}/predicate_analysis.txt"
PROMPTS_FILE  = f"{OUTPUT_SUBDIR}/prompts.jsonl"

LABEL_COL     = "relation_present"
PREDICATE_COL = "predicate"

OPENROUTER_BASE_URL    = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_SITE_URL    = "https://github.com/nlp/hebrew_RE"
OPENROUTER_APP_NAME    = "Hebrew-RE"

RELATION_COLS = {
    "triplet":  "basic_relation",
    "template": "template_relation",
}

# Prices per 1M tokens (USD) — verify current rates on openrouter.ai
API_MODELS = [
    {
        "id":  "openai/gpt-5.1",
        "tag": "gpt51",
        "prices_per_1m": {"input": 0.518, "output": 10.0},
    },
    {
        "id":  "google/gemini-3-pro",
        "tag": "gemini3_pro",
        "prices_per_1m": {"input": 1.25, "output": 10.0},
    },
    {
        "id":  "anthropic/claude-opus-4-6",
        "tag": "claude_opus46",
        "prices_per_1m": {"input": 15.0, "output": 75.0},
    },
]

PROMPT_LANGS   = ["he", "en"]
RELATION_TYPES = ["triplet"]
SHOT_CONFIGS   = [0, 5]
COT_CONFIGS    = [False]

LLM_MAX_TOKENS = 16

MAX_WORKERS      = 8
MAX_RETRIES      = 5
RETRY_BASE_DELAY = 2.0
RETRY_429_DELAY  = 20.0

MIN_PRED_EXAMPLES = 5
SHOT_2_INDICES    = [0, 3]   # 1 positive + 1 world-knowledge-leakage negative

# ---------------------------------------------------------------------------
# Few-shot example pool  (3 positive + 2 negative)
# ---------------------------------------------------------------------------

_POOL = [
    # 0 — POSITIVE: birthplace
    {
        ("he", "triplet"): {
            "text":      "ויליאם שייקספיר (1564–1616) נולד בסטרטפורד-אפון-אייבון, אנגליה. הוא נחשב לגדול המחזאים של כל הזמנים.",
            "subject":   "שייקספיר",
            "predicate": "מקום לידה",
            "object":    "סטרטפורד-אפון-אייבון",
            "answer":    "כן",
            "reasoning": "הטקסט מציין במפורש שוויליאם שייקספיר נולד בסטרטפורד-אפון-אייבון, ולכן יחס מקום הלידה מופיע.",
        },
        ("en", "triplet"): {
            "text":      "William Shakespeare (1564–1616) was born in Stratford-upon-Avon, England. He is considered the greatest playwright of all time.",
            "subject":   "Shakespeare",
            "predicate": "place of birth",
            "object":    "Stratford-upon-Avon",
            "answer":    "yes",
            "reasoning": "The text explicitly states Shakespeare was born in Stratford-upon-Avon, so the relation is expressed.",
        },
        ("he", "template"): {
            "text":      "ויליאם שייקספיר (1564–1616) נולד בסטרטפורד-אפון-אייבון, אנגליה. הוא נחשב לגדול המחזאים של כל הזמנים.",
            "statement": "מקום הלידה של שייקספיר הוא סטרטפורד-אפון-אייבון",
            "answer":    "כן",
            "reasoning": "הטקסט מאשר במפורש שנולד שם.",
        },
        ("en", "template"): {
            "text":      "William Shakespeare (1564–1616) was born in Stratford-upon-Avon, England. He is considered the greatest playwright of all time.",
            "statement": "The birthplace of Shakespeare is Stratford-upon-Avon",
            "answer":    "yes",
            "reasoning": "The text directly states he was born there.",
        },
    },
    # 1 — POSITIVE: founder
    {
        ("he", "triplet"): {
            "text":      "מייקרוסופט נוסדה ב-1975 על ידי ביל גייטס ופול אלן. המטה ממוקם ברדמונד, וושינגטון.",
            "subject":   "מייקרוסופט",
            "predicate": "מייסד",
            "object":    "ביל גייטס",
            "answer":    "כן",
            "reasoning": "הטקסט אומר 'נוסדה על ידי ביל גייטס', ולכן יחס המייסד מופיע.",
        },
        ("en", "triplet"): {
            "text":      "Microsoft was founded in 1975 by Bill Gates and Paul Allen. Its headquarters are in Redmond, Washington.",
            "subject":   "Microsoft",
            "predicate": "founder",
            "object":    "Bill Gates",
            "answer":    "yes",
            "reasoning": "The text says 'founded by Bill Gates', so the founder relation is expressed.",
        },
        ("he", "template"): {
            "text":      "מייקרוסופט נוסדה ב-1975 על ידי ביל גייטס ופול אלן. המטה ממוקם ברדמונד, וושינגטון.",
            "statement": "המייסד של מייקרוסופט הוא ביל גייטס",
            "answer":    "כן",
            "reasoning": "הטקסט מאשר שגייטס ייסד את החברה.",
        },
        ("en", "template"): {
            "text":      "Microsoft was founded in 1975 by Bill Gates and Paul Allen. Its headquarters are in Redmond, Washington.",
            "statement": "The founder of Microsoft is Bill Gates",
            "answer":    "yes",
            "reasoning": "The text confirms Gates co-founded the company.",
        },
    },
    # 2 — POSITIVE: occupation
    {
        ("he", "triplet"): {
            "text":      "מארי קירי הייתה פיזיקאית וכימאית פולנית-צרפתית. היא זכתה בפרס נובל פעמיים.",
            "subject":   "מארי קירי",
            "predicate": "עיסוק",
            "object":    "פיזיקאית",
            "answer":    "כן",
            "reasoning": "הטקסט מתאר אותה במפורש כ'פיזיקאית', ולכן יחס העיסוק מופיע.",
        },
        ("en", "triplet"): {
            "text":      "Marie Curie was a Polish-French physicist and chemist. She won the Nobel Prize twice.",
            "subject":   "Marie Curie",
            "predicate": "occupation",
            "object":    "physicist",
            "answer":    "yes",
            "reasoning": "The text describes her as a 'physicist', so the occupation relation is present.",
        },
        ("he", "template"): {
            "text":      "מארי קירי הייתה פיזיקאית וכימאית פולנית-צרפתית. היא זכתה בפרס נובל פעמיים.",
            "statement": "העיסוק של מארי קירי הוא פיזיקאית",
            "answer":    "כן",
            "reasoning": "הטקסט מציין שהיא פיזיקאית.",
        },
        ("en", "template"): {
            "text":      "Marie Curie was a Polish-French physicist and chemist. She won the Nobel Prize twice.",
            "statement": "The occupation of Marie Curie is physicist",
            "answer":    "yes",
            "reasoning": "The text explicitly says she was a physicist.",
        },
    },
    # 3 — NEGATIVE: official language (world-knowledge leakage)
    {
        ("he", "triplet"): {
            "text":      'ברזיל היא המדינה הגדולה ביותר בדרום אמריקה, עם שטח של כ-8.5 מיליון קמ"ר ואוכלוסייה של כ-215 מיליון.',
            "subject":   "ברזיל",
            "predicate": "שפה רשמית",
            "object":    "פורטוגזית",
            "answer":    "לא",
            "reasoning": "הטקסט עוסק בגודל ברזיל ואוכלוסייתה, אך לא מזכיר את שפתה הרשמית. גם אם פורטוגזית היא השפה הרשמית, עובדה זו לא מופיעה בקטע.",
        },
        ("en", "triplet"): {
            "text":      "Brazil is the largest country in South America, with an area of about 8.5 million km² and a population of about 215 million.",
            "subject":   "Brazil",
            "predicate": "official language",
            "object":    "Portuguese",
            "answer":    "no",
            "reasoning": "The text covers Brazil's size and population but never mentions its official language. Even though Portuguese is correct globally, it is not stated in this text.",
        },
        ("he", "template"): {
            "text":      'ברזיל היא המדינה הגדולה ביותר בדרום אמריקה, עם שטח של כ-8.5 מיליון קמ"ר ואוכלוסייה של כ-215 מיליון.',
            "statement": "השפה הרשמית של ברזיל היא פורטוגזית",
            "answer":    "לא",
            "reasoning": "הטקסט לא מזכיר שפה. למרות שהעובדה נכונה בעולם, היא לא מופיעה בקטע.",
        },
        ("en", "template"): {
            "text":      "Brazil is the largest country in South America, with an area of about 8.5 million km² and a population of about 215 million.",
            "statement": "The official language of Brazil is Portuguese",
            "answer":    "no",
            "reasoning": "The text does not mention any language. The fact is true globally but not expressed in this passage.",
        },
    },
    # 4 — NEGATIVE: borders (world-knowledge leakage)
    {
        ("he", "triplet"): {
            "text":      "הנמר האמורי הוא תת-מין נדיר של נמר החי ביערות רוסיה ובצפון סין. אוכלוסייתו פחות ממאה פרטים.",
            "subject":   "רוסיה",
            "predicate": "גובל עם",
            "object":    "סין",
            "answer":    "לא",
            "reasoning": "הטקסט מציין שהנמר חי ברוסיה ובסין, אך אינו מתאר יחסי גבול ביניהן. גם אם ידוע שהמדינות גובלות, עובדה זו לא מוזכרת.",
        },
        ("en", "triplet"): {
            "text":      "The Amur leopard is a rare subspecies of leopard living in the forests of Russia and northern China. Its population is estimated at fewer than one hundred individuals.",
            "subject":   "Russia",
            "predicate": "borders with",
            "object":    "China",
            "answer":    "no",
            "reasoning": "The text mentions Russia and China as habitats of the leopard but says nothing about a border between them.",
        },
        ("he", "template"): {
            "text":      "הנמר האמורי הוא תת-מין נדיר של נמר החי ביערות רוסיה ובצפון סין. אוכלוסייתו פחות ממאה פרטים.",
            "statement": "רוסיה גובלת ב-סין",
            "answer":    "לא",
            "reasoning": "הטקסט לא מדבר על גבולות. הנמר פשוט חי בשתי המדינות.",
        },
        ("en", "template"): {
            "text":      "The Amur leopard is a rare subspecies of leopard living in the forests of Russia and northern China. Its population is estimated at fewer than one hundred individuals.",
            "statement": "Russia borders China",
            "answer":    "no",
            "reasoning": "The text does not describe any border relationship; it only mentions both countries as leopard habitats.",
        },
    },
]


def get_shot_examples(lang: str, rel_type: str, n_shot: int) -> list:
    key = (lang, rel_type)
    if n_shot == 0:
        return []
    indices = SHOT_2_INDICES if n_shot == 2 else list(range(5))
    return [_POOL[i][key] for i in indices[:n_shot]]


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
    logger = logging.getLogger("clean_dataset_with_api_llm_v2")
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


def _col(prefix: str, tag: str, lang: str, rel: str, shot: int, cot: bool) -> str:
    cot_s = "cot" if cot else "nc"
    return f"{prefix}_{tag}_{lang}_{rel}_{shot}s_{cot_s}"


def _estimate_cost(in_tok: int, out_tok: int, prices: dict) -> float:
    return (in_tok * prices["input"] + out_tok * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _system_text(lang: str, rel_type: str, use_cot: bool) -> str:
    cot_suffix_he = (
        " לפני מתן תשובתך הסופית, הסבר בקצרה את הנימוקים שלך."
        ' לאחר מכן, כתוב את תשובתך הסופית בשורה האחרונה: "כן" או "לא" בלבד.'
    ) if use_cot else ""
    cot_suffix_en = (
        " Before giving your final answer, briefly explain your reasoning."
        ' Then write your final answer on the last line as just "yes" or "no".'
    ) if use_cot else ""

    if lang == "he":
        one_word = "" if use_cot else " ענה במילה אחת בלבד."
        if rel_type == "triplet":
            return (
                "אתה מסייע לזיהוי יחסים בטקסטים בעברית. "
                "יחס הוא קשר סמנטי בין שתי ישויות: נושא (subject), סוג היחס (predicate) ואובייקט (object). "
                "תפקידך לקרוא טקסט ולקבוע האם היחס הנתון מופיע או נובע מהטקסט — "
                "כלומר האם ניתן להסיק מהטקסט שהנושא קשור לאובייקט באמצעות היחס המצוין. "
                'ענה "כן" אם היחס מופיע בטקסט, או "לא" אם אינו מופיע.'
                + one_word + cot_suffix_he
            )
        else:
            return (
                "אתה מסייע לזיהוי יחסים בטקסטים בעברית. "
                "יחס בין שתי ישויות מנוסח כמשפט. "
                "תפקידך לקרוא טקסט ולקבוע האם המשפט הנתון נובע מהטקסט או מופיע בו. "
                'ענה "כן" אם המשפט נובע מהטקסט, או "לא" אחרת.'
                + one_word + cot_suffix_he
            )
    else:
        one_word = "" if use_cot else " One word only."
        if rel_type == "triplet":
            return (
                "You are a relation extraction assistant. "
                "A relation is a semantic connection between two entities described by a subject, a predicate (relation type), and an object. "
                "Your task is to read a text and determine whether the given relation is expressed or can be inferred from it — "
                "i.e., whether the text indicates that the subject is connected to the object via the given predicate. "
                'Answer "yes" if the relation is expressed in the text, "no" if it is not.'
                + one_word + cot_suffix_en
            )
        else:
            return (
                "You are a textual entailment assistant. "
                "A relation between two entities is expressed as a natural-language statement. "
                "Your task is to read a text and determine whether the given statement is entailed by or expressed in it. "
                'Answer "yes" if entailed, "no" if not.'
                + one_word + cot_suffix_en
            )


def _user_msg(lang: str, rel_type: str, item: dict, use_cot: bool) -> str:
    text = item["text"]
    if rel_type == "triplet":
        subject  = item["subject"]
        pred     = item["predicate"]
        obj      = item["object"]
        if lang == "he":
            ans_instr = ('הסבר את הנימוק, ולאחר מכן ענה "כן" או "לא" בשורה האחרונה.'
                         if use_cot else 'ענה "כן" או "לא" בלבד.')
            return (
                f"טקסט:\n{text}\n\n"
                f"יחס לבדיקה:\n"
                f"  נושא (subject):   {subject}\n"
                f"  יחס (predicate):  {pred}\n"
                f"  אובייקט (object): {obj}\n\n"
                f'חפש בטקסט האם הקשר בין "{subject}" ל-"{obj}" דרך היחס "{pred}" מוזכר או נובע ממנו.\n'
                + ans_instr
            )
        else:
            ans_instr = ('Explain your reasoning, then answer "yes" or "no" on the last line.'
                         if use_cot else 'Answer "yes" or "no" only.')
            return (
                f"Text:\n{text}\n\n"
                f"Relation to check:\n"
                f"  Subject:   {subject}\n"
                f"  Predicate: {pred}\n"
                f"  Object:    {obj}\n\n"
                f'Look in the text for whether the connection between "{subject}" and "{obj}" '
                f'via the relation "{pred}" is mentioned or can be inferred.\n'
                + ans_instr
            )
    else:
        relation = item.get("statement") or item.get(RELATION_COLS["template"]) or ""
        if lang == "he":
            ans_instr = ('הסבר את הנימוק, ולאחר מכן ענה "כן" או "לא" בשורה האחרונה.'
                         if use_cot else 'ענה "כן" או "לא" בלבד.')
            return (
                f"טקסט:\n{text}\n\n"
                f"משפט לבדיקה: {relation}\n\n"
                f'חפש בטקסט האם המשפט הנ"ל נובע ממנו או מופיע בו.\n'
                + ans_instr
            )
        else:
            ans_instr = ('Explain your reasoning, then answer "yes" or "no" on the last line.'
                         if use_cot else 'Answer "yes" or "no" only.')
            return (
                f"Text:\n{text}\n\n"
                f"Statement to check: {relation}\n\n"
                "Look in the text for whether the above statement is expressed or can be inferred from it.\n"
                + ans_instr
            )


def build_messages(lang: str, rel_type: str, row: dict,
                   n_shot: int, use_cot: bool) -> list:
    examples = get_shot_examples(lang, rel_type, n_shot)
    messages = [{"role": "system", "content": _system_text(lang, rel_type, use_cot)}]
    for ex in examples:
        messages.append({"role": "user",      "content": _user_msg(lang, rel_type, ex, use_cot)})
        asst = (ex["reasoning"] + "\n" + ex["answer"]) if use_cot else ex["answer"]
        messages.append({"role": "assistant", "content": asst})
    messages.append({"role": "user", "content": _user_msg(lang, rel_type, row, use_cot)})
    return messages


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_YES_TOKENS = {"yes", "כן"}
_NO_TOKENS  = {"no",  "לא"}


def _tokenise(text: str) -> list:
    return [t.lower() for t in re.split(r"[\s.,!?:;()\[\]\"']+", text) if t]


def parse_yes_no(raw: str, cot_mode: bool = False) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cot_mode:
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        for line in reversed(lines[-3:]):
            tokens = _tokenise(line)
            for t in tokens:
                if t in _YES_TOKENS: return "1"
                if t in _NO_TOKENS:  return "0"
        for line in reversed(lines):
            if re.search(r"\byes\b", line.lower()) or "כן" in line: return "1"
            if re.search(r"\bno\b",  line.lower()) or "לא" in line: return "0"
        return "unknown"
    tokens = _tokenise(cleaned)
    if tokens:
        if tokens[0] in _YES_TOKENS: return "1"
        if tokens[0] in _NO_TOKENS:  return "0"
    lower = cleaned.lower()
    if re.search(r"\byes\b", lower) or "כן" in lower: return "1"
    if re.search(r"\bno\b",  lower) or "לא" in lower: return "0"
    return "unknown"


# ---------------------------------------------------------------------------
# API call with retry
# ---------------------------------------------------------------------------

def call_api(client, model_id: str, messages: list, max_tokens: int) -> tuple:
    """
    Returns (raw_text, in_tokens, out_tokens).
    Retries up to MAX_RETRIES with exponential backoff; 429 uses RETRY_429_DELAY.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id, messages=messages,
                max_tokens=max_tokens, temperature=0,
            )
            raw   = response.choices[0].message.content or ""
            usage = response.usage
            return raw, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            err_str = str(e)
            is_last = attempt == MAX_RETRIES - 1
            if is_last:
                return f"ERROR: {e}", 0, 0
            if "429" in err_str or "rate limit" in err_str.lower():
                time.sleep(RETRY_429_DELAY)
            else:
                time.sleep(RETRY_BASE_DELAY * (2 ** attempt))


# ---------------------------------------------------------------------------
# Classify all rows for one combo — concurrent
# ---------------------------------------------------------------------------

def classify_rows(rows: list, client, model_id: str,
                  lang: str, rel_type: str, n_shot: int, use_cot: bool,
                  max_tokens: int, n_workers: int,
                  log: logging.Logger, desc: str) -> tuple:
    """
    Returns (parsed_labels, raw_outputs, total_in_tok, total_out_tok).
    """
    n             = len(rows)
    parsed        = [None] * n
    raws          = [None] * n
    total_in_tok  = 0
    total_out_tok = 0

    def _process(idx_row):
        idx, row = idx_row
        msgs = build_messages(lang, rel_type, row, n_shot, use_cot)
        raw, in_tok, out_tok = call_api(client, model_id, msgs, max_tokens)
        return idx, raw, in_tok, out_tok

    log_every = max(1, n // 4)
    t_start   = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process, (i, r)): i for i, r in enumerate(rows)}
        for done_count, future in enumerate(
            tqdm(concurrent.futures.as_completed(futures), total=n,
                 desc=f"    {desc}", leave=False, file=TqdmToLogger(log)), 1
        ):
            idx, raw, in_tok, out_tok = future.result()
            parsed[idx]  = parse_yes_no(raw, cot_mode=use_cot)
            raws[idx]    = raw.strip()
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
                    f"tokens: in={total_in_tok:,}, out={total_out_tok:,}"
                )

    return parsed, raws, total_in_tok, total_out_tok


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_hard(labels: list, gold: list) -> dict:
    TP = FP = FN = TN = unknown = 0
    for pred, g in zip(labels, gold):
        if pred == "unknown":
            unknown += 1
        pos      = pred == "1"
        gold_pos = g == "1"
        if pos and gold_pos:       TP += 1
        elif pos and not gold_pos: FP += 1
        elif not pos and gold_pos: FN += 1
        else:                      TN += 1
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total             if total             else 0
    precision = TP / (TP + FP)               if (TP + FP)         else 0
    recall    = TP / (TP + FN)               if (TP + FN)         else 0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "unknown": unknown}


# ---------------------------------------------------------------------------
# Majority / ensemble voting
# ---------------------------------------------------------------------------

def compute_majority(rows: list, col_names: list) -> list:
    results = []
    for row in rows:
        votes = [row[c] for c in col_names if c in row and row[c] in ("1", "0")]
        ones  = votes.count("1")
        zeros = votes.count("0")
        if ones > zeros:    results.append("1")
        elif zeros > ones:  results.append("0")
        else:               results.append("unknown")
    return results


def compute_weighted_majority(rows: list, col_names: list, weights: list,
                               threshold: float = 0.5) -> list:
    results = []
    for row in rows:
        total_w = total_v = 0.0
        for col, w in zip(col_names, weights):
            pred = row.get(col, "unknown")
            if pred == "1":
                total_v += w
                total_w += w
            elif pred == "0":
                total_w += w
        if total_w == 0:
            results.append("unknown")
        elif total_v / total_w >= threshold:
            results.append("1")
        else:
            results.append("0")
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(run_stats: list, majority_stats: list, weighted_stats: list,
                  summary_path: str, log: logging.Logger, total: float):
    W = 130
    lines = []
    lines.append("=" * W)
    lines.append("API LLM CLASSIFICATION SUMMARY  (v2)")
    lines.append(f"Total wall time: {_fmt_duration(total)}")
    lines.append("=" * W)

    # Per-model cost / timing totals
    model_agg: dict = {}
    for s in run_stats:
        agg = model_agg.setdefault(s["model"], {"time": 0.0, "rows": 0,
                                                  "cost": 0.0, "in": 0, "out": 0})
        agg["time"] += s["time"]
        agg["rows"] += s["n_rows"]
        agg["cost"] += s["cost"]
        agg["in"]   += s["in_tok"]
        agg["out"]  += s["out_tok"]
    lines.append("")
    lines.append("Per-model totals:")
    for model, agg in model_agg.items():
        rps = agg["rows"] / agg["time"] if agg["time"] > 0 else float("inf")
        lines.append(
            f"  {model:<16}  time={_fmt_duration(agg['time'])}  "
            f"rows/s={rps:.1f}  cost=${agg['cost']:.4f}  "
            f"tokens: in={agg['in']:,}, out={agg['out']:,}"
        )

    # ── HARD-LABEL TABLE ──────────────────────────────────────────────────────
    lines.append("")
    lines.append("Hard-label metrics:")
    hdr = (
        f"  {'model':<16} {'lang':<4} {'rel':<10} {'shot':>4}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'unk':>4}  {'rows/s':>7}  {'cost':>9}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in run_stats:
        m   = s["metrics"]
        rps = s["n_rows"] / s["time"] if s["time"] > 0 else float("inf")
        lines.append(
            f"  {s['model']:<16} {s['lang']:<4} {s['rel_type']:<10} {s['shot']:>4}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['unknown']:>4}  {rps:>7.1f}  ${s['cost']:>8.4f}  {_fmt_duration(s['time']):>8}"
        )

    # ── UNIFORM MAJORITY TABLE ────────────────────────────────────────────────
    if majority_stats:
        lines.append("")
        lines.append("Uniform majority-vote columns:")
        hdr3 = (
            f"  {'column':<55}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
            f"{'unk':>4}  {'voters':>6}"
        )
        lines.append(hdr3)
        lines.append("  " + "-" * (len(hdr3) - 2))
        for s in majority_stats:
            m = s["metrics"]
            lines.append(
                f"  {s['col']:<55}  "
                f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
                f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
                f"{m['unknown']:>4}  {s['n_voters']:>6}"
            )

    # ── WEIGHTED MAJORITY TABLE ───────────────────────────────────────────────
    if weighted_stats:
        lines.append("")
        lines.append("F1-weighted majority-vote columns:")
        for s in weighted_stats:
            m = s["metrics"]
            lines.append(
                f"  {s['col']:<55}  "
                f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
                f"unk={m['unknown']}  voters={s['n_voters']}"
            )

    lines.append("=" * W)
    text = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def write_error_analysis(rows: list, run_stats: list, gold: list,
                          label_col: str, error_path: str,
                          log: logging.Logger, max_examples: int = 10):
    W = 110
    n_rows = len(rows)
    n_pos  = gold.count("1")

    best_stat = max(run_stats, key=lambda s: s["metrics"]["f1"])
    best_clean = _col("llm_clean", best_stat["model"], best_stat["lang"],
                       best_stat["rel_type"], best_stat["shot"], best_stat["cot"])

    lines = []
    lines.append("=" * W)
    lines.append("  API LLM ERROR ANALYSIS  (v2)")
    lines.append(f"  Total rows         : {n_rows}")
    lines.append(f"  Positive / Negative: {n_pos} / {n_rows-n_pos}  "
                 f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")
    lines.append(f"  Best combo (F1={best_stat['metrics']['f1']:.3f}): "
                 f"{best_stat['model']} / {best_stat['lang']} / {best_stat['rel_type']} / "
                 f"{best_stat['shot']}-shot / {'CoT' if best_stat['cot'] else 'no-CoT'}")
    lines.append("=" * W)

    # Section 1: per-combo error counts
    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 1 — PER-COMBO ERROR COUNTS")
    lines.append("=" * W)
    hdr = (f"  {'combo':<60} {'F1':>6} {'FP':>5} {'FN':>5} {'unk':>5}  "
           "FP-rate  FN-rate")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in sorted(run_stats, key=lambda x: x["metrics"]["f1"], reverse=True):
        m     = s["metrics"]
        combo = f"{s['model']}/{s['lang']}/{s['rel_type']}/{s['shot']}s"
        fp_r = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) else 0
        fn_r = m["FN"] / (m["FN"] + m["TP"]) if (m["FN"] + m["TP"]) else 0
        lines.append(
            f"  {combo:<60} {m['f1']:>6.3f} {m['FP']:>5} {m['FN']:>5} {m['unknown']:>5}"
            f"  {fp_r:.3f}    {fn_r:.3f}"
        )

    # Section 2: shot comparison
    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 2 — SHOT-LEVEL F1 COMPARISON")
    lines.append("=" * W)
    shot_f1: dict = defaultdict(list)
    for s in run_stats:
        shot_f1[s["shot"]].append(s["metrics"]["f1"])
    for shot, vals in sorted(shot_f1.items()):
        lines.append(f"  {shot}-shot:  mean F1={sum(vals)/len(vals):.4f}  ({len(vals)} combos)")

    # Sections 3 & 4: FP/FN examples for best combo
    for pred_val, gold_val, title in [
        ("1", "0", "SECTION 3 — FALSE POSITIVES  [best combo]"),
        ("0", "1", "SECTION 4 — FALSE NEGATIVES  [best combo]"),
    ]:
        examples = [(i, r) for i, r in enumerate(rows)
                    if r.get(best_clean) == pred_val and r[label_col] == gold_val]
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  {title}  — {len(examples)} total")
        lines.append("=" * W)
        for _, (i, row) in enumerate(examples[:max_examples]):
            snip = row["text"].replace("\n", " ")[:150]
            rel  = row.get(RELATION_COLS[best_stat["rel_type"]], "")
            raw  = row.get(_col("llm_raw", best_stat["model"], best_stat["lang"],
                               best_stat["rel_type"], best_stat["shot"], False), "")
            lines.append(f"  [row {i}]")
            lines.append(f"    text      : {snip!r}")
            lines.append(f"    relation  : {rel!r}")
            lines.append(f"    raw output: {raw!r}")
        if len(examples) > max_examples:
            lines.append(f"  ... and {len(examples) - max_examples} more")

    # Section 5: Hard rows — all combos wrong
    all_clean = [_col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
                 for s in run_stats]
    hard = [i for i, r in enumerate(rows)
            if all(r.get(c, "unknown") != gold[i] for c in all_clean if c in r)]
    lines.append("")
    lines.append("=" * W)
    lines.append(f"  SECTION 5 — HARD ROWS: all combos wrong — "
                 f"{len(hard)} ({100*len(hard)/n_rows:.1f}%)")
    lines.append("=" * W)
    for i in hard[:max_examples]:
        snip = rows[i]["text"].replace("\n", " ")[:150]
        lines.append(f"  [row {i}]  gold={gold[i]}  text={snip!r}")
    if len(hard) > max_examples:
        lines.append(f"  ... and {len(hard) - max_examples} more")

    lines.append("=" * W)
    with open(error_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Error analysis written to {error_path}")


# ---------------------------------------------------------------------------
# Predicate-stratified analysis
# ---------------------------------------------------------------------------

def write_predicate_stratified(rows: list, run_stats: list, gold: list,
                                predicate_col: str, pred_path: str,
                                log: logging.Logger):
    W = 130
    n_rows = len(rows)

    pred_groups: dict = defaultdict(list)
    for i, row in enumerate(rows):
        pred_groups[row.get(predicate_col, "UNKNOWN")].append(i)

    top_combos = sorted(run_stats, key=lambda s: s["metrics"]["f1"], reverse=True)[:10]
    sorted_preds = sorted(pred_groups.items(), key=lambda kv: len(kv[1]), reverse=True)

    lines = []
    lines.append("=" * W)
    lines.append("  API LLM PREDICATE-STRATIFIED METRICS")
    lines.append(f"  Predicates with >= {MIN_PRED_EXAMPLES} examples: "
                 f"{sum(1 for v in pred_groups.values() if len(v) >= MIN_PRED_EXAMPLES)}")
    lines.append("=" * W)

    # Overall predicate distribution
    lines.append("")
    lines.append("Predicate distribution (top 20 by count):")
    hdr0 = f"  {'predicate':<35} {'n':>5} {'pos':>5} {'neg':>5} {'pos%':>6}"
    lines.append(hdr0)
    lines.append("  " + "-" * (len(hdr0) - 2))
    pred_pos = {p: sum(1 for i in idxs if gold[i] == "1") for p, idxs in pred_groups.items()}
    for p, idxs in sorted_preds[:20]:
        n, ps = len(idxs), pred_pos[p]
        lines.append(f"  {p:<35} {n:>5} {ps:>5} {n-ps:>5} {100*ps/n:>5.0f}%")

    # Per-predicate F1 for top combos
    lines.append("")
    lines.append("=" * W)
    lines.append(f"  PER-PREDICATE F1  (predicates with ≥{MIN_PRED_EXAMPLES}, top combos)")
    lines.append("=" * W)
    combo_labels = [
        f"{s['model']}/{s['lang']}/{s['rel_type']}/{s['shot']}s"
        for s in top_combos
    ]
    hdr_parts = [f"  {'predicate':<35} {'n':>5}"]
    for lbl in combo_labels:
        hdr_parts.append(f" {lbl[:16]:>16}")
    lines.append("".join(hdr_parts))
    lines.append("  " + "-" * (len("".join(hdr_parts)) - 2))

    pred_f1_summary: list = []
    for pred, idxs in sorted_preds:
        if len(idxs) < MIN_PRED_EXAMPLES:
            continue
        pred_golds = [gold[i] for i in idxs]
        combo_f1s, row_parts = [], [f"  {pred:<35} {len(idxs):>5}"]
        for s in top_combos:
            clean_col   = _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
            pred_labels = [rows[i].get(clean_col, "unknown") for i in idxs]
            m = compute_metrics_hard(pred_labels, pred_golds)
            combo_f1s.append(m["f1"])
            row_parts.append(f" {m['f1']:>16.3f}")
        mean_f1 = sum(combo_f1s) / len(combo_f1s) if combo_f1s else 0
        pred_f1_summary.append((pred, mean_f1, len(idxs)))
        lines.append("".join(row_parts))

    lines.append("")
    lines.append("=" * W)
    lines.append("  EASIEST predicates (highest mean F1):")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: -x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")
    lines.append("")
    lines.append("  HARDEST predicates (lowest mean F1):")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")

    # Shot comparison per predicate
    lines.append("")
    lines.append("=" * W)
    lines.append("  SHOT COMPARISON PER PREDICATE  (mean F1 across all models/lang/rel)")
    lines.append("=" * W)
    hdr5 = f"  {'predicate':<35} {'n':>5}" + "".join(f" {'shot'+str(k)+'s':>8}" for k in [0, 5])
    lines.append(hdr5)
    lines.append("  " + "-" * (len(hdr5) - 2))
    for pred, _, n in sorted(pred_f1_summary, key=lambda x: -x[2])[:20]:
        idxs = pred_groups[pred]
        pred_golds = [gold[i] for i in idxs]
        row_parts = [f"  {pred:<35} {n:>5}"]
        for shot_k in [0, 5]:
            shot_runs = [s for s in run_stats if s["shot"] == shot_k]
            f1s = []
            for s in shot_runs:
                clean_col   = _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
                pred_labels = [rows[i].get(clean_col, "unknown") for i in idxs]
                f1s.append(compute_metrics_hard(pred_labels, pred_golds)["f1"])
            row_parts.append(f" {sum(f1s)/len(f1s) if f1s else 0:>8.3f}")
        lines.append("".join(row_parts))

    lines.append("=" * W)
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Predicate analysis written to {pred_path}")


# ---------------------------------------------------------------------------
# Archive helper
# ---------------------------------------------------------------------------

def archive_old_outputs(base_dir: str, log: logging.Logger):
    outputs_dir = os.path.join(base_dir, "outputs")
    archive_dir = os.path.join(outputs_dir, "ARCHIVE")
    if not os.path.isdir(outputs_dir):
        return
    os.makedirs(archive_dir, exist_ok=True)

    protected_dirs  = {"ARCHIVE", "API_LLM_clean", "opensource_LLM_clean",
                       "NLI_clean", "dataset_statistics"}
    protected_files = {
        "prepared_gold_dataset_gemma_3_27b_it.csv",
        "prepared_gold_dataset_gemma_3_27b_it.log",
    }

    moved = 0
    for name in os.listdir(outputs_dir):
        src = os.path.join(outputs_dir, name)
        if name in protected_dirs:
            continue
        if os.path.isfile(src) and name in protected_files:
            continue
        dst = os.path.join(archive_dir, name)
        if os.path.exists(dst):
            log.info(f"[archive]  {name} already in ARCHIVE — skipping")
            continue
        shutil.move(src, dst)
        log.info(f"[archive]  {name} → ARCHIVE/")
        moved += 1
    if moved == 0:
        log.info("[archive]  nothing to move (outputs/ already clean)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced API LLM relation classifier v2 via OpenRouter"
    )
    parser.add_argument("--input",      default=INPUT_FILE)
    parser.add_argument("--output",     default=OUTPUT_FILE)
    parser.add_argument("--log",        default=LOG_FILE)
    parser.add_argument("--summary",    default=SUMMARY_FILE)
    parser.add_argument("--error",      default=ERROR_FILE)
    parser.add_argument("--pred-file",  default=PRED_FILE)
    parser.add_argument("--prompts",    default=PROMPTS_FILE)
    parser.add_argument("--label-col",  default=LABEL_COL)
    parser.add_argument("--workers",    type=int, default=MAX_WORKERS)
    parser.add_argument("--shots",      type=int, nargs="+", default=SHOT_CONFIGS,
                        help="Shot counts (default: 0 5)")
    parser.add_argument("--debug",      type=int, default=0,
                        help="Run on first N rows only (0 = full dataset)")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip archiving old outputs/")
    args = parser.parse_args()

    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise EnvironmentError(
            f"Environment variable {OPENROUTER_API_KEY_ENV} is not set.\n"
            f"Run: export {OPENROUTER_API_KEY_ENV}=sk-or-..."
        )

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path   = os.path.join(base, args.input)
    output_path  = os.path.join(base, args.output)
    log_path     = os.path.join(base, args.log)
    sum_path     = os.path.join(base, args.summary)
    err_path     = os.path.join(base, args.error)
    pred_path    = os.path.join(base, args.pred_file)
    prompts_path = os.path.join(base, args.prompts)

    for p in (output_path, log_path, sum_path, err_path, pred_path, prompts_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    if not args.no_archive:
        archive_old_outputs(base, log)

    log.info("=" * 70)
    log.info("clean_dataset_with_api_llm.py  v2  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output dir: {os.path.join(base, OUTPUT_SUBDIR)}")
    log.info(f"  workers:    {args.workers}")
    log.info(f"  shots:      {args.shots}")
    log.info(f"  debug rows: {args.debug if args.debug else 'all'}")
    log.info(f"  models ({len(API_MODELS)}):")
    for m in API_MODELS:
        log.info(f"    [{m['tag']}]  {m['id']}  "
                 f"(in=${m['prices_per_1m']['input']}/1M, out=${m['prices_per_1m']['output']}/1M)")
    log.info("=" * 70)

    # Load CSV
    t0 = time.time()
    with open(input_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if args.debug:
        rows = rows[:args.debug]
        log.info(f"[debug]  truncated to first {args.debug} rows")
    n_rows = len(rows)
    gold   = [r[args.label_col] for r in rows]
    n_pos  = gold.count("1")
    log.info(f"[load]  {n_rows} rows  ({_fmt_duration(time.time()-t0)})")
    log.info(f"[load]  positive={n_pos}  negative={n_rows-n_pos}  "
             f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")

    for col in (args.label_col, PREDICATE_COL):
        if col not in rows[0]:
            raise ValueError(f"Column '{col}' not found.")
    for rel_col in RELATION_COLS.values():
        if rel_col not in rows[0]:
            raise ValueError(f"Relation column '{rel_col}' not found.")

    # OpenRouter client
    from openai import OpenAI
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title":      OPENROUTER_APP_NAME,
        },
    )

    run_stats:     list = []
    prompt_records: list = []

    # ── Main loop ──────────────────────────────────────────────────────────────
    for model_idx, model_cfg in enumerate(API_MODELS, 1):
        tag      = model_cfg["tag"]
        model_id = model_cfg["id"]
        prices   = model_cfg["prices_per_1m"]

        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(API_MODELS)}]  {tag}  ({model_id})")
        t_model = time.time()

        for lang in PROMPT_LANGS:
            for rel_type in RELATION_TYPES:
                for n_shot in args.shots:
                        clean_col = _col("llm_clean", tag, lang, rel_type, n_shot, False)
                        raw_col   = _col("llm_raw",   tag, lang, rel_type, n_shot, False)
                        desc      = f"{tag}/{lang}/{rel_type}/{n_shot}s/nc"

                        log.info("*" * 70)
                        log.info(f"  COMBO: {desc}")
                        log.info(f"    output cols: {clean_col} | {raw_col}")

                        # Log example prompt
                        ex_msgs = build_messages(lang, rel_type, rows[0], n_shot, False)
                        log.info(f"    example prompt (row 0):\n{'-'*40}")
                        for msg in ex_msgs:
                            log.info(f"    [{msg['role']}] {msg['content'][:300]}")
                        log.info(f"    {'-'*40}")

                        t0 = time.time()
                        parsed, raws, in_tok, out_tok = classify_rows(
                            rows, client, model_id, lang, rel_type,
                            n_shot, False, LLM_MAX_TOKENS,
                            n_workers=args.workers, log=log, desc=desc,
                        )
                        elapsed = time.time() - t0

                        for r, label, raw in zip(rows, parsed, raws):
                            r[clean_col] = label
                            r[raw_col]   = raw
                            prompt_records.append({
                                "row_idx":  rows.index(r),
                                "model": tag, "lang": lang, "rel_type": rel_type,
                                "shot": n_shot,
                                "messages": build_messages(lang, rel_type, r, n_shot, False),
                                "raw": raw, "parsed": label,
                                "gold": r[args.label_col],
                            })

                        metrics = compute_metrics_hard(parsed, gold)
                        cost = _estimate_cost(in_tok, out_tok, prices)
                        rps  = n_rows / elapsed if elapsed > 0 else float("inf")
                        m    = metrics

                        log.info(f"  {'='*66}")
                        log.info(f"  RESULT  {desc}")
                        log.info(f"  {'─'*66}")
                        log.info(f"  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                                 f"rec={m['recall']:.3f}  f1={m['f1']:.3f}")
                        log.info(f"  TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']} "
                                 f"unk={m['unknown']}  {rps:.1f} rows/s  "
                                 f"{_fmt_duration(elapsed)}  cost=${cost:.4f}")
                        log.info(f"  tokens: in={in_tok:,}, out={out_tok:,}")
                        log.info(f"  {'='*66}")

                        log.info(f"  examples (first 3):")
                        for r, label, raw in zip(rows[:3], parsed[:3], raws[:3]):
                            log.info(f"    hyp={r[RELATION_COLS[rel_type]]!r}")
                            log.info(f"    raw={raw!r}  parsed={label}  gold={r[args.label_col]}")

                        run_stats.append({
                            "model": tag, "lang": lang, "rel_type": rel_type,
                            "shot": n_shot,
                            "time": elapsed, "n_rows": n_rows,
                            "in_tok": in_tok, "out_tok": out_tok, "cost": cost,
                            "metrics": metrics,
                        })

        log.info(f"  [{tag}] total time: {_fmt_duration(time.time()-t_model)}")

        # Checkpoint save after each model
        t0 = time.time()
        fieldnames = list(dict.fromkeys(rows[0].keys()))
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"[checkpoint save]  {n_rows} rows → {output_path}  "
                 f"({_fmt_duration(time.time()-t0)})")

    # ── Majority / ensemble voting ─────────────────────────────────────────────
    log.info("=" * 70)
    log.info("[voting]  computing majority / ensemble columns")

    majority_stats: list = []
    weighted_stats: list = []

    all_clean_cols = [
        _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
        for s in run_stats
    ]
    all_f1s = [s["metrics"]["f1"] for s in run_stats]

    # 1. Per (lang, rel, shot) across all models
    seen: set = set()
    for s in run_stats:
        key = (s["lang"], s["rel_type"], s["shot"])
        if key in seen:
            continue
        seen.add(key)
        src_cols = [_col("llm_clean", mc["tag"], s["lang"], s["rel_type"], s["shot"], False)
                    for mc in API_MODELS
                    if _col("llm_clean", mc["tag"], s["lang"], s["rel_type"], s["shot"], False)
                    in rows[0]]
        if not src_cols:
            continue
        col_name = _col("llm_majority", "all_models",
                        s["lang"], s["rel_type"], s["shot"], False)
        labels = compute_majority(rows, src_cols)
        for r, lbl in zip(rows, labels):
            r[col_name] = lbl
        m = compute_metrics_hard(labels, gold)
        majority_stats.append({"col": col_name, "n_voters": len(src_cols), "metrics": m})
        log.info(f"  {col_name}: f1={m['f1']:.3f}")

    # 2. Best-per-model: uniform + weighted
    best_per_model_cols, best_per_model_w = [], []
    for mc in API_MODELS:
        model_runs = [s for s in run_stats if s["model"] == mc["tag"]]
        if not model_runs:
            continue
        best = max(model_runs, key=lambda s: s["metrics"]["f1"])
        c = _col("llm_clean", mc["tag"], best["lang"], best["rel_type"], best["shot"], False)
        best_per_model_cols.append(c)
        best_per_model_w.append(best["metrics"]["f1"])
        log.info(f"  best combo for {mc['tag']}: {best['lang']}/{best['rel_type']}/"
                 f"{best['shot']}s  F1={best['metrics']['f1']:.3f}")

    col = "llm_majority_best_per_model"
    labels = compute_majority(rows, best_per_model_cols)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    majority_stats.append({"col": col, "n_voters": len(best_per_model_cols), "metrics": m})

    col = "llm_wmajority_best_per_model"
    labels = compute_weighted_majority(rows, best_per_model_cols, best_per_model_w)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    weighted_stats.append({"col": col, "n_voters": len(best_per_model_cols), "metrics": m})
    log.info(f"  {col}: f1={m['f1']:.3f}")

    # 3. All combos: uniform + weighted
    col = "llm_majority_all"
    labels = compute_majority(rows, all_clean_cols)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    majority_stats.append({"col": col, "n_voters": len(all_clean_cols), "metrics": m})

    col = "llm_wmajority_all"
    labels = compute_weighted_majority(rows, all_clean_cols, all_f1s)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    weighted_stats.append({"col": col, "n_voters": len(all_clean_cols), "metrics": m})
    log.info(f"  {col}: f1={m['f1']:.3f}")

    # Final CSV save
    t0 = time.time()
    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows written  ({_fmt_duration(time.time()-t0)})")

    # Prompts JSONL
    with open(prompts_path, "w", encoding="utf-8") as f:
        for rec in prompt_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"Prompts log written  ({len(prompt_records)} records)")

    # Reports
    total = time.time() - wall_start
    write_summary(run_stats, majority_stats, weighted_stats, sum_path, log, total)
    write_error_analysis(rows, run_stats, gold, args.label_col, err_path, log)
    write_predicate_stratified(rows, run_stats, gold, PREDICATE_COL, pred_path, log)

    log.info("=" * 70)
    log.info(f"DONE.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  classified.csv     → {output_path}")
    log.info(f"  summary.txt        → {sum_path}")
    log.info(f"  error_analysis.txt → {err_path}")
    log.info(f"  predicate_analysis → {pred_path}")
    log.info(f"  prompts.jsonl      → {prompts_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()