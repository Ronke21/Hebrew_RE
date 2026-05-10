"""
Enhanced open-source LLM relation classifier — v2.

Dimensions compared (on top of original model × lang × rel_type):
  * shot   : 0 / 2 / 5-shot in-context examples
  * cot    : chain-of-thought on/off  (instruct models only; skipped for base)
  * scores : soft P(yes)/(P(yes)+P(no)) from next-token logits + threshold sweep
  + weighted majority vote  (F1-weighted across combos)
  + predicate-stratified metrics

Output  : outputs/opensource_LLM_clean/
Archive : existing outputs/ files moved to outputs/ARCHIVE/ (skipped if already done)

Usage
-----
  CUDA_VISIBLE_DEVICES=6,7 python clean_data/clean_dataset_with_opensource_llm.py
  CUDA_VISIBLE_DEVICES=6,7 python clean_data/clean_dataset_with_opensource_llm.py --shots 0 2 --cot-modes nocot
  CUDA_VISIBLE_DEVICES=6,7 python clean_data/clean_dataset_with_opensource_llm.py --debug 20
  CUDA_VISIBLE_DEVICES=6,7 python clean_data/clean_dataset_with_opensource_llm.py --no-archive
"""

import gc
import os
import re
import csv
import shutil
import time
import logging
import argparse
from collections import defaultdict

import torch
import transformers
from tqdm import tqdm

# --- vLLM / transformers compatibility patch ---
if not hasattr(transformers.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    transformers.PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens
    )

try:
    from vllm import LLM as vLLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths / macros
# ---------------------------------------------------------------------------

INPUT_FILE    = "outputs/ARCHIVE/prepared_gold_dataset_gemma_3_27b_it.csv"
OUTPUT_SUBDIR = "outputs/opensource_LLM_clean"
OUTPUT_FILE   = f"{OUTPUT_SUBDIR}/classified.csv"
LOG_FILE      = f"{OUTPUT_SUBDIR}/classify.log"
SUMMARY_FILE  = f"{OUTPUT_SUBDIR}/summary.txt"
ERROR_FILE    = f"{OUTPUT_SUBDIR}/error_analysis.txt"
PRED_FILE     = f"{OUTPUT_SUBDIR}/predicate_analysis.txt"

LABEL_COL     = "relation_present"
PREDICATE_COL = "predicate"

RELATION_COLS = {
    "triplet":  "basic_relation",
    "template": "template_relation",
}

LLM_MODELS = [
    {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507", "tag": "qwen3",         "type": "instruct"},
    {"id": "dicta-il/DictaLM-3.0-24B-Base",    "tag": "dictalm3",      "type": "base"},
    {"id": "CohereLabs/aya-expanse-32b",        "tag": "cohere_aya32b", "type": "instruct"},
    {"id": "google/gemma-3-27b-it",             "tag": "gemma3",        "type": "instruct"},
    {"id": "google/gemma-4-26B-A4B-it",        "tag": "gemma4_it",     "type": "instruct"},
    {"id": "google/gemma-4-31B-it",            "tag": "gemma4_31b_it",    "type": "instruct"},
    {"id": "Qwen/Qwen3.5-35B-A3B-Base",        "tag": "qwen35_base",      "type": "base"},
    {"id": "mistralai/Mistral-Small-24B-Instruct-2501", "tag": "mistral_small24b", "type": "instruct"},
]

PROMPT_LANGS   = ["he", "en"]
RELATION_TYPES = ["triplet", "template"]
SHOT_CONFIGS   = [0, 2, 5]
COT_CONFIGS    = [False]

LLM_BATCH_SIZE         = 32
LLM_MAX_NEW_TOKENS = 32
LLM_MAX_INPUT_LEN      = 2048

SOFT_THRESHOLDS   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_PRED_EXAMPLES = 5

# For 2-shot: pick 1 positive + 1 world-knowledge-leakage negative
SHOT_2_INDICES = [0, 3]

# ---------------------------------------------------------------------------
# Few-shot example pool  (3 positive + 2 negative)
# ---------------------------------------------------------------------------
# _POOL[i][(lang, rel_type)] → dict with text, relation info, answer, reasoning

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
    logger = logging.getLogger("clean_dataset_with_opensource_llm_v2")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
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
            suffix = '\nחפש בטקסט האם הקשר בין "{s}" ל-"{o}" דרך היחס "{p}" מוזכר או נובע ממנו.\n{ans}'.format(
                s=subject, o=obj, p=pred,
                ans='ענה "כן" או "לא" בלבד.' if not use_cot else 'הסבר את הנימוק, ולאחר מכן ענה "כן" או "לא" בשורה האחרונה.'
            )
            return (
                f"טקסט:\n{text}\n\n"
                f"יחס לבדיקה:\n"
                f"  נושא (subject):   {subject}\n"
                f"  יחס (predicate):  {pred}\n"
                f"  אובייקט (object): {obj}\n"
                + suffix
            )
        else:
            suffix = '\nLook in the text for whether the connection between "{s}" and "{o}" via the relation "{p}" is mentioned or can be inferred.\n{ans}'.format(
                s=subject, o=obj, p=pred,
                ans='Answer "yes" or "no" only.' if not use_cot else 'Explain your reasoning, then answer "yes" or "no" on the last line.'
            )
            return (
                f"Text:\n{text}\n\n"
                f"Relation to check:\n"
                f"  Subject:   {subject}\n"
                f"  Predicate: {pred}\n"
                f"  Object:    {obj}\n"
                + suffix
            )
    else:
        relation = item.get("statement") or item.get(RELATION_COLS["template"]) or ""
        if lang == "he":
            suffix = '\nחפש בטקסט האם המשפט הנ"ל נובע ממנו או מופיע בו.\n{ans}'.format(
                ans='ענה "כן" או "לא" בלבד.' if not use_cot else 'הסבר את הנימוק, ולאחר מכן ענה "כן" או "לא" בשורה האחרונה.'
            )
            return f"טקסט:\n{text}\n\nמשפט לבדיקה: {relation}\n" + suffix
        else:
            suffix = '\nLook in the text for whether the above statement is expressed or can be inferred from it.\n{ans}'.format(
                ans='Answer "yes" or "no" only.' if not use_cot else 'Explain your reasoning, then answer "yes" or "no" on the last line.'
            )
            return f"Text:\n{text}\n\nStatement to check: {relation}\n" + suffix


def _build_prompt_instruct(lang: str, rel_type: str, row: dict, tokenizer,
                           n_shot: int, use_cot: bool) -> str:
    examples = get_shot_examples(lang, rel_type, n_shot)
    messages = [{"role": "system", "content": _system_text(lang, rel_type, use_cot)}]
    for ex in examples:
        messages.append({"role": "user",      "content": _user_msg(lang, rel_type, ex, use_cot)})
        if use_cot:
            asst = ex["reasoning"] + "\n" + ex["answer"]
        else:
            asst = ex["answer"]
        messages.append({"role": "assistant", "content": asst})
    messages.append({"role": "user", "content": _user_msg(lang, rel_type, row, use_cot)})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_prompt_base(lang: str, rel_type: str, row: dict, n_shot: int) -> str:
    examples = get_shot_examples(lang, rel_type, n_shot)

    if lang == "he":
        if rel_type == "triplet":
            intro = (
                "קרא את הטקסט הבא וענה האם היחס הנתון מופיע בו.\n"
                "יחס הוא קשר סמנטי בין שתי ישויות המתואר על ידי נושא, סוג יחס ואובייקט.\n"
                "ענה 'כן' אם היחס מופיע, 'לא' אם אינו מופיע."
            )
            def fmt_example(ex):
                return (
                    f"טקסט: {ex['text']}\n"
                    f"נושא: {ex['subject']} | יחס: {ex['predicate']} | אובייקט: {ex['object']}\n"
                    f"תשובה: {ex['answer']}"
                )
            def fmt_query():
                return (
                    f"טקסט: {row['text']}\n"
                    f"נושא: {row['subject']} | יחס: {row['predicate']} | אובייקט: {row['object']}\n"
                    "תשובה:"
                )
        else:
            intro = (
                "קרא את הטקסט הבא וענה האם המשפט הנתון נובע ממנו.\n"
                "ענה 'כן' אם המשפט נובע מהטקסט, 'לא' אחרת."
            )
            def fmt_example(ex):
                return (
                    f"טקסט: {ex['text']}\n"
                    f"משפט: {ex['statement']}\n"
                    f"תשובה: {ex['answer']}"
                )
            def fmt_query():
                return (
                    f"טקסט: {row['text']}\n"
                    f"משפט: {row[RELATION_COLS['template']]}\n"
                    "תשובה:"
                )
    else:
        if rel_type == "triplet":
            intro = (
                "Read the following text and answer whether the given relation is expressed in it.\n"
                "A relation is described by a subject, a predicate, and an object.\n"
                "Answer 'yes' if expressed, 'no' if not."
            )
            def fmt_example(ex):
                return (
                    f"Text: {ex['text']}\n"
                    f"Subject: {ex['subject']} | Predicate: {ex['predicate']} | Object: {ex['object']}\n"
                    f"Answer: {ex['answer']}"
                )
            def fmt_query():
                return (
                    f"Text: {row['text']}\n"
                    f"Subject: {row['subject']} | Predicate: {row['predicate']} | Object: {row['object']}\n"
                    "Answer:"
                )
        else:
            intro = (
                "Read the following text and answer whether the given statement is entailed by it.\n"
                "Answer 'yes' if entailed, 'no' if not."
            )
            def fmt_example(ex):
                return (
                    f"Text: {ex['text']}\n"
                    f"Statement: {ex['statement']}\n"
                    f"Answer: {ex['answer']}"
                )
            def fmt_query():
                return (
                    f"Text: {row['text']}\n"
                    f"Statement: {row[RELATION_COLS['template']]}\n"
                    "Answer:"
                )

    parts = [intro]
    for ex in examples:
        parts.append(fmt_example(ex))
    parts.append(fmt_query())
    return "\n\n".join(parts)


def build_prompt(model_type: str, lang: str, rel_type: str, row: dict,
                 tokenizer, n_shot: int, use_cot: bool) -> str:
    if model_type == "instruct":
        return _build_prompt_instruct(lang, rel_type, row, tokenizer, n_shot, use_cot)
    return _build_prompt_base(lang, rel_type, row, n_shot)


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
        # For CoT: look at the last few lines first
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
    # Normal mode: first token wins
    tokens = _tokenise(cleaned)
    if tokens:
        if tokens[0] in _YES_TOKENS: return "1"
        if tokens[0] in _NO_TOKENS:  return "0"
    lower = cleaned.lower()
    if re.search(r"\byes\b", lower) or "כן" in lower: return "1"
    if re.search(r"\bno\b",  lower) or "לא" in lower: return "0"
    return "unknown"


# ---------------------------------------------------------------------------
# Soft score helpers
# ---------------------------------------------------------------------------

def get_yn_token_ids(tokenizer) -> tuple[list, list]:
    """Return (yes_ids, no_ids) — single-token ids for all yes/no surface forms."""
    yes_vars = ["yes", "Yes", "YES", "כן", " yes", " Yes", " כן", "▁yes", "▁Yes"]
    no_vars  = ["no",  "No",  "NO",  "לא", " no",  " No",  " לא", "▁no",  "▁No"]
    yes_ids, no_ids = set(), set()
    for v in yes_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            yes_ids.add(toks[0])
    for v in no_vars:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            no_ids.add(toks[0])
    return sorted(yes_ids), sorted(no_ids)


def compute_soft_scores_from_logits(logits, yes_ids: list, no_ids: list) -> list:
    """
    logits : (batch, vocab) float tensor — first-token logits from generate(output_scores=True).
    Returns list of floats in [0,1]: P(yes)/(P(yes)+P(no)).
    """
    if not yes_ids and not no_ids:
        return [0.5] * logits.shape[0]
    probs = torch.softmax(logits.float(), dim=-1)
    yes_t = torch.tensor(yes_ids, device=logits.device) if yes_ids else None
    no_t  = torch.tensor(no_ids,  device=logits.device) if no_ids  else None
    p_yes = probs[:, yes_t].sum(dim=-1) if yes_t is not None else torch.zeros(logits.shape[0], device=logits.device)
    p_no  = probs[:, no_t ].sum(dim=-1) if no_t  is not None else torch.zeros(logits.shape[0], device=logits.device)
    soft  = (p_yes / (p_yes + p_no + 1e-10)).cpu().tolist()
    return soft


def _soft_from_vllm_logprobs(first_token_lp: dict, yes_ids: list, no_ids: list) -> float:
    """
    first_token_lp: dict {token_id: vllm.Logprob} for the first generated token.
    Returns P(yes)/(P(yes)+P(no)), or 0.5 if neither found.
    """
    import math
    p_yes = p_no = 0.0
    for tid, lp_obj in first_token_lp.items():
        p = math.exp(lp_obj.logprob)
        if tid in yes_ids:
            p_yes += p
        elif tid in no_ids:
            p_no += p
    denom = p_yes + p_no
    return p_yes / denom if denom > 1e-10 else 0.5


# ---------------------------------------------------------------------------
# Model load / unload
# ---------------------------------------------------------------------------

def load_llm(model_id: str, log: logging.Logger):
    """
    Returns (model, tokenizer, backend) where backend is 'vllm' or 'hf'.
    Tries vLLM first (if available) for maximum throughput; falls back to HF.
    HF path uses flash_attention_2 + torch.compile when available.
    """
    log.info(f"    path: {model_id}")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── vLLM path ──────────────────────────────────────────────────────────────
    if VLLM_AVAILABLE and n_gpus > 0:
        try:
            tp = max(1, n_gpus)
            llm = vLLM(
                model=model_id,
                dtype="bfloat16",
                tensor_parallel_size=tp,
                trust_remote_code=True,
                max_model_len=LLM_MAX_INPUT_LEN + 64,
                gpu_memory_utilization=0.90,
            )
            log.info(f"    vLLM backend  (tensor_parallel_size={tp})")
            return llm, tokenizer, "vllm"
        except Exception as e:
            log.warning(f"    vLLM load failed ({e}), falling back to HF")

    # ── HF path ────────────────────────────────────────────────────────────────
    if n_gpus > 0:
        major, _ = torch.cuda.get_device_capability()
        pt_dtype = torch.bfloat16 if major >= 8 else torch.float16
        log.info(f"    HF backend  GPU={torch.cuda.get_device_name(0)}  dtype={pt_dtype}")
    else:
        pt_dtype = torch.float16
        log.warning("    No CUDA detected — running on CPU (very slow)")

    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=pt_dtype,
            trust_remote_code=True, attn_implementation="flash_attention_2",
        )
        log.info("    flash_attention_2 enabled")
    except Exception as e:
        log.warning(f"    flash_attention_2 / CausalLM failed ({type(e).__name__}), retrying")
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
            )
        except ValueError:
            # Multimodal/VLM model — not a CausalLM; use ConditionalGeneration
            log.info("    Falling back to Gemma4ForConditionalGeneration (VLM model)")
            from transformers import Gemma4ForConditionalGeneration
            try:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id, device_map="auto", torch_dtype=pt_dtype,
                    trust_remote_code=True, attn_implementation="flash_attention_2",
                )
                log.info("    flash_attention_2 enabled (VLM)")
            except Exception:
                model = Gemma4ForConditionalGeneration.from_pretrained(
                    model_id, device_map="auto", torch_dtype=pt_dtype, trust_remote_code=True,
                )

    model.eval()

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log.info("    torch.compile enabled (mode=reduce-overhead)")
        except Exception as e:
            log.warning(f"    torch.compile failed ({e})")

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    log.info(f"    ready  ({n_params:.1f}B params, device_map=auto)")
    return model, tokenizer, "hf"


def unload_llm(model, backend: str, log: logging.Logger):
    if backend == "vllm":
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("    GPU cache cleared")


def _input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Generation + soft score extraction
# ---------------------------------------------------------------------------

_BASE_STOP = ["\n", "כן", "לא", "yes", "no"]


def _stop_ids(tokenizer, stops: list) -> list:
    ids: set = set()
    for s in stops:
        for variant in [s, " " + s]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return list(ids)


def classify_rows(
    rows: list,
    model,
    tokenizer,
    model_type: str,
    backend: str,
    lang: str,
    rel_type: str,
    n_shot: int,
    use_cot: bool,
    yes_ids: list,
    no_ids: list,
    batch_size: int,
    log: logging.Logger,
    desc: str,
    max_new_tokens: int,
) -> tuple:
    """Returns (parsed_labels, raw_outputs, soft_scores)."""
    prompts = [build_prompt(model_type, lang, rel_type, r, tokenizer, n_shot, use_cot)
               for r in rows]
    n = len(prompts)

    if backend == "vllm":
        return _classify_vllm(model, prompts, yes_ids, no_ids, max_new_tokens, log, desc, n)
    else:
        return _classify_hf(model, tokenizer, model_type, prompts,
                            yes_ids, no_ids, batch_size, log, desc, max_new_tokens, use_cot)


def _classify_vllm(llm, prompts, yes_ids, no_ids, max_new_tokens, log, desc, n):
    """vLLM path: single-call batch generation with first-token logprobs for soft scores."""
    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=max(len(yes_ids) + len(no_ids) + 5, 20),
    )
    log.info(f"    [{desc}] vLLM generating {n} prompts …")
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    log.info(f"    [{desc}] done  {n/(time.time()-t0):.1f} rows/s")

    yes_set = set(yes_ids)
    no_set  = set(no_ids)
    parsed_all, raws_all, soft_all = [], [], []
    for out in outputs:
        text = out.outputs[0].text.strip()
        raws_all.append(text)
        parsed_all.append(parse_yes_no(text))
        lps = out.outputs[0].logprobs
        soft = _soft_from_vllm_logprobs(lps[0], yes_set, no_set) if lps else 0.5
        soft_all.append(soft)
    return parsed_all, raws_all, soft_all


def _classify_hf(model, tokenizer, model_type, prompts,
                 yes_ids, no_ids, batch_size, log, desc, max_new_tokens, use_cot):
    """HF path: batched generation with output_scores=True for single-pass soft scores."""
    n = len(prompts)
    parsed_all, raws_all, soft_all = [], [], []
    log_every = max(1, (n // batch_size) // 4)

    hf_stop_kwargs = {}
    if model_type == "base":
        extra = _stop_ids(tokenizer, _BASE_STOP)
        if extra:
            hf_stop_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + extra

    device = _input_device(model)
    effective_batch = batch_size  # may shrink on OOM

    for batch_idx, start in enumerate(
        tqdm(range(0, n, batch_size), desc=f"    {desc}", leave=False,
             file=TqdmToLogger(log)), 1
    ):
        batch_prompts = prompts[start : start + batch_size]

        # Process batch in sub-batches; retry with smaller size on OOM
        sub_parsed, sub_raws, sub_soft = [], [], []
        sub_start = 0
        while sub_start < len(batch_prompts):
            sub = batch_prompts[sub_start : sub_start + effective_batch]
            encodings = tokenizer(
                sub,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=LLM_MAX_INPUT_LEN,
            )
            enc_on_device = {k: v.to(device) for k, v in encodings.items()
                             if k != "token_type_ids"}
            input_len = enc_on_device["input_ids"].shape[1]
            try:
                with torch.no_grad():
                    output = model.generate(
                        **enc_on_device,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **hf_stop_kwargs,
                    )
                sub_soft.extend(compute_soft_scores_from_logits(output.scores[0], yes_ids, no_ids))
                for seq in output.sequences:
                    raw = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
                    sub_raws.append(raw)
                    sub_parsed.append(parse_yes_no(raw, cot_mode=use_cot))
                sub_start += effective_batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                effective_batch = max(1, effective_batch // 2)
                log.warning(f"    OOM — reduced batch_size to {effective_batch}, retrying")

        soft_all.extend(sub_soft)
        raws_all.extend(sub_raws)
        parsed_all.extend(sub_parsed)

        if batch_idx % log_every == 0 or (start + batch_size) >= n:
            done = min(start + batch_size, n)
            log.info(f"    progress: {done}/{n} ({100*done/n:.0f}%)")

    return parsed_all, raws_all, soft_all


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


def compute_metrics_soft(scores: list, gold: list, threshold: float) -> dict:
    TP = FP = FN = TN = 0
    for sc, g in zip(scores, gold):
        pred     = sc >= threshold
        gold_pos = g == "1"
        if pred and gold_pos:       TP += 1
        elif pred and not gold_pos: FP += 1
        elif not pred and gold_pos: FN += 1
        else:                       TN += 1
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total             if total             else 0
    precision = TP / (TP + FP)               if (TP + FP)         else 0
    recall    = TP / (TP + FN)               if (TP + FN)         else 0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) else 0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


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
    """
    F1-weighted vote.  Unknown predictions are skipped.
    weighted_score = sum(w_i * vote_i) / sum(w_i)  where vote_i∈{0,1}
    """
    results = []
    for row in rows:
        total_w = 0.0
        total_v = 0.0
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


def compute_soft_ensemble(rows: list, col_names: list) -> list:
    """Average soft scores across col_names."""
    results = []
    for row in rows:
        vals = []
        for c in col_names:
            v = row.get(c)
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        results.append(sum(vals) / len(vals) if vals else 0.5)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(run_stats: list, majority_stats: list, weighted_stats: list,
                  soft_ensemble_stats: list, summary_path: str,
                  log: logging.Logger, total: float):
    lines = []
    W = 120

    lines.append("=" * W)
    lines.append("OPEN-SOURCE LLM CLASSIFICATION SUMMARY")
    lines.append(f"Total wall time: {_fmt_duration(total)}")
    lines.append("=" * W)

    # Per-model timing
    model_times: dict = {}
    for s in run_stats:
        model_times.setdefault(s["model"], 0.0)
        model_times[s["model"]] += s["time"]
    lines.append("")
    lines.append("Per-model total inference time:")
    for m, t in model_times.items():
        r = sum(s["n_rows"] for s in run_stats if s["model"] == m)
        rps = r / t if t > 0 else float("inf")
        lines.append(f"  {m:<14}  {_fmt_duration(t)}  ({rps:.1f} rows/s across all combos)")

    # ── HARD-LABEL METRICS TABLE ──────────────────────────────────────────────
    lines.append("")
    lines.append("Hard-label metrics (per combo):")
    hdr = (
        f"  {'model':<14} {'lang':<4} {'rel':<10} {'shot':>4}  "
        f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
        f"{'unk':>4}  {'rows/s':>7}  {'time':>8}"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in run_stats:
        m   = s["metrics"]
        rps = s["n_rows"] / s["time"] if s["time"] > 0 else float("inf")
        lines.append(
            f"  {s['model']:<14} {s['lang']:<4} {s['rel_type']:<10} {s['shot']:>4}  "
            f"{m['accuracy']:>6.3f} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
            f"{m['TP']:>5} {m['FP']:>5} {m['FN']:>5} {m['TN']:>5}  "
            f"{m['unknown']:>4}  {rps:>7.1f}  {_fmt_duration(s['time']):>8}"
        )

    # ── SOFT-SCORE THRESHOLD TABLE (best threshold per combo) ─────────────────
    lines.append("")
    lines.append("Soft-score metrics (best F1 threshold per combo):")
    hdr2 = (
        f"  {'model':<14} {'lang':<4} {'rel':<10} {'shot':>4}  "
        f"{'best_thr':>8}  {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
        f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}"
    )
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))
    for s in run_stats:
        if not s["soft_metrics"]:
            continue
        best_t, best_m = max(s["soft_metrics"].items(), key=lambda kv: kv[1]["f1"])
        lines.append(
            f"  {s['model']:<14} {s['lang']:<4} {s['rel_type']:<10} {s['shot']:>4}  "
            f"{best_t:>8.2f}  "
            f"{best_m['accuracy']:>6.3f} {best_m['precision']:>6.3f} "
            f"{best_m['recall']:>6.3f} {best_m['f1']:>6.3f}  "
            f"{best_m['TP']:>5} {best_m['FP']:>5} {best_m['FN']:>5} {best_m['TN']:>5}"
        )

    # ── UNIFORM MAJORITY TABLE ────────────────────────────────────────────────
    if majority_stats:
        lines.append("")
        lines.append("Uniform majority-vote columns:")
        hdr3 = (
            f"  {'column':<50}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  "
            f"{'unk':>4}  {'voters':>6}"
        )
        lines.append(hdr3)
        lines.append("  " + "-" * (len(hdr3) - 2))
        for s in majority_stats:
            m = s["metrics"]
            lines.append(
                f"  {s['col']:<50}  "
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
                f"  {s['col']:<50}  "
                f"acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  "
                f"unk={m['unknown']}  voters={s['n_voters']}"
            )

    # ── SOFT ENSEMBLE TABLE ───────────────────────────────────────────────────
    if soft_ensemble_stats:
        lines.append("")
        lines.append("Soft-score ensemble columns (best F1 threshold):")
        hdr4 = (
            f"  {'column':<50}  {'best_thr':>8}  "
            f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}  "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}  {'voters':>6}"
        )
        lines.append(hdr4)
        lines.append("  " + "-" * (len(hdr4) - 2))
        for s in soft_ensemble_stats:
            best_t, best_m = max(s["metrics_by_threshold"].items(), key=lambda kv: kv[1]["f1"])
            lines.append(
                f"  {s['col']:<50}  {best_t:>8.2f}  "
                f"{best_m['accuracy']:>6.3f} {best_m['precision']:>6.3f} "
                f"{best_m['recall']:>6.3f} {best_m['f1']:>6.3f}  "
                f"{best_m['TP']:>5} {best_m['FP']:>5} {best_m['FN']:>5} {best_m['TN']:>5}  "
                f"{s['n_voters']:>6}"
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
                         label_col: str, error_path: str, log: logging.Logger,
                         max_examples: int = 10):
    W = 110
    n_rows = len(rows)
    n_pos  = gold.count("1")

    # Identify the best combo by F1 for focused examples
    best_stat = max(run_stats, key=lambda s: s["metrics"]["f1"])
    best_clean_col = _col("llm_clean", best_stat["model"], best_stat["lang"],
                          best_stat["rel_type"], best_stat["shot"], False)

    lines = []
    lines.append("=" * W)
    lines.append("  OPEN-SOURCE LLM ERROR ANALYSIS")
    lines.append(f"  Total rows         : {n_rows}")
    lines.append(f"  Positive / Negative: {n_pos} / {n_rows-n_pos}  "
                 f"({100*n_pos/n_rows:.1f}% / {100*(n_rows-n_pos)/n_rows:.1f}%)")
    lines.append(f"  Best combo (F1={best_stat['metrics']['f1']:.3f}): "
                 f"{best_stat['model']} / {best_stat['lang']} / {best_stat['rel_type']} / "
                 f"{best_stat['shot']}-shot / no-CoT")
    lines.append("=" * W)

    # ── Section 1: per-combo error counts ─────────────────────────────────────
    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 1 — PER-COMBO ERROR COUNTS")
    lines.append("=" * W)
    hdr = (
        f"  {'combo':<55} {'F1':>6} {'FP':>5} {'FN':>5} {'unk':>5}  "
        "FP-rate  FN-rate"
    )
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for s in sorted(run_stats, key=lambda x: x["metrics"]["f1"], reverse=True):
        m     = s["metrics"]
        combo = f"{s['model']}/{s['lang']}/{s['rel_type']}/{s['shot']}s"
        fp_rate = m["FP"] / (m["FP"] + m["TN"]) if (m["FP"] + m["TN"]) else 0
        fn_rate = m["FN"] / (m["FN"] + m["TP"]) if (m["FN"] + m["TP"]) else 0
        lines.append(
            f"  {combo:<55} {m['f1']:>6.3f} {m['FP']:>5} {m['FN']:>5} {m['unknown']:>5}"
            f"  {fp_rate:.3f}    {fn_rate:.3f}"
        )

    # ── Section 2: Shot-level comparison (aggregated) ─────────────────────────
    lines.append("")
    lines.append("=" * W)
    lines.append("  SECTION 2 — SHOT-LEVEL F1 COMPARISON")
    lines.append("=" * W)
    shot_f1: dict = defaultdict(list)
    for s in run_stats:
        shot_f1[s["shot"]].append(s["metrics"]["f1"])
    for shot, vals in sorted(shot_f1.items()):
        mean_f1 = sum(vals) / len(vals)
        lines.append(f"  {shot}-shot:  mean F1={mean_f1:.4f}  (across {len(vals)} combos)")

    # ── Section 3: FP/FN examples for best combo ──────────────────────────────
    for section, pred_val, gold_val, title in [
        ("FALSE POSITIVES (pred=1, gold=0)", "1", "0", "SECTION 3 — FALSE POSITIVES  [best combo]"),
        ("FALSE NEGATIVES (pred=0, gold=1)", "0", "1", "SECTION 4 — FALSE NEGATIVES  [best combo]"),
    ]:
        examples = [(i, r) for i, r in enumerate(rows)
                    if r.get(best_clean_col) == pred_val and r[label_col] == gold_val]
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

    # ── Section 6: Hard rows — all non-CoT combos wrong ───────────────────────
    nc_clean_cols = [_col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
                     for s in run_stats]
    hard = [i for i, r in enumerate(rows)
            if all(r.get(c, "unknown") != gold[i] for c in nc_clean_cols if c in r)]
    lines.append("")
    lines.append("=" * W)
    lines.append(f"  SECTION 5 — HARD ROWS: all combos wrong — {len(hard)} ({100*len(hard)/n_rows:.1f}%)")
    lines.append("=" * W)
    for i in hard[:max_examples]:
        row  = rows[i]
        snip = row["text"].replace("\n", " ")[:150]
        lines.append(f"  [row {i}]  gold={gold[i]}")
        lines.append(f"    text: {snip!r}")
    if len(hard) > max_examples:
        lines.append(f"  ... and {len(hard) - max_examples} more")

    # ── Section 7: Soft-score confidence distribution for best-soft combo ─────
    best_soft = max((s for s in run_stats if s["soft_metrics"]),
                    key=lambda s: max(m["f1"] for m in s["soft_metrics"].values()),
                    default=None)
    if best_soft:
        score_col = _col("llm_score", best_soft["model"], best_soft["lang"],
                         best_soft["rel_type"], best_soft["shot"], False)
        lines.append("")
        lines.append("=" * W)
        lines.append(f"  SECTION 6 — SOFT SCORE DISTRIBUTION  [{score_col}]")
        lines.append("=" * W)
        bins = [(0.0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"),
                (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, "≥0.9")]
        for lo, hi, label in bins:
            pos_n = sum(1 for r, g in zip(rows, gold)
                        if lo <= float(r.get(score_col, 0.5)) < hi and g == "1")
            neg_n = sum(1 for r, g in zip(rows, gold)
                        if lo <= float(r.get(score_col, 0.5)) < hi and g == "0")
            lines.append(f"  [{label}]  pos={pos_n}  neg={neg_n}")

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

    # Group by predicate
    pred_groups: dict = defaultdict(list)
    for i, row in enumerate(rows):
        pred_groups[row.get(predicate_col, "UNKNOWN")].append(i)

    # Take top combos by F1 for detailed analysis (max 10 to keep output readable)
    top_combos = sorted(run_stats, key=lambda s: s["metrics"]["f1"], reverse=True)[:10]

    lines = []
    lines.append("=" * W)
    lines.append("  PREDICATE-STRATIFIED METRICS")
    lines.append(f"  Predicates with >= {MIN_PRED_EXAMPLES} examples: "
                 f"{sum(1 for v in pred_groups.values() if len(v) >= MIN_PRED_EXAMPLES)}")
    lines.append("=" * W)

    # ── Section 1: overall predicate stats ────────────────────────────────────
    lines.append("")
    lines.append("Predicate distribution (top 20 by count):")
    pred_pos: dict = {}
    for pred, idxs in pred_groups.items():
        pred_pos[pred] = sum(1 for i in idxs if gold[i] == "1")
    sorted_preds = sorted(pred_groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    hdr0 = f"  {'predicate':<35} {'n':>5} {'pos':>5} {'neg':>5} {'pos%':>6}"
    lines.append(hdr0)
    lines.append("  " + "-" * (len(hdr0) - 2))
    for p, idxs in sorted_preds[:20]:
        n  = len(idxs)
        ps = pred_pos[p]
        ns = n - ps
        lines.append(f"  {p:<35} {n:>5} {ps:>5} {ns:>5} {100*ps/n:>5.0f}%")

    # ── Section 2: per-predicate F1 for top combos ────────────────────────────
    lines.append("")
    lines.append("=" * W)
    lines.append("  PER-PREDICATE F1  (predicates with ≥ {} examples, top combos)".format(MIN_PRED_EXAMPLES))
    lines.append("=" * W)

    combo_labels = []
    for s in top_combos:
        combo_labels.append(
            f"{s['model']}/{s['lang']}/{s['rel_type']}/{s['shot']}s"
        )

    # Header
    hdr_parts = [f"  {'predicate':<35} {'n':>5}"]
    for lbl in combo_labels:
        hdr_parts.append(f" {lbl[:16]:>16}")
    lines.append("".join(hdr_parts))
    lines.append("  " + "-" * (len("".join(hdr_parts)) - 2))

    pred_f1_summary: list = []  # (pred, mean_f1) for sorting
    for pred, idxs in sorted_preds:
        if len(idxs) < MIN_PRED_EXAMPLES:
            continue
        pred_golds = [gold[i] for i in idxs]
        combo_f1s  = []
        row_parts  = [f"  {pred:<35} {len(idxs):>5}"]
        for s in top_combos:
            clean_col = _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
            pred_labels = [rows[i].get(clean_col, "unknown") for i in idxs]
            m = compute_metrics_hard(pred_labels, pred_golds)
            combo_f1s.append(m["f1"])
            row_parts.append(f" {m['f1']:>16.3f}")
        mean_f1 = sum(combo_f1s) / len(combo_f1s) if combo_f1s else 0
        pred_f1_summary.append((pred, mean_f1, len(idxs)))
        lines.append("".join(row_parts))

    # ── Section 3: easiest / hardest predicates ───────────────────────────────
    lines.append("")
    lines.append("=" * W)
    lines.append("  EASIEST PREDICATES (highest mean F1 across top combos):")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: -x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")
    lines.append("")
    lines.append("  HARDEST PREDICATES (lowest mean F1 across top combos):")
    for pred, mean_f1, n in sorted(pred_f1_summary, key=lambda x: x[1])[:10]:
        lines.append(f"    {pred:<35}  mean F1={mean_f1:.3f}  n={n}")

    # ── Section 4: shot comparison per predicate ──────────────────────────────
    lines.append("")
    lines.append("=" * W)
    lines.append("  SHOT COMPARISON PER PREDICATE  (mean F1 across all models, lang, rel)")
    lines.append("=" * W)
    hdr5 = f"  {'predicate':<35} {'n':>5}" + "".join(f" {'shot'+str(k)+'s':>8}" for k in [0, 2, 5])
    lines.append(hdr5)
    lines.append("  " + "-" * (len(hdr5) - 2))
    for pred, _, n in sorted(pred_f1_summary, key=lambda x: -x[2])[:20]:
        idxs = pred_groups[pred]
        pred_golds = [gold[i] for i in idxs]
        row_parts = [f"  {pred:<35} {n:>5}"]
        for shot_k in [0, 2, 5]:
            shot_runs = [s for s in run_stats if s["shot"] == shot_k]
            f1s = []
            for s in shot_runs:
                clean_col = _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
                pred_labels = [rows[i].get(clean_col, "unknown") for i in idxs]
                m = compute_metrics_hard(pred_labels, pred_golds)
                f1s.append(m["f1"])
            mean_f1 = sum(f1s) / len(f1s) if f1s else 0
            row_parts.append(f" {mean_f1:>8.3f}")
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

    protected_dirs  = {"ARCHIVE", "opensource_LLM_clean", "NLI_clean", "dataset_statistics"}
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
        description="Enhanced open-source LLM relation classifier (v2)"
    )
    parser.add_argument("--input",       default=INPUT_FILE)
    parser.add_argument("--output",      default=OUTPUT_FILE)
    parser.add_argument("--log",         default=LOG_FILE)
    parser.add_argument("--summary",     default=SUMMARY_FILE)
    parser.add_argument("--error",       default=ERROR_FILE)
    parser.add_argument("--pred-file",   default=PRED_FILE)
    parser.add_argument("--label-col",   default=LABEL_COL)
    parser.add_argument("--batch-size",  type=int, default=LLM_BATCH_SIZE)
    parser.add_argument("--shots",       type=int, nargs="+", default=SHOT_CONFIGS,
                        help="Shot counts to evaluate (default: 0 2 5)")
    parser.add_argument("--debug",       type=int, default=0,
                        help="Run on first N rows only (0 = full dataset)")
    parser.add_argument("--no-archive",  action="store_true",
                        help="Skip archiving old outputs/")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path  = os.path.join(base, args.input)
    output_path = os.path.join(base, args.output)
    log_path    = os.path.join(base, args.log)
    sum_path    = os.path.join(base, args.summary)
    err_path    = os.path.join(base, args.error)
    pred_path   = os.path.join(base, args.pred_file)

    for p in (output_path, log_path, sum_path, err_path, pred_path):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    log = setup_logger(log_path)
    wall_start = time.time()

    # Archive old outputs
    if not args.no_archive:
        archive_old_outputs(base, log)

    # GPU info
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info("=" * 70)
    log.info("clean_dataset_with_opensource_llm.py  v2  started")
    log.info(f"  input:      {input_path}")
    log.info(f"  output dir: {os.path.join(base, OUTPUT_SUBDIR)}")
    log.info(f"  label col:  {args.label_col}")
    log.info(f"  GPUs:       {n_gpus}")
    log.info(f"  backend:    {'vLLM' if VLLM_AVAILABLE and n_gpus > 0 else 'HuggingFace'}")
    log.info(f"  shots:      {args.shots}")
    log.info(f"  debug rows: {args.debug if args.debug else 'all'}")
    log.info(f"  models ({len(LLM_MODELS)}):")
    for m in LLM_MODELS:
        log.info(f"    [{m['tag']}]  {m['id']}  (type={m['type']})")
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

    available = set(rows[0].keys())
    for col in (args.label_col, PREDICATE_COL):
        if col not in available:
            raise ValueError(f"Column '{col}' not found. Available: {sorted(available)}")
    for rel_col in RELATION_COLS.values():
        if rel_col not in available:
            raise ValueError(f"Relation column '{rel_col}' not found.")

    # Resume: merge previously-computed columns from checkpoint into rows
    if os.path.exists(output_path):
        log.info(f"[resume]  checkpoint found: {output_path}")
        with open(output_path, encoding="utf-8-sig") as f:
            ckpt_rows = list(csv.DictReader(f))
        if args.debug:
            ckpt_rows = ckpt_rows[:args.debug]
        if len(ckpt_rows) == n_rows:
            ckpt_extra = set(ckpt_rows[0].keys()) - set(rows[0].keys())
            if ckpt_extra:
                for r, cr in zip(rows, ckpt_rows):
                    for col in ckpt_extra:
                        r[col] = cr[col]
                log.info(f"[resume]  merged {len(ckpt_extra)} columns from checkpoint — will skip completed combos")
            else:
                log.info("[resume]  checkpoint has no extra columns — starting fresh")
        else:
            log.warning(f"[resume]  row-count mismatch ({len(ckpt_rows)} vs {n_rows}) — ignoring checkpoint")

    run_stats: list = []

    # ── Main loop ──────────────────────────────────────────────────────────────
    for model_idx, model_cfg in enumerate(LLM_MODELS, 1):
        tag        = model_cfg["tag"]
        model_id   = model_cfg["id"]
        model_type = model_cfg["type"]

        # Skip model load entirely if every combo for this model is already in the checkpoint
        _model_combos = [
            _col("llm_clean", tag, la, rt, ns, False)
            for la in PROMPT_LANGS for rt in RELATION_TYPES for ns in args.shots
        ]
        if all(rows[0].get(c, "") != "" for c in _model_combos):
            log.info(f"[model {model_idx}/{len(LLM_MODELS)}]  {tag}  — all combos in checkpoint, skipping")
            for la in PROMPT_LANGS:
                for rt in RELATION_TYPES:
                    for ns in args.shots:
                        _cc = _col("llm_clean", tag, la, rt, ns, False)
                        _sc = _col("llm_score", tag, la, rt, ns, False)
                        _p  = [r.get(_cc, "") if r.get(_cc, "") in ("0", "1") else "unknown" for r in rows]
                        _ss = [float(r[_sc]) if r.get(_sc, "") != "" else 0.5 for r in rows]
                        _m  = compute_metrics_hard(_p, gold)
                        _sm = {thr: compute_metrics_soft(_ss, gold, thr) for thr in SOFT_THRESHOLDS}
                        run_stats.append({"model": tag, "lang": la, "rel_type": rt, "shot": ns,
                                          "time": 0.0, "n_rows": n_rows, "metrics": _m, "soft_metrics": _sm})
                        log.info(f"  [skip]  {tag}/{la}/{rt}/{ns}s/nc  F1={_m['f1']:.3f}")
            continue

        log.info("-" * 70)
        log.info(f"[model {model_idx}/{len(LLM_MODELS)}]  {tag}  ({model_type})")
        t_model = time.time()
        model, tokenizer, backend = load_llm(model_id, log)
        yes_ids, no_ids  = get_yn_token_ids(tokenizer)
        log.info(f"    yes_ids ({len(yes_ids)}): {yes_ids[:8]}")
        log.info(f"    no_ids  ({len(no_ids)}):  {no_ids[:8]}")

        for lang in PROMPT_LANGS:
            for rel_type in RELATION_TYPES:
                for n_shot in args.shots:
                        clean_col = _col("llm_clean", tag, lang, rel_type, n_shot, False)
                        raw_col   = _col("llm_raw",   tag, lang, rel_type, n_shot, False)
                        score_col = _col("llm_score", tag, lang, rel_type, n_shot, False)
                        desc      = f"{tag}/{lang}/{rel_type}/{n_shot}s/nc"

                        # Skip combo if already computed (partial-model resume)
                        if rows[0].get(clean_col, "") != "":
                            log.info(f"  [skip]  {desc}  (already in checkpoint)")
                            _p  = [r.get(clean_col, "") if r.get(clean_col, "") in ("0", "1") else "unknown" for r in rows]
                            _ss = [float(r[score_col]) if r.get(score_col, "") != "" else 0.5 for r in rows]
                            _m  = compute_metrics_hard(_p, gold)
                            _sm = {thr: compute_metrics_soft(_ss, gold, thr) for thr in SOFT_THRESHOLDS}
                            run_stats.append({"model": tag, "lang": lang, "rel_type": rel_type,
                                              "shot": n_shot, "time": 0.0, "n_rows": n_rows,
                                              "metrics": _m, "soft_metrics": _sm})
                            continue

                        log.info("*" * 70)
                        log.info(f"  COMBO: {desc}")
                        log.info(f"    output cols: {clean_col} | {raw_col} | {score_col}")

                        # Log example prompt
                        ex_prompt = build_prompt(model_type, lang, rel_type,
                                                 rows[0], tokenizer, n_shot, False)
                        log.info(f"    example prompt (row 0):\n{'-'*40}\n"
                                 f"{ex_prompt[:600]}\n{'-'*40}")

                        t0 = time.time()
                        parsed, raws, soft_scores = classify_rows(
                            rows, model, tokenizer, model_type, backend,
                            lang, rel_type, n_shot, False,
                            yes_ids, no_ids,
                            batch_size=args.batch_size, log=log,
                            desc=desc, max_new_tokens=LLM_MAX_NEW_TOKENS,
                        )
                        elapsed = time.time() - t0

                        for r, label, raw, sc in zip(rows, parsed, raws, soft_scores):
                            r[clean_col] = label
                            r[raw_col]   = raw
                            r[score_col] = f"{sc:.4f}"

                        metrics = compute_metrics_hard(parsed, gold)
                        soft_metrics = {
                            thr: compute_metrics_soft(soft_scores, gold, thr)
                            for thr in SOFT_THRESHOLDS
                        }

                        rps    = n_rows / elapsed if elapsed > 0 else float("inf")
                        m = metrics
                        log.info(f"  {'='*66}")
                        log.info(f"  RESULT  {desc}")
                        log.info(f"  {'─'*66}")
                        log.info(f"  acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
                                 f"rec={m['recall']:.3f}  f1={m['f1']:.3f}")
                        log.info(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}  "
                                 f"unk={m['unknown']}  {rps:.1f} rows/s  {_fmt_duration(elapsed)}")
                        best_soft_thr, best_soft_m = max(soft_metrics.items(),
                                                         key=lambda kv: kv[1]["f1"])
                        log.info(f"  best soft: thresh={best_soft_thr}  "
                                 f"f1={best_soft_m['f1']:.3f}  "
                                 f"prec={best_soft_m['precision']:.3f}  "
                                 f"rec={best_soft_m['recall']:.3f}")
                        log.info(f"  {'='*66}")

                        log.info(f"  examples (first 3):")
                        for r, label, raw, sc in zip(rows[:3], parsed[:3], raws[:3], soft_scores[:3]):
                            log.info(f"    hyp={r[RELATION_COLS[rel_type]]!r}")
                            log.info(f"    raw={raw!r}  parsed={label}  score={sc:.3f}  gold={r[args.label_col]}")

                        run_stats.append({
                            "model": tag, "lang": lang, "rel_type": rel_type,
                            "shot": n_shot,
                            "time": elapsed, "n_rows": n_rows,
                            "metrics": metrics, "soft_metrics": soft_metrics,
                        })

        model_total = time.time() - t_model
        unload_llm(model, backend, log)
        log.info(f"  [{tag}] total time: {_fmt_duration(model_total)}")

        # Checkpoint save after each model
        t0 = time.time()
        fieldnames = list(dict.fromkeys(rows[0].keys()))
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"[checkpoint save]  {n_rows} rows → {output_path}  ({_fmt_duration(time.time()-t0)})")

    # ── Majority / ensemble voting ─────────────────────────────────────────────
    log.info("=" * 70)
    log.info("[voting]  computing majority / ensemble columns")

    majority_stats:       list = []
    weighted_stats:       list = []
    soft_ensemble_stats:  list = []

    # Build all combos list
    all_clean_cols = [
        _col("llm_clean", s["model"], s["lang"], s["rel_type"], s["shot"], False)
        for s in run_stats
    ]
    all_f1s = [s["metrics"]["f1"] for s in run_stats]

    # 1. Per (lang, rel_type, shot, cot): uniform majority across all models
    seen_settings: set = set()
    for s in run_stats:
        key = (s["lang"], s["rel_type"], s["shot"], False)
        if key in seen_settings:
            continue
        seen_settings.add(key)
        src_cols = [_col("llm_clean", mc["tag"], s["lang"], s["rel_type"], s["shot"], False)
                    for mc in LLM_MODELS
                    if _col("llm_clean", mc["tag"], s["lang"], s["rel_type"], s["shot"], False)
                    in rows[0]]
        if not src_cols:
            continue
        col_name = _col("llm_majority", "all_models", s["lang"], s["rel_type"], s["shot"], False)
        labels   = compute_majority(rows, src_cols)
        for r, lbl in zip(rows, labels):
            r[col_name] = lbl
        m = compute_metrics_hard(labels, gold)
        majority_stats.append({"col": col_name, "n_voters": len(src_cols), "metrics": m})
        log.info(f"  {col_name}: f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    # 2. Best-per-model (0-shot no-CoT baseline): pick each model's best combo by F1
    best_per_model_cols, best_per_model_w = [], []
    for mc in LLM_MODELS:
        model_runs = [s for s in run_stats if s["model"] == mc["tag"]]
        if not model_runs:
            continue
        best = max(model_runs, key=lambda s: s["metrics"]["f1"])
        c = _col("llm_clean", mc["tag"], best["lang"], best["rel_type"], best["shot"], False)
        best_per_model_cols.append(c)
        best_per_model_w.append(best["metrics"]["f1"])
        log.info(f"  best combo for {mc['tag']}: {best['lang']}/{best['rel_type']}/"
                 f"{best['shot']}s/nc  F1={best['metrics']['f1']:.3f}")

    # Uniform majority across best-per-model
    col = "llm_majority_best_per_model"
    labels = compute_majority(rows, best_per_model_cols)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    majority_stats.append({"col": col, "n_voters": len(best_per_model_cols), "metrics": m})
    log.info(f"  {col}: f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    # F1-weighted majority across best-per-model
    col = "llm_wmajority_best_per_model"
    labels = compute_weighted_majority(rows, best_per_model_cols, best_per_model_w)
    for r, lbl in zip(rows, labels):
        r[col] = lbl
    m = compute_metrics_hard(labels, gold)
    weighted_stats.append({"col": col, "n_voters": len(best_per_model_cols), "metrics": m})
    log.info(f"  {col}: f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    # 3. All combos: uniform + F1-weighted
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
    log.info(f"  {col}: f1={m['f1']:.3f}  prec={m['precision']:.3f}  rec={m['recall']:.3f}")

    # 4. Soft ensembles — per (lang, rel, shot, cot) + all-nocot + all
    for key in seen_settings:
        lang, rel_type, shot, cot = key
        src_score_cols = [
            _col("llm_score", mc["tag"], lang, rel_type, shot, cot)
            for mc in LLM_MODELS
            if _col("llm_score", mc["tag"], lang, rel_type, shot, cot) in rows[0]
        ]
        if not src_score_cols:
            continue
        ens_col = _col("llm_soft_ens", "all_models", lang, rel_type, shot, cot)
        scores  = compute_soft_ensemble(rows, src_score_cols)
        for r, sc in zip(rows, scores):
            r[ens_col] = f"{sc:.4f}"
        m_by_t = {t: compute_metrics_soft(scores, gold, t) for t in SOFT_THRESHOLDS}
        soft_ensemble_stats.append({"col": ens_col, "n_voters": len(src_score_cols),
                                    "metrics_by_threshold": m_by_t})

    # All no-CoT soft ensemble
    nocot_score_cols = [
        _col("llm_score", s["model"], s["lang"], s["rel_type"], s["shot"], False)
        for s in run_stats
        if _col("llm_score", s["model"], s["lang"], s["rel_type"], s["shot"], False) in rows[0]
    ]
    if nocot_score_cols:
        ens_col = "llm_soft_ens_all_nocot"
        scores  = compute_soft_ensemble(rows, nocot_score_cols)
        for r, sc in zip(rows, scores):
            r[ens_col] = f"{sc:.4f}"
        m_by_t = {t: compute_metrics_soft(scores, gold, t) for t in SOFT_THRESHOLDS}
        soft_ensemble_stats.append({"col": ens_col, "n_voters": len(nocot_score_cols),
                                    "metrics_by_threshold": m_by_t})
        best_t, best_m = max(m_by_t.items(), key=lambda kv: kv[1]["f1"])
        log.info(f"  {ens_col}: best f1={best_m['f1']:.3f}@{best_t}")

    # ── Final CSV save ─────────────────────────────────────────────────────────
    t0 = time.time()
    fieldnames = list(dict.fromkeys(rows[0].keys()))
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"[save]  {n_rows} rows written to {output_path}  ({_fmt_duration(time.time()-t0)})")

    # ── Reports ────────────────────────────────────────────────────────────────
    total = time.time() - wall_start
    write_summary(run_stats, majority_stats, weighted_stats, soft_ensemble_stats,
                  sum_path, log, total)
    write_error_analysis(rows, run_stats, gold, args.label_col, err_path, log)
    write_predicate_stratified(rows, run_stats, gold, PREDICATE_COL, pred_path, log)

    log.info("=" * 70)
    log.info(f"DONE.  Total wall time: {_fmt_duration(total)}")
    log.info(f"  classified.csv      → {output_path}")
    log.info(f"  summary.txt         → {sum_path}")
    log.info(f"  error_analysis.txt  → {err_path}")
    log.info(f"  predicate_analysis  → {pred_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()