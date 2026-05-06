# Paper Outline — Hebrew RE Data Cleaning

## 1. Introduction
- Distant supervision for RE is noisy; Hebrew is low-resource
- We construct, annotate, and clean a Hebrew RE dataset from Wikidata + Wikipedia
- Overview of contributions

## 2. Dataset Construction
- Extracting (subject, predicate, object) triplets from Wikidata for Hebrew entities
- Matching triplets to Hebrew Wikipedia passages (CROCO pipeline)
- Statistics on the full silver dataset: size, predicate distribution, text length

## 3. Gold Annotation
- Sampling 500 examples for human annotation
- Annotation task and guidelines: is the relation explicitly stated in the text?
- Inter-annotator agreement
- Gold dataset statistics: class balance, predicate-level noise rates, hard predicates

## 4. NLI Finetuning
- Finetuning multilingual encoders (mmBERT, me5-large, XLM-R, NeoDictaBERT) on Hebrew NLI (mrl_eval)
- Hypothesis templates: basic / structured / LLM-generated
- Results on gold set: precision, recall, F1; ensemble; error analysis

## 5. LLM-Based Classification
- Open-source LLMs (Gemma3, Qwen3, Aya, DictaLM) with Hebrew/English prompts
- API LLMs (GPT-5.1, Gemini 2.5 Pro) with majority voting
- Results on gold set: comparison to NLI; cost and runtime trade-offs; error analysis

## 6. Silver Dataset Analysis
- Applying best classifier to the full ~3.1M rows
- Statistics on predicted label distribution per predicate
- Comparison of silver (distant-supervision) vs. predicted labels — noise rate estimates

## 7. Conclusion
- LLMs outperform NLI encoders; world-knowledge leakage is the main failure mode
- Future: text-grounded evidence extraction, predicate-specific models
