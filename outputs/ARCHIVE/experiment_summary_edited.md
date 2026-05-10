
**Gold set:** 500 manually labelled examples; 355 positive (71%) / 145 negative (29%)  
**Full dataset:** ~3.1M rows (noisy distant-supervision labels, no gold annotation)

## Dataset Snapshot

| Statistic | Value |
|---|---|
| Total gold examples | 500 |
| Positive (relation present) | 355 (71%) |
| Negative (relation absent) | 145 (29%) |
| Unique predicates | 97 |
| Mean text length | ~79 words / ~492 chars |
| Most frequent predicate | מדינה (country) — 79 examples |

**Predicate-level signal:** Several predicates are near-always positive in the gold set (e.g., אזרחות 100%, אמן מבצע 100%, מקום לידה 100%, חלוקה משנית 100%), while others are near-always negative (ההפך מ 0%, שפה שבשימוש 6%, שפה רשמית 16%). This is a direct consequence of distant-supervision noise — factually-true but unmentioned relations cluster in specific predicate types.

---

## Method 1: Finetuned NLI Encoders

### What was tested

- **4 multilingual encoder models**, all finetuned on Hebrew NLI data:
  - mmBERT-base, multilingual-e5-large (me5large), NeoDictaBERT, XLM-RoBERTa-large
- **3 hypothesis templates** per model (12 model×hypothesis combinations):
  - `basic_relation` — raw triplet concatenation ("X predicate Y")
  - `template_relation` — structured sentence ("X is the Y of Z")
  - `llm_relation` — LLM-generated natural Hebrew sentence
- **4 confidence thresholds** evaluated: 0.6, 0.7, 0.8, 0.9
- **Ensemble** variants: per-hypothesis average, per-model best, full ensemble (all 12)
- Runtime: ~11 min total (local GPU); inference per model: 9 s – 1 min 18 s

### Results (threshold = 0.7)

**Per model, best hypothesis (F1 ranked):**

| Model | Best Hyp | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|
| me5large | template_relation | 0.704 | 0.708 | 0.992 | **0.826** |
| me5large | basic_relation | 0.704 | 0.709 | 0.989 | 0.826 |
| mmbert | basic_relation | 0.696 | 0.707 | 0.977 | 0.820 |
| xlmroberta | template_relation | 0.698 | 0.707 | 0.980 | 0.822 |
| neodictabert | template_relation | 0.694 | 0.707 | 0.972 | 0.819 |

**Ensemble variants (threshold = 0.6):**

| Ensemble | Acc | Prec | Rec | F1 | Voters |
|---|---|---|---|---|---|
| nli_ensemble_template_relation | 0.710 | 0.710 | **1.000** | **0.830** | 4 |
| nli_ensemble_llm_relation | 0.710 | 0.710 | 1.000 | 0.830 | 4 |
| nli_ensemble_best_per_model | 0.710 | 0.710 | 1.000 | 0.830 | 4 |
| nli_ensemble_all (12 combos) | 0.710 | 0.710 | 1.000 | 0.830 | 12 |

### Key conclusions

- **Recall-dominated behaviour:** All models achieve near-perfect recall (0.96–1.0) but low precision (~0.70), indicating a strong positive bias. The models almost never miss a true positive but generate many false positives.
- **Hypothesis template has negligible effect:** F1 differences across the three hypothesis formats are <0.01 for every model. The NLI framing is largely invariant to how the hypothesis is phrased.
- **High inter-model agreement (96–99%):** Models agree on predictions in almost every case — ensemble averaging does not meaningfully reduce errors. The 4 models effectively make the same errors.
- **Ensemble recall = 1.0 but accuracy is capped at 71%:** At threshold 0.6, the ensemble never predicts 0 — it classifies everything as positive. This is close to a majority-class baseline.
- **Hard error class — confident false positives:** 133 rows (26.6%) are wrong for every combination. Nearly all are false positives where the relation is *factually true* but not explicitly stated in the text. Models apply world knowledge rather than grounding in the text.
- **Longer texts → lower accuracy:** Accuracy drops from 0.76 (texts 100–200 chars) to 0.66–0.68 (400+ chars), suggesting models struggle with longer, denser passages.
- **NLI finetuning may have introduced over-sensitivity:** Models consistently predict "entailment" regardless of whether the hypothesis is grounded in the premise, suggesting the NLI training signal did not transfer well to this task's strict localization requirement.

---

## Method 2: Open-Source LLMs

### What was tested

- **4 large language models** (loaded locally with 2 GPUs, bfloat16):
  - Qwen3-31B-it, DictaLM-3.0-24B-Base, Cohere Aya-32B, Gemma-3-27B-it
- **2 prompt languages** × **2 relation formats** = 4 prompt variants per model:
  - Languages: Hebrew (he), English (en)
  - Formats: `triplet` (raw triplet), `template` (structured sentence)
- **Majority vote** ensembles over all/subset of models
- Runtime: ~1 h 12 min total (GPU inference dominates)

### Results (majority vote, best configurations)

**Per model, best single configuration (F1 ranked):**

| Model | Lang | Format | Acc | Prec | Rec | F1 |
|---|---|---|---|---|---|---|
| gemma3 | he | triplet | 0.834 | 0.845 | 0.938 | **0.889** |
| gemma3 | en | template | 0.828 | 0.851 | 0.918 | 0.883 |
| cohere_aya32b | en | triplet | 0.812 | 0.804 | 0.972 | 0.880 |
| qwen3 | en | triplet | 0.786 | 0.802 | 0.927 | 0.860 |
| dictalm3 | en | triplet | 0.814 | 0.883 | 0.851 | 0.867 |
| dictalm3 | he | triplet | 0.760 | 0.912 | 0.732 | 0.812 |
| dictalm3 | he | template | 0.582 | 0.987 | 0.417 | **0.586** |

**Majority vote ensembles:**

| Ensemble | Acc | Prec | Rec | F1 | Unk | Voters |
|---|---|---|---|---|---|---|
| llm_majority_best_per_model | 0.840 | 0.857 | 0.930 | **0.892** | 39 | 4 |
| llm_majority_en_triplet | 0.832 | 0.843 | 0.938 | 0.888 | 36 | 4 |
| llm_majority_he_triplet | 0.834 | 0.876 | 0.893 | 0.884 | 60 | 4 |
| llm_majority_all (16 combos) | 0.834 | 0.852 | 0.927 | 0.888 | 20 | 16 |

### Key conclusions

- **LLMs substantially outperform NLI encoders:** Best F1 of 0.889 (Gemma3) vs. 0.826 (NLI), and best ensemble F1 of 0.892 vs. 0.830, with much better precision (0.857 vs. 0.710).
- **Gemma3 is the clear top performer:** Both Hebrew and English prompts yield F1 > 0.875. The instruct-tuned model is better calibrated for this binary task.
- **DictaLM (base model) collapses in Hebrew template format (F1 = 0.586):** A base (non-instruct) model is poorly suited to structured prompt following. Hebrew template format caused DictaLM to almost always predict negative (recall = 0.417, precision = 0.987). English triplet works reasonably (F1 = 0.867).
- **English prompts tend to outperform Hebrew** for most models, including Qwen3 and DictaLM. Only Gemma3 performs equally well in Hebrew. This may reflect stronger English instruction-following training.
- **Cohere Aya has the highest recall (0.969–0.972) but weaker precision (~0.80):** Good for high-coverage use cases but not for precision-oriented data cleaning.
- **Unknown / abstention rate:** ~20–60 rows per ensemble abstain due to disagreement. The `llm_majority_all` ensemble reduces unknowns to 20 by leveraging 16 votes.
- **Runtime trade-off:** 1h 12 min for 500 examples on 2 GPUs vs. 11 min for NLI — significant cost for local inference, but no API fees and full data privacy.

---

## Method 3: API LLMs (OpenRouter)

### What was tested

- **2 frontier API models** via OpenRouter:
  - GPT-5.1 (`gpt51`) — OpenAI
  - Gemini 2.5 Pro (`gemini25_pro`) — Google
- **2 prompt languages**: Hebrew (he), English (en)
- **Prompt format**: `triplet` (raw triplet only — single format tested)
- **Majority vote** over all 4 combos
- Runtime: ~7 min total; Cost: GPT-5.1 $0.32, Gemini 2.5 Pro $0.52 (total $0.84)

### Results

**Per model:**

| Model | Lang | Acc | Prec | Rec | F1 | Unk | rows/s | Cost |
|---|---|---|---|---|---|---|---|---|
| gpt51 | he | 0.854 | 0.920 | 0.870 | **0.894** | 0 | 7.7 | $0.164 |
| gpt51 | en | 0.854 | 0.912 | 0.879 | **0.895** | 0 | 7.8 | $0.154 |
| gemini25_pro | en | 0.826 | 0.874 | 0.882 | 0.878 | 24 | 3.9 | $0.242 |
| gemini25_pro | he | 0.790 | 0.898 | 0.794 | 0.843 | 57 | 3.6 | $0.274 |

**Majority vote ensembles:**

| Ensemble | Acc | Prec | Rec | F1 | Unk | Voters |
|---|---|---|---|---|---|---|
| llm_majority_all | **0.864** | 0.926 | 0.879 | **0.902** | 33 | 4 |
| llm_majority_he_triplet | 0.862 | 0.944 | 0.856 | 0.898 | 50 | 2 |
| llm_majority_en_triplet | 0.860 | 0.928 | 0.870 | 0.898 | 51 | 2 |
| llm_majority_best_per_model | 0.860 | 0.928 | 0.870 | 0.898 | 51 | 2 |

### Key conclusions

- **GPT-5.1 is the single best model across all methods:** F1 = 0.894–0.895, Acc = 0.854, with zero unknown responses and ~8 rows/s throughput.
- **API ensemble achieves best overall F1 = 0.902** and best accuracy = 0.864 across all three methods.
- **Gemini 2.5 Pro underperforms GPT-5.1** and produces many abstentions in Hebrew (57 unknowns, ~11% of the set). English prompts reduce unknowns to 24 but still lag GPT-5.1 by ~1.5 F1 points.
- **Prompt language (he vs. en) has less effect for GPT-5.1** (F1 identical at 0.894–0.895) but matters more for Gemini (0.843 he vs. 0.878 en), suggesting GPT-5.1 is more robustly multilingual.
- **Shared false-positive patterns with NLI models:** Both API LLMs and NLI models flag the same "world-knowledge leakage" examples — where the fact is true but not stated in the text (language-of-country, diplomatic relations, borders). This is a dataset-level issue, not a model-level one.
- **Cost vs. quality:** $0.84 total for the gold set. Extrapolating to the full 3.1M-row dataset → roughly $5,000–$6,000 — prohibitive for cleaning the full set. API models are practical for gold evaluation, not bulk labelling.
- **Speed advantage:** 7 min vs. 1h 12 min for open-source LLMs, with better F1. For annotation pipelines where cost is not the constraint, API models are clearly preferable.

---

## Cross-Method Comparison

### Final metrics (best configuration per method)

| Method | Best Config | Acc | Prec | Rec | F1 | Runtime | Cost |
|---|---|---|---|---|---|---|---|
| NLI encoders (ensemble) | nli_ensemble_template @0.6 | 0.710 | 0.710 | **1.000** | 0.830 | 11 min | Free |
| Open-source LLM (ensemble) | majority_best_per_model | 0.840 | 0.857 | 0.930 | 0.892 | 1h 12min | Free |
| API LLM (ensemble) | majority_all | **0.864** | 0.926 | 0.879 | **0.902** | 7 min | $0.84 |
| API LLM (single best) | GPT-5.1 / en | 0.854 | **0.944** | 0.856 | 0.898 | — | $0.15 |

### Shared error patterns across all methods

**Systematic false positives — world-knowledge leakage:**  
The dominant error type across all three methods is predicting 1 when the relation is factually true globally but *not explicitly stated in the passage*. Recurring categories:
- Language-of-country (e.g., "Hungarian is spoken in Hungary" — text is about a village in Hungary)
- Diplomatic relations (e.g., "Belize and Guatemala have diplomatic relations" — text is about a reptile found in both)
- Geographic borders (e.g., "Brazil borders Bolivia" — text is about a bird species in both countries)
- Antonyms / conceptual pairs (e.g., "East is the opposite of South" — text mentions both directions)

These 133 "impossible" cases (26.6% of the dataset where every NLI combo is wrong) represent a **ceiling on NLI-style approaches** and are also challenging for LLMs. They constitute a hard subset that may require explicit text-grounding mechanisms.

**Systematic false negatives — implicit expression:**  
Relations that are present in the text but expressed indirectly or requiring inference are frequently missed:
- "Instance of" / membership relations expressed through parenthetical apposition
- Biographical relations embedded in complex multi-clause sentences
- Relations where the predicate is paraphrased rather than stated literally

---

## Future Research Proposals

### 1. Text-grounded classification with span extraction
Current methods classify at the (text, triplet) level without identifying *where* in the text the relation is expressed. Adding a span-extraction step (e.g., ask the model to quote the evidence sentence before classifying) would force text grounding and directly address the world-knowledge leakage error type. This could be implemented as a chain-of-thought prompt or a dedicated extractive QA step.

### 2. Hard-negative mining for NLI finetuning
The NLI models are overfit to predicting "entailment" because their training data may have lacked sufficient hard negatives (factually-true-but-unmentioned). Collecting or synthesizing hard negative examples — specifically triplets that are globally true but absent from a given passage — and retraining the NLI models on this augmented set could improve precision dramatically.

### 3. Cross-method ensemble (NLI + LLM)
The NLI ensemble and the LLM methods make different errors: NLI has near-perfect recall (catches everything) while LLMs have higher precision (fewer false positives). Combining them — using NLI as a recall-oriented first pass and LLMs for precision-oriented re-ranking — could yield a system with both high recall and high precision. A simple threshold on the LLM confidence when NLI flags positive would be a low-cost starting point.

### 4. Scaling to the full 3.1M-row dataset with open-source LLMs
API models are too expensive at scale (~$5,000 for the full set). However, Gemma3 achieves F1 = 0.889 locally in ~1.5 min/500-rows pace, which extrapolates to ~10–15 days for the full set on 2 GPUs. Batching, quantization (int4/int8), and selective inference (only on rows where distant-supervision label is uncertain) could make this tractable.

### 5. Predicate-stratified analysis and predicate-specific classifiers
The dataset shows extreme predicate-level variation in noise rates (0% positive for ההפך מ vs. 100% for אזרחות). A predicate-aware classifier — either a per-predicate model or a model that conditions on predicate type — could exploit this structure. The hardest predicates (diplomatic relations, language-of, borders) might benefit from retrieval-augmented approaches that explicitly look for the relation in the text.

### 6. Active learning / human-in-the-loop for ambiguous examples
The ~83 rows (16.6%) where the API LLM majority vote is uncertain or wrong are good candidates for active learning. Routing uncertain predictions to human annotators — rather than discarding or blindly accepting them — would yield a cleaner gold set for retraining and a more reliable estimate of the true classifier ceiling.

### 7. Evaluation on predicate subsets with known noise profiles
Rather than reporting aggregate metrics, a finer-grained evaluation bucketing results by predicate-level noise rate would reveal which relation types each method handles well. This would guide which predicates in the full 3.1M dataset can be reliably cleaned automatically vs. which require manual attention.
