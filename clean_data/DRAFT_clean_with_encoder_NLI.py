# # CUDA_VISIBLE_DEVICES=2 python clean_with_encoder_NLI.py --local_checkpoint neodictabert_hebnli/checkpoint-4500 2>&1 | tee clean_neodictabert.txt

import logging
import os
import re
import torch
import jsonlines
import transformers
import argparse
from tqdm import tqdm

import json
from json.decoder import JSONDecodeError

ENTAILMENT_THRESHOLD = 0.75

SENTENCE_SPLIT_REGEX = re.compile("(?<!\\d)\\.(?!\\d)")


# Batch size configuration
DEFAULT_BATCH_SIZE = 64
LARGE_TEXT_THRESHOLD = 256
LARGE_TEXT_BATCH_SIZE = 12


def iter_jsonl(path, encoding='utf-8'):
    """
    Yield (line_no, obj) for each valid JSON object in the file.
    Skips empty lines and logs (yields None) for invalid lines.
    """
    with open(path, 'r', encoding=encoding) as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                yield i, obj
            except JSONDecodeError as e:
                print(f"[WARN] Invalid JSON in {path} at line {i}: {e}. Skipping line.")
                continue


def count_jsonl_lines(path, encoding='utf-8'):
    """Count valid JSON lines in a file for progress tracking."""
    count = 0
    with open(path, 'r', encoding=encoding) as fh:
        for line in fh:
            if line.strip():
                try:
                    json.loads(line)
                    count += 1
                except JSONDecodeError:
                    pass
    return count


def get_case_insensitive_key_value(input_dict, key):
    return next((value for dict_key, value in input_dict.items() if dict_key.lower() == key.lower()), None)


def get_adaptive_batch_size(texts):
    """
    Determine batch size based on the maximum text length in the batch.
    Returns appropriate batch size to avoid memory issues.
    """
    if not texts:
        return DEFAULT_BATCH_SIZE
    
    max_length = max(len(text[0]) + len(text[1]) for text in texts)
    
    if max_length > LARGE_TEXT_THRESHOLD:
        return LARGE_TEXT_BATCH_SIZE
    else:
        return DEFAULT_BATCH_SIZE


def filter_triples(model, tokenizer, texts, device):
    """
    Filter triples using entailment model.
    Returns confidence scores for each triple.
    """
    if not texts:
        return torch.tensor([])
    
    batch_size = get_adaptive_batch_size(texts)
    result = []
    
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        encoded_input = tokenizer(
            [ex[0] for ex in batch_texts],
            [ex[1] for ex in batch_texts],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=256,
            padding='longest',
            return_token_type_ids=False,
            truncation=True
        )

        for tensor in encoded_input:
            encoded_input[tensor] = encoded_input[tensor].to(device)
        
        with torch.no_grad():
            outputs = model(**encoded_input, return_dict=True)
        
        result.append(outputs['logits'].softmax(dim=1))
        del outputs
    
    if not result:
        return torch.tensor([])
    
    logits = torch.cat(result)
    return logits[:, 1] if logits.dim() > 1 else logits.unsqueeze(0)


def prepare_triplet(subject_entity, object_entity, article_text, predicate):
    text_triplet = ''
    text_triplet += SENTENCE_SPLIT_REGEX.split(
            article_text[:min(subject_entity['boundaries'][0], object_entity['boundaries'][0])]
        )[-1]
    text_triplet += article_text[min(subject_entity['boundaries'][0], object_entity['boundaries'][0]):
                                 max(subject_entity['boundaries'][1], object_entity['boundaries'][1])]
    text_triplet += SENTENCE_SPLIT_REGEX.split(
            article_text[max(subject_entity['boundaries'][1], object_entity['boundaries'][1]):]
        )[0]
    return (
        text_triplet.strip('\n'),
        ' '.join([str(subject_entity['surfaceform']),
                  str(predicate['surfaceform']),
                  str(object_entity['surfaceform'])])
    )


def main(folder_input, folder_output, model_name_or_path, model_tag="", threshold=ENTAILMENT_THRESHOLD, name=""):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_config = transformers.AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=False,
        output_attentions=False,
        trust_remote_code=True
    )
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True, use_safetensors=True)
    # if no safetensors file, load normally
    except Exception as e:
        print(f"[WARN] Could not load model with safetensors: {e}. Trying normal load.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.half()

    # Replace any invalid filename characters (like \ / : * ? " < > |) with '_'
    safe_name = re.sub(r'[\\/:"*?<>|]+', '_', name)
    out_path = os.path.join(folder_output, f"heb_clean_{safe_name}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = jsonlines.open(out_path, mode='a')

    # --- Counters ---
    rows_processed = 0           # number of articles
    triples_processed = 0        # number of triples total
    clean_triples_count = 0      # number of triples above threshold
    files_processed = 0

    for k, j, y in os.walk(folder_input):
        for file_name in y:
            in_path = os.path.join(k, file_name)

            print("Parsing file", in_path)
            
            # Count lines for accurate progress tracking
            total_lines = count_jsonl_lines(in_path)
            
            # Use tqdm with total count
            for line_num, article in tqdm(iter_jsonl(in_path), total=total_lines, desc="Processing articles"):
                if article is None:
                    continue

                rows_processed += 1
                previous = []
                triples_list = []
                texts = []

                for triple in article['triples']:
                    triples_processed += 1   # count every triple

                    if triple['subject']['boundaries'] and triple['object']['boundaries'] and \
                       (triple['subject']['boundaries'], triple['object']['boundaries']) not in previous:
                        previous.append((triple['subject']['boundaries'], triple['object']['boundaries']))
                        triples_list.append(triple)
                        texts.append(prepare_triplet(triple['subject'], triple['object'], article['text'], triple["predicate"]))
                    elif (triple['subject']['boundaries'], triple['object']['boundaries']) not in previous:
                        distance = float("inf")
                        for entity in article['entities']:
                            if entity['uri'] == triple['subject']['uri']:
                                new_distance = abs(min(triple['object']['boundaries']) - min(entity['boundaries']))
                                if new_distance < distance:
                                    subject_entity = entity
                                    distance = new_distance
                        triple['subject'] = subject_entity
                        previous.append((triple['subject']['boundaries'], triple['object']['boundaries']))
                        triples_list.append(triple)
                        texts.append(prepare_triplet(subject_entity, triple['object'], article['text'], triple["predicate"]))

                if not texts:
                    continue

                indexes = filter_triples(model, tokenizer, texts, device)
                
                # Fix: Check tensor size properly
                if indexes.numel() == 0:
                    continue
                
                # Add confidence score and clean flag to ALL triples
                for pred, trip in zip(indexes, triples_list):
                    trip['confidence'] = pred.item()
                    trip['clean'] = "yes" if pred.item() > threshold else "no"
                    if trip['clean'] == "yes":
                        clean_triples_count += 1

                # Keep ALL triples (no filtering)
                article['triples'] = triples_list
                writer.write(article)

            files_processed += 1

            if files_processed % 5 == 0:
                print(f"Processed {files_processed} files, "
                      f"{rows_processed} articles, "
                      f"{triples_processed} triples, "
                      f"{clean_triples_count} marked as clean.")

    writer.close()
    print(f"FINAL: Processed {files_processed} files, "
          f"{rows_processed} articles, "
          f"{triples_processed} triples, "
          f"{clean_triples_count} marked as clean (entailment > {threshold}).")


if __name__ == "__main__":
    # defaults
    main_dir = "/home/nlp/ronke21/crocodile/"
    default_input = os.path.join(main_dir, "out/he/AA")
    default_output = os.path.join(main_dir, "out_clean/he/")
    local_base = "/home/nlp/ronke21/heb_nli_mrl_eval/output/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=default_input,
                        help=f"Input dir (default: {default_input})")
    parser.add_argument("--output_dir", type=str, default=default_output,
                        help=f"Output dir (default: {default_output})")
    parser.add_argument("--model_checkpoint", type=str, default=None,
                        help="HF model name or path (e.g. MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)")
    parser.add_argument("--local_checkpoint", type=str, default=None,
                        help="Relative path inside local output dir (e.g. mt5-xl_hebnli/checkpoint-6000)")
    parser.add_argument("--model_tag", type=str, default="", help="Tag for output filename (default empty)")
    parser.add_argument("--threshold", type=float, default=ENTAILMENT_THRESHOLD, help="Entailment confidence threshold")
    args = parser.parse_args()

    # decide which checkpoint to use
    if args.local_checkpoint and args.model_checkpoint:
        raise ValueError("Use only one of --model_checkpoint or --local_checkpoint")
    elif args.local_checkpoint:
        model_path = os.path.join(local_base, args.local_checkpoint)
        tag = args.model_tag or os.path.basename(args.local_checkpoint)
        name = args.local_checkpoint
    elif args.model_checkpoint:
        model_path = args.model_checkpoint
        tag = args.model_tag or os.path.basename(args.model_checkpoint)
        name = args.model_checkpoint
    else:
        raise ValueError("You must provide either --model_checkpoint or --local_checkpoint")

    main(args.input_dir, args.output_dir, model_path, tag, args.threshold, name)