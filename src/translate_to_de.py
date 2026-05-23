"""
Translate EN ConflictQA data to DE (German) for cross-lingual knowledge conflict research.
Uses Llama-3.1-8B-Instruct for translation.

Usage:
    python src/translate_to_de.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --input_dir data/llama31/ \
        --output_dir data/llama31/ \
        --batch_size 4
"""

import argparse
import json
import os
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


TRANSLATE_PROMPT = """You are a professional English-to-German translator. Translate the following text to German.

Rules:
1. Preserve all proper nouns (person names, place names, organization names) in their standard German translations. If no standard translation exists, keep the original.
2. Preserve the semantic meaning precisely — do not add, omit, or interpret.
3. For short answer phrases (1-5 words), translate concisely.
4. Output ONLY the German translation, nothing else.

English text:
{text}

German translation:"""

FIELDS_TO_TRANSLATE = ["question", "context", "parametric_answer", "contextual_answer"]


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def translate_text(model, tokenizer, text: str, max_new_tokens: int = 512) -> str:
    prompt = TRANSLATE_PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return result


def translate_file(model, tokenizer, input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    partial_path = output_path + ".partial"
    translated = []
    start_idx = 0

    if os.path.exists(partial_path):
        with open(partial_path, "r", encoding="utf-8") as f:
            translated = [json.loads(line) for line in f if line.strip()]
        start_idx = len(translated)
        print(f"  Resuming from checkpoint: {start_idx}/{len(items)}")

    total = len(items)
    for i in range(start_idx, total):
        item = items[i]
        de_item = {}
        for key, value in item.items():
            if key in FIELDS_TO_TRANSLATE:
                de_item[key] = translate_text(model, tokenizer, value)
            elif key == "language":
                de_item[key] = "de"
            else:
                de_item[key] = value
        translated.append(de_item)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{total}] translated")

        if len(translated) % 50 == 0:
            with open(partial_path, "w", encoding="utf-8") as f:
                for t in translated:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            print(f"  Progress saved at {len(translated)}/{total}")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in translated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"Saved {len(translated)} items to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--input_dir", default="data/llama31/")
    parser.add_argument("--output_dir", default="data/llama31/")
    parser.add_argument("--batch_size", type=int, default=4, help="(reserved for future batch translation)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(args.model)
    print("Model loaded.")

    file_pairs = [
        ("en_conflict_train.jsonl", "de_conflict_train.jsonl"),
        ("en_conflict_test.jsonl", "de_conflict_test.jsonl"),
    ]

    for en_file, de_file in file_pairs:
        input_path = os.path.join(args.input_dir, en_file)
        output_path = os.path.join(args.output_dir, de_file)
        if not os.path.exists(input_path):
            print(f"Skipping {en_file} (not found)")
            continue
        print(f"\nTranslating {en_file} -> {de_file}")
        translate_file(model, tokenizer, input_path, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
