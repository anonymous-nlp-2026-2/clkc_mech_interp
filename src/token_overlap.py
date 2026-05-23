import json
import os
from pathlib import Path
from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

DATA_ROOT = Path("./data")
MODELS = {
    "Qwen3-8B": {
        "tokenizer_path": "Qwen/Qwen3-8B",
        "data_dir": DATA_ROOT / "qwen3",
    },
    "Llama-3.1-8B-Instruct": {
        "tokenizer_path": "meta-llama/Llama-3.1-8B-Instruct",
        "data_dir": DATA_ROOT / "llama31",
    },
}

def load_prompts(filepaths):
    texts = []
    for fp in filepaths:
        with open(fp) as f:
            for line in f:
                row = json.loads(line)
                texts.append(row["question"] + " " + row["context"])
    return texts

def collect_token_ids(tokenizer, texts):
    ids = set()
    for t in texts:
        ids.update(tokenizer.encode(t, add_special_tokens=False))
    return ids

def main():
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_path"], trust_remote_code=True)
        print(f"Vocab size: {tokenizer.vocab_size}")

        d = cfg["data_dir"]
        en_files = sorted(d.glob("en_conflict_*_verified.jsonl"))
        zh_files = sorted(d.glob("zh_conflict_*.jsonl"))
        zh_files = [f for f in zh_files if "activations" not in f.name]

        en_texts = load_prompts(en_files)
        zh_texts = load_prompts(zh_files)
        print(f"EN prompts: {len(en_texts)}, ZH prompts: {len(zh_texts)}")

        en_ids = collect_token_ids(tokenizer, en_texts)
        zh_ids = collect_token_ids(tokenizer, zh_texts)

        intersection = en_ids & zh_ids
        union = en_ids | zh_ids

        jaccard = len(intersection) / len(union) if union else 0
        overlap_en = len(intersection) / len(en_ids) if en_ids else 0
        overlap_zh = len(intersection) / len(zh_ids) if zh_ids else 0

        print(f"Unique EN token IDs: {len(en_ids)}")
        print(f"Unique ZH token IDs: {len(zh_ids)}")
        print(f"Intersection: {len(intersection)}")
        print(f"Union: {len(union)}")
        print(f"Jaccard (|EN∩ZH| / |EN∪ZH|): {jaccard:.4f}")
        print(f"Overlap/EN (|EN∩ZH| / |EN|): {overlap_en:.4f}")
        print(f"Overlap/ZH (|EN∩ZH| / |ZH|): {overlap_zh:.4f}")

        en_only = en_ids - zh_ids
        zh_only = zh_ids - en_ids
        print(f"EN-only tokens: {len(en_only)}")
        print(f"ZH-only tokens: {len(zh_only)}")

        print("\nSample EN-only tokens (first 20):")
        samples = sorted(list(en_only))[:20]
        for tid in samples:
            print(f"  {tid}: {repr(tokenizer.decode([tid]))}")

        print("Sample ZH-only tokens (first 20):")
        samples = sorted(list(zh_only))[:20]
        for tid in samples:
            print(f"  {tid}: {repr(tokenizer.decode([tid]))}")

        print(f"\nOverlap tokens as % of vocab: {len(intersection)/tokenizer.vocab_size*100:.2f}%")

if __name__ == "__main__":
    main()
