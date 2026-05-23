#!/bin/bash
# Activate your environment
cd "$(dirname "$0")"

echo "=== Qwen probe ===" >> logs/probe_progress.log
CUDA_VISIBLE_DEVICES="" python -u src/train_probe.py \
    --activation_dir data/xnli/qwen3/ \
    --output_dir data/xnli/qwen3/ \
    --seeds 42,123,456,789,1024 \
    --skip_permutation >> logs/xnli_probe_qwen.log 2>&1
echo "QWEN_DONE" >> logs/probe_progress.log

echo "=== Llama probe ===" >> logs/probe_progress.log
CUDA_VISIBLE_DEVICES="" python -u src/train_probe.py \
    --activation_dir data/xnli/llama31/ \
    --output_dir data/xnli/llama31/ \
    --seeds 42,123,456,789,1024 \
    --skip_permutation >> logs/xnli_probe_llama.log 2>&1
echo "LLAMA_DONE" >> logs/probe_progress.log

echo "ALL_DONE" >> logs/probe_progress.log
