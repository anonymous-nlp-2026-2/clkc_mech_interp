# English-Chinese Knowledge Conflict Signals Are Consistent With General Cross-Lingual Asymmetries

Code for the anonymous EMNLP 2026 ARR submission.

## Abstract

When multilingual large language models encounter conflicting information between languages, presentation language can determine the answer, but the internal mechanism remains unknown. We probe residual streams in Llama-3.1-8B-Instruct, Qwen3-8B, and Gemma-2-9B-IT on English–Chinese conflicts, combining cross-lingual transfer, Cross-lingual Natural Language Inference (XNLI) non-conflict baselines, confound exclusion, and representation steering. Conflict signals are linearly detectable across both languages, and cross-lingual transfer exhibits a directional asymmetry: Chinese-trained probes transfer to English ~7 percentage points more accurately than the reverse, matching the XNLI baseline and consistent with general multilingual structure rather than conflict-specific encoding. A dissociation emerges between detection and decision: the probe signal is broadly distributed across mid-to-upper layers, yet representation steering achieves causal influence only at a mid-layer (L13, 41% depth) in Llama; analogous Qwen and Gemma steering yields no significant effect. These findings indicate that linear detectability does not imply causal relevance, with implications for probe-based intervention.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

- `src/` — Main source code
  - Probe training and evaluation
  - Steering experiments
  - Bootstrap analysis
  - Cross-task transfer analysis
- `figures/` — Figure generation scripts

## Reproducing Results

### 1. Data Collection
Collect residual stream activations from multilingual QA datasets with knowledge conflicts.

### 2. Probe Training
```bash
python src/train_probe.py --model llama --data_dir data/llama31/
```

### 3. Bootstrap Analysis
```bash
python src/bootstrap_probe.py --model llama --n_bootstrap 2000
```

### 4. Steering Experiments
```bash
python src/steering_experiment.py --model llama --layer 13 --alphas 5 10 20 --position last
```

### 5. Cross-Task Transfer
```bash
python src/cross_task_transfer.py
```

See individual scripts for full argument documentation.
