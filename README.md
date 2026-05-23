# Cross-Lingual Knowledge Conflict Signals: A Mechanistic Interpretability Study

Code for the anonymous EMNLP 2026 submission.

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
