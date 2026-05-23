"""Representation steering via linear probe weights (plan_107).

Uses the linear probe's weight vector as a steering direction to causally
intervene in the residual stream during knowledge-conflict processing.
Measures whether the intervention flips model output between parametric
and contextual answers.

Input:
  - Model (Qwen3-8B or Llama-3.1-8B-Instruct)
  - Activation files (en_activations.pt from collect_activations.py)
  - Test data (en_conflict_test_verified.jsonl)

Output:
  - steering_results.json: flip rates, Fisher exact test per alpha
"""

import argparse
import json
import re
import string
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Probe weight extraction
# ---------------------------------------------------------------------------

def load_steering_direction(activation_path: str, layer: int):
    """Train a probe on the given layer and extract the steering direction.

    The probe is trained with StandardScaler (matching train_probe.py).
    The weight is then mapped back to original activation space:
        w_orig = w_scaled / scaler.scale_
    so steering in the residual stream is geometrically correct.
    """
    data = torch.load(activation_path, map_location="cpu", weights_only=True)
    n_train = data["split_sizes"]["train"]
    acts = data["activations"]
    if acts.dtype == torch.bfloat16:
        acts = acts.float()

    train_X = acts[:n_train, layer, :].numpy()
    train_y = data["labels"][:n_train].numpy()
    test_X = acts[n_train:, layer, :].numpy()
    test_y = data["labels"][n_train:].numpy()

    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    probe = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42
    )
    probe.fit(train_X_scaled, train_y)

    acc = float(np.mean(probe.predict(test_X_scaled) == test_y))
    print(f"Probe accuracy at layer {layer}: {acc:.4f}")

    # Map weight back to original space
    w_scaled = probe.coef_[0]  # shape: [hidden_dim]
    w_orig = w_scaled / scaler.scale_
    d = w_orig / np.linalg.norm(w_orig)
    return d.astype(np.float32), acc


def random_orthogonal_direction(d: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generate a random unit vector orthogonal to d (control condition)."""
    rng = np.random.RandomState(seed)
    v = rng.randn(len(d)).astype(np.float32)
    v -= np.dot(v, d) * d  # Gram-Schmidt
    v /= np.linalg.norm(v)
    return v


# ---------------------------------------------------------------------------
# Peak layer lookup
# ---------------------------------------------------------------------------

def get_peak_layer(probe_results_path: str, direction: str = "EN→EN") -> int:
    """Read peak layer from probe_results.json (handles both formats)."""
    with open(probe_results_path) as f:
        data = json.load(f)

    # Llama format: {"EN→EN": {"peak_layer": 28, ...}}
    if direction in data and "peak_layer" in data[direction]:
        return data[direction]["peak_layer"]

    # Qwen format: {"summary": {"EN→EN": {"best_layer": 24, ...}}}
    if "summary" in data and direction in data["summary"]:
        return data["summary"][direction]["best_layer"]

    raise ValueError(f"Cannot find peak layer for {direction} in {probe_results_path}")


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------

class SteeringHook:
    """Forward hook: injects during prefill only, skips KV-cache decode steps."""

    def __init__(self, direction, alpha, position="last"):
        self.direction = torch.from_numpy(direction)
        self.alpha = alpha
        self.position = position
        self._handle = None
        self.applied = False

    def __call__(self, module, input, output):
        if self.applied:
            return output

        h = output[0]
        d = self.direction.to(h.device, dtype=h.dtype)

        if h.dim() == 3:
            if self.position == "last":
                h[:, -1, :] += self.alpha * d
            elif self.position == "all":
                h += self.alpha * d
            self.applied = True
        elif h.dim() == 2:
            h[-1] += self.alpha * d
            self.applied = True

        return output

    def reset(self):
        self.applied = False

    def register(self, module):
        self._handle = module.register_forward_hook(self)

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None

# ---------------------------------------------------------------------------
# Answer classification
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def extract_answer_entity(sentence: str) -> str:
    patterns = [
        r"(?:is|was|were|are)\s+(.+?)\.?\s*$",
        r"born in\s+(.+?)\.?\s*$",
        r"located in\s+(.+?)\.?\s*$",
        r"(?:written|directed|produced|created|composed|founded|invented) by\s+(.+?)\.?\s*$",
    ]
    for p in patterns:
        m = re.search(p, sentence, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".")
    return sentence


def answers_match(candidate: str, reference: str) -> bool:
    c = normalize_answer(candidate)
    r = normalize_answer(reference)
    if not c or not r:
        return False
    if c == r or r in c or c in r:
        return True
    c_last = c.split()[-1] if c.split() else ""
    r_last = r.split()[-1] if r.split() else ""
    if len(c_last) > 2 and len(r_last) > 2 and c_last == r_last:
        return True
    return False


def classify_answer(response: str, parametric: str, contextual: str) -> str:
    """Classify model response as parametric, contextual, both, or neither."""
    response = strip_think_tags(response)

    # Extract entity from full-sentence contextual answer
    ctx_entity = extract_answer_entity(contextual)

    param_match = answers_match(response, parametric)
    ctx_match = answers_match(response, ctx_entity)

    if ctx_match and not param_match:
        return "contextual"
    elif param_match and not ctx_match:
        return "parametric"
    elif ctx_match and param_match:
        return "both"
    return "neither"


def strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_conflict_prompt(item: dict) -> str:
    """Build the same prompt format used during activation collection."""
    return (
        f"Context: {item['context']}\n"
        f"Question: {item['question']}\n"
        f"Answer:"
    )


def generate_answer(
    model, tokenizer, prompt: str, max_new_tokens: int = 100,
    use_chat_template: bool = True,
) -> str:
    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_steering_experiment(
    model, tokenizer, layer_module, test_data,
    steering_dir, control_dir, alphas,
    use_chat_template=True,
    position="last",
):
    results = {}

    # Baseline
    print(f"Baseline ({len(test_data)} samples)...")
    baseline = []
    for item in test_data:
        prompt = build_conflict_prompt(item)
        resp = generate_answer(model, tokenizer, prompt, use_chat_template=use_chat_template)
        atype = classify_answer(resp, item["parametric_answer"], item["contextual_answer"])
        baseline.append({"response": resp, "answer_type": atype})

    baseline_dist = {}
    for b in baseline:
        baseline_dist[b["answer_type"]] = baseline_dist.get(b["answer_type"], 0) + 1
    print(f"  Baseline distribution: {baseline_dist}")

    n = len(test_data)
    for alpha in alphas:
        print(f"\nalpha = {alpha:+.1f}")

        # Steering
        hook = SteeringHook(steering_dir, alpha, position=position)
        hook.register(layer_module)
        steered = []
        for item in test_data:
            hook.reset()
            prompt = build_conflict_prompt(item)
            resp = generate_answer(model, tokenizer, prompt, use_chat_template=use_chat_template)
            atype = classify_answer(resp, item["parametric_answer"], item["contextual_answer"])
            steered.append({"response": resp, "answer_type": atype})
        hook.remove()

        # Control (random orthogonal direction, same magnitude)
        chook = SteeringHook(control_dir, alpha, position=position)
        chook.register(layer_module)
        control = []
        for item in test_data:
            chook.reset()
            prompt = build_conflict_prompt(item)
            resp = generate_answer(model, tokenizer, prompt, use_chat_template=use_chat_template)
            atype = classify_answer(resp, item["parametric_answer"], item["contextual_answer"])
            control.append({"response": resp, "answer_type": atype})
        chook.remove()

        # Flip counts
        s_flips = sum(1 for b, s in zip(baseline, steered) if b["answer_type"] != s["answer_type"])
        c_flips = sum(1 for b, c in zip(baseline, control) if b["answer_type"] != c["answer_type"])

        # Directional flips: parametric→contextual and contextual→parametric
        s_p2c = sum(1 for b, s in zip(baseline, steered)
                    if b["answer_type"] == "parametric" and s["answer_type"] == "contextual")
        s_c2p = sum(1 for b, s in zip(baseline, steered)
                    if b["answer_type"] == "contextual" and s["answer_type"] == "parametric")

        # Fisher exact test
        table = [[s_flips, n - s_flips], [c_flips, n - c_flips]]
        _, fisher_p = stats.fisher_exact(table)

        # Steered distribution
        steered_dist = {}
        for s in steered:
            steered_dist[s["answer_type"]] = steered_dist.get(s["answer_type"], 0) + 1

        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "n": n,
            "steering_flips": s_flips,
            "control_flips": c_flips,
            "steering_flip_rate": round(s_flips / n, 4),
            "control_flip_rate": round(c_flips / n, 4),
            "steering_p2c": s_p2c,
            "steering_c2p": s_c2p,
            "fisher_p": round(fisher_p, 6),
            "steered_distribution": steered_dist,
        }
        print(f"  Steering flips: {s_flips}/{n} ({s_flips/n:.1%})  "
              f"[p→c: {s_p2c}, c→p: {s_c2p}]")
        print(f"  Control  flips: {c_flips}/{n} ({c_flips/n:.1%})")
        print(f"  Fisher p = {fisher_p:.4f}")

    return results, baseline


def main():
    parser = argparse.ArgumentParser(
        description="Representation steering experiment (plan_107)"
    )
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--data_dir", required=True,
                        help="Directory with en_activations.pt and en_conflict_test_verified.jsonl")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (defaults to data_dir)")
    parser.add_argument("--peak_layer", type=int, default=None,
                        help="Layer to steer (auto-detected from probe_results.json if omitted)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[-20, -10, -5, -2, -1, 1, 2, 5, 10, 20],
                        help="Steering strengths (positive = toward conflict direction)")
    parser.add_argument("--position", choices=["last", "all"], default="last",
                        help="Which token positions to steer")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix for output filename (e.g. _L13_last)")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Use raw prompt instead of chat template")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only validate setup, do not run generation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    # --- Resolve peak layer ---
    if args.peak_layer is not None:
        peak_layer = args.peak_layer
    else:
        probe_path = data_dir / "probe_results.json"
        peak_layer = get_peak_layer(str(probe_path))
    print(f"Steering layer: {peak_layer}")

    # --- Load steering direction from activations ---
    act_path = data_dir / "en_activations.pt"
    steering_dir, probe_acc = load_steering_direction(str(act_path), peak_layer)
    control_dir = random_orthogonal_direction(steering_dir)
    print(f"Steering direction dim: {len(steering_dir)}")
    print(f"Orthogonality check: dot={np.dot(steering_dir, control_dir):.6f}")

    # --- Load test data (conflict only) ---
    test_path = data_dir / "en_conflict_test_verified.jsonl"
    with open(test_path) as f:
        all_data = [json.loads(line) for line in f]
    test_data = [d for d in all_data if d["label"] == "conflict"]
    print(f"Test samples: {len(test_data)} conflict / {len(all_data)} total")

    if args.dry_run:
        print("\n=== DRY RUN: setup validated, skipping model load and generation ===")
        print(f"  Model: {args.model}")
        print(f"  Peak layer: {peak_layer}")
        print(f"  Probe accuracy: {probe_acc:.4f}")
        print(f"  Alphas: {args.alphas}")
        print(f"  Position: {args.position}")
        print(f"  Test samples: {len(test_data)}")
        print(f"  Sample prompt preview:")
        print(f"    {build_conflict_prompt(test_data[0])[:200]}...")
        return

    # --- Load model ---
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Access target layer
    layer_module = model.model.layers[peak_layer]
    print(f"Hooked layer: model.model.layers[{peak_layer}]")

    # --- Run experiment ---
    use_chat = not args.no_chat_template
    results, baseline = run_steering_experiment(
        model, tokenizer, layer_module, test_data,
        steering_dir, control_dir, args.alphas,
        use_chat_template=use_chat,
        position=args.position,
    )

    # --- Save results ---
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    out_path = output_dir / f"steering_results{suffix}.json"
    output = {
        "config": {
            "model": args.model,
            "peak_layer": peak_layer,
            "probe_accuracy": probe_acc,
            "position": args.position,
            "alphas": args.alphas,
            "n_test": len(test_data),
            "use_chat_template": use_chat,
        },
        "baseline": {
            "distribution": {},
            "samples": baseline[:5],
        },
        "results": results,
    }

    # Baseline distribution
    for b in baseline:
        t = b["answer_type"]
        output["baseline"]["distribution"][t] = output["baseline"]["distribution"].get(t, 0) + 1

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
