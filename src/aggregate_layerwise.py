"""Aggregate per-layer probe curves for Paper Agent figure generation."""
import json

DIR_MAP = {"EN→EN": "en_en", "ZH→ZH": "zh_zh", "EN→ZH": "en_zh", "ZH→EN": "zh_en",
           "EN→EN": "en_en", "ZH→ZH": "zh_zh", "EN→ZH": "en_zh", "ZH→EN": "zh_en"}

def extract_curves(per_layer_data):
    curves = {}
    for direction, entries in per_layer_data.items():
        key = DIR_MAP.get(direction, direction.lower().replace("→", "_").replace("→", "_"))
        if isinstance(entries, list):
            curves[key] = {str(e["layer"]): e["balanced_accuracy"] for e in entries}
        elif isinstance(entries, dict):
            curves[key] = {}
            for layer_key, stats in sorted(entries.items(), key=lambda x: int(x[0].replace("layer_", ""))):
                layer_num = layer_key.replace("layer_", "")
                if isinstance(stats, dict) and "mean" in stats:
                    curves[key][layer_num] = stats["mean"]
                elif isinstance(stats, dict) and "balanced_accuracy" in stats:
                    curves[key][layer_num] = stats["balanced_accuracy"]
    return curves

def find_peak(curves):
    results = {}
    for direction, layer_acc in curves.items():
        best_layer = max(layer_acc, key=layer_acc.get)
        results[direction] = {"peak_layer": int(best_layer), "peak_acc": layer_acc[best_layer]}
    return results

output = {}

# Llama31 residual stream
with open("./data/llama31/probe_results_perlayer.json") as f:
    llama_res = json.load(f)
llama_curves = extract_curves(llama_res["per_layer"])
output["llama31"] = {
    "n_layers": llama_res.get("num_layers", 32),
    "probe_type": "residual_stream",
    "directions": llama_curves,
    "peaks": find_peak(llama_curves),
}

# Llama31 MLP
with open("./data/llama31/mlp_probe_results.json") as f:
    llama_mlp = json.load(f)
llama_mlp_curves = extract_curves(llama_mlp["per_layer"])
output["llama31_mlp"] = {
    "n_layers": llama_mlp.get("num_layers", 32),
    "probe_type": "mlp",
    "directions": llama_mlp_curves,
    "peaks": find_peak(llama_mlp_curves),
}

# Qwen3 residual stream
with open("./data/qwen3/probe_results.json") as f:
    qwen_res = json.load(f)
qwen_curves = extract_curves(qwen_res["per_layer"])
output["qwen3"] = {
    "n_layers": qwen_res.get("num_layers", 36),
    "probe_type": "residual_stream",
    "directions": qwen_curves,
    "peaks": find_peak(qwen_curves),
}

# Qwen3 MLP
with open("./data/qwen3/mlp_probe_results.json") as f:
    qwen_mlp = json.load(f)
qwen_mlp_curves = extract_curves(qwen_mlp["per_layer"])
output["qwen3_mlp"] = {
    "n_layers": qwen_mlp.get("num_layers", 36),
    "probe_type": "mlp",
    "directions": qwen_mlp_curves,
    "peaks": find_peak(qwen_mlp_curves),
}

# Also include bootstrap CI data if available
try:
    with open("./data/qwen3/probe_results_bootstrap_n2000.json") as f:
        qwen_boot = json.load(f)
    qwen_boot_curves = {}
    for direction, layers in qwen_boot["per_layer"].items():
        key = DIR_MAP.get(direction, direction)
        ci_data = {}
        for layer_key, stats in sorted(layers.items(), key=lambda x: int(x[0].replace("layer_", ""))):
            layer_num = layer_key.replace("layer_", "")
            ci_data[layer_num] = {"mean": stats["mean"], "ci_lower": stats["ci_lower"], "ci_upper": stats["ci_upper"]}
        qwen_boot_curves[key] = ci_data
    output["qwen3_bootstrap_ci"] = {"n_seeds": 2000, "directions": qwen_boot_curves}
except Exception as e:
    print(f"Qwen3 bootstrap CI: {e}")

out_path = "./data/layerwise_probe_curves.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved to {out_path}")

# Print summary
for model_key in ["llama31", "qwen3", "llama31_mlp", "qwen3_mlp"]:
    m = output[model_key]
    print(f"\n{model_key} ({m['probe_type']}, {m['n_layers']} layers):")
    for d, p in m["peaks"].items():
        print(f"  {d}: peak layer {p['peak_layer']}, acc {p['peak_acc']:.4f}")

# Gemma2 residual stream (from bootstrap means)
DIR_MAP_BOOT = {"en_en": "en_en", "zh_zh": "zh_zh", "en_zh": "en_zh", "zh_en": "zh_en"}
try:
    with open("./data/gemma2/probe_results_bootstrap_n2000.json") as f:
        gemma_boot = json.load(f)
    gemma_curves = {}
    gemma_ci_curves = {}
    for direction, layers in gemma_boot["per_layer"].items():
        key = DIR_MAP_BOOT.get(direction, direction)
        curve = {}
        ci_data = {}
        for layer_key, stats in sorted(layers.items(), key=lambda x: int(x[0].replace("layer_", ""))):
            layer_num = layer_key.replace("layer_", "")
            curve[layer_num] = stats["mean"]
            ci_data[layer_num] = {"mean": stats["mean"], "ci_lower": stats["ci_lower"], "ci_upper": stats["ci_upper"]}
        gemma_curves[key] = curve
        gemma_ci_curves[key] = ci_data
    output["gemma2"] = {
        "n_layers": 42,
        "probe_type": "residual_stream",
        "directions": gemma_curves,
        "peaks": find_peak(gemma_curves),
    }
    output["gemma2_bootstrap_ci"] = {"n_seeds": 2000, "directions": gemma_ci_curves}
    print("Added Gemma2 data")
except Exception as e:
    print(f"Gemma2: {e}")

# Re-save
with open("./data/layerwise_probe_curves.json", "w") as f:
    json.dump(output, f, indent=2)
print("Re-saved with Gemma2")

# Print Gemma summary
if "gemma2" in output:
    m = output["gemma2"]
    print(f"\ngemma2 ({m['probe_type']}, {m['n_layers']} layers):")
    for d, p in m["peaks"].items():
        print(f"  {d}: peak layer {p['peak_layer']}, acc {p['peak_acc']:.4f}")
