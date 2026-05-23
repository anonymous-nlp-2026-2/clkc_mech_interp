"""Extract per-layer probe balanced accuracy from activations (lightweight)."""
import json, sys, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

DIRECTIONS = ["EN→EN", "ZH→ZH", "EN→ZH", "ZH→EN"]

def load_act(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    n = d["split_sizes"]["train"]
    a = d["activations"].float() if d["activations"].dtype == torch.bfloat16 else d["activations"]
    return {"tr_X": a[:n], "tr_y": d["labels"][:n].numpy(), "te_X": a[n:], "te_y": d["labels"][n:].numpy()}

def probe_layer(tr_X, tr_y, te_X, te_y):
    sc = StandardScaler()
    tr_X = sc.fit_transform(tr_X)
    te_X = sc.transform(te_X)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=42)
    clf.fit(tr_X, tr_y)
    pred = clf.predict(te_X)
    return round(balanced_accuracy_score(te_y, pred), 4), round(f1_score(te_y, pred), 4)

def run(en_path, zh_path, n_layers):
    print(f"Loading {en_path}...", flush=True)
    en = load_act(en_path)
    print(f"Loading {zh_path}...", flush=True)
    zh = load_act(zh_path)
    pairs = {
        "EN→EN": (en, en), "ZH→ZH": (zh, zh),
        "EN→ZH": (en, zh), "ZH→EN": (zh, en),
    }
    result = {}
    for d_name in DIRECTIONS:
        train_d, test_d = pairs[d_name]
        layer_results = []
        for L in range(n_layers):
            ba, f1 = probe_layer(
                train_d["tr_X"][:, L, :].numpy(), train_d["tr_y"],
                test_d["te_X"][:, L, :].numpy(), test_d["te_y"])
            layer_results.append({"layer": L, "balanced_accuracy": ba, "f1": f1})
            if L % 8 == 0:
                print(f"  {d_name} layer {L}: ba={ba}", flush=True)
        result[d_name] = layer_results
    return result

if __name__ == "__main__":
    base = "./data/llama31"
    res = run(f"{base}/en_activations.pt", f"{base}/zh_activations.pt", 32)
    out = {"per_layer": res, "num_layers": 32}
    out_path = f"{base}/probe_results_perlayer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")
