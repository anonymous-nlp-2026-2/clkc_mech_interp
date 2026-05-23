"""Bootstrap MLP vs Linear probe delta for C3 statistical validation.
Uses PyTorch MLP (consistent with train_mlp_probe.py) and sklearn LogisticRegression."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json, time, warnings, os
warnings.filterwarnings("ignore")

CONDITIONS = {
    "qwen3_conflict": {"en_path": "data/qwen3/en_activations.pt", "zh_path": "data/qwen3/zh_activations.pt", "peak_layer": 23},
    "llama31_conflict": {"en_path": "data/llama31/en_activations.pt", "zh_path": "data/llama31/zh_activations.pt", "peak_layer": 5},
    "qwen3_xnli": {"en_path": "data/xnli/qwen3/en_activations.pt", "zh_path": "data/xnli/qwen3/zh_activations.pt", "peak_layer": 22},
    "llama31_xnli": {"en_path": "data/xnli/llama31/en_activations.pt", "zh_path": "data/xnli/llama31/zh_activations.pt", "peak_layer": 17},
}
N_BOOTSTRAP = 1000
SEED = 42


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_once(X_train, y_train, X_test, y_test, seed=42, epochs=50, lr=1e-3, bs=64):
    torch.manual_seed(seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    n_pos = y_tr_t.sum().item()
    n_neg = len(y_tr_t) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])

    model = MLPProbe(X_tr_t.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_val_t), y_val_t).item()
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = (model(X_test_t) > 0).numpy().astype(int)
    return balanced_accuracy_score(y_test, preds)


def load_split(path, peak_layer):
    d = torch.load(path, map_location="cpu", weights_only=True)
    act = d["activations"][:, peak_layer, :]
    if act.dtype == torch.bfloat16:
        act = act.float()
    act = act.numpy()
    labels = d["labels"].numpy()
    n_train = d["split_sizes"]["train"]
    return act[:n_train], labels[:n_train], act[n_train:], labels[n_train:]


def run_bootstrap(name, cfg):
    print(f"\n{'='*60}")
    print(f"  {name}  (layer {cfg['peak_layer']}, n_boot={N_BOOTSTRAP})")
    print(f"{'='*60}")

    en_train, en_labels, _, _ = load_split(cfg["en_path"], cfg["peak_layer"])
    _, _, zh_test, zh_labels = load_split(cfg["zh_path"], cfg["peak_layer"])
    print(f"  EN train: {en_train.shape}, ZH test: {zh_test.shape}")
    print(f"  Label dist - EN: {np.bincount(en_labels)}, ZH: {np.bincount(zh_labels)}", flush=True)

    rng = np.random.RandomState(SEED)
    deltas, linear_accs, mlp_accs = [], [], []
    t0 = time.time()

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(len(en_train), size=len(en_train), replace=True)
        X_boot = en_train[idx]
        y_boot = en_labels[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        scaler = StandardScaler()
        X_boot_s = scaler.fit_transform(X_boot)
        zh_test_s = scaler.transform(zh_test)

        lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", C=1.0)
        lr.fit(X_boot_s, y_boot)
        l_acc = balanced_accuracy_score(zh_labels, lr.predict(zh_test_s))

        m_acc = train_mlp_once(X_boot_s, y_boot, zh_test_s, zh_labels, seed=SEED+i)

        deltas.append(m_acc - l_acc)
        linear_accs.append(l_acc)
        mlp_accs.append(m_acc)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_BOOTSTRAP - i - 1)
            print(f"  [{i+1:4d}/{N_BOOTSTRAP}] delta={np.mean(deltas):.4f}  "
                  f"lr={np.mean(linear_accs):.4f}  mlp={np.mean(mlp_accs):.4f}  "
                  f"ETA={eta:.0f}s", flush=True)

    deltas = np.array(deltas)
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    mean_d = float(np.mean(deltas))

    result = {
        "mean_delta": round(mean_d, 6),
        "std_delta": round(float(np.std(deltas)), 6),
        "ci_lower": round(ci_lo, 6),
        "ci_upper": round(ci_hi, 6),
        "excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        "ci_sign": "positive" if ci_lo > 0 else ("negative" if ci_hi < 0 else "contains_zero"),
        "mean_linear_acc": round(float(np.mean(linear_accs)), 6),
        "mean_mlp_acc": round(float(np.mean(mlp_accs)), 6),
        "n_valid_boots": len(deltas),
        "peak_layer": cfg["peak_layer"],
    }

    print(f"\n  RESULT: delta={mean_d:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"excludes_zero={result['excludes_zero']}  "
          f"linear={result['mean_linear_acc']:.4f}  mlp={result['mean_mlp_acc']:.4f}", flush=True)
    return result


if __name__ == "__main__":
    # os.chdir to project root if needed
    os.makedirs("logs", exist_ok=True)

    all_results = {}
    t_start = time.time()

    for name, cfg in CONDITIONS.items():
        all_results[name] = run_bootstrap(name, cfg)
        with open("data/c3_mlp_bootstrap_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  All done in {total:.0f}s ({total/60:.1f}min)")
    print(f"{'='*60}")
    print("\n  Summary:")
    for name, r in all_results.items():
        tag = "***SIG***" if r["excludes_zero"] else "   n.s.  "
        print(f"  {tag} {name:20s}: delta={r['mean_delta']:+.4f} [{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}]")
    print(f"\n  Saved to data/c3_mlp_bootstrap_results.json")
