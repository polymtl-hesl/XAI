"""
Experiment 3 – Explanation Consistency
=======================================
Claim: DeepSHAP produces identical explanations across repeated runs on the
       same inputs (determinism), and explanations change only gradually as
       inputs are slightly perturbed (stability).

What it does:
  Part A – Determinism:
    Runs DeepSHAP twice on the exact same set of anomaly instances from the
    Landing-attack / flight-0 scenario. Computes the L2 distance between the
    two SHAP vectors for each instance. A max L2 of 0.000 confirms full
    determinism.

  Part B – Stability:
    Adds zero-mean Gaussian noise with increasing standard deviation
    (eps = 0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00) to the same instances and
    measures how much the SHAP explanations shift (mean L2 distance vs clean).
    Gradual increase is expected; a sudden jump would indicate fragility.

Outputs:
  exp3_consistency.png / .pdf
  exp3_consistency.json   (numerical results)
"""

import sys, os, pickle, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from model import ExplainableModel

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ── paths ────────────────────────────────────────────────────────────────────
DATA_FOLDER = REPO_ROOT / "data" / "processed" / "default"
MODEL_PATH  = REPO_ROOT / "trained_model" / "default.pth"
OUT         = REPO_ROOT / "outputs"
OUT.mkdir(exist_ok=True)

WINDOW_SIZE = 60
THRESHOLD   = 0.5

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data …")
with open(DATA_FOLDER / "X_safe_1.pickle",  "rb") as f: X_safe_1  = pickle.load(f)
with open(DATA_FOLDER / "y_safe_1.pickle",  "rb") as f: y_safe_1  = pickle.load(f)
with open(DATA_FOLDER / "X_safe_2.pickle",  "rb") as f: X_safe_2  = pickle.load(f)
with open(DATA_FOLDER / "y_safe_2.pickle",  "rb") as f: y_safe_2  = pickle.load(f)
with open(DATA_FOLDER / "X_attack.pickle",  "rb") as f: X_attack  = pickle.load(f)
with open(DATA_FOLDER / "y_attack.pickle",  "rb") as f: y_attack  = pickle.load(f)

scaler     = StandardScaler()
X_safe_1_n = scaler.fit_transform(X_safe_1)
X_safe_2_n = scaler.fit_transform(X_safe_2)
X_attack_n = [scaler.fit_transform(X_attack[k]) for k in range(len(X_attack))]

X_all = X_attack_n.copy(); y_all = y_attack.copy()
X_all.append(X_safe_1_n); y_all.append(y_safe_1)
X_all.append(X_safe_2_n); y_all.append(y_safe_2)
X_n = np.vstack(X_all)
y   = np.concatenate(y_all)
X_train, X_test, y_train, y_test = train_test_split(X_n, y, test_size=0.2,  random_state=42)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# ── load model ─────────────────────────────────────────────────────────────────
print("Loading model …")
config_stub = {"model": {"input_layer": 128, "hidden_layer_1": 64,
                         "hidden_layer_2": 32, "init_range": 0.1}}
model = ExplainableModel(X_train.shape[1], config_stub)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
model.eval()

# ── flight index (pre-computed, no CSV reads needed) ─────────────────────────
with open(REPO_ROOT / "data" / "flight_index.json") as _f:
    _idx = json.load(_f)
flight_names  = _idx["flight_names"]
flight_cumlen = _idx["flight_cumlen"]

def get_anomalies(attack_idx, flight_idx):
    s, e = flight_cumlen[flight_idx], flight_cumlen[flight_idx + 1]
    X = X_attack_n[attack_idx][s:e]
    X_safe = X_safe_1_n[s:min(e, len(X_safe_1_n))]
    with torch.no_grad():
        preds = model.forward(torch.FloatTensor(X)).squeeze(-1).numpy()
    idx = np.where(preds > THRESHOLD)[0]
    return X[idx], X_safe

# ── EXPERIMENT 3 ──────────────────────────────────────────────────────────────
print("\n=== Experiment 3: Explanation Consistency ===")

# use Landing attack (index 1), flight 0
a_idx, f_idx = 1, 0
X_anom, X_safe = get_anomalies(a_idx, f_idx)
X_anom = X_anom[:50] if len(X_anom) > 50 else X_anom
print(f"  Using {len(X_anom)} anomaly instances from flight {flight_names[f_idx]}")

background = torch.FloatTensor(np.mean(X_safe, axis=0, keepdims=True))
explainer  = shap.DeepExplainer(model, background)

# Part A: Determinism
print("  Part A: Determinism (two identical runs) …")
sv1 = explainer(torch.FloatTensor(X_anom)).values[:, :, 0]
sv2 = explainer(torch.FloatTensor(X_anom)).values[:, :, 0]
l2_diffs = np.linalg.norm(sv1 - sv2, axis=1)
print(f"    Max L2 between runs:  {l2_diffs.max():.2e}")
print(f"    Mean L2 between runs: {l2_diffs.mean():.2e}")
det_result = {"max_l2": float(l2_diffs.max()), "mean_l2": float(l2_diffs.mean())}

# Part B: Stability under perturbation
print("  Part B: Stability under noise …")
epsilons        = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
mean_l2_per_eps = []
for eps in epsilons:
    noise    = np.random.normal(0, eps, X_anom.shape)
    X_noisy  = X_anom + noise
    sv_noisy = explainer(torch.FloatTensor(X_noisy)).values[:, :, 0]
    diffs    = np.linalg.norm(sv1 - sv_noisy, axis=1)
    mean_l2_per_eps.append(float(diffs.mean()))
    print(f"    eps={eps:.2f}  mean L2 shift = {diffs.mean():.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Subplot A: determinism histogram
ax1.hist(l2_diffs, bins=20, color="#4d79ff", edgecolor="white")
ax1.set_xlabel("L2 distance between the two SHAP runs", fontsize=9)
ax1.set_ylabel("Instance count", fontsize=9)
ax1.set_title("(a) Determinism: SHAP distance\nbetween two identical runs", fontsize=9)
ax1.axvline(l2_diffs.max(), color="red", linestyle="--", linewidth=1,
            label=f"max = {l2_diffs.max():.2e}")
ax1.legend(fontsize=8)

# Subplot B: stability curve
ax2.plot(epsilons, mean_l2_per_eps, marker="o", color="#ff4d4d", linewidth=2)
ax2.set_xlabel("Perturbation magnitude  (std of added Gaussian noise)", fontsize=9)
ax2.set_ylabel("Mean L2 shift of SHAP values", fontsize=9)
ax2.set_title("(b) Stability: Explanation shift\nunder input perturbation", fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle("Explanation Consistency Analysis", fontsize=11)
plt.tight_layout()
plt.savefig(str(OUT / "exp3_consistency.pdf"), bbox_inches="tight")
plt.savefig(str(OUT / "exp3_consistency.png"), dpi=180, bbox_inches="tight")
plt.close()

results = {
    "determinism": det_result,
    "stability":   dict(zip([str(e) for e in epsilons], mean_l2_per_eps)),
}
with open(OUT / "exp3_consistency.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"  Figure saved to {OUT}/exp3_consistency.png")
print(f"  Results saved to {OUT}/exp3_consistency.json")
print("Experiment 3 complete.")
