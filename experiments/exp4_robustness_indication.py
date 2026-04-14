"""
Experiment 4 – Robustness Indication
=====================================
Claim: The SHAP explanations are faithful (removing the features they
       highlight degrades detection) and attribution rankings remain
       stable under realistic sensor noise.

What it does:
  Part A – Faithfulness (feature-removal test):
    Pools all anomaly instances across all attack types and flights.
    For k = 0 … 13, replaces the top-k SHAP-ranked features with their
    standardised baseline value (0) and re-measures the detection rate.
    A steep drop confirms that the highlighted features are genuinely
    responsible for the model's decision.

  Part B – Attribution robustness (cosine similarity vs noise):
    Takes 30 anomaly instances from the Landing/flight-0 scenario.
    Adds Gaussian noise at 7 levels (std = 0 … 1.0) and measures how
    similar the resulting SHAP vectors are to the clean-input SHAP vectors
    (mean cosine similarity). High similarity at low noise = stable ranking.

Outputs:
  exp4_robustness.png / .pdf
  exp4_robustness.json   (numerical results)
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

ATTACK_NAMES = ["Noise", "Landing", "Departing", "Manoeuvre"]
WINDOW_SIZE  = 60
THRESHOLD    = 0.5

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
N_FLIGHTS = len(flight_names)

def get_anomalies(attack_idx, flight_idx):
    s, e = flight_cumlen[flight_idx], flight_cumlen[flight_idx + 1]
    X = X_attack_n[attack_idx][s:e]
    X_safe = X_safe_1_n[s:min(e, len(X_safe_1_n))]
    with torch.no_grad():
        preds = model.forward(torch.FloatTensor(X)).squeeze(-1).numpy()
    idx = np.where(preds > THRESHOLD)[0]
    return X[idx], X_safe

# ── EXPERIMENT 4 ──────────────────────────────────────────────────────────────
print("\n=== Experiment 4: Robustness Indication ===")

# Part A: faithfulness — pool all anomaly instances
print("  Part A: Faithfulness (top-k feature removal) …")
all_X_anom, all_sv = [], []
for a_idx in range(len(ATTACK_NAMES)):
    for f_idx in range(N_FLIGHTS):
        X_a, X_s = get_anomalies(a_idx, f_idx)
        if len(X_a) < 2:
            continue
        bg  = torch.FloatTensor(np.mean(X_s, axis=0, keepdims=True))
        exp = shap.DeepExplainer(model, bg)
        sv  = exp(torch.FloatTensor(X_a)).values[:, :, 0]
        all_X_anom.append(X_a)
        all_sv.append(sv)

X_pool  = np.vstack(all_X_anom)
sv_pool = np.vstack(all_sv)
print(f"  Pooled {len(X_pool)} anomaly instances across all scenarios")

def detection_rate_after_removal(X, sv, k):
    """Zero-out top-k SHAP features and re-run the model."""
    X_mod = X.copy()
    for i in range(len(X)):
        top_k = np.argsort(np.abs(sv[i]))[::-1][:k]
        X_mod[i, top_k] = 0.0   # replace with standardised baseline
    with torch.no_grad():
        preds = model.forward(torch.FloatTensor(X_mod)).squeeze(-1).numpy()
    return float((preds > THRESHOLD).mean())

k_values  = list(range(0, 14))
acc_curve = [detection_rate_after_removal(X_pool, sv_pool, k) for k in k_values]
for k, acc in zip(k_values, acc_curve):
    print(f"    k={k:2d}: detection rate = {acc:.4f}")

# Part B: attribution robustness — cosine similarity vs noise
print("  Part B: Attribution robustness (cosine similarity) …")
X_a, X_s = get_anomalies(1, 0)   # Landing attack, flight 0
X_a = X_a[:30] if len(X_a) > 30 else X_a
bg  = torch.FloatTensor(np.mean(X_s, axis=0, keepdims=True))
exp = shap.DeepExplainer(model, bg)
sv_clean = exp(torch.FloatTensor(X_a)).values[:, :, 0]

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
cosine_sims  = []
for nl in noise_levels:
    X_noisy = X_a + np.random.normal(0, nl, X_a.shape)
    sv_n    = exp(torch.FloatTensor(X_noisy)).values[:, :, 0]
    dots    = (sv_clean * sv_n).sum(axis=1)
    norms   = np.linalg.norm(sv_clean, axis=1) * np.linalg.norm(sv_n, axis=1) + 1e-9
    cs      = float((dots / norms).mean())
    cosine_sims.append(cs)
    print(f"    noise std={nl:.2f}  cosine sim = {cs:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(k_values, acc_curve, marker="s", color="#4d79ff", linewidth=2)
ax1.axhline(acc_curve[0], color="grey", linestyle="--", linewidth=1, label="Baseline (k=0)")
ax1.fill_between(k_values, acc_curve, acc_curve[0], alpha=0.15, color="#4d79ff")
ax1.set_xlabel("Number of top SHAP features zeroed out (k)", fontsize=9)
ax1.set_ylabel("Anomaly detection rate", fontsize=9)
ax1.set_title("(a) Faithfulness: Detection rate\nwhen removing top-k SHAP features", fontsize=9)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
for k_mark in [1, 3, 5]:
    ax1.annotate(f"k={k_mark}\n{acc_curve[k_mark]:.2f}",
                 xy=(k_mark, acc_curve[k_mark]),
                 xytext=(k_mark + 0.4, acc_curve[k_mark] + 0.03),
                 fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))

ax2.plot(noise_levels, cosine_sims, marker="o", color="#ff4d4d", linewidth=2)
ax2.set_ylim(0, 1.05)
ax2.set_xlabel("Input noise magnitude (std of Gaussian noise)", fontsize=9)
ax2.set_ylabel("Mean cosine similarity with clean SHAP", fontsize=9)
ax2.set_title("(b) Attribution robustness: Cosine similarity\nbetween clean and perturbed SHAP", fontsize=9)
ax2.axhline(0.9, color="green", linestyle="--", linewidth=1, alpha=0.7, label="0.9 threshold")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

plt.suptitle("Robustness Indication Analysis", fontsize=11)
plt.tight_layout()
plt.savefig(str(OUT / "exp4_robustness.pdf"), bbox_inches="tight")
plt.savefig(str(OUT / "exp4_robustness.png"), dpi=180, bbox_inches="tight")
plt.close()

results = {
    "faithfulness": {str(k): float(v) for k, v in zip(k_values, acc_curve)},
    "cosine_sim":   {str(nl): float(cs) for nl, cs in zip(noise_levels, cosine_sims)},
}
with open(OUT / "exp4_robustness.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"  Figure saved to {OUT}/exp4_robustness.png")
print(f"  Results saved to {OUT}/exp4_robustness.json")
print("Experiment 4 complete.")
