"""
Experiment 1 – Behavioral Plausibility
=======================================
Claim: The features that DeepSHAP highlights as most important for anomaly
       detection match what a domain expert would expect for each attack type.

What it does:
  - Runs DeepSHAP on all detected anomalies for every (flight, attack) pair.
  - Computes the mean absolute SHAP value per feature, per attack type,
    aggregated across all 7 test flights.
  - Produces a row-normalised heatmap (attack types × 25 features).
  - Saves the top-3 most important features per attack type to a JSON file.

Outputs:
  exp1_plausibility_heatmap.png / .pdf
  exp1_top3.json
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

ATTACK_NAMES  = ["Noise", "Landing", "Departing", "Manoeuvre"]
ATTACK_LABELS = ["Gaussian Noise", "Landing Injection",
                 "Departing Injection", "Manoeuvre Injection"]
WINDOW_SIZE   = 60
THRESHOLD     = 0.5

# ── feature names ─────────────────────────────────────────────────────────────
features   = ["altitude", "groundspeed", "vertical_rate", "x", "y"]
stats      = ["std", "max", "min", "median", "mean_abs_change"]
feat_names = [f"{f}_{s}" for f in features for s in stats]   # 25 names

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
print(f"Model accuracy: {model.evaluate(X_test, y_test)*100:.2f}%")

# ── flight index (pre-computed, no CSV reads needed) ─────────────────────────
with open(REPO_ROOT / "data" / "flight_index.json") as _f:
    _idx = json.load(_f)
flight_names  = _idx["flight_names"]
flight_cumlen = _idx["flight_cumlen"]
N_FLIGHTS = len(flight_names)
print(f"Flights: {flight_names}")

def get_anomalies(attack_idx, flight_idx):
    s, e = flight_cumlen[flight_idx], flight_cumlen[flight_idx + 1]
    X = X_attack_n[attack_idx][s:e]
    X_safe = X_safe_1_n[s:min(e, len(X_safe_1_n))]
    with torch.no_grad():
        preds = model.forward(torch.FloatTensor(X)).squeeze(-1).numpy()
    idx = np.where(preds > THRESHOLD)[0]
    return X[idx], X_safe

# ── EXPERIMENT 1 ──────────────────────────────────────────────────────────────
print("\n=== Experiment 1: Behavioral Plausibility ===")

mean_shap_per_attack = {}

for a_idx, a_name in enumerate(ATTACK_NAMES):
    all_shap = []
    for f_idx in range(N_FLIGHTS):
        X_anom, X_safe = get_anomalies(a_idx, f_idx)
        if len(X_anom) < 3:
            continue
        background = torch.FloatTensor(np.mean(X_safe, axis=0, keepdims=True))
        explainer  = shap.DeepExplainer(model, background)
        sv = explainer(torch.FloatTensor(X_anom))
        all_shap.append(np.abs(sv.values[:, :, 0]))
    if all_shap:
        stacked = np.vstack(all_shap)
        mean_shap_per_attack[a_idx] = stacked.mean(axis=0)
    print(f"  {a_name}: {len(all_shap)} flights processed")

# heatmap
matrix      = np.array([mean_shap_per_attack[i] for i in range(len(ATTACK_NAMES))])
matrix_norm = matrix / matrix.max(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(matrix_norm, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(25))
ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(ATTACK_LABELS)))
ax.set_yticklabels(ATTACK_LABELS, fontsize=9)
ax.set_title("Normalised Mean |SHAP| per Feature per Attack Type", fontsize=11)
plt.colorbar(im, ax=ax, label="Normalised importance")
plt.tight_layout()
plt.savefig(str(OUT / "exp1_plausibility_heatmap.pdf"), bbox_inches="tight")
plt.savefig(str(OUT / "exp1_plausibility_heatmap.png"), dpi=180, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved to {OUT}/exp1_plausibility_heatmap.png")

# top-3 per attack
top3 = {}
for a_idx in range(len(ATTACK_NAMES)):
    vals    = mean_shap_per_attack[a_idx]
    top_idx = np.argsort(vals)[::-1][:3]
    top3[ATTACK_NAMES[a_idx]] = [(feat_names[i], float(vals[i])) for i in top_idx]
    print(f"  Top-3 [{ATTACK_NAMES[a_idx]}]: {top3[ATTACK_NAMES[a_idx]]}")

with open(OUT / "exp1_top3.json", "w") as f:
    json.dump(top3, f, indent=2)
print(f"  Top-3 saved to {OUT}/exp1_top3.json")
print("Experiment 1 complete.")
