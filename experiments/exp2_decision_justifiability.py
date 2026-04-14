"""
Experiment 2 – Decision Justifiability
=======================================
Claim: For any individual alert, the system can produce a human-readable
       explanation showing which features drove the anomaly decision and
       by how much.

What it does:
  - For each of the 4 attack types, selects the first flight that has at
    least one detected anomaly.
  - Runs DeepSHAP on the first anomalous instance from that flight.
  - Produces a 2x2 waterfall bar-chart showing the top-8 contributing
    features (positive = pushes toward anomaly, negative = pushes toward normal).

Outputs:
  exp2_justifiability_waterfall.png / .pdf
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

features   = ["altitude", "groundspeed", "vertical_rate", "x", "y"]
stats      = ["std", "max", "min", "median", "mean_abs_change"]
feat_names = [f"{f}_{s}" for f in features for s in stats]

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
    return X[idx], X_safe, flight_names[flight_idx]

# ── EXPERIMENT 2 ──────────────────────────────────────────────────────────────
print("\n=== Experiment 2: Decision Justifiability ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for a_idx, a_name in enumerate(ATTACK_NAMES):
    # find first flight with anomalies
    for f_idx in range(N_FLIGHTS):
        X_anom, X_safe, fname = get_anomalies(a_idx, f_idx)
        if len(X_anom) >= 1:
            break

    print(f"  {a_name}: using flight {fname}, {len(X_anom)} anomaly instances")

    background = torch.FloatTensor(np.mean(X_safe, axis=0, keepdims=True))
    explainer  = shap.DeepExplainer(model, background)
    sv         = explainer(torch.FloatTensor(X_anom[:1]))

    shap_vals = sv.values[0, :, 0]
    top_n     = 8
    order     = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    rest      = shap_vals[np.argsort(np.abs(shap_vals))[::-1][top_n:]].sum()

    names_top = [feat_names[i] for i in order]
    vals_top  = shap_vals[order]

    bar_names = [f"All others ({len(shap_vals) - top_n})"] + names_top[::-1].tolist()
    bar_vals  = [rest] + list(vals_top[::-1])
    colors    = ["#d0d0d0"] + ["#ff4d4d" if v > 0 else "#4d79ff" for v in bar_vals[1:]]

    ax = axes[a_idx]
    ax.barh(bar_names, bar_vals, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{ATTACK_LABELS[a_idx]}\n(flight: {fname})", fontsize=9)
    ax.set_xlabel("SHAP value  (red = toward anomaly, blue = toward normal)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", labelsize=7)

plt.suptitle("Per-Alert Feature Attribution  —  Decision Justifiability", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(str(OUT / "exp2_justifiability_waterfall.pdf"), bbox_inches="tight")
plt.savefig(str(OUT / "exp2_justifiability_waterfall.png"), dpi=180, bbox_inches="tight")
plt.close()
print(f"  Figure saved to {OUT}/exp2_justifiability_waterfall.png")
print("Experiment 2 complete.")
