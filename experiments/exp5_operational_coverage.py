"""
Experiment 5 – Operational Coverage
=====================================
Claim: The explanation system can produce a valid SHAP artifact for every
       (flight, attack-type) scenario in the test set, with no gap in
       coverage.

What it does:
  Iterates over all 7 test flights and all 4 attack types (28 scenarios).
  For each scenario:
    1. Detects anomaly instances using the trained model.
    2. Runs DeepSHAP on those instances.
    3. Records the count of explained instances and the top-1 attributed
       feature.

  Produces a two-panel figure:
    Left  — heatmap of anomaly instance counts per cell.
    Right — heatmap showing the top-1 SHAP feature label per cell.

  Saves a JSON file with the full coverage matrix and summary statistics.

Outputs:
  exp5_coverage.png / .pdf
  exp5_coverage.json   (numerical results + top-1 feature matrix)
"""

import sys, os, pickle, json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "model"))

import numpy as np
import matplotlib
import matplotlib.colors
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
print(f"Flights: {flight_names}")

def get_anomalies(attack_idx, flight_idx):
    s, e = flight_cumlen[flight_idx], flight_cumlen[flight_idx + 1]
    X = X_attack_n[attack_idx][s:e]
    X_safe = X_safe_1_n[s:min(e, len(X_safe_1_n))]
    with torch.no_grad():
        preds = model.forward(torch.FloatTensor(X)).squeeze(-1).numpy()
    idx = np.where(preds > THRESHOLD)[0]
    return X[idx], X_safe

# ── EXPERIMENT 5 ──────────────────────────────────────────────────────────────
print("\n=== Experiment 5: Operational Coverage ===")

count_matrix = np.zeros((N_FLIGHTS, len(ATTACK_NAMES)), dtype=int)
top1_matrix  = np.full((N_FLIGHTS, len(ATTACK_NAMES)), "", dtype=object)
covered      = np.zeros((N_FLIGHTS, len(ATTACK_NAMES)), dtype=int)

for f_idx in range(N_FLIGHTS):
    for a_idx, a_name in enumerate(ATTACK_NAMES):
        X_a, X_s = get_anomalies(a_idx, f_idx)
        count_matrix[f_idx, a_idx] = len(X_a)
        if len(X_a) >= 1:
            bg  = torch.FloatTensor(np.mean(X_s, axis=0, keepdims=True))
            exp = shap.DeepExplainer(model, bg)
            sv  = exp(torch.FloatTensor(X_a)).values[:, :, 0]
            top1_matrix[f_idx, a_idx] = feat_names[np.argmax(np.abs(sv).mean(axis=0))]
            covered[f_idx, a_idx] = 1
        print(f"  Flight {flight_names[f_idx]:10s}  attack {a_name:12s}  "
              f"anomalies={len(X_a):4d}  top1={top1_matrix[f_idx, a_idx]}")

total     = N_FLIGHTS * len(ATTACK_NAMES)
explained = int(covered.sum())
total_inst = int(count_matrix.sum())
print(f"\n  Coverage: {explained}/{total} scenarios")
print(f"  Total anomaly instances explained: {total_inst}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: anomaly count heatmap
im1 = axes[0].imshow(count_matrix, aspect="auto", cmap="Blues")
axes[0].set_xticks(range(len(ATTACK_NAMES)))
axes[0].set_xticklabels(ATTACK_LABELS, fontsize=8, rotation=20, ha="right")
axes[0].set_yticks(range(N_FLIGHTS))
axes[0].set_yticklabels(flight_names, fontsize=8)
axes[0].set_title("(a) Anomaly instance count\nper (flight, attack) scenario", fontsize=9)
plt.colorbar(im1, ax=axes[0], label="# anomaly instances")
for fi in range(N_FLIGHTS):
    for ai in range(len(ATTACK_NAMES)):
        val = count_matrix[fi, ai]
        axes[0].text(ai, fi, str(val), ha="center", va="center", fontsize=7,
                     color="white" if val > count_matrix.max() * 0.6 else "black")

# Right: top-1 feature label overlay — solid green cells, white bold text
BG_COLOR = "#1b5e20"   # deep green: high contrast against white text
cell_arr  = np.ones((N_FLIGHTS, len(ATTACK_NAMES)))
axes[1].imshow(cell_arr, aspect="auto",
               cmap=matplotlib.colors.ListedColormap([BG_COLOR]), vmin=0, vmax=1)
for x in np.arange(-0.5, len(ATTACK_NAMES), 1):
    axes[1].axvline(x, color="white", linewidth=0.8)
for y in np.arange(-0.5, N_FLIGHTS, 1):
    axes[1].axhline(y, color="white", linewidth=0.8)
axes[1].set_xticks(range(len(ATTACK_NAMES)))
axes[1].set_xticklabels(ATTACK_LABELS, fontsize=8, rotation=20, ha="right")
axes[1].set_yticks(range(N_FLIGHTS))
axes[1].set_yticklabels(flight_names, fontsize=8)
axes[1].set_title("(b) Top-1 attributed feature\nper (flight, attack) pair", fontsize=9)
ABBREV = {"vertical_rate": "vrate", "groundspeed": "gspd",
          "altitude": "alt", "mean_abs_change": "mac"}
for fi in range(N_FLIGHTS):
    for ai in range(len(ATTACK_NAMES)):
        txt = top1_matrix[fi, ai]
        if txt:
            short = txt
            for long, abbr in ABBREV.items():
                short = short.replace(long, abbr)
            axes[1].text(ai, fi, short, ha="center", va="center",
                         fontsize=7.5, color="white", fontweight="bold")

plt.suptitle("Operational Coverage: Explanation Artifacts Across All 28 Scenarios",
             fontsize=11)
plt.tight_layout()
plt.savefig(str(OUT / "exp5_coverage.pdf"), bbox_inches="tight")
plt.savefig(str(OUT / "exp5_coverage.png"), dpi=180, bbox_inches="tight")
plt.close()

results = {
    "total_scenarios":      total,
    "explained_scenarios":  explained,
    "total_anomaly_instances": total_inst,
    "coverage_rate":        explained / total,
    "anomaly_counts":       count_matrix.tolist(),
    "top1_features":        top1_matrix.tolist(),
    "flight_names":         flight_names,
    "attack_names":         ATTACK_NAMES,
}
with open(OUT / "exp5_coverage.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n  Figure saved to {OUT}/exp5_coverage.png")
print(f"  Results saved to {OUT}/exp5_coverage.json")
print("Experiment 5 complete.")
