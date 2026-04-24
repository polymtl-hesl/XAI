# XAI4ADS-B

Reproducibility code for the DASC 2026 paper:
**"From Explainability to Certifiability: XAI-Driven Assurance for AI-Based Avionics Systems"**
*Adem Hmissa, Charles de Malefette, Jean-Yves Ouattara, Felipe Magalhaes, Ahmad Shahnejat Bushehri, Rim Zrelli, Adel Abusitta, Gabriela Nicolescu*

Five standalone experiments evaluate DeepSHAP explanations as certification evidence for an ADS-B intrusion detection model, each targeting one assurance claim.

---

## 1. Requirements

- Python 3.9 or later
- ~500 MB disk space (model weights + processed data)

---

## 2. Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3. Running the Experiments

Run each script from the **repo root**. All outputs are saved to `outputs/`.

### Experiment 1 — Behavioral Plausibility

> Do the top-attributed features match what a domain expert would expect for each attack type?

```bash
python experiments/exp1_behavioral_plausibility.py
```

**Outputs:**
```
outputs/exp1_plausibility_heatmap.png   # 4×25 heatmap of mean |SHAP| per feature per attack
outputs/exp1_plausibility_heatmap.pdf
outputs/exp1_top3.json                  # top-3 features + scores per attack type
```

**Expected result:** `altitude_std` leads for Gaussian noise, `vertical_rate_min` for landing injection, `groundspeed_max` for trajectory injections.

---

### Experiment 2 — Decision Justifiability

> Can every individual detection be accompanied by a human-reviewable rationale?

```bash
python experiments/exp2_decision_justifiability.py
```

**Outputs:**
```
outputs/exp2_justifiability_waterfall.png   # per-alert attribution bar charts (4 attack types)
outputs/exp2_justifiability_waterfall.pdf
```

**Expected result:** Each panel decomposes a detection into feature-level contributions, ordered by magnitude, with top features physically consistent with the attack.

---

### Experiment 3 — Explanation Consistency

> Are DeepSHAP explanations reproducible and stable under sensor noise?

```bash
python experiments/exp3_explanation_consistency.py
```

**Outputs:**
```
outputs/exp3_consistency.png   # (a) L2 determinism distribution; (b) stability vs noise magnitude
outputs/exp3_consistency.pdf
outputs/exp3_consistency.json  # max_l2, stability values per epsilon
```

**Expected result:** L2 distance between two identical runs = `0.0` (bit-for-bit deterministic). Mean L2 shift at ε=0.01 ≈ `0.031`.

---

### Experiment 4 — Robustness Indication

> Are the top-attributed features genuinely decisive, and do attributions hold up under perturbation?

```bash
python experiments/exp4_robustness_indication.py
```

**Outputs:**
```
outputs/exp4_robustness.png   # (a) faithfulness curve DR(k); (b) cosine similarity vs noise
outputs/exp4_robustness.pdf
outputs/exp4_robustness.json  # DR values for k=0..13, cosine similarity per epsilon
```

**Expected result:** Detection rate drops from `1.000` → `0.366` when top-3 features are removed (63.4% drop). Cosine similarity stays above `0.9` for ε ≤ 0.15.

---

### Experiment 5 — Operational Coverage

> Are explanation artifacts generated for every flight/attack combination?

```bash
python experiments/exp5_operational_coverage.py
```

**Outputs:**
```
outputs/exp5_coverage.png    # (a) anomaly count heatmap; (b) top-1 feature per cell
outputs/exp5_coverage.pdf
outputs/exp5_coverage.json   # 28×2 matrix of counts and top-1 features
```

**Expected result:** All 28 of 28 (flight, attack) pairs produce explained anomalies. Total: 18,269 explained instances.

---

## 4. Run All Experiments

```bash
for exp in 1 2 3 4 5; do
    python experiments/exp${exp}_*.py
done
```

Or on Windows:
```powershell
foreach ($exp in 1..5) {
    python (Get-ChildItem experiments/exp${exp}_*.py)
}
```

---

## 5. Repository Structure

```
XAI4ADS-B/
├── model/
│   ├── model.py          # MLP architecture (25 → 128 → 64 → 32 → 1, sigmoid)
│   ├── utils.py          # Legacy preprocessing utilities (not used by experiments)
│   └── explanation.py    # Legacy SHAP exploration utilities (not used by experiments)
├── experiments/
│   ├── exp1_behavioral_plausibility.py
│   ├── exp2_decision_justifiability.py
│   ├── exp3_explanation_consistency.py
│   ├── exp4_robustness_indication.py
│   └── exp5_operational_coverage.py
├── data/
│   ├── processed/default/    # Pre-processed feature windows (pickle)
│   └── flight_index.json     # Flight names and window offsets (no CSVs needed)
├── trained_model/
│   └── default.pth           # Pre-trained weights (99.89% test accuracy)
├── outputs/                  # Created at runtime — figures and JSON results
├── requirements.txt
└── .gitignore
```

---

## 6. Citation

```bibtex
@inproceedings{hmissa2026xai,
  title     = {From Explainability to Certifiability: {XAI}-Driven Assurance
               for {AI}-Based Avionics Systems},
  author    = {Hmissa, Adem and de Malefette, Charles and Ouattara, Jean-Yves
               and Magalhaes, Felipe and Shahnejat Bushehri, Ahmad
               and Zrelli, Rim and Abusitta, Adel and Nicolescu, Gabriela},
  booktitle = {Proceedings of the AIAA/IEEE Digital Avionics Systems Conference (DASC)},
  year      = {2026}
}
```
