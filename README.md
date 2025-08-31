# AI-Assisted Mole Detection for Teledermatology Triage
**Triaging system via Vision Transformer (ViT) and Nested Hierarchical Transformer (NesT) to detect nevi from dermatological images**
*Companion codebase to the published study in Informatics in Medicine Unlocked (2023). Please find the paper at https://doi.org/10.1016/j.imu.2023.101311*

---

## Overview

This repository implements an AI triage system that identifies **mole (nevus) presence** in patient-submitted skin photos to support online dermatology workflows.  
The goal is to act as a **first-line filter** that prioritizes recall on mole-positive images to minimize false negatives in clinical triage.

---

## Key Contributions

- **Clinical triage framing:** Optimizes for **high recall** on mole-positive cases while maintaining strong precision and accuracy.  
- **Transformer-based baselines:** Comparison of **NesT vs SOTA models** (prior to 2022) for this teledermatology use case.  
- **Reproducible splits:** Repository includes initial train/val/test CSV structures to standardize experiments

---

## Repository Structure
mole-detection/
├─ src/ # Model code, data pipeline, training/eval logic
├─ train.sh # Example training launcher
├─ requirements.txt # Python dependencies
├─ setup.py # Optional installable package metadata
├─ train_set_initial.csv # Train split manifest
├─ validation_set.csv # Validation split manifest
└─ test_set_initial.csv # Test split manifest


---

## Installation

```bash
git clone https://github.com/Debarpan98/mole-detection.git
cd mole-detection

python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---
## Data

The repository ships with CSV manifests for the initial splits:

train_set_initial.csv

validation_set.csv

test_set_initial.csv

Each CSV should map images to labels (binary: mole vs. no_mole).
If you are using your own data, replicate the same CSV format and update paths accordingly.

Tip: Store images in any folder structure you prefer; the CSVs should contain absolute or project-relative paths plus the label column.
Keep class imbalance in mind when reporting metrics.


