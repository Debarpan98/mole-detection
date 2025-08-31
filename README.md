# AI-Assisted Mole Detection for Teledermatology Triage
**Triaging system via Vision Transformer (ViT) and Nested Hierarchical Transformer (NesT) to detect nevi from dermatological images**
*Companion codebase to the published study in Informatics in Medicine Unlocked (2023)*

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
