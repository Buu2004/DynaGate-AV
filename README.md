# Dynamic FFIA: Lightweight Gated Multimodal Fusion for Fish Feeding Intensity Assessment

The first dynamic gated expert-selection framework specifically designed for FFIA in aquaculture.

## Dataset
The dataset used in this project is [U-FFIA](https://zenodo.org/records/11059975)

## Features
- Supports **2-branch** (Audio + Video only) and **3-branch** with 4 fusion experts (Cross Attention, Simple Cross Attention, MBT, Self Attention).
- Dynamic lightweight CNN gating mechanism.
- FLOPs-aware regularization.
- Reproducible training & evaluation.

## How to Use

### 1. Training

Install the required dependencies:

```bash
pip install -r requirements.txt
````

Run training with cross-attention fusion:

```bash
python scripts/train.py --fusion_type cross_attn --n_runs 2 --epochs 50
```

For **2-branch only (no fusion)**:

```bash
python scripts/train.py --fusion_type none
```

---

### 2. Inference (One-Switch)

Run inference with cross-attention fusion:

```bash
python scripts/inference.py --fusion_type cross_attn --model_path best_model_run_1_cross_attn.pth
```

For **2-branch only (no fusion)**:

```bash
python scripts/inference.py --fusion_type none --model_path best_model_run_1_2branch.pth
```
