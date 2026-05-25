# Mosquito Flight Trajectory Prediction

**Dacon Monthly AI Competition** — Predict mosquito 3D position +80ms ahead from 11 observations (40ms interval).

- **Metric**: R-Hit@1cm (fraction of predictions within 1cm of ground truth)
- **Current Best OOF**: 0.6659 (v3_D)
- **Target**: LB ≥ 0.69

## Problem Setup

```
Input:  11 3D coordinates (x, y, z) at t = -400ms, -360ms, ..., -40ms, 0ms
Output: 3D coordinate at t = +80ms
```

## Pipeline

```
Raw trajectory
     │
     ▼
Kalman Filter (CV model, σ_obs=0.000267)  →  baseline OOF: 0.5964
     │
     │  residual (y - Kalman prediction)
     ▼
GRU v3_D (BiGRU-1L + Mish + curvature features + focal loss γ=2)
     │
     │  predict residual in yaw-rotated body frame
     ▼
Calibration (per-axis α search)
     │
     ▼
Final Prediction  →  OOF: 0.6659
```

## Key Architecture (v3_D)

| Component | Details |
|-----------|---------|
| Backbone | 1-layer Bidirectional GRU, hidden=64 |
| Activation | Mish |
| Sequence input | (N, 11, 13) — velocity, curvature, angular_vel, turn_rate |
| Scalar input | (N, 73) — motion stats + wavelet DWT + tier3 |
| Output | Kalman residual in body frame (tanh clip ±3cm) |
| Aux heads | head_F (disp from last obs) + head_W (weak Kalman residual) |
| Loss | Focal Euclidean (γ=2.0) + SoftHit |
| Training | 5-fold CV, 3 seeds, AdamW + CosineAnnealingLR, 400 epochs |

## Experiment Results

| Experiment | OOF | Notes |
|-----------|-----|-------|
| Kalman CV (baseline) | 0.5964 | sigma_obs=0.000267 |
| GRU v3_D | **0.6659** | Best overall |
| Transformer (v5) | 0.6620 | Too complex for 10K samples |
| Rotation Augmentation (v6) | 0.6608 | Redundant with yaw normalization |
| Multi-step Aux Task (v7) | 0.6635 | Marginal benefit |
| LightGBM (scalar features) | 0.5939 | Worse than Kalman |
| Kalman CA | 0.4489 | Wrong motion model |

## Key Insights

1. **CV Kalman is the right physics model**: at 40ms scale, constant velocity fits mosquito flight much better than constant acceleration
2. **Yaw normalization makes rotation augmentation redundant**: model already learns in body frame
3. **Shallow GRU beats Transformer**: with only 10K samples, 1-layer BiGRU generalizes better
4. **Focal loss helps with hard samples**: upweights sharp-turn trajectories (C4 cluster)
5. **All GRU models have Pearson correlation ≈ 1.0**: Kalman dominates prediction variance, making ensembling ineffective
6. **LightGBM fails**: sequential temporal patterns cannot be captured by 73 scalar features alone

## Files

```
├── train_v3.py      # GRU with Optuna — best model (v3_D config)
├── train_v5.py      # Transformer + Y-flip TTA + Optuna
├── train_v6.py      # Rotation augmentation experiment
├── train_v7.py      # Multi-step self-supervised auxiliary task
├── train_lgbm.py    # LightGBM baseline
├── ensemble_v2.py   # Ensemble experiments (correlation analysis)
├── INSIGHTS.md      # Detailed experiment log and lessons learned
└── submissions/     # All submission CSVs
```

## Environment

- GPU: NVIDIA RTX 4070 Ti Super (16GB)
- Python 3.11, PyTorch cu124
- Conda env: mosquito

## Usage

```bash
conda activate mosquito

# Train best model (v3_D)
python train_v3.py

# Best submission: submissions/sub_v3_D_+both_OOF0.6659.csv
```
