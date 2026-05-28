# 🦟 Mosquito Flight Trajectory Prediction
### 모기 비행 궤적 예측 (데이콘 월간 AI 경진대회)

> **Task**: Predict a mosquito's 3D position **+80ms** ahead from 11 observations at 40ms intervals  
> **과제**: 40ms 간격 11시점 3D 좌표 관측 → 80ms 후 위치 예측  
> **Metric**: R-Hit@1cm — 예측값이 실제값에서 1cm 이내면 적중  
> **Best OOF**: **0.6669 (v3_aug)** | **LB (Public)**: **~0.6780**

---

## 🤖 Multi-Agent Orchestration & AI-Driven Development
본 프로젝트는 단순한 모델링을 넘어, **두 개의 강력한 LLM 에이전트(Claude, Gemini)를 병렬로 지휘(Orchestration)하며 복잡한 아키텍처를 단기간에 구현한 "AI-Native 개발 파이프라인"**의 결과물입니다.

* **Tech Lead (Human)**: 논문 리서치(IaKNN 등)를 통한 도메인 지식(곡률, 비틀림) 설계, 리스크 관리 및 최종 의사결정
* **Agent 1 (Claude 3.5)**: 안전한 파라미터 기반 앙상블 로직 구축 및 모델 안정성 검증
* **Agent 2 (Gemini 1.5 - Antigravity)**: 하이 리스크-하이 리턴 아키텍처(Adaptive Kalman Filter) 개발 및 백그라운드 학습 자동화
* **System Prompting (`AGENTS.md`)**: 멀티 에이전트 간의 환각(Hallucination) 방지 및 공통 코딩 컨벤션 강제를 위한 CI/CD급 규칙 문서화

👉 **AI의 실패와 한계, 그리고 이를 극복한 과정은 [INSIGHTS.md](INSIGHTS.md)와 [AGENTS.md](AGENTS.md)에 상세히 기록되어 있습니다.**

---

## 📐 문제 설정 (Problem Setup)

```
관측 (Input)                                       예측 (Predict)
──────────────────────────────────────────────────────────────
t= -400ms  -360ms  -320ms  ...  -40ms    0ms   →   +80ms
    •──────•──────•──────  ...  •────────•           ★
   (x,y,z) (x,y,z) ...              (x,y,z)      어디에?
                                                  Where?
   ←──────── 11개 관측점 (11 observations) ────→

각 관측점: 3D 좌표 (x, y, z) in meters
모기 평균 속도: ~0.5–2 m/s  |  평균 가속도: ~10–20 m/s²
```

---

## 🔁 파이프라인 (Pipeline)

```
  원본 궤적 (Raw Trajectory)
         │
         │
  ┌──────┴──────┐
  │             │
  ↓             ↓
[칼만 필터]   [피처 엔지니어링]
 Kalman CV     Feature Engineering
 σ=0.000267    seq: (N,11,13)  ← 곡률/속도/가속도
 R-Hit=0.5964  scal: (N,73)   ← 통계/Wavelet/Tier3
  │             │
  │  잔차 예측   │
  │  (residual) │
  │             ↓
  │       [GRU v3_D]
  │        BiGRU 1층
  │        hidden=64
  │        Focal Loss γ=2
  │             │
  │    잔차 + 역회전
  │    (body frame → world frame)
  │             │
  └──────┬──────┘
         ↓
  [최종 예측 = 칼만 + GRU 잔차]
  [Calibration: per-axis α]
         │
         ↓
   R-Hit@1cm = 0.6659
```

---

## 🏆 모델 성능 비교 (Results)

| 모델 | OOF R-Hit | 비고 / Notes |
|------|:---------:|-------------|
| Kalman CV (물리 베이스라인) | 0.5964 | 등속도 칼만, σ=0.000267 |
| Kalman CA (등가속도) | 0.4489 ❌ | CV보다 훨씬 나쁨 |
| GRU v3_B (Mish 추가) | 0.6631 | — |
| GRU v3_C (+곡률 피처) | 0.6641 | — |
| **GRU v3_D (+Focal Loss)** | **0.6659 ⭐** | **현재 최고 / Best** |
| Transformer v5 | 0.6620 ❌ | 데이터 부족으로 과적합 |
| Rotation Aug v6 | 0.6608 ❌ | yaw norm과 중복 |
| Multi-step Aux v7 | 0.6635 | 미개선 |
| LightGBM | 0.5939 ❌ | 시계열 패턴 포착 불가 |

---

## 🧠 v3_D가 최고인 이유 (Why v3_D Wins)

**4가지 핵심 설계 결정 / 4 Key Design Decisions:**

**① 칼만 잔차 예측** — GRU가 절대 좌표가 아닌 "칼만 오차"만 보정  
*GRU corrects only the Kalman error, not the full position*

**② Body Frame 정규화** — 마지막 속도 방향 기준으로 좌표 회전 → 방향 불변  
*Rotate coordinates to body frame → direction-invariant learning*

**③ 곡률(Curvature) 피처** — 급기동 샘플을 미리 감지  
*Detect sharp-turn trajectories before they happen*

**④ Focal Loss γ=2** — 어려운 샘플(급기동)에 높은 가중치  
*Upweight hard samples (sharp turns) during training*

→ **상세 분석은 [INSIGHTS.md](INSIGHTS.md) 참조**

---

## ⚙️ v3_D 최적 설정 (Optimal Config)

```python
# 칼만 필터 / Kalman Filter
sigma_obs  = 0.000267   # Optuna로 탐색 / found by Optuna
sigma_proc = 1.0

# GRU 아키텍처 / GRU Architecture
hidden       = 64         # 128은 과적합 / 128 overfits
num_layers   = 1          # 2층보다 1층이 우수 / 1L > 2L
bidirectional = True
fc_hidden    = 128
dropout      = 0.25

# 학습 / Training
lr            = 0.000989  # CosineAnnealingLR
weight_decay  = 0.000886
focal_gamma   = 2.0       # 핵심! / Key!
lF, lW        = 0.3, 0.3  # 보조 헤드 가중치 / aux head weights

# 검증 / Validation
n_folds = 5   # KFold
n_seeds = 3   # 안정성을 위한 다중 seed / for stability
```

---

## 🗂️ 코드 구조 (Code Structure)

```
mosquitoes/
├── 🏆 train_v3.py      Best model (v3_D) — Kalman + BiGRU + Focal Loss
├── 🔬 train_v5.py      Transformer experiment (failed)
├── 🔬 train_v6.py      Rotation augmentation (failed)
├── 🔬 train_v7.py      Multi-step self-supervised (no improvement)
├── 🔬 train_lgbm.py    LightGBM baseline (failed)
├── 📊 ensemble_v2.py   Ensemble analysis (corr=1.0, no benefit)
├── 📖 INSIGHTS.md      Full experiment log & lessons learned
└── open/               Data (gitignored)
```

---

## 🚀 실행 방법 (Usage)

```bash
# 환경 활성화 / Activate environment
conda activate mosquito

# 최고 모델 학습 / Train best model
python train_v3.py

# 결과물 / Output
# submissions/sub_v3_D_+both_OOF0.6659.csv
```

**환경 / Environment**: Python 3.11 · PyTorch cu124 · NVIDIA RTX 4070 Ti Super (16GB)

---

## 📚 참고 논문 (References)

| 논문 | 아이디어 적용 |
|-----|-------------|
| VECTOR (arXiv:2410.23305) | 속도 기반 GRU 입력, Whitening 정규화 |
| IaKNN (arXiv:1902.10928) | Kalman + 딥러닝 결합, 보조 헤드 |
| WTFTP (Nature Comm. 2023) | Wavelet 시간-주파수 피처 |
