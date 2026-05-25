# 🦟 모기 비행 궤적 예측 — 실험 인사이트 (Experiment Insights)

> **대회 / Competition**: 데이콘 월간 AI 경진대회  
> **과제 / Task**: 모기 3D 궤적 11점(40ms 간격) 관측 → 80ms 후 위치 예측  
> **Predict**: 3D mosquito position +80ms ahead from 11 observations (40ms apart)  
> **지표 / Metric**: R-Hit@1cm (예측값이 실제값 1cm 이내 = 적중)  
> **현재 최고 / Best**: OOF **0.6659**, LB Public ~**0.6780**

---

## 📊 전체 실험 결과 한눈에 (All Results at a Glance)

```
R-Hit@1cm (높을수록 좋음 / Higher is better)

  0.70 ┤
       │
  0.69 ┤  ← 목표 / Target
       │
  0.68 ┤
       │
  0.67 ┤              ██████  ← 최고! v3_D (0.6659)
       │         ████  ████
  0.66 ┤    ████  ████  ████  ████
       │    ████  ████  ████  ████
  0.65 ┤    ████  ████  ████  ████
       │    ████  ████  ████  ████
  0.60 ┤  ████████████████████████████████████
       │  ████████████████████████████████████
  0.59 ┤  ████  ████████████████████████████
       │  ████
  0.59 ┤  ████                               ████
       └──────────────────────────────────────────
         Kalman  v3_B   v3_C  [v3_D]  v5    v6    v7   LGBM
         (CV)   (Mish) (+곡률) (최고) (Tr.) (Aug) (ms)
```

| 모델 / Model | OOF R-Hit | 이전 대비 | 핵심 변경점 |
|-------------|:---------:|:-------:|------------|
| Kalman CV (물리 베이스라인) | 0.5964 | — | 등속도 칼만 필터 |
| Kalman CA (등가속도) | 0.4489 | **-0.1475** ❌ | 잘못된 물리 모델 |
| GRU v3_baseline | 0.6550 | +0.0586 | GRU + Whitening |
| GRU v3_B | 0.6631 | +0.0081 | Mish 활성화 함수 추가 |
| GRU v3_C | 0.6641 | +0.0010 | 곡률(curvature) 피처 추가 |
| **GRU v3_D** ⭐ | **0.6659** | **+0.0018** | **Focal Loss γ=2.0 추가** |
| Transformer v5 | 0.6620 | -0.0039 ❌ | Transformer 아키텍처 |
| Rotation Aug v6 | 0.6608 | -0.0051 ❌ | 4방향 회전 증강 |
| Multi-step v7 | 0.6635 | -0.0024 ❌ | 자기지도 보조 task |
| LightGBM | 0.5939 | -0.0025 ❌ | 트리 기반 모델 |

---

## 🏆 왜 v3_D가 가장 좋은가? (Why is v3_D the Best?)

v3_D는 네 가지 핵심 설계 결정이 맞물려서 최고 성능을 냅니다.  
*v3_D achieves the best score through four interconnected design decisions.*

---

### 결정 1️⃣ — 칼만 필터를 "바닥"으로 쓴다 (Use Kalman as the Floor)

**문제 / Problem**: 모기 위치를 절대 좌표로 직접 예측하면, 모델이 공간 전체를 학습해야 함  
**해결 / Solution**: 칼만 필터가 먼저 물리 기반 예측을 하고, GRU는 그 **잔차(오차)만** 보정

```
[입력 궤적]
    │
    ├──→ 칼만 필터 (등속도 모델)  ────────────────────────┐
    │    σ_obs = 0.000267                                  │
    │    80ms 외삽                                          │
    │    R-Hit = 0.5964                                    ↓
    │                                              [최종 예측] = 칼만 + GRU 잔차
    └──→ GRU (잔차만 학습)        ────────────────────────┘
         "칼만이 얼마나 틀렸나?"
         평균 잔차 크기 ≈ 1.2cm
```

**왜 효과적인가? / Why it works**:
- 칼만 필터가 전체 분산의 ~90%를 설명 → GRU는 나머지 ~10%만 집중
- 작은 목표값 → 학습 안정, 과적합 감소
- 물리 법칙(등속도) 위반 불가 → 황당한 예측 없음

---

### 결정 2️⃣ — Body Frame 회전 정규화 (Yaw Rotation Normalization)

**문제 / Problem**: 모기가 북쪽, 남쪽, 동쪽 어디로 날든 패턴은 같다 → 절대 방향이 무의미  
**해결 / Solution**: 마지막 속도 방향을 기준으로 좌표계를 회전시킨 뒤 잔차를 예측

```
절대 좌표계 (Before)          Body Frame (After)
  y                              "앞" (forward)
  ↑  모기가 북동쪽으로 날면?       ↑
  │   →  (dx, dy) = (0.7, 0.7)   │   → (d_fwd, d_side) = (1.0, 0.0)
  │                               │
  └───→ x                         └───→ "옆" (side)

  방향에 따라 값이 달라짐          항상 같은 표현 → 학습 쉬움
```

**왜 효과적인가? / Why it works**:
- 모델이 "어느 방향으로 가는가"가 아니라 "얼마나 직진/회전하는가"만 배움
- 데이터 10,000개가 사실상 "방향 정규화된" 무한 데이터처럼 활용됨
- → **이 결정 때문에 회전 증강(v6)이 오히려 역효과**: 이미 방향 불변이라 정보 중복

---

### 결정 3️⃣ — 곡률 피처로 "급기동"을 잡는다 (Curvature Features for Sharp Turns)

**문제 / Problem**: 어떤 모기는 갑자기 방향을 바꿈 → 단순 속도 피처로 구분 불가  
**해결 / Solution**: 물리량 `curvature`, `angular_velocity`, `turn_rate`를 log 변환 후 입력

```python
# 곡률 = 속도의 변화율 / Curvature = rate of change of velocity direction
curvature   = ‖v × a‖ / ‖v‖³     # 경로 곡률 (path curvature)
angular_vel = ‖v × a‖ / ‖v‖²     # 각속도 (angular velocity)
turn_rate   = arccos(v̂ₜ · v̂ₜ₊₁)  # 연속 방향 변화 (consecutive direction change)
```

**왜 효과적인가? / Why it works**:
- 직선 비행: curvature ≈ 0 → 칼만이 이미 잘 맞춤 → GRU 잔차 ≈ 0
- 급기동: curvature 크게 증가 → GRU가 "이건 칼만이 틀릴 것"을 미리 감지
- log1p 변환 → 극단값(급기동) 압축, 수치 안정

---

### 결정 4️⃣ — Focal Loss로 "어려운 샘플"에 집중 (Focal Loss for Hard Samples)

**문제 / Problem**: 쉬운 샘플(직선 비행)이 많아 → loss가 쉬운 샘플에 지배됨 → 급기동 예측 안 됨  
**해결 / Solution**: 오차가 큰 샘플일수록 가중치를 높이는 Focal Loss

```
일반 Loss:
  샘플 A (직선, 오차=0.3cm)  가중치: 1.0  ← 쉬운 샘플이 gradient 지배
  샘플 B (급기동, 오차=3cm)  가중치: 1.0

Focal Loss (γ=2.0):
  샘플 A (직선, 오차=0.3cm)  가중치: 0.1  ← 낮춤
  샘플 B (급기동, 오차=3cm)  가중치: 0.9  ← 높임 ★

  w = sigmoid((err - 1cm) / 0.2cm) ^ γ
```

**왜 효과적인가? / Why it works**:
- Fold 2처럼 급기동 샘플이 집중된 폴드에서 특히 효과적
- γ=2.0이 γ=1.0, γ=0(일반 loss) 대비 일관되게 우수
- 전체 R-Hit 향상: 쉬운 샘플을 더 맞추는 것보다 어려운 샘플을 "덜 틀리는" 것이 지표에 직접 기여

---

### 결정 5️⃣ (보너스) — 1층 BiGRU가 최적인 이유 (Why 1-Layer BiGRU is Optimal)

데이터 10,000개 × 시계열 길이 11 → 단순한 모델이 이깁니다.  
*With only 10K samples and sequence length 11, simpler models win.*

```
모델 복잡도 vs. 성능 (complexity vs. performance)

성능
0.666 ┤         ★ 1L BiGRU (v3_D)
      │       ·   ·
0.662 ┤     ·        ·  2L BiGRU
      │   ·              ·
0.658 ┤  ·                 ·  Transformer
      │
      └─────────────────────────────
        단순  →→→→→→→→→  복잡
        Simple           Complex

"10K 데이터에서 Transformer = 과적합의 시작"
```

| 아키텍처 | 파라미터 수 | OOF | 판정 |
|---------|:----------:|:---:|:----:|
| 1L BiGRU (hidden=64) | ~85K | 0.6659 | ✅ 최적 |
| 2L BiGRU (hidden=64) | ~170K | 0.6631 | ❌ 과적합 |
| Transformer (4head, 2L) | ~250K | 0.6620 | ❌ 과적합 |

---

## 🔬 실험별 상세 분석 (Detailed Experiment Analysis)

### ❌ 실패 1: Constant Acceleration Kalman
```
시도: 등가속도 칼만 (CV → CA)
결과: 0.4489 (CV 0.5964 대비 -0.1475)

실패 원인 / Why it failed:
  40ms 스케일에서 모기 가속도 ≈ 노이즈
  CA 모델은 존재하지 않는 가속도 추세를 "만들어냄"
  → 오히려 예측이 실제 궤적에서 멀어짐

교훈: 물리 모델은 데이터의 실제 스케일에 맞춰야 함
Lesson: The physics model must match the actual timescale of the data.
```

### ❌ 실패 2: 회전 증강 (Rotation Augmentation, v6)
```
시도: 0°, 90°, 180°, 270° 회전 → 훈련 데이터 4배
결과: 0.6608 (v3_D 대비 -0.0051)

실패 원인 / Why it failed:
  v3_D는 이미 yaw 회전 정규화(body frame)를 사용
  → 모델은 이미 방향 불변(rotation-invariant)
  → 회전 증강 = 중복 정보 + 학습 시간만 4배

  "이미 답을 알고 있는 문제를 4번 푸는 것과 같음"
  "Like solving a problem you already know the answer to, 4 times."

교훈: 모델 설계에서 처리한 불변성은 증강으로 다시 처리하지 말 것
Lesson: Don't augment for invariances already handled in model design.
```

### ❌ 실패 3: Y-flip TTA (v5)
```
시도: 예측값의 y좌표를 뒤집어서 평균 (Test-Time Augmentation)
결과: OOF 하락

실패 원인 / Why it failed:
  EDA 발견: 모기는 앞으로 나아가는 편향이 있음 (-0.360cm forward bias)
  → Y축 대칭성이 없음
  → Y-flip은 편향을 반전시켜서 오히려 예측 악화

  "모기는 한 방향으로 날지, 양방향으로 대칭되지 않음"
  "Mosquitoes fly in one direction; they are not y-symmetric."
```

### ❌ 실패 4: 앙상블 (Ensemble)
```
시도: 여러 GRU 모델 가중 평균
결과: 모든 모델 쌍 Pearson correlation = 1.0000

실패 원인 / Why it failed:
  칼만 필터 예측값이 전체 분산의 ~90% 설명
  GRU 잔차 ≈ 전체의 ~10% 뿐
  → 칼만이 같으면 최종 예측도 거의 동일

  잔차 std: x=1.36cm, y=1.33cm, z=0.94cm (칼만 기준)
  → 잔차끼리는 다양하지만, 최종 좌표 기준으로는 corr=1.0

교훈: 앙상블 효과 = 예측 다양성. 같은 기반을 쓰면 다양성이 없음
Lesson: Ensembling needs diverse predictions; shared Kalman base removes diversity.
```

### ❌ 실패 5: LightGBM
```
시도: 79차원 스칼라 피처 → LightGBM으로 칼만 잔차 예측
결과: 0.5939 (칼만 단독 0.5979보다 낮음!)

실패 원인 / Why it failed:
  스칼라 통계는 시계열 순서 정보를 버림
  "속도 평균 = 1.2m/s"는 알지만, "1번→2번→3번 관측의 가속도 변화"는 모름
  → 트리 모델은 패턴의 시간 의존성을 포착 불가

  GRU: 시계열을 순서대로 처리 → 시간 패턴 포착 ✅
  LightGBM: 통계값만 → 순서 정보 손실 ❌

교훈: 시계열 예측은 시계열 모델(GRU, LSTM)이 필요
Lesson: Time series prediction needs sequential models (GRU, LSTM), not tabular ones.
```

---

## 🧩 최종 파이프라인 구조 (Final Pipeline)

```
┌─────────────────────────────────────────────────────────────┐
│                      입력 데이터 / Input                      │
│  X_train: (10000, 11, 3)  — 11시점 × 3D 좌표              │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴─────────────┐
              ↓                          ↓
  ┌─────────────────────┐    ┌─────────────────────────────┐
  │  시계열 피처 빌드      │    │   스칼라 피처 빌드           │
  │  build_seq_curv()   │    │   build_scalar()            │
  │  (N, 11, 13)        │    │   + build_tier3()           │
  │                     │    │   + build_wavelet()         │
  │  · 상대위치 (3)      │    │   (N, 73)                   │
  │  · 속도 (3)         │    │                             │
  │  · 가속도 (3)       │    │   · 속도/가속도 통계 (26)    │
  │  · 속력 (1)         │    │   · 구간별 path length (19)  │
  │  · 곡률 (1)         │    │   · Wavelet DWT 계수 (27)   │
  │  · 각속도 (1)       │    │   · 노이즈 추정 (LOO, poly2)│
  │  · 방향변화율 (1)   │    └──────────────┬──────────────┘
  └──────────┬──────────┘                   │
             │                              │
             │  Whitening 정규화            │  StandardScaler
             ↓                             ↓
  ┌──────────────────────────────────────────────────────────┐
  │                   GRU v3_D 모델                          │
  │                                                          │
  │   Seq Input →  [BiGRU 1층, hidden=64]  → GRU Output    │
  │                        ↓                                 │
  │   Scal Input → [Linear 73→64, Mish]   → Scal Emb       │
  │                        ↓                                 │
  │        [Concat → FC(192→128, Mish) → Dropout(0.25)]     │
  │                        ↓                                 │
  │               ┌────────┴────────┐                       │
  │               ↓                 ↓                        │
  │          head_main         head_F / head_W              │
  │         (잔차 예측)         (보조 학습)                  │
  │       tanh × 0.03m                                      │
  └──────────────────────┬───────────────────────────────────┘
                         │
                         ↓ GRU 잔차 예측 (body frame)
  ┌──────────────────────────────────────────────────────────┐
  │                칼만 필터 예측 + 잔차 = 최종 예측          │
  │                                                          │
  │  ① 잔차를 역회전 (inv_rotate_xy)                        │
  │  ② 칼만 예측 + 역회전된 잔차                            │
  │  ③ Per-axis Calibration (α 탐색)                        │
  └──────────────────────────────────────────────────────────┘
                         │
                         ↓
              최종 제출 / Final Submission
              OOF: 0.6659  |  LB: ~0.6780
```

---

## 🔧 핵심 하이퍼파라미터 (Key Hyperparameters)

Optuna 80회 탐색으로 결정된 최적값입니다.  
*Optimal values determined by 80 Optuna trials.*

| 파라미터 | 최적값 | 탐색 범위 | 왜 이 값인가? |
|---------|:-----:|:--------:|-------------|
| `sigma_obs` (칼만 노이즈) | **0.000267** | 1e-4 ~ 5e-3 | 너무 크면 칼만이 과도하게 평탄화, 너무 작으면 노이즈에 민감 |
| `hidden` (GRU 은닉층) | **64** | 32, 64, 128 | 128은 10K 데이터에서 과적합 |
| `num_layers` | **1** | 1, 2 | 2층은 gradient 소실, 일관되게 1층이 우수 |
| `fc_hidden` | **128** | 64, 128, 256 | 64는 표현력 부족, 256은 과적합 |
| `dropout` | **0.25** | 0.1, 0.25, 0.4 | 0.1은 과적합, 0.4는 underfitting |
| `focal_gamma` | **2.0** | 0.0, 1.0, 2.0 | 급기동 샘플 업가중치 최적 |
| `lr` | **0.000989** | 1e-4 ~ 5e-3 | CosineAnnealingLR과 함께 ~1e-3이 안정적 |
| `lF`, `lW` (보조 헤드) | **0.3** | 0.1 ~ 0.5 | 너무 크면 주 task 방해, 너무 작으면 정규화 효과 없음 |

---

## 📈 다음에 시도할 것 (What to Try Next)

### 🟢 가능성 높음 / High Potential
| 아이디어 | 예상 효과 | 이유 |
|---------|:--------:|-----|
| **Stacking** (GRU OOF → meta-learner) | +0.002~0.005 | GRU와 Kalman의 다른 귀납 편향 결합 |
| **여러 σ_obs 칼만 앙상블** | +0.001~0.003 | 강/약 칼만이 서로 다른 오차 패턴 |
| **Physics-Informed Loss** | +0.001~0.002 | 속도/가속도 연속성 제약으로 불물리적 예측 방지 |

### 🟡 중간 가능성 / Medium Potential
| 아이디어 | 예상 효과 | 이유 |
|---------|:--------:|-----|
| **Conformer** (Conv+GRU) | ±0.001 | Local 패턴(CNN) + Global 패턴(GRU) |
| **더 많은 데이터 증강** (노이즈 주입) | ±0.001 | 측정 노이즈 시뮬레이션 |

### 🔴 이미 시도, 실패 / Already Tried, Failed
| 아이디어 | OOF 결과 | 실패 이유 |
|---------|:--------:|---------|
| ~~Rotation Augmentation~~ | 0.6608 ↓ | yaw norm과 중복 |
| ~~Y-flip TTA~~ | 하락 ↓ | 모기 비행 비대칭 |
| ~~CA Kalman~~ | 0.4489 ↓ | 잘못된 물리 모델 |
| ~~Transformer~~ | 0.6620 ↓ | 10K 데이터로 과적합 |
| ~~GRU 앙상블~~ | ≈동일 | Pearson corr = 1.0 |
| ~~LightGBM~~ | 0.5939 ↓ | 시계열 순서 정보 손실 |

---

## 💡 핵심 교훈 요약 (Key Lessons Summary)

```
┌─────────────────────────────────────────────────────────────┐
│  1. 물리 모델 먼저                                           │
│     Physics model first                                     │
│     "칼만 CV 0.5964는 복잡한 딥러닝보다 더 강력한 시작점"   │
│     "Kalman CV 0.5964 is a stronger start than raw DL"      │
├─────────────────────────────────────────────────────────────┤
│  2. 데이터 크기 = 모델 복잡도 상한                           │
│     Data size = ceiling for model complexity                │
│     "10K 샘플 → 1층 GRU. 파라미터 줄이면 성능 오름"         │
│     "10K samples → 1-layer GRU. Fewer params = better."    │
├─────────────────────────────────────────────────────────────┤
│  3. 증강 전 물리 대칭성 확인                                 │
│     Check physical symmetry before augmenting              │
│     "Y-flip 실패: 모기는 앞으로만 난다"                      │
│     "Y-flip failed: mosquitoes are not y-symmetric."        │
├─────────────────────────────────────────────────────────────┤
│  4. 중복 불변성 처리 금지                                    │
│     No redundant invariance handling                        │
│     "yaw norm 이미 함 → 회전 증강 하면 역효과"              │
│     "yaw norm already done → rotation aug = harmful."       │
├─────────────────────────────────────────────────────────────┤
│  5. 앙상블 ≠ 항상 좋음                                      │
│     Ensemble ≠ always better                                │
│     "corr=1.0이면 앙상블 무의미. 다양성이 전제조건"          │
│     "corr=1.0 means no benefit. Diversity is prerequisite." │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 코드 구조 (Code Structure)

```
mosquitoes/
│
├── 🏆 train_v3.py          ← 최고 성능 / Best model (v3_D, OOF 0.6659)
│                              Kalman + BiGRU + 곡률 피처 + Focal Loss
│
├── 🔬 train_v5.py          ← Transformer 실험 (OOF 0.6620, 실패)
│                              CLS token, Pre-LN, Y-flip TTA
│
├── 🔬 train_v6.py          ← 회전 증강 실험 (OOF 0.6608, 실패)
│                              0°/90°/180°/270° 회전, 4x 데이터
│
├── 🔬 train_v7.py          ← 자기지도 보조 task (OOF 0.6635, 미개선)
│                              10스텝 → 11번째 관측 변위 예측
│
├── 🔬 train_lgbm.py        ← LightGBM 실험 (OOF 0.5939, 실패)
│                              79차원 스칼라 피처 기반 트리 모델
│
├── 📊 ensemble_v2.py       ← 앙상블 실험 (corr=1.0, 무의미 확인)
│
├── 📖 INSIGHTS.md          ← 이 파일 / This file
│
└── open/                   ← 데이터 (gitignore)
    ├── train/              TRAIN_00001~10000.csv (11×3 좌표)
    ├── test/               TEST_00001~10000.csv
    └── train_labels.csv    정답 좌표 (x, y, z)
```
