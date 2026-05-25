# 모기 비행 궤적 예측 — 실험 인사이트 총정리

> **대회**: 데이콘 월간 AI 경진대회  
> **목표**: 11시점(40ms 간격) 3D 관측 → 80ms 후 위치 예측 (R-Hit@1cm)  
> **기간**: 2025–2026  
> **최고 점수**: LB 0.6780 (Public), OOF 기준 최고 모델: v3_D (0.6659)

---

## 1. 데이터 EDA — 핵심 발견

### 1-1. 스케일
| 항목 | 값 |
|------|-----|
| 관측 시간 | -400ms ~ 0ms (11점, 40ms 간격) |
| 예측 시점 | +80ms |
| x/y/z 이동 범위 | 수 cm 단위 |
| 평균 속도 | ~0.5–2 m/s |
| 평균 가속도 | ~10–20 m/s² |

### 1-2. 비행 패턴 분류
모기 궤적은 크게 4가지 클러스터로 분류:
- **C1**: 직선형 고속 비행 (easy, R-Hit 높음)
- **C2**: 완만한 곡선 비행 (보통)
- **C3**: 저속 선회 비행 (보통)
- **C4**: 급격한 방향 전환 (hard, R-Hit 낮음, Fold 2에 집중)

### 1-3. Y축 비대칭
- 모기는 **순방향 편향** = -0.360cm (앞으로 가는 경향)
- **Y-flip TTA 실패 원인**: 모기 비행에 y축 대칭성이 없음. Y-flip은 비행 방향을 반전시켜 오히려 예측 악화
- 결론: 데이터 증강 시 실제 물리적 대칭성 확인 필수

### 1-4. Fold 2 난이도
모든 실험에서 Fold 2가 ~0.645–0.652로 일관되게 낮음 → 급기동 샘플(C4) 집중

---

## 2. 물리 기반 예측 (Kalman Filter)

### 2-1. Constant Velocity (CV) Kalman — 베이스라인
```
sigma_obs = 0.3e-3  (관측 노이즈)
sigma_proc = 1.0    (프로세스 노이즈)
OOF R-Hit = 0.5964
```
- **축별 독립** 처리 (x, y, z 각각 스칼라 Kalman)
- `sigma_obs`는 Optuna로 탐색 → 0.000267이 최적
- 이 값이 딥러닝 모델의 "상한"을 결정

### 2-2. Constant Acceleration (CA) Kalman — 실패
```
OOF R-Hit = 0.4489  (CV 대비 -0.15, 대폭 하락)
```
- 40ms 스케일에서 모기 비행은 등속 모델이 등가속 모델보다 훨씬 우수
- **원인**: 40ms 구간에서 모기 가속도는 노이즈 수준 → CA 모델은 잘못된 가속도 추정

### 2-3. RTS Smoother — 수학적으로 동치
- 미래 예측에서 `x_smooth[T-1] = x_filter[T-1]` (항상 동일)
- RTS smoother는 과거 추정 개선에만 효과적; 미래 예측에는 무의미
- **결론**: Kalman → 80ms 외삽에 RTS 적용은 효과 없음

---

## 3. 딥러닝 — GRU 모델 아키텍처 진화

### 3-1. 모델 버전별 OOF 비교

| 버전 | 아키텍처 | OOF | 비고 |
|------|----------|-----|------|
| v3_baseline | GRU + Whitening | 0.655 | 기준점 |
| v3_B | Mish + 1L BiGRU | 0.6631 | activation 개선 |
| v3_C | curvature feature | 0.6641 | 피처 개선 |
| **v3_D** | **+focal_gamma=2** | **0.6659** | **최고 OOF** |
| v5 (Transformer) | CLS token, Pre-LN | 0.6620 | GRU보다 열등 |
| v6 (rotation aug) | 4x 회전 증강 | 0.6608 | 증강 역효과 |
| v7 lS=0.3 | multi-step aux | 0.6635 | 미개선 |

### 3-2. v3_D 최적 설정

```python
CFG = {
    'hidden':       64,
    'num_layers':   1,
    'bidirectional': True,
    'fc_hidden':    128,
    'dropout':      0.25,
    'main_clip':    0.03,       # tanh clip: ±2cm → ±3cm 잔차
    'lr':           0.000989,
    'wd':           0.000886,
    'focal_gamma':  2.0,
    'lF':           0.3,        # aux head_F (displacement from last obs) weight
    'lW':           0.3,        # aux head_W (weak Kalman residual) weight
    'sigma_obs':    0.000267,
}
```

### 3-3. 입력 피처 구조

**시계열 (seq)**: `(N, 11, 13)` — build_seq_curv
- rel (3): X - X[-1] (상대 위치)
- velocity (3): v = diff(X)/DT
- acceleration (3): a = diff(v)/DT
- speed (1): ‖v‖
- log(1+curvature) (1): 곡률
- log(1+angular_vel) (1): 각속도
- turn_rate (1): 방향 변화율

**스칼라**: `(N, 73)` = scalar(26) + tier3(19) + wavelet(27) + [1 extra]
- scalar (26): 속도/가속도 통계 + 노이즈 추정 (poly2, savgol, LOO-spline)
- tier3 (19): rolling mean speed + cumulative path
- wavelet (27): db4 Level-2 DWT 계수 통계 (3축 × 3단계 × 3 stats)

**출력**: Kalman 잔차를 yaw-회전 좌표계에서 예측
```
target = rotate_xy(y_train - kalman_pred, theta_yaw)
theta_yaw = arctan2(v_y, v_x)  # 마지막 속도 방향 기준
```

### 3-4. 보조 헤드 (Auxiliary Heads)
```
head_main: Kalman 잔차 (primary output, tanh clip ±0.03m)
head_F:    X[-1]에서의 변위 (displacement from last obs)
head_W:    약한 Kalman(sigma_obs=1e-3) 잔차 (smoother baseline)
```
- `total_loss = loss_main + lF*loss_F + lW*loss_W`
- 보조 헤드는 여러 기준점으로 학습을 정규화하는 효과

### 3-5. Focal Loss 효과
```python
def loss_focal_euclid(p, t, gamma=2.0):
    err = ‖p - t‖
    p_miss = sigmoid((err - 0.01) / 0.002)  # miss probability
    w = p_miss^gamma                          # upweight hard samples
    return (w * err).mean()
```
- 오류가 큰 샘플(>1cm)에 더 큰 가중치 부여
- gamma=2.0 (v3_D) > gamma=1.0 (v5) > gamma=0.0 (no focal)

---

## 4. 실패한 실험들과 이유

### 4-1. Rotation Augmentation (v6) — 실패
**실험**: X-Y 평면 4방향 회전 (0°, 90°, 180°, 270°) → 4x 데이터  
**결과**: OOF 0.6608 (v3_D 대비 -0.0051)

**실패 원인**:
- **yaw normalization이 이미 방향 불변성을 확보**
  - 모델이 body frame(yaw 기준)에서 학습 → 절대 방향 정보 불필요
  - 회전 증강은 정보적으로 완전히 중복
- 4x 데이터 증가로 학습 시간만 증가, 새 정보는 없음
- **교훈**: 이미 모델 설계에서 다룬 불변성을 데이터 증강으로 다시 처리하지 말 것

### 4-2. Y-flip TTA (v5) — 실패
**실험**: 예측 시 y 좌표 반전 후 평균  
**결과**: OOF 하락

**실패 원인**:
- 모기 비행에는 Y축 대칭성이 없음
- EDA: 순방향 편향 = -0.360cm → 방향성 있는 비행
- Y-flip은 오히려 편향 반전 → 예측 악화

### 4-3. Transformer (v5) — GRU보다 열등
**실험**: CLS token + Pre-LN Transformer  
**결과**: OOF 0.6620 (GRU 0.6659 대비 낮음)

**실패 원인**:
- 10,000 샘플 데이터셋에서 Transformer의 많은 파라미터 = 과적합
- 시계열 길이 11 (매우 짧음) → Self-attention의 이점 없음
- GRU가 단기 순서 의존성 포착에 더 효율적

### 4-4. Multi-step Self-Supervised (v7) — 미개선
**실험**: 10스텝 → 11번째 관측 변위 예측 보조 task  
**결과**: lS=0.3 → OOF 0.6635 (v3_D 0.6659 대비 낮음)

**실패 원인**:
- GRU가 이미 내부적으로 short-term pattern 학습
- 보조 task loss가 주 task 학습 간섭
- 10K 데이터에서 추가 generalization 효과 미미

### 4-5. 앙상블 — 상관관계 문제
**실험**: v3_D, v4r1, v4ens, v5r4, v5ens 등 다양한 모델 앙상블  
**결과**: 모든 모델 쌍 Pearson correlation = 1.0000

**원인**:
- Kalman filter가 예측 분산의 대부분 설명
- GRU는 작은 잔차만 보정 → 모든 GRU 모델 예측이 유사
- **결론**: 같은 Kalman 기반 모델끼리 앙상블은 무의미. 다른 기반(e.g., 순수 딥러닝, 물리 모델) 필요

---

## 5. 피처 엔지니어링 인사이트

### 5-1. 가장 유용한 피처
1. **curvature / angular_vel** (log1p 변환): 급기동 샘플 식별
2. **noise_loo (LOO spline)**: 측정 노이즈 추정, hard 샘플 가중치
3. **wavelet DWT coefficients**: 주파수 도메인에서 비행 패턴 구분
4. **last velocity vector** (v_last): 방향 + 속도 크기

### 5-2. 덜 유용한 피처
- straightness: GRU가 시계열에서 이미 파악
- high_speed / high_acc binary: 연속형이 더 유용

### 5-3. Whitening vs StandardScaler
- **Whitening (Cholesky)**: seq 피처에 더 효과적 (VECTOR 논문)
- **StandardScaler**: scalar 피처에 충분
- 두 방법의 성능 차이는 미미 (~0.001 OOF)

---

## 6. 하이퍼파라미터 탐색 (Optuna)

### 6-1. 주요 탐색 결과
| 파라미터 | 범위 | 최적 |
|----------|------|------|
| sigma_obs | 1e-4 ~ 5e-3 | 0.000267 |
| lr | 1e-4 ~ 5e-3 | 0.000989 |
| weight_decay | 1e-5 ~ 1e-3 | 0.000886 |
| hidden | 32, 64, 128 | 64 |
| num_layers | 1, 2 | 1 |
| fc_hidden | 64, 128, 256 | 128 |
| dropout | 0.1, 0.25, 0.4 | 0.25 |
| focal_gamma | 0.0, 1.0, 2.0 | 2.0 |

### 6-2. 핵심 교훈
- **shallow > deep**: GRU 1층이 2층보다 좋음 (10K 데이터 한계)
- **작은 lr 유리**: CosineAnnealingLR과 함께 0.001 수준
- **focal loss 효과적**: gamma=2.0이 일관되게 최고

---

## 7. 앞으로의 방향 (미탐색)

### 7-1. 높은 가능성
- ~~**LightGBM**~~: **실패** — OOF 0.5939 (Kalman 0.5979보다 낮음). 시계열 패턴을 스칼라 피처 79개로 포착 불가. Tree model은 순서 정보 손실
- **XGBoost with lag features**: LightGBM보다는 lag feature 수동 설계 필요
- **서로 다른 Kalman sigma_obs 조합**: 강/약 Kalman 예측값 자체를 피처로 사용
- **Stacking**: GRU OOF 예측 → meta-learner

### 7-2. 중간 가능성
- **Conformer (Conv+Attention+GRU)**: CNN으로 local pattern 포착 후 GRU
- **TCN (Temporal Convolutional Network)**: dilated causal convolution
- **Physics-informed loss**: 속도/가속도 continuity 제약

### 7-3. 낮은 가능성 (이미 시도, 실패)
- ~~회전 augmentation~~ (이미 yaw normalization이 처리)
- ~~Y-flip TTA~~ (비대칭 데이터)
- ~~CA Kalman~~ (CV가 우월)
- ~~Transformer~~ (데이터 부족)
- ~~GRU 앙상블~~ (correlation=1.0)

---

## 8. 제출 파일 목록

| 파일명 | OOF | LB (Public) | 비고 |
|--------|-----|-------------|------|
| sub_v3_D_+both_OOF0.6659.csv | 0.6659 | 0.6780 (추정) | **현재 최고** |
| sub_v5_rank1.csv | 0.6620 | — | Transformer, 낮음 |
| sub_v6_rotaug_OOF0.6608.csv | 0.6608 | — | 회전 aug, 더 낮음 |
| sub_v7_lS0.3_OOF0.6635.csv | 0.6635 | — | multi-step |
| sub_lgbm_*.csv | 0.5939 | — | LightGBM, Kalman보다 열등 |

---

## 9. 코드 구조

```
mosquitoes/
├── open/                    # 데이터
│   ├── train/               # TRAIN_00001~10000.csv
│   ├── test/                # TEST_00001~10000.csv
│   └── train_labels.csv
├── cache/                   # 전처리 캐시
│   ├── noise_cache.npz      # poly2/savgol/LOO 노이즈
│   └── noise_cache_10step.npz
├── submissions/             # 제출 파일
├── train_v3.py              # GRU (v3_D 포함, 최고 성능)
├── train_v5.py              # Transformer + Optuna
├── train_v6.py              # Rotation augmentation
├── train_v7.py              # Multi-step self-supervised
├── train_lgbm.py            # LightGBM baseline
├── ensemble_v2.py           # 앙상블 실험
└── INSIGHTS.md              # 이 파일
```

---

## 10. 핵심 교훈 요약

1. **물리 모델이 강력한 베이스라인**: Kalman CV 0.5964는 단순하지만 강력. GRU는 ~0.07 개선
2. **데이터 크기가 아키텍처를 결정**: 10K 샘플 → 단순한 1층 GRU가 최적
3. **대칭성 가정 검증 필수**: Y-flip TTA 실패처럼 데이터의 물리적 특성 먼저 확인
4. **중복 불변성 처리 금지**: yaw normalization + 회전 augmentation은 redundant
5. **Focal loss는 불균형 샘플에 효과적**: hard case(급기동) 업가중치로 전체 R-Hit 향상
6. **앙상블 전 상관관계 확인**: corr=1.0 이면 앙상블 무의미
7. **Kalman 잔차 body frame 예측**: 절대 좌표가 아닌 상대 잔차 + 회전 정규화가 학습 안정화
