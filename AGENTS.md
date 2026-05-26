# Agent Guidelines

AI 에이전트(Antigravity, Claude Code 등)를 위한 문서입니다.
작업 전 반드시 읽어주세요.

---

## 프로젝트 맥락 (Context)

- **대회**: 데이콘 월간 AI 경진대회 — 모기 3D 궤적 예측
- **마감**: 2026년 6월 1일
- **현재 최고 점수**: OOF **0.6659** / LB Public **~0.6780**
- **목표**: LB **0.69+** (현재 1위 ~0.70)
- **상세 맥락**: `CLAUDE.md` (환경·데이터·파이프라인 구조)
- **실험 기록**: `INSIGHTS.md` (실패 이유 포함 전체 실험 로그)

> **핵심 구조 요약**: 칼만 필터(물리 베이스라인 0.5964) + BiGRU가 칼만 잔차만 예측 → 최종 예측 = 칼만 + GRU 잔차. 최고 모델은 `train_v3.py`의 `v3_D` 설정.

---

## 환경 세팅 (Environment)

```bash
conda activate mosquito   # Python 3.11, PyTorch cu124
# 데이터 경로: D:\hyeon\공부\mosquitoes\open\
# GPU: RTX 4070 Ti Super 16GB
```

---

## 이미 시도해서 실패한 것들 — 다시 시도하지 말 것

| 방법 | 결과 | 실패 이유 |
|------|:----:|----------|
| Rotation Augmentation (v6) | 0.6608 ↓ | yaw 정규화와 중복 |
| Y-flip TTA (v5) | 하락 ↓ | 모기 비행 비대칭 |
| CA Kalman | 0.4489 ↓ | 잘못된 물리 모델 |
| Transformer (v5) | 0.6620 ↓ | 10K 데이터로 과적합 |
| GRU 단순 앙상블 | ≈변화없음 | Pearson corr = 1.0 |
| LightGBM | 0.5939 ↓ | 시계열 순서 손실 |
| 2층 BiGRU | 0.6631 ↓ | 과적합 |

---

## 지금 시도할 것 — 우선순위 순서대로

### 🥇 1순위: 여러 σ_obs 칼만 앙상블

**왜 유망한가**: 단순 GRU 앙상블은 corr=1.0이라 무의미했지만, **칼만 σ_obs가 다르면 출발점 자체가 달라져** GRU 잔차도 달라짐 → 진짜 다양성 확보 가능.

```python
# 시도할 σ_obs 값들
sigmas = [0.0001, 0.000267, 0.0005, 0.001, 0.002]

# 각 sigma로 칼만 예측 → train_v3.py v3_D 구조로 잔차 학습 → OOF 수집
# 최적 가중치는 Optuna나 grid search로 탐색
# 예상 효과: +0.001 ~ +0.003
```

**실행**: `train_v3.py`에서 `SIGMA_OBS` 상수를 바꿔가며 5개 모델 학습 → `ensemble_v2.py` 참고해서 corr 확인 → corr < 0.98이면 앙상블 의미 있음.

---

### 🥈 2순위: Stacking (OOF 기반 메타 러너)

**왜 유망한가**: 칼만과 GRU는 귀납 편향이 완전히 다름. OOF 예측을 피처로 쓰는 메타 러너가 "이 샘플은 칼만이 맞고, 저 샘플은 GRU가 맞다"를 학습할 수 있음.

```python
# Level 1: 이미 있는 것들
# - kalman_oof: (10000, 3)  — 칼만 예측값
# - gru_oof:    (10000, 3)  — v3_D OOF 예측값

# Level 2: 메타 러너 입력 피처 후보
meta_features = np.concatenate([
    kalman_oof,          # (10000, 3)
    gru_oof,             # (10000, 3)
    curvature_features,  # 곡률 (급기동 여부 — 어느 모델이 더 좋을지 판단)
    kalman_residual_std, # 칼만 불확실성 추정
], axis=1)

# 메타 러너: Ridge regression 또는 작은 MLP (오버피팅 주의)
# 예상 효과: +0.002 ~ +0.005
```

---

### 🥉 3순위: Calibration 개선 (궤적 유형별)

**현재**: per-axis α (x, y, z 각각 스칼라 하나).
**문제**: 직선 비행 vs 급기동 샘플에 같은 α 적용 → 최적이 아님.

```python
# 시도: 곡률 기준으로 두 그룹으로 나누기
low_curv  = curvature < median_curvature   # 직선 비행
high_curv = curvature >= median_curvature  # 급기동

# 각 그룹별로 α 탐색 → 6개 α (3축 × 2그룹)
# 예상 효과: +0.001 ~ +0.002
```

---

### 4순위: Physics-Informed Loss 추가

**아이디어**: 예측 위치가 마지막 관측 속도와 심하게 어긋나면 패널티.

```python
def physics_loss(pred, last_vel, dt=0.08):
    # 물리적으로 가능한 범위: 마지막 속도 × dt ± 최대 가속도 × dt²
    expected = last_pos + last_vel * dt
    physics_penalty = F.mse_loss(pred, expected) * 0.1
    return main_loss + physics_penalty
# 예상 효과: +0.001 ~ +0.002 (급기동 샘플 개선)
```

---

## 작업 완료 기준 (Done Criteria)

- OOF R-Hit > **0.670** 이면 제출 가치 있음
- OOF R-Hit > **0.675** 이면 LB 0.69+ 기대 가능
- 새 실험 결과는 `INSIGHTS.md` 테이블에 추가할 것

## 주의사항

- `open/` 데이터 디렉토리는 gitignore — 절대 커밋하지 말 것
- 새 모델은 `train_v8.py`, `train_v9.py` 식으로 버전 관리
- 기존 `train_v3.py` (최고 모델) 수정 금지 — 새 파일로 실험
