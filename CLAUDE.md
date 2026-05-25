# 모기 비행 궤적 예측 AI 대회 — 프로젝트 컨텍스트

## 대회 개요
- **주최**: 데이콘 월간 AI 경진대회
- **목표**: 40ms 간격 11시점 관측된 모기 3D 좌표 → 80ms 후 위치 예측
- **평가**: R-Hit@1cm (예측값이 실제값 1cm 이내면 적중)
- **현재 점수**: LB 0.6746 (Public), 20위권 목표 0.6968+

## 데이터 구조
```
D:\hyeon\공부\mosquitoes\open\
├── train\          TRAIN_00001.csv ~ TRAIN_10000.csv
├── test\           TEST_00001.csv  ~ TEST_10000.csv
├── train_labels.csv   (id, x, y, z) — 정답
└── sample_submission.csv
```

### 각 샘플 CSV 구조
- 11행 × 3열 (x, y, z)
- 시간축: -400ms, -360ms, ..., -40ms, 0ms (40ms 간격)
- 출력: +80ms 후 좌표

## 환경 세팅
- **GPU**: NVIDIA RTX 4070 Ti Super (16GB VRAM)
- **CUDA**: 13.1 (PyTorch cu124로 작동)
- **Python**: 3.11
- **Conda 환경**: mosquito
- **주요 패키지**: torch, numpy, pandas, scikit-learn, optuna, PyWavelets

## 현재 파이프라인

### Step 1 — 데이터 로드
```python
DT     = 0.040   # 40ms
T_PRED = 0.080   # 예측 시점

X_train  # (10000, 11, 3) — train 궤적
X_test   # (10000, 11, 3) — test 궤적
y_train  # (10000, 3)     — 정답 좌표
```

### Step 2 — Kalman Filter (물리 기반 예측)
- 등속도 모델 (CV), 축별 독립 처리
- 파라미터: sigma_obs=0.3e-3, sigma_proc=1.0
- 80ms 외삽
- OOF R-Hit: **0.5964**

### Step 3 — GRUv2 (딥러닝 잔차 보정)
- **입력 (시계열)**: 속도 기반 11채널 seq (N, 11, 11) — VECTOR 논문 아이디어
- **입력 (스칼라)**: Wavelet + 운동통계 44채널 — WTFTP 논문 아이디어
- **출력**: Kalman 잔차를 회전 좌표계에서 예측 (±2cm tanh)
- **보조 헤드**: 직접 변위 예측 (±5cm) — IaKNN 논문 아이디어
- **정규화**: Whitening (VECTOR) 또는 StandardScaler
- **학습**: 5-Fold CV, AdamW, CosineAnnealingLR
- OOF R-Hit: **~0.67**

### Step 4 — Calibration
- Per-axis α 탐색 [0.85~1.1]
- y축 약 0.95 배율이 최적 경향

### Step 5 — 앙상블 + 제출
- GRU 여러 셋업 가중 평균
- Optuna로 하이퍼파라미터 탐색 중

## 핵심 함수들

### Kalman Filter
```python
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    # 등속도 모델, 축별 독립, 80ms 외삽
    ...

def yaw_angle(v):
    return np.arctan2(v[:, 1], v[:, 0])

def rotate_xy(vec, theta):    # 회전 정규화
def inv_rotate_xy(vec, theta): # 역회전
```

### 피처 빌더
```python
def build_seq_velocity(X):
    # VECTOR: 속도+가속도+상대위치+속도크기 → (N,11,11)

def build_wavelet_features(X, wavelet='db4', level=2):
    # WTFTP: DWT 계수 통계 → (N,27)

def build_scal_v2(X):
    # 운동통계 17 + Wavelet 27 → (N,44)

class WhiteningScaler:
    # Cholesky 기반 Whitening 정규화
```

### 모델
```python
class GRUv2(nn.Module):
    # GRU(2-layer) + scal_emb + fc + head_main + head_aux
    # 입력: (N,11,SEQ_DIM), (N,SCAL_DIM)
    # 출력: main(N,3), aux(N,3)
```

## 참고 논문 (프로젝트 지식에 PDF 있음)
1. **VECTOR** (arXiv:2410.23305): 속도 입력 GRU, Whitening 정규화
2. **IaKNN** (arXiv:1902.10928): Kalman + 딥러닝 3-layer, 보조 가속도 학습
3. **WTFTP** (Nature Comm. 2023): Wavelet 시간-주파수 분해로 급기동 예측

## 다음 목표
- [ ] Optuna 하이퍼파라미터 탐색 완료 후 재제출
- [ ] 기존 LB 0.6780 모델과 앙상블
- [ ] 0.69대 진입 목표
