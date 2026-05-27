# AI Agents Coordination Log (Antigravity & Claude)

이 문서는 Antigravity와 Claude Code 등 다양한 AI 에이전트들이 협업하면서 발생했던 **실수(Mistakes)**를 복기하고, **주의해야 할 점(Pitfalls)**을 공유하여 같은 실수를 반복하지 않도록 하기 위해 작성된 로그입니다. 에이전트들은 작업 시작 시 이 문서를 먼저 확인하여 컨텍스트를 동기화해야 합니다.

---

## 📝 Error Logs & Lessons Learned

### [2026-05-27] 파이썬 가상환경(Conda) 실행 오류
- **문제 발생**: 터미널(Run Command)에서 단순히 `python train_v11.py`를 실행했다가 `pywt` (PyWavelets) 모듈이 없다는 에러 발생.
- **원인**: 기본 시스템 파이썬이 실행되었음. Windows PowerShell 환경에서는 `conda activate mosquito`가 쉘 초기화 문제로 실패할 수 있음.
- **해결 및 주의사항**: 
  - 학습 스크립트를 실행할 때는 시스템 파이썬 대신 **가상환경 파이썬 실행 파일의 절대 경로**를 사용할 것.
  - 예: `C:\Users\hyeon\miniconda3\envs\mosquito\python.exe train_v11.py`

### [2026-05-27] K-Fold 모델 체크포인트 캐시(Cache) 중복 로딩 실수
- **문제 발생**: `train_v10.py`를 복사하여 새로운 피처를 추가한 `train_v11_multi_kalman.py`를 작성하고 실행했으나, 모델이 학습을 진행하지 않고 1초 만에 완료됨.
- **원인**: 복사한 스크립트 내부에 하드코딩 되어 있던 `CKPT_DIR = ... / 'cache' / 'v10_ckpt'` 경로를 수정하지 않았음. 이로 인해 기존 v10의 체크포인트(`fold0_oof.npy` 등)가 존재한다고 판단하고, 새 모델 구조에 대한 훈련을 스킵한 채 과거 OOF 데이터를 그대로 반환함.
- **해결 및 주의사항**:
  - 모델 버전업(예: `v10` → `v11`)을 위해 스크립트를 복사(Copy)할 경우, 반드시 **체크포인트 경로(`CKPT_DIR`)와 제출 파일 이름(`fname`)을 새로운 버전에 맞게 수정**할 것.
  - 모델 구조가 바뀌었는데 이전 캐시를 불러오면 심각한 성능 측정 오류(False Positive/Negative)를 유발할 수 있음.

### [2026-05-27] Stacking 모델(Ridge Regression) 과도한 정규화(L2 Penalty) 실수
- **문제 발생**: v9, v10, v11 OOF 앙상블을 위해 Ridge Regression(L2 규제)를 적용했더니 최종 성능이 베이스라인(0.59)으로 수직 낙하함.
- **원인**: v9, v10, v11 모델들의 예측값이 너무 비슷(Highly Correlated)해서 다중공선성(Multicollinearity) 문제가 생김. 이 경우 L2 규제가 작동하면 계수(Coefficients)들을 0에 가깝게 축소(Shrinkage)시켜 버림. 그 결과 잔차 예측값이 0이 되어, 결국 베이스 칼만 필터 예측만 남게 됨.
- **해결 및 주의사항**:
  - 앙상블 인풋들의 상관관계가 1.0에 육박할 경우, 일반적인 `Ridge(alpha=1.0)`를 쓰면 모든 가중치가 죽어버림.
  - `LinearRegression(fit_intercept=False)`로 절편 없이 단순 선형 결합을 하거나, 차라리 **단순 평균(Simple Average)**을 내는 편이 훨씬 안전함.

### [2026-05-27] $\sigma_{obs}$ 파라미터 변형을 통한 앙상블 다양성 확보 실패 (다중공선성)
- **문제 발생**: 칼만 필터의 $\sigma_{obs}$를 0.0001 ~ 0.002까지 다양하게 주어 5개의 베이스 모델을 학습시켰으나, OOF 예측값 간의 Pearson 상관계수가 모두 0.995 이상으로 나옴.
- **원인**: 물리 엔진(칼만 필터)의 세팅을 다르게 주어도, 뒤에 붙은 강력한 신경망(GRU)이 최종 목표물(GT)을 향해 궤적을 보정하면서 결국 똑같은 지점으로 수렴해 버림.
- **결론**: 신경망 기반 잔차 예측 구조에서는, 단순한 전처리/상수값 튜닝만으로는 앙상블에 필요한 '다양성(Diversity)'을 만들어낼 수 없음. 완전히 다른 아키텍처(예: Transformer, Adaptive Kalman)가 필요함.

---

## 💡 Architectural Decisions & Tips
- (추가 예정)
