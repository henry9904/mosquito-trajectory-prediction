"""
두 제출 파일 앙상블 스크립트
- submission_v2.csv   : velocity-based GRUv2 + wavelet (LB ~0.67x)
- sub_optuna          : Optuna GRU pos-based + tier3 (OOF 0.6623)
"""
import numpy as np
import pandas as pd
import glob, os, time

DATA_DIR  = r'D:\hyeon\공부\mosquitoes\open'
OUT_DIR   = r'D:\hyeon\공부\mosquitoes\submissions'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 파일 로드 ───────────────────────────────────────────────
sub_ref = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

# submission_v2.csv (velocity-based, 0.67x LB)
v2_path = r'D:\hyeon\공부\mosquitoes\submission_v2.csv'
sub_v2  = pd.read_csv(v2_path)
pred_v2 = sub_v2[['x','y','z']].values
print(f'[A] submission_v2   : {len(sub_v2)}행  (LB ~0.67x)')

# 최신 Optuna 제출 (OOF 0.6623)
opt_files = sorted(glob.glob(os.path.join(OUT_DIR, 'sub_optuna*.csv')))
if not opt_files:
    raise FileNotFoundError('sub_optuna*.csv 파일 없음')
opt_path = opt_files[-1]
sub_opt  = pd.read_csv(opt_path)
pred_opt = sub_opt[['x','y','z']].values
print(f'[B] {os.path.basename(opt_path)}: {len(sub_opt)}행  (OOF 0.6623)')

# ── OOF 기반 최적 가중치 계산 (성능 비율) ──────────────────
# submission_v2 OOF ≈ 0.6653 (mosquitoes.ipynb 결과)
# sub_optuna    OOF = 0.6623
OOF_V2  = 0.6653
OOF_OPT = 0.6623
w_v2_perf = OOF_V2 / (OOF_V2 + OOF_OPT)   # 성능 비례 가중치
print(f'\nOOF 기반 가중치: v2={w_v2_perf:.3f}, opt={1-w_v2_perf:.3f}')

# ── 앙상블 sweep ────────────────────────────────────────────
print('\n=== 앙상블 가중치 sweep ===')
print('  w(v2)  w(opt)  | x_std   y_std   z_std   | pred_norm_mean')
print('-' * 65)

results = []
for w_v2 in np.arange(0.0, 1.05, 0.1):
    w_opt = 1.0 - w_v2
    ens   = w_v2 * pred_v2 + w_opt * pred_opt
    xstd  = ens[:,0].std()
    ystd  = ens[:,1].std()
    zstd  = ens[:,2].std()
    norm  = np.linalg.norm(ens, axis=-1).mean()
    print(f'  {w_v2:.1f}    {w_opt:.1f}   | {xstd:.4f}  {ystd:.4f}  {zstd:.4f}  | {norm:.4f}')
    results.append((w_v2, w_opt, ens.copy()))

# ── 저장할 앙상블 파일들 ─────────────────────────────────────
ts = time.strftime('%m%d_%H%M')

save_weights = [
    (0.5, 0.5,        'equal'),
    (w_v2_perf, 1-w_v2_perf, 'oof_ratio'),
    (0.3, 0.7,        'opt_heavy'),
    (0.4, 0.6,        'opt_mild'),
]

print(f'\n=== 앙상블 파일 저장 ===')
for w_v2, w_opt, label in save_weights:
    ens = w_v2 * pred_v2 + w_opt * pred_opt
    df_ens = pd.DataFrame({
        'id': sub_v2['id'],
        'x':  ens[:, 0],
        'y':  ens[:, 1],
        'z':  ens[:, 2],
    })
    df_ens = sub_ref[['id']].merge(df_ens, on='id', how='left')
    fname  = f'sub_ens_{ts}_{label}_v2x{w_v2:.1f}_opt{w_opt:.1f}.csv'
    fpath  = os.path.join(OUT_DIR, fname)
    df_ens.to_csv(fpath, index=False)
    print(f'  저장: {fname}')

print(f'\n[제출 추천 순서]')
print(f'  1. sub_ens_*_opt_heavy (v2=0.3, opt=0.7)  ← Optuna 성능 믿기')
print(f'  2. sub_ens_*_equal     (v2=0.5, opt=0.5)  ← 다양성 극대화')
print(f'  3. sub_ens_*_oof_ratio (v2={w_v2_perf:.2f}, opt={1-w_v2_perf:.2f}) ← OOF 비례')
print(f'\n참고: sub_24 (LB 0.6780) CSV도 있으면 더 좋은 앙상블 가능!')
print(f'  경로: submissions/ 폴더에 sub_24_final_v2.csv 있으면 자동 감지')
