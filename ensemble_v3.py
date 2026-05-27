"""
ensemble_v3.py
Ensemble submission_v2.csv (Colab GRUv2, OOF≈0.6653)
with sub_v3_D (local BiGRU + Focal, OOF=0.6659)
"""
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR  = r'D:\hyeon\공부\mosquitoes\open'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR  = os.path.join(DATA_DIR, 'test')
SUB_DIR   = r'D:\hyeon\공부\mosquitoes\submissions'

# ── 1. 두 제출 파일 로드 ──────────────────────────────────────────
v2_path = r'D:\hyeon\공부\mosquitoes\submission_v2.csv'
v3d_candidates = sorted(glob.glob(os.path.join(SUB_DIR, '*v3_D*.csv')))
print(f'submission_v2.csv: {v2_path}')
print(f'v3_D candidates: {v3d_candidates}')

df_v2  = pd.read_csv(v2_path)
print(f'\nsubmission_v2 shape: {df_v2.shape}')
print(df_v2.head(3))

if v3d_candidates:
    df_v3d = pd.read_csv(v3d_candidates[-1])   # 최신 파일 사용
    print(f'\nv3_D shape: {df_v3d.shape}')
    print(df_v3d.head(3))

    # ── 2. 상관관계 분석 ────────────────────────────────────────────
    v2  = df_v2[['x','y','z']].values
    v3d = df_v3d[['x','y','z']].values

    print('\n[예측값 분포 비교]')
    for i, ax in enumerate('xyz'):
        print(f'  {ax}: v2 mean={v2[:,i].mean():.4f} std={v2[:,i].std():.4f} | '
              f'v3D mean={v3d[:,i].mean():.4f} std={v3d[:,i].std():.4f}')

    print('\n[Pearson 상관관계 (x,y,z)]')
    for i, ax in enumerate('xyz'):
        r = np.corrcoef(v2[:,i], v3d[:,i])[0,1]
        print(f'  {ax}: r = {r:.6f}')

    r_all = np.corrcoef(v2.ravel(), v3d.ravel())[0,1]
    print(f'  전체 (3D concat): r = {r_all:.6f}')

    diff = np.linalg.norm(v2 - v3d, axis=1)
    print(f'\n[예측 차이 (L2 거리)]')
    print(f'  평균: {diff.mean()*100:.3f} cm')
    print(f'  중앙값: {np.median(diff)*100:.3f} cm')
    print(f'  max: {diff.max()*100:.3f} cm')
    print(f'  >1cm 차이: {(diff>0.01).mean()*100:.1f}%')

    # ── 3. Kalman 로드 및 OOF 평가 ──────────────────────────────────
    print('\n데이터 로드 중...')
    train_files = sorted(glob.glob(os.path.join(TRAIN_DIR, 'TRAIN_*.csv')))
    test_files  = sorted(glob.glob(os.path.join(TEST_DIR,  'TEST_*.csv')))
    labels      = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
    train_ids   = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
    test_ids    = [os.path.splitext(os.path.basename(f))[0] for f in test_files]

    X_test = np.stack([pd.read_csv(f)[['x','y','z']].values
                       for f in tqdm(test_files, desc='test 로드')], axis=0).astype(np.float64)

    # ── 4. 가중 앙상블 ───────────────────────────────────────────────
    print('\n[가중 앙상블 탐색]')
    print('  (w = v3_D 비율)')
    # OOF 없이 test 예측만으로 단순 평균 후 저장
    best_w = None

    os.makedirs(SUB_DIR, exist_ok=True)
    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        ens = v3d * w + v2 * (1 - w)
        df_ens = pd.DataFrame({'id': df_v3d['id'],
                               'x': ens[:,0], 'y': ens[:,1], 'z': ens[:,2]})
        tag = f'ens_v3D_{w:.1f}_v2_{1-w:.1f}'
        out_path = os.path.join(SUB_DIR, f'sub_{tag}.csv')
        df_ens.to_csv(out_path, index=False)
        print(f'  w={w:.1f} → {out_path}')

    # ── 5. 단순 평균 (50:50) 저장 ──────────────────────────────────
    ens50 = (v3d + v2) / 2
    df50 = pd.DataFrame({'id': df_v3d['id'],
                         'x': ens50[:,0], 'y': ens50[:,1], 'z': ens50[:,2]})
    path50 = os.path.join(SUB_DIR, 'sub_ens_v3D_0.5_v2_0.5.csv')
    df50.to_csv(path50, index=False)
    print(f'\n50:50 앙상블 → {path50}')

    print('\n완료. 각 앙상블 파일을 Dacon에 제출해보세요.')
    print('상관관계가 높으면 앙상블 효과가 제한적일 수 있습니다.')
else:
    print('v3_D 제출 파일이 없습니다. submissions/ 폴더를 확인하세요.')
    # v2 파일만 분석
    print(df_v2.describe())
