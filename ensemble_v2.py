# -*- coding: utf-8 -*-
"""
ensemble_v2.py — v3_D / v4 / v5 앙상블 그리드 탐색
  - 각 모델 OOF 점수를 가중치 프록시로 사용
  - 3-way 그리드 탐색: v3 / v4ensemble / v5
  - 상관계수 분석으로 다양성 확인
  - 예상 OOF 기준 상위 파일 저장
"""
import numpy as np, pandas as pd, os, time, itertools

SUB_DIR = r'D:\hyeon\공부\mosquitoes\submissions'
OUT_DIR = SUB_DIR

# ── 모델 정의 (파일명, OOF) ─────────────────────────────────────
MODELS = [
    # (태그, 파일명, OOF)
    ('v3D',   'sub_v3_D_+both_OOF0.6659.csv',              0.6659),
    ('v4r1',  'sub_v4_rank1_t70_OOF0.6652.csv',             0.6652),
    ('v4ens', 'sub_v4_topK_weighted_0525_0310_OOF0.6657.csv',0.6657),
    ('v5r4',  'sub_v5_rank4_t34_gru_OOF0.6629.csv',         0.6629),
    ('v5ens', 'sub_v5_topK_weighted_0525_0518_OOF0.6622.csv',0.6622),
]

print('=== 모델 로드 ===')
preds = {}
ids   = None
for tag, fname, oof in MODELS:
    path = os.path.join(SUB_DIR, fname)
    df = pd.read_csv(path)
    preds[tag] = df[['x','y','z']].values.astype(np.float64)
    if ids is None:
        ids = df['id'].values
    print(f'  {tag:6s}  OOF={oof:.4f}  rows={len(df)}')

# ── 예측 간 상관 분석 ──────────────────────────────────────────
print('\n=== 예측 상관계수 (유클리드 거리 기준 Pearson) ===')
tags = [m[0] for m in MODELS]
mats = np.stack([preds[t] for t in tags], 0)  # (M, N, 3)
flat = mats.reshape(len(tags), -1)             # (M, N*3)
corr = np.corrcoef(flat)
print('      ' + '  '.join(f'{t:>6}' for t in tags))
for i, ti in enumerate(tags):
    row = '  '.join(f'{corr[i,j]:6.4f}' for j in range(len(tags)))
    print(f'  {ti:6s}  {row}')

# ── OOF 점수 정규화 ───────────────────────────────────────────
oofs = {m[0]: m[2] for m in MODELS}

def oof_weighted(combo):
    """combo: list of (tag, weight). weight sum=1."""
    return sum(oofs[t]*w for t,w in combo)

def make_pred(combo):
    """가중 평균 예측."""
    out = np.zeros_like(preds[tags[0]], dtype=np.float64)
    for t, w in combo:
        out += w * preds[t]
    return out

# ── 2-way 그리드: v3D vs X ─────────────────────────────────────
print('\n=== 2-way 앙상블: v3D + 각 모델 ===')
print(f'  {"pair":20s}  w_v3D  w_other  exp_OOF')
pair_results = []
for tag, _, oof in MODELS[1:]:
    for w1 in np.arange(0.3, 0.75, 0.05):
        w2 = 1.0 - w1
        combo = [('v3D', w1), (tag, w2)]
        eoof = oof_weighted(combo)
        pair_results.append((eoof, combo, f'v3D+{tag}'))
pair_results.sort(key=lambda x: -x[0])
seen = set()
for eoof, combo, label in pair_results[:20]:
    key = label + f'{combo[0][1]:.2f}'
    if key not in seen:
        seen.add(key)
        w1, w2 = combo[0][1], combo[1][1]
        print(f'  {label:20s}  {w1:.2f}   {w2:.2f}   {eoof:.4f}')

# ── 3-way 그리드: v3D + v4ens + v5r4 ─────────────────────────
print('\n=== 3-way 그리드: v3D / v4ens / v5r4 ===')
grid_results = []
step = 0.05
ws = np.arange(0.0, 1.01, step)
for w1 in ws:
    for w2 in ws:
        w3 = 1.0 - w1 - w2
        if w3 < -1e-9 or w3 > 1.0 + 1e-9: continue
        w3 = max(0.0, min(1.0, w3))
        if w1 < 0.1: continue  # v3D는 최소 10% 이상
        combo = [('v3D', w1), ('v4ens', w2), ('v5r4', w3)]
        eoof = oof_weighted(combo)
        grid_results.append((eoof, w1, w2, w3, combo))

grid_results.sort(key=lambda x: -x[0])
print(f'  {"w_v3D":>6} {"w_v4ens":>7} {"w_v5r4":>7}  exp_OOF')
for eoof, w1, w2, w3, combo in grid_results[:15]:
    print(f'  {w1:6.2f}  {w2:7.2f}  {w3:7.2f}   {eoof:.4f}')

# ── 저장할 앙상블 후보 선정 ───────────────────────────────────
ts = time.strftime('%m%d_%H%M')
sample_ref = pd.read_csv(os.path.join(r'D:\hyeon\공부\mosquitoes\open', 'sample_submission.csv'))

candidates = [
    # tag, combo, label
    ('v3D_only',   [('v3D', 1.0)],                              'v3D_only'),
    ('v3+v4ens_50',[('v3D', 0.50), ('v4ens', 0.50)],           'v3D0.50_v4ens0.50'),
    ('v3+v4ens_60',[('v3D', 0.60), ('v4ens', 0.40)],           'v3D0.60_v4ens0.40'),
    ('v3+v5r4_70', [('v3D', 0.70), ('v5r4', 0.30)],            'v3D0.70_v5r40.30'),
    ('v3+v5r4_60', [('v3D', 0.60), ('v5r4', 0.40)],            'v3D0.60_v5r40.40'),
    ('v3+v5r4_80', [('v3D', 0.80), ('v5r4', 0.20)],            'v3D0.80_v5r40.20'),
    ('3way_60_30_10',[('v3D',0.60),('v4ens',0.30),('v5r4',0.10)],'v3D0.60_v4ens0.30_v5r40.10'),
    ('3way_50_30_20',[('v3D',0.50),('v4ens',0.30),('v5r4',0.20)],'v3D0.50_v4ens0.30_v5r40.20'),
    ('3way_55_35_10',[('v3D',0.55),('v4ens',0.35),('v5r4',0.10)],'v3D0.55_v4ens0.35_v5r40.10'),
    ('3way_65_25_10',[('v3D',0.65),('v4ens',0.25),('v5r4',0.10)],'v3D0.65_v4ens0.25_v5r40.10'),
    # best grid top-1
]

# 3-way grid top-3 추가
added = set()
for eoof, w1, w2, w3, combo in grid_results[:30]:
    k = f'{w1:.2f}_{w2:.2f}_{w3:.2f}'
    if k not in added:
        added.add(k)
        lbl = f'grid_v3D{w1:.2f}_v4ens{w2:.2f}_v5r4{w3:.2f}'
        candidates.append((lbl, combo, lbl))
    if len(added) >= 3:
        break

print('\n=== 앙상블 파일 저장 ===')
saved_info = []
for tag, combo, label in candidates:
    pred = make_pred(combo)
    eoof = oof_weighted(combo)
    df_out = pd.DataFrame({'id': ids, 'x': pred[:,0], 'y': pred[:,1], 'z': pred[:,2]})
    df_out = sample_ref[['id']].merge(df_out, on='id', how='left')
    fname = f'sub_ens_{ts}_{label}_eOOF{eoof:.4f}.csv'
    fpath = os.path.join(OUT_DIR, fname)
    df_out.to_csv(fpath, index=False)
    saved_info.append((eoof, fname, combo))
    print(f'  eOOF={eoof:.4f}  {fname}')

print('\n=== 제출 추천 순서 (예상 OOF 높은순) ===')
saved_info.sort(key=lambda x: -x[0])
for i, (eoof, fname, combo) in enumerate(saved_info, 1):
    weights_str = ', '.join(f'{t}:{w:.2f}' for t,w in combo)
    print(f'  {i:2d}. eOOF={eoof:.4f}  [{weights_str}]')
    print(f'      {fname}')

print('\n완료.')
