# -*- coding: utf-8 -*-
"""
train_lgbm.py — LightGBM on 73-dim scalar features
  - Feature: build_scalar (26) + build_tier3 (19) + build_wavelet (27) = 72-dim + Kalman pred (3) = 75-dim
  - Target: Kalman residual (y_train - kalman_train) per axis
  - 5-fold CV, axis-independent models (x, y, z separately)
  - Calibration + ensemble with v3_D submission
"""
import os, gc, warnings, pathlib, glob
import numpy as np, pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pywt, lightgbm as lgb
warnings.filterwarnings('ignore')

import random
SEED = 0
random.seed(SEED); np.random.seed(SEED)

DT, T_PRED = 0.040, 0.080
t_obs = np.arange(-400, 1, 40) / 1000.0

DATA_DIR  = str(pathlib.Path(__file__).parent / 'open')
CACHE_DIR = str(pathlib.Path(__file__).parent / 'cache')
OUT_DIR   = str(pathlib.Path(__file__).parent / 'submissions')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

# ── 데이터 로드 ─────────────────────────────────────────────────
train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.csv')))
test_files  = sorted(glob.glob(os.path.join(DATA_DIR, 'test',  '*.csv')))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))
sub    = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

def load_stack(files, desc):
    return np.stack([pd.read_csv(f)[['x','y','z']].values
                     for f in tqdm(files, desc=desc)]).astype(np.float64)

X_train = load_stack(train_files, 'train')
X_test  = load_stack(test_files,  'test')
train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
test_ids  = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
y_train = labels.set_index('id').loc[train_ids][['x','y','z']].values.astype(np.float64)
N = len(X_train)
print(f'X_train {X_train.shape}  y_train {y_train.shape}')

# ── 기본 유틸 ──────────────────────────────────────────────────
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N_, T, _ = X.shape
    F_  = np.array([[1,DT],[0,1]]); Fp = np.array([[1,T_PRED],[0,1]])
    Q   = sigma_proc**2 * np.array([[DT**4/4, DT**3/2],[DT**3/2, DT**2]])
    R   = sigma_obs**2; pred = np.zeros((N_,3))
    for j in range(3):
        z=X[:,:,j]; s=np.zeros((N_,2)); s[:,0]=z[:,0]; P=np.eye(2)
        for t in range(1,T):
            s=s@F_.T; P=F_@P@F_.T+Q
            inn=z[:,t]-s[:,0]; S_=P[0,0]+R; K=P[:,0]/S_
            s=s+np.outer(inn,K); P=P-np.outer(K,P[0,:])
        pred[:,j]=(s@Fp.T)[:,0]
    return pred

def r_hit(pred, true, thr=0.01):
    return float((np.linalg.norm(pred-true, axis=-1) <= thr).mean())

def cos_safe(a, b):
    return np.clip((a*b).sum(-1) /
                   np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1), 1e-12), -1, 1)

LOG_COLS = ['mean_speed','max_speed','speed_std','mean_acc','max_acc','max_jerk',
            'net_disp','|v_last|','|a_last|','|a_recent|','jerk_last','jerk_recent',
            'noise_poly2','noise_savgol','noise_loo']

def build_scalar(X, np_a, ns_a, nl_a=None):
    d=np.diff(X,axis=1); v=d/DT; a=np.diff(v,axis=1)/DT; jk=np.diff(a,axis=1)/DT
    sp=np.linalg.norm(v,axis=-1); ac=np.linalg.norm(a,axis=-1); jm=np.linalg.norm(jk,axis=-1)
    vl=v[:,-1]; al=a[:,-1]; ar=a[:,-3:].mean(1)
    nd=np.linalg.norm(X[:,-1]-X[:,0],axis=-1); pl=np.linalg.norm(d,axis=-1).sum(1)
    st=np.where(pl>1e-12, nd/np.maximum(pl,1e-12), 0.)
    tc=cos_safe(vl, v[:,:-1].mean(1))
    if nl_a is None: nl_a=ns_a
    df=pd.DataFrame({
        'mean_speed':sp.mean(1),'max_speed':sp.max(1),'speed_std':sp.std(1),
        'mean_acc':ac.mean(1),'max_acc':ac.max(1),'max_jerk':jm.max(1),
        'straightness':st,'net_disp':nd,'turn_cos':tc,
        '|v_last|':np.linalg.norm(vl,axis=-1),'|a_last|':np.linalg.norm(al,axis=-1),
        '|a_recent|':np.linalg.norm(ar,axis=-1),'jerk_last':jm[:,-1],
        'jerk_recent':jm[:,-3:].mean(1),
        'noise_poly2':np_a,'noise_savgol':ns_a,'noise_loo':nl_a,
        'hard_turn':(tc<0.5).astype(np.float32),
        'high_speed':(np.linalg.norm(vl,axis=-1)>1.).astype(np.float32),
        'high_acc':(ac.max(1)>15).astype(np.float32),
        'log_max_acc':np.log1p(ac.max(1))})
    for c in LOG_COLS: df[c]=np.log1p(df[c])
    cross=np.cross(v[:,:-1],a); cm=np.linalg.norm(cross,axis=-1)
    sm=np.linalg.norm(v[:,:-1],axis=-1)
    kappa=cm/(sm**3+1e-12); omega=cm/(sm**2+1e-12)
    v_unit=v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-12)
    tr_=np.arccos((v_unit[:,:-1]*v_unit[:,1:]).sum(-1).clip(-1,1))
    df['mean_curvature']=np.log1p(kappa.mean(1)); df['max_curvature']=np.log1p(kappa.max(1))
    df['mean_angvel']=np.log1p(omega.mean(1));    df['max_angvel']=np.log1p(omega.max(1))
    df['mean_turnrate']=tr_.mean(1);              df['max_turnrate']=tr_.max(1)
    return df.values.astype(np.float32)

def build_tier3(X):
    d=np.diff(X,axis=1); v=d/DT; sp=np.linalg.norm(v,axis=-1)
    sr=np.stack([sp[:,i:i+3].mean(1) for i in range(8)],1)*DT
    cp=np.concatenate([np.zeros((X.shape[0],1)),np.cumsum(np.linalg.norm(d,axis=-1),1)],1)
    return np.concatenate([sr,cp],1).astype(np.float32)

def build_wavelet(X):
    feats=[]
    for ax in range(3):
        for c in pywt.wavedec(X[:,:,ax],'db4',level=2,axis=1):
            feats+=[c.mean(1),c.std(1),np.abs(c).max(1)]
    return np.column_stack(feats).astype(np.float32)

def noise_poly2(X):
    V=np.vander(t_obs[:X.shape[1]],3,increasing=False); o=np.zeros(X.shape[0])
    for j in range(3):
        c=np.linalg.lstsq(V,X[:,:,j].T,rcond=None)[0]
        o+=(X[:,:,j]-(V@c).T).std(1)
    return o/3

def noise_savgol(X):
    return (X-savgol_filter(X,5,2,axis=1)).std(1).mean(-1)

def noise_loo_spline(X, t_obs_sub):
    N_,T,_=X.shape; o=np.zeros(N_); idx=np.arange(T)
    for i in tqdm(range(N_),desc='LOO spline'):
        s=0
        for k in range(1,T-1):
            m=idx!=k
            for j in range(3):
                cs=CubicSpline(t_obs_sub[m],X[i,m,j])
                s+=(X[i,k,j]-cs(t_obs_sub[k]))**2
        o[i]=np.sqrt(s/((T-2)*3))
    return o

# ── 노이즈 캐시 로드 ─────────────────────────────────────────────
CACHE_FILE = os.path.join(CACHE_DIR, 'noise_cache.npz')
if os.path.exists(CACHE_FILE):
    nc=np.load(CACHE_FILE)
    noise_p,noise_s,noise_l=nc['np_'],nc['ns_'],nc['nl_']
    noise_p_te,noise_s_te=nc['np_te'],nc['ns_te']
    print('noise cache loaded')
else:
    noise_p=noise_poly2(X_train); noise_s=noise_savgol(X_train)
    noise_l=noise_loo_spline(X_train, t_obs)
    noise_p_te=noise_poly2(X_test); noise_s_te=noise_savgol(X_test)
    np.savez(CACHE_FILE, np_=noise_p, ns_=noise_s, nl_=noise_l,
             np_te=noise_p_te, ns_te=noise_s_te)

# ── Feature 빌드 ──────────────────────────────────────────────
print('Building features...')
scal_tr  = build_scalar(X_train, noise_p, noise_s, noise_l)    # (N, 26)
scal_te  = build_scalar(X_test,  noise_p_te, noise_s_te)       # (N, 26)
t3_tr    = build_tier3(X_train)                                  # (N, 19)
t3_te    = build_tier3(X_test)
wav_tr   = build_wavelet(X_train)                                # (N, 27)
wav_te   = build_wavelet(X_test)

# Kalman predictions (sigma_obs best = 0.000267)
so_best = 0.000267
kal_tr   = kalman_predict(X_train, so_best, 1.0)                 # (N, 3)
kal_te   = kalman_predict(X_test,  so_best, 1.0)

# Last velocity vector (raw features for direction)
v_last_tr = (X_train[:,-1] - X_train[:,-2]) / DT               # (N, 3)
v_last_te = (X_test[:,-1]  - X_test[:,-2])  / DT

# Final feature matrix: scal + tier3 + wavelet + kalman_pred + last_velocity
feat_tr = np.concatenate([scal_tr, t3_tr, wav_tr, kal_tr, v_last_tr], axis=1).astype(np.float32)
feat_te = np.concatenate([scal_te, t3_te, wav_te, kal_te, v_last_te], axis=1).astype(np.float32)
print(f'feat_tr shape: {feat_tr.shape}')  # (10000, 78)

# Target: raw y_train (predict absolute coordinate directly)
# Also compute Kalman residual
resid_tr = (y_train - kal_tr).astype(np.float32)  # (N, 3)
print(f'Kalman R-Hit (train): {r_hit(kal_tr, y_train):.4f}')
print(f'Residual per axis std (cm): x={resid_tr[:,0].std()*100:.3f}, y={resid_tr[:,1].std()*100:.3f}, z={resid_tr[:,2].std()*100:.3f}')

# ── LightGBM 학습 (5-Fold CV) ────────────────────────────────
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

lgb_params = {
    'objective':       'regression',
    'metric':          'rmse',
    'learning_rate':   0.05,
    'num_leaves':      63,
    'max_depth':       -1,
    'min_child_samples': 20,
    'subsample':       0.8,
    'subsample_freq':  1,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'n_estimators':    2000,
    'early_stopping_rounds': 100,
    'verbose':         -1,
    'n_jobs':          -1,
    'random_state':    SEED,
}

oof_pred = np.zeros((N, 3), dtype=np.float64)
test_preds = []  # will hold (N_folds, N_test, 3) for averaging

print('\n=== LightGBM 5-Fold CV ===')
for fold, (tri, vai) in enumerate(kf.split(feat_tr)):
    X_tr, X_va = feat_tr[tri], feat_tr[vai]
    y_tr_res, y_va_res = resid_tr[tri], resid_tr[vai]
    kal_va = kal_tr[vai]

    fold_test_pred = np.zeros((len(feat_te), 3))
    for ax, ax_name in enumerate(['x', 'y', 'z']):
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_tr_res[:, ax],
            eval_set=[(X_va, y_va_res[:, ax])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
        )
        oof_pred[vai, ax] = kal_va[:, ax] + model.predict(X_va)
        fold_test_pred[:, ax] = kal_te[:, ax] + model.predict(feat_te)

    fold_rhit = r_hit(oof_pred[vai], y_train[vai])
    print(f'  Fold {fold+1}: R-Hit={fold_rhit:.4f}')
    test_preds.append(fold_test_pred)

oof_rhit = r_hit(oof_pred, y_train)
print(f'\nOOF R-Hit (LightGBM): {oof_rhit:.4f}')
print(f'Kalman OOF R-Hit:      {r_hit(kal_tr, y_train):.4f}')

# ── 제출 파일 저장 ────────────────────────────────────────────
test_pred_mean = np.mean(test_preds, axis=0)  # (N_test, 3) fold-averaged

out_df = sub.copy()
for i, tid in enumerate(test_ids):
    row_idx = out_df.index[out_df['id'] == tid][0]
    out_df.loc[row_idx, 'x'] = test_pred_mean[i, 0]
    out_df.loc[row_idx, 'y'] = test_pred_mean[i, 1]
    out_df.loc[row_idx, 'z'] = test_pred_mean[i, 2]

sub_path = os.path.join(OUT_DIR, f'sub_lgbm_OOF{oof_rhit:.4f}.csv')
out_df.to_csv(sub_path, index=False)
print(f'Saved: {sub_path}')

# ── Calibration ────────────────────────────────────────────────
print('\n=== Calibration ===')
best_cal = oof_rhit
best_alpha = np.ones(3)
for ax in range(3):
    for alpha in np.arange(0.85, 1.11, 0.01):
        tmp = oof_pred.copy()
        tmp[:, ax] = kal_tr[:, ax] + alpha * (oof_pred[:, ax] - kal_tr[:, ax])
        rh = r_hit(tmp, y_train)
        if rh > best_cal:
            best_cal = rh
            best_alpha[ax] = alpha

print(f'Best alpha: {best_alpha}')
print(f'Cal OOF R-Hit: {best_cal:.4f}')

# Apply calibration to test predictions
test_pred_cal = test_pred_mean.copy()
for ax in range(3):
    test_pred_cal[:, ax] = kal_te[:, ax] + best_alpha[ax] * (test_pred_mean[:, ax] - kal_te[:, ax])

out_cal = sub.copy()
for i, tid in enumerate(test_ids):
    row_idx = out_cal.index[out_cal['id'] == tid][0]
    out_cal.loc[row_idx, 'x'] = test_pred_cal[i, 0]
    out_cal.loc[row_idx, 'y'] = test_pred_cal[i, 1]
    out_cal.loc[row_idx, 'z'] = test_pred_cal[i, 2]

cal_path = os.path.join(OUT_DIR, f'sub_lgbm_cal_OOF{best_cal:.4f}.csv')
out_cal.to_csv(cal_path, index=False)
print(f'Saved calibrated: {cal_path}')

# ── v3_D와 앙상블 ──────────────────────────────────────────────
import glob as glob_
v3d_files = glob_.glob(os.path.join(OUT_DIR, 'sub_v3_D_+both_OOF*.csv'))
if v3d_files:
    v3d_file = v3d_files[0]
    print(f'\n=== Ensemble with {os.path.basename(v3d_file)} ===')
    v3d_sub = pd.read_csv(v3d_file)

    # Align by id
    v3d_pred = np.zeros((len(test_ids), 3))
    for i, tid in enumerate(test_ids):
        row = v3d_sub[v3d_sub['id'] == tid]
        if len(row) > 0:
            v3d_pred[i] = row[['x','y','z']].values[0]

    for w_lgbm in [0.1, 0.2, 0.3, 0.4, 0.5]:
        w_v3d = 1.0 - w_lgbm
        ens_pred = w_v3d * v3d_pred + w_lgbm * test_pred_cal

        ens_out = sub.copy()
        for i, tid in enumerate(test_ids):
            row_idx = ens_out.index[ens_out['id'] == tid][0]
            ens_out.loc[row_idx, 'x'] = ens_pred[i, 0]
            ens_out.loc[row_idx, 'y'] = ens_pred[i, 1]
            ens_out.loc[row_idx, 'z'] = ens_pred[i, 2]

        ens_path = os.path.join(OUT_DIR, f'sub_lgbm_ens_w{int(w_lgbm*10)}_v3d{int(w_v3d*10)}.csv')
        ens_out.to_csv(ens_path, index=False)
        print(f'  w_lgbm={w_lgbm:.1f}: saved {os.path.basename(ens_path)}')
else:
    print('\nv3_D submission not found, skipping ensemble')

print('\nDone!')
