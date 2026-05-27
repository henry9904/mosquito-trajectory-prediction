import os, glob, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import torch

DATA_DIR  = str(pathlib.Path(__file__).parent / 'open')
CACHE_DIR = str(pathlib.Path(__file__).parent / 'cache')
OUT_DIR   = str(pathlib.Path(__file__).parent / 'submissions')

train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.csv')))
test_files  = sorted(glob.glob(os.path.join(DATA_DIR, 'test',  '*.csv')))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))

print('데이터 로딩 중...')
def load_stack(files):
    return np.stack([pd.read_csv(f)[['x','y','z']].values for f in files]).astype(np.float64)

X_train = load_stack(train_files)
X_test  = load_stack(test_files)
train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
test_ids  = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
y_train   = labels.set_index('id').loc[train_ids][['x','y','z']].values.astype(np.float64)

DT, T_PRED = 0.040, 0.080
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N, T, _ = X.shape
    F_  = np.array([[1,DT],[0,1]]); Fp = np.array([[1,T_PRED],[0,1]])
    Q   = sigma_proc**2 * np.array([[DT**4/4, DT**3/2],[DT**3/2, DT**2]])
    R   = sigma_obs**2; pred = np.zeros((N,3))
    for j in range(3):
        z=X[:,:,j]; s=np.zeros((N,2)); s[:,0]=z[:,0]; P=np.eye(2)
        for t in range(1,T):
            s=s@F_.T; P=F_@P@F_.T+Q
            inn=z[:,t]-s[:,0]; S=P[0,0]+R; K=P[:,0]/S
            s=s+np.outer(inn,K); P=P-np.outer(K,P[0,:])
        pred[:,j]=(s@Fp.T)[:,0]
    return pred

def r_hit(pred, true, thr=0.01):
    return float((np.linalg.norm(pred-true, axis=-1) <= thr).mean())

def yaw_angle(v): return np.arctan2(v[:,1], v[:,0])
def rotate_xy(vec, th):
    c,s = np.cos(th), np.sin(th)
    return np.stack([vec[:,0]*c+vec[:,1]*s, -vec[:,0]*s+vec[:,1]*c, vec[:,2]], -1)
def inv_rot(vec, th):
    c,s = np.cos(th), np.sin(th)
    return np.stack([vec[:,0]*c-vec[:,1]*s, vec[:,0]*s+vec[:,1]*c, vec[:,2]], -1)

so_best, sp_best = 0.000267, 1.0
kal_tr_best = kalman_predict(X_train, so_best, sp_best)
kal_te_best = kalman_predict(X_test,  so_best, sp_best)

theta_tr = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)
target_main = rotate_xy(y_train - kal_tr_best, theta_tr).astype(np.float32)

kf = KFold(n_splits=5, shuffle=True, random_state=0)

models = ['v9', 'v10', 'v11']
oof_preds = {m: np.zeros((len(X_train), 3)) for m in models}
test_folds = {m: [] for m in models}

print('OOF 로딩 및 테스트 예측 로딩 중...')
for fi, (tri, vai) in enumerate(kf.split(np.arange(len(X_train)))):
    for m in models:
        ckpt_oof = os.path.join(CACHE_DIR, f'{m}_ckpt', f'fold{fi}_oof.npy')
        ckpt_te  = os.path.join(CACHE_DIR, f'{m}_ckpt', f'fold{fi}_te.npy')
        
        oof_preds[m][vai] = np.load(ckpt_oof)
        test_folds[m].append(np.load(ckpt_te))

test_preds = {m: np.mean(test_folds[m], 0) for m in models}

for m in models:
    rh = r_hit(kal_tr_best + inv_rot(oof_preds[m], theta_tr), y_train)
    print(f'Model {m} OOF R-Hit: {rh:.4f}')

from sklearn.linear_model import LinearRegression

# Meta Learner (Linear Regression without intercept)
meta_oof = np.zeros_like(target_main)
meta_test = np.zeros_like(test_preds['v9'])

print('Meta-Learner 학습 중 (LinearRegression)...')
for ax in range(3):
    X_meta_tr = np.column_stack([oof_preds[m][:, ax] for m in models])
    y_meta_tr = target_main[:, ax]
    X_meta_te = np.column_stack([test_preds[m][:, ax] for m in models])
    
    for fi, (tri, vai) in enumerate(kf.split(np.arange(len(X_train)))):
        clf = LinearRegression(fit_intercept=False)
        clf.fit(X_meta_tr[tri], y_meta_tr[tri])
        meta_oof[vai, ax] = clf.predict(X_meta_tr[vai])
        
    # Full fit for test
    clf = LinearRegression(fit_intercept=False)
    clf.fit(X_meta_tr, y_meta_tr)
    meta_test[:, ax] = clf.predict(X_meta_te)
    print(f'Axis {ax} Coefs: {clf.coef_} (Sum: {clf.coef_.sum():.4f})')

# Simple Average Baseline
avg_oof = sum(oof_preds[m] for m in models) / len(models)
avg_rh = r_hit(kal_tr_best + inv_rot(avg_oof, theta_tr), y_train)
print(f'Simple Average OOF R-Hit: {avg_rh:.4f}')

avg_test_pred = kal_te_best + inv_rot(sum(test_preds[m] for m in models) / len(models), theta_te)
avg_fname = f'sub_simple_average_OOF{avg_rh:.4f}.csv'
pd.DataFrame({'id': test_ids,
              'x': avg_test_pred[:,0], 'y': avg_test_pred[:,1], 'z': avg_test_pred[:,2]}
).to_csv(os.path.join(OUT_DIR, avg_fname), index=False)
print(f'저장 완료: {avg_fname}')

final_oof_pred = kal_tr_best + inv_rot(meta_oof, theta_tr)
final_rh = r_hit(final_oof_pred, y_train)
print(f'\nMeta-Learner OOF R-Hit: {final_rh:.4f}')

best_cal, best_a = -1, np.ones(3)
for ax in np.arange(0.9, 1.1, 0.05):
    for ay in np.arange(0.9, 1.1, 0.05):
        for az in np.arange(0.9, 1.1, 0.05):
            a = np.array([ax, ay, az])
            rc = r_hit(kal_tr_best + inv_rot(meta_oof*a, theta_tr), y_train)
            if rc > best_cal: best_cal, best_a = rc, a.copy()
print(f'Calibrated Meta OOF: {best_cal:.4f} with alpha={best_a}')

final_test_pred = kal_te_best + inv_rot(meta_test * best_a, theta_te)
fname = f'sub_meta_ridge_OOF{final_rh:.4f}_Cal{best_cal:.4f}.csv'
pd.DataFrame({'id': test_ids,
              'x': final_test_pred[:,0], 'y': final_test_pred[:,1], 'z': final_test_pred[:,2]}
).to_csv(os.path.join(OUT_DIR, fname), index=False)
print(f'저장 완료: {fname}')
