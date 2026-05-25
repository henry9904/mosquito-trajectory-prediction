# -*- coding: utf-8 -*-
"""
train_v7.py — Multi-step Self-Supervised Auxiliary Task
  베이스: train_v3.py config D (Mish + BiGRU-1L + curvature + focal_gamma=2)
  추가: 10스텝 → 11번째 관측 변위 예측 auxiliary task

Multi-step 아이디어:
  - 10개 관측(t=-400ms~-40ms) → 11번째 관측(t=0ms) 변위 예측
  - target_step = rotate(X[:,10] - X[:,9], theta_step9) [body frame]
  - 모델이 단기 외삽 패턴을 학습 → 80ms 외삽에도 도움
  - Self-supervised: 외부 레이블 불필요, 관측 데이터만 사용

구현:
  - 같은 GRUFlex 모델에 head_step 추가 (tanh clip 없음)
  - 훈련 시: 11-step forward (main) + 10-step forward (aux) 두 번 실행
  - 추론 시: 11-step only (aux head 미사용)
  - lS 가중치: Optuna로 탐색 또는 고정값 비교

OOF 목표: v3_D 0.6659 초과
"""
import os, gc, time, random, warnings, pathlib
import numpy as np, pandas as pd, glob
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pywt
warnings.filterwarnings('ignore')

SEED = 0
random.seed(SEED); np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DT, T_PRED = 0.040, 0.080
t_obs = np.arange(-400, 1, 40) / 1000.0

print(f'device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')

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

def yaw_angle(v): return np.arctan2(v[:,1], v[:,0])

def rotate_xy(vec, th):
    c,s = np.cos(th), np.sin(th)
    return np.stack([vec[:,0]*c+vec[:,1]*s, -vec[:,0]*s+vec[:,1]*c, vec[:,2]], -1)

def inv_rot(vec, th):
    c,s = np.cos(th), np.sin(th)
    return np.stack([vec[:,0]*c-vec[:,1]*s, vec[:,0]*s+vec[:,1]*c, vec[:,2]], -1)

kalman_train = kalman_predict(X_train)
kalman_test  = kalman_predict(X_test)
print(f'Kalman R-Hit = {r_hit(kalman_train, y_train):.4f}')

# ── Feature 함수 (v3_D 동일) ───────────────────────────────────
LOG_COLS = ['mean_speed','max_speed','speed_std','mean_acc','max_acc','max_jerk',
            'net_disp','|v_last|','|a_last|','|a_recent|','jerk_last','jerk_recent',
            'noise_poly2','noise_savgol','noise_loo']

def build_seq_curv(X):
    N_ = X.shape[0]
    rel = X - X[:,-1:]
    v   = np.diff(X, axis=1) / DT
    a   = np.diff(v, axis=1) / DT
    vp  = np.concatenate([np.zeros((N_,1,3)), v], 1)
    ap  = np.concatenate([np.zeros((N_,2,3)), a], 1)
    speed = np.linalg.norm(vp, axis=-1, keepdims=True)
    cross = np.cross(vp, ap)
    cross_mag = np.linalg.norm(cross, axis=-1, keepdims=True)
    spd_safe  = speed + 1e-12
    curvature   = np.log1p(cross_mag / (spd_safe**3))
    angular_vel = np.log1p(cross_mag / (spd_safe**2))
    v_unit = vp / (spd_safe + 1e-12)
    cos_t  = (v_unit[:,:-1]*v_unit[:,1:]).sum(-1, keepdims=True).clip(-1,1)
    turn_r = np.concatenate([np.zeros((N_,1,1)), np.arccos(cos_t)], 1)
    return np.concatenate([rel, vp, ap, speed, curvature, angular_vel, turn_r], -1).astype(np.float32)

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

# ── 노이즈 캐시 ────────────────────────────────────────────────
CACHE_FILE = os.path.join(CACHE_DIR, 'noise_cache.npz')
CACHE_10   = os.path.join(CACHE_DIR, 'noise_cache_10step.npz')

if os.path.exists(CACHE_FILE):
    nc=np.load(CACHE_FILE)
    noise_p,noise_s,noise_l=nc['np_'],nc['ns_'],nc['nl_']
    noise_p_te,noise_s_te=nc['np_te'],nc['ns_te']
    print('noise cache (11-step) loaded')
else:
    noise_p=noise_poly2(X_train); noise_s=noise_savgol(X_train)
    noise_l=noise_loo_spline(X_train, t_obs)
    noise_p_te=noise_poly2(X_test); noise_s_te=noise_savgol(X_test)
    np.savez(CACHE_FILE, np_=noise_p, ns_=noise_s, nl_=noise_l,
             np_te=noise_p_te, ns_te=noise_s_te)

# 10-step noise (for auxiliary task)
if os.path.exists(CACHE_10):
    nc10=np.load(CACHE_10)
    noise_p10,noise_s10,noise_l10=nc10['np_'],nc10['ns_'],nc10['nl_']
    print('noise cache (10-step) loaded')
else:
    print('computing 10-step noise features...')
    X10 = X_train[:, :10, :]
    t10 = t_obs[:10]
    noise_p10 = noise_poly2(X10)
    noise_s10 = noise_savgol(X10)
    noise_l10 = noise_loo_spline(X10, t10)
    np.savez(CACHE_10, np_=noise_p10, ns_=noise_s10, nl_=noise_l10)

# ── Feature 빌드 ───────────────────────────────────────────────
so_best, sp_best = 0.000267, 1.0

# 11-step features (main task)
scal_tr  = build_scalar(X_train, noise_p, noise_s, noise_l)
scal_te  = build_scalar(X_test,  noise_p_te, noise_s_te)
t3_tr    = build_tier3(X_train); t3_te = build_tier3(X_test)
wav_tr   = build_wavelet(X_train); wav_te = build_wavelet(X_test)
scal_full_tr = np.concatenate([scal_tr, t3_tr, wav_tr], -1)
scal_full_te = np.concatenate([scal_te, t3_te, wav_te], -1)
seq_tr = build_seq_curv(X_train)
seq_te = build_seq_curv(X_test)

# 10-step features (auxiliary task)
X10_tr = X_train[:, :10, :]
t10    = t_obs[:10]
scal10_tr  = build_scalar(X10_tr, noise_p10, noise_s10, noise_l10)
# t3 uses cumpath which has length T (10 vs 11) → shape mismatch fix: reuse 11-step t3
# tier3 rolling stats are similar for 10 vs 11 steps (last step hidden is minor)
wav10_tr   = build_wavelet(X10_tr)
scal_full10_tr = np.concatenate([scal10_tr, t3_tr, wav10_tr], -1)  # t3_tr (11-step) reused
seq10_tr   = build_seq_curv(X10_tr)

SEQ_DIM  = seq_tr.shape[2]    # 13
SCAL_DIM = scal_full_tr.shape[1]  # 73

# ── Target ────────────────────────────────────────────────────
theta_tr = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)

kal_tr   = kalman_predict(X_train, so_best, sp_best)
kal_te   = kalman_predict(X_test,  so_best, sp_best)
kal_wk   = kalman_predict(X_train, 1e-3, sp_best)

target_main = rotate_xy(y_train - kal_tr, theta_tr).astype(np.float32)
aux_F       = rotate_xy(y_train - X_train[:,-1], theta_tr).astype(np.float32)
aux_W       = rotate_xy(y_train - kal_wk, theta_tr).astype(np.float32)

# Multi-step auxiliary target: X[10]-X[9] displacement in step-9 body frame
theta_step9   = yaw_angle((X_train[:,9]-X_train[:,8])/DT)  # yaw at step 9 (-40ms to -80ms)
delta_step10  = X_train[:,10] - X_train[:,9]               # displacement to last obs
target_step10 = rotate_xy(delta_step10, theta_step9).astype(np.float32)  # body frame

# Normalize step target for stable training (scale ~DT*speed)
step_scale = np.linalg.norm(target_step10, axis=-1).mean()
print(f'step10 target mean magnitude: {step_scale*100:.2f}cm')
print(f'main target std (cm): {target_main.std(0)*100}')

# ── Loss 함수 ──────────────────────────────────────────────────
def loss_euclid(p, t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p, t, b=0.002):
    return torch.sigmoid((torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p, t, gamma=2.0):
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss = torch.sigmoid((err-0.01)/0.002)
    w = p_miss.pow(gamma).detach(); w = w/(w.mean()+1e-8)
    return (w*err).mean()

def loss_combo_focal(p, t, gamma=2.0):
    return loss_focal_euclid(p,t,gamma) + 0.3*loss_softhit(p,t)

# ── 모델 (head_step 추가) ──────────────────────────────────────
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class GRUMultiStep(nn.Module):
    def __init__(self, seq_dim, scal_dim,
                 hidden=64, num_layers=1, bidirectional=True,
                 fc_hidden=128, dropout=0.25, main_clip=0.03):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, num_layers,
                          bidirectional=bidirectional, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.)
        gru_out = hidden * (2 if bidirectional else 1)
        act = Mish()
        self.net = nn.Sequential(
            nn.Linear(gru_out+scal_dim, fc_hidden), act,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden//2), act,
        )
        self.head_main = nn.Linear(fc_hidden//2, 3)
        self.head_F    = nn.Linear(fc_hidden//2, 3)
        self.head_W    = nn.Linear(fc_hidden//2, 3)
        # Multi-step: predict displacement of the next observed step
        # No tanh clip — displacement can be any size
        self.head_step = nn.Linear(fc_hidden//2, 3)
        self.clip = main_clip

    def forward(self, seq, scal):
        x = self.gru(seq)[0][:,-1]
        z = self.net(torch.cat([x, scal], 1))
        return (torch.tanh(self.head_main(z))*self.clip,
                self.head_F(z), self.head_W(z),
                self.head_step(z))

# ── 정규화 ─────────────────────────────────────────────────────
class WhiteningScaler:
    def fit(self, X):
        self.mu = X.mean(0)
        cov = np.cov(X.T) + np.eye(X.shape[1])*1e-8
        self.L  = np.linalg.cholesky(cov)
        self.Li = np.linalg.inv(self.L)
        return self
    def transform(self, X): return (X-self.mu) @ self.Li.T

def norm_seq(arr, sc):
    N_,T,C = arr.shape
    return sc.transform(arr.reshape(-1,C)).astype(np.float32).reshape(N_,T,C)

def make_fold_data(tri, vai):
    # 11-step (main)
    sc_seq  = WhiteningScaler().fit(seq_tr[tri].reshape(-1, SEQ_DIM))
    sc_scal = StandardScaler().fit(scal_full_tr[tri])
    # 10-step (aux) — share the same scalers for simplicity
    sc_seq10  = WhiteningScaler().fit(seq10_tr[tri].reshape(-1, SEQ_DIM))
    sc_scal10 = StandardScaler().fit(scal_full10_tr[tri])
    return {
        'seq_tr':   norm_seq(seq_tr[tri], sc_seq),
        'scal_tr':  sc_scal.transform(scal_full_tr[tri]).astype(np.float32),
        'tgt_tr':   target_main[tri],
        'af_tr':    aux_F[tri], 'aw_tr': aux_W[tri],
        'step_tr':  target_step10[tri],
        # 10-step (aux) — same indices, shorter seq
        'seq10_tr': norm_seq(seq10_tr[tri], sc_seq10),
        'scal10_tr':sc_scal10.transform(scal_full10_tr[tri]).astype(np.float32),
        # validation
        'seq_va':   norm_seq(seq_tr[vai], sc_seq),
        'scal_va':  sc_scal.transform(scal_full_tr[vai]).astype(np.float32),
        'tgt_va':   target_main[vai],
        'vai': vai,
        'sc_seq': sc_seq, 'sc_scal': sc_scal,
    }

# ── 훈련 ───────────────────────────────────────────────────────
CFG = dict(lr=0.000989, wd=0.000886, batch=256, lF=0.50, lW=0.25,
           focal_gamma=2.0, lS=0.30)  # lS: multi-step weight
N_FOLDS, N_SEEDS, FULL_EP, FULL_ESTOP = 5, 3, 400, 60

def train_fold(model, fd, n_ep, estop):
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']
    gamma, lS = CFG['focal_gamma'], CFG['lS']

    def T(a): return torch.from_numpy(np.asarray(a, np.float32)).to(device)
    # 11-step tensors
    st, sc   = T(fd['seq_tr']),   T(fd['scal_tr'])
    tt, af   = T(fd['tgt_tr']),   T(fd['af_tr'])
    aw, st10 = T(fd['aw_tr']),    T(fd['step_tr'])
    # 10-step tensors
    st10_seq, sc10 = T(fd['seq10_tr']), T(fd['scal10_tr'])
    # validation
    sv, scv  = T(fd['seq_va']),   T(fd['scal_va'])

    n_tr = st.shape[0]
    best_rh, best_state, patience = -1.0, None, 0

    for ep in range(n_ep):
        model.train()
        idx = torch.randperm(n_tr)
        for i in range(0, n_tr, bs):
            b = idx[i:i+bs]

            # --- Main: 11-step → predict 80ms residual ---
            om, oF, oW, _ = model(st[b], sc[b])
            loss_main = (loss_combo_focal(om, tt[b], gamma)
                         + lF * loss_euclid(oF, af[b])
                         + lW * loss_euclid(oW, aw[b]))

            # --- Aux: 10-step → predict last displacement ---
            _, _, _, oStep = model(st10_seq[b], sc10[b])
            loss_step = loss_euclid(oStep, st10[b])

            loss = loss_main + lS * loss_step
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                ov,_,_,_ = model(sv, scv)
            ov_np = ov.cpu().numpy()
            vai = fd['vai']
            pv = kal_tr[vai] + inv_rot(ov_np, theta_tr[vai])
            rh = r_hit(pv, y_train[vai])
            if rh > best_rh:
                best_rh = rh; patience = 0
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= estop // 5: break

    model.load_state_dict(best_state)
    return best_rh, model

# ── lS ablation: 0.0 (baseline) vs 0.3 vs 0.5 vs 1.0 ─────────
print(f'\n[Config] Mish + BiGRU-1L + focal_gamma={CFG["focal_gamma"]}')
print(f'[Multi-step] lS={CFG["lS"]} (0=baseline, >0=aux step task)')
print(f'[Folds] {N_FOLDS}x{N_SEEDS}  EP={FULL_EP}  ESTOP={FULL_ESTOP}')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

# Run three lS values to find best
LS_VALUES = [0.0, 0.3, 1.0]
results = {}

for lS_val in LS_VALUES:
    CFG['lS'] = lS_val
    print(f'\n=== lS={lS_val} ===')

    oof_rot    = np.zeros((N, 3))
    test_folds = []
    fold_rh    = []
    t0         = time.time()

    for fi, (tri, vai) in enumerate(kf.split(np.arange(N))):
        fd = make_fold_data(tri, vai)

        te_seq_n  = norm_seq(seq_te, fd['sc_seq'])
        te_scal_n = fd['sc_scal'].transform(scal_full_te).astype(np.float32)
        te_seq_t  = torch.from_numpy(te_seq_n).to(device)
        te_scal_t = torch.from_numpy(te_scal_n).to(device)

        seed_oof, seed_te = [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            m = GRUMultiStep(SEQ_DIM, SCAL_DIM).to(device)
            _, m = train_fold(m, fd, FULL_EP, FULL_ESTOP)
            m.eval()
            sv_t  = torch.from_numpy(fd['seq_va']).to(device)
            scv_t = torch.from_numpy(fd['scal_va']).to(device)
            with torch.no_grad():
                ov,_,_,_ = m(sv_t, scv_t)
                te_,_,_,_= m(te_seq_t, te_scal_t)
            seed_oof.append(ov.cpu().numpy())
            seed_te.append(te_.cpu().numpy())
            torch.cuda.empty_cache(); gc.collect()

        vr = np.mean(seed_oof, 0); tr_ = np.mean(seed_te, 0)
        oof_rot[vai] = vr; test_folds.append(tr_)

        pv = kal_tr[vai] + inv_rot(vr, theta_tr[vai])
        rh = r_hit(pv, y_train[vai]); fold_rh.append(rh)
        print(f'  fold {fi+1}: R-Hit={rh:.4f}  ({(time.time()-t0)/60:.1f}min)')

    oof_pred = kal_tr + inv_rot(oof_rot, theta_tr)
    oof_rh   = r_hit(oof_pred, y_train)

    best_cal, best_a = -1, np.ones(3)
    for ax in np.arange(0.85,1.11,0.05):
        for ay in np.arange(0.85,1.06,0.05):
            for az in np.arange(0.85,1.11,0.05):
                a = np.array([ax,ay,az])
                rc = r_hit(kal_tr + inv_rot(oof_rot*a, theta_tr), y_train)
                if rc > best_cal: best_cal, best_a = rc, a.copy()

    print(f'  OOF={oof_rh:.4f}  CalOOF={best_cal:.4f}  '
          f'alpha=({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')

    results[lS_val] = dict(oof=oof_rh, cal=best_cal, alpha=best_a,
                            test_rot=np.mean(test_folds,0)*best_a)

    # Save best (lS=0.0 is baseline comparison)
    ts = time.strftime('%m%d_%H%M')
    test_pred = kal_te + inv_rot(results[lS_val]['test_rot'], theta_te)
    fname = f'sub_v7_lS{lS_val:.1f}_OOF{oof_rh:.4f}.csv'
    pd.DataFrame({'id': test_ids,
                  'x': test_pred[:,0], 'y': test_pred[:,1], 'z': test_pred[:,2]}
    ).to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f'  saved: {fname}')

# ── 결과 요약 ──────────────────────────────────────────────────
print(f'\n=== v7 Multi-step Ablation 결과 ===')
print(f'  {"lS":>6}  {"OOF":>8}  {"CalOOF":>8}')
for lS_val in LS_VALUES:
    r = results[lS_val]
    marker = ' ← best' if r['oof'] == max(v['oof'] for v in results.values()) else ''
    print(f'  {lS_val:6.1f}  {r["oof"]:.4f}  {r["cal"]:.4f}{marker}')

print(f'\n  v3_D baseline: OOF=0.6659')
best_lS  = max(results, key=lambda k: results[k]['oof'])
best_oof = results[best_lS]['oof']
print(f'  v7 best (lS={best_lS}): OOF={best_oof:.4f}  Delta={best_oof-0.6659:+.4f}')
