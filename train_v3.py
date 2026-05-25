# -*- coding: utf-8 -*-
"""
train_v3.py — Ablation study on top of v2 best config (trial #62)
  A. baseline  : v2 best (Mish + BiGRU + hidden=64) re-run
  B. +features : + curvature / angular-velocity seq features
  C. +focal    : + focal regression loss (hard-example upweight)
  D. +both     : B + C

EDA 근거:
  - Residual kurtosis=28~30 → heavy-tail → focal loss 유효
  - C1/C4 cluster (급기동 20%) hit rate 27~35% → curvature feature 필요
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

# ── 재현성 ─────────────────────────────────────────────────────
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
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── 경로 ───────────────────────────────────────────────────────
DATA_DIR  = str(pathlib.Path(__file__).parent / 'open')
CACHE_DIR = str(pathlib.Path(__file__).parent / 'cache')
OUT_DIR   = str(pathlib.Path(__file__).parent / 'submissions')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
print(f'DATA_DIR: {DATA_DIR}')

# ── 데이터 로드 ────────────────────────────────────────────────
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
y_train   = labels.set_index('id').loc[train_ids][['x','y','z']].values.astype(np.float64)
print(f'X_train {X_train.shape}  y_train {y_train.shape}')

# ── 기본 유틸 ─────────────────────────────────────────────────
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N, T, _ = X.shape
    F  = np.array([[1,DT],[0,1]]); Fp = np.array([[1,T_PRED],[0,1]])
    Q  = sigma_proc**2 * np.array([[DT**4/4, DT**3/2],[DT**3/2, DT**2]])
    R  = sigma_obs**2; pred = np.zeros((N,3))
    for j in range(3):
        z=X[:,:,j]; s=np.zeros((N,2)); s[:,0]=z[:,0]; P=np.eye(2)
        for t in range(1,T):
            s=s@F.T; P=F@P@F.T+Q
            inn=z[:,t]-s[:,0]; S=P[0,0]+R; K=P[:,0]/S
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
    return np.stack([vec[:,0]*c-vec[:,1]*s,  vec[:,0]*s+vec[:,1]*c, vec[:,2]], -1)

kalman_train = kalman_predict(X_train)
kalman_test  = kalman_predict(X_test)
print(f'Kalman R-Hit = {r_hit(kalman_train, y_train):.4f}')

# ════════════════════════════════════════════════════════════════
# Feature engineering
# ════════════════════════════════════════════════════════════════

# ── 기존 seq (9 dim) ───────────────────────────────────────────
def build_seq_base(X):
    N = X.shape[0]
    rel = X - X[:,-1:]
    v   = np.diff(X, axis=1) / DT
    vp  = np.concatenate([np.zeros((N,1,3)), v], 1)
    a   = np.diff(v, axis=1) / DT
    ap  = np.concatenate([np.zeros((N,2,3)), a], 1)
    return np.concatenate([rel, vp, ap], -1).astype(np.float32)  # (N,11,9)

# ── 새 seq (13 dim): +곡률·각속도·회전율·속도크기 ─────────────
def build_seq_curv(X):
    """EDA 근거: C1/C4 급기동 클러스터 탐지를 위한 곡률/각속도 feature 추가
    추가 4채널:
      speed       : |v|         — 순간 속력
      curvature   : |v × a|/|v|^3  — 경로 곡률 (급선회일수록 큰 값)
      angular_vel : |v × a|/|v|^2  — 각속도 크기
      turn_rate   : acos(cos_turn)  — 인접 속도벡터 각도 변화 [rad]
    """
    N = X.shape[0]
    rel = X - X[:,-1:]                          # (N,11,3)
    v   = np.diff(X, axis=1) / DT              # (N,10,3)
    a   = np.diff(v, axis=1) / DT              # (N,9,3)

    # pad 기준: 앞쪽 0 패딩으로 11 timestep 맞추기
    vp  = np.concatenate([np.zeros((N,1,3)), v], 1)  # (N,11,3)
    ap  = np.concatenate([np.zeros((N,2,3)), a], 1)  # (N,11,3)

    # speed (N,11,1)
    speed = np.linalg.norm(vp, axis=-1, keepdims=True)  # (N,11,1)

    # cross product v × a — 곡률·각속도 공통 분자
    # v와 a를 11 dim으로 정렬: ap는 t=2~10에 가속도, t=0~1은 0
    cross = np.cross(vp, ap)                             # (N,11,3)
    cross_mag = np.linalg.norm(cross, axis=-1, keepdims=True)  # (N,11,1)
    spd_safe  = speed + 1e-12

    curvature   = cross_mag / (spd_safe**3)              # (N,11,1)
    angular_vel = cross_mag / (spd_safe**2)              # (N,11,1)

    # 인접 속도 벡터 각도 (turn rate)
    v_unit = vp / (spd_safe + 1e-12)                    # (N,11,3)
    cos_t  = (v_unit[:,:-1] * v_unit[:,1:]).sum(-1, keepdims=True).clip(-1,1)  # (N,10,1)
    turn_r = np.arccos(cos_t)                           # (N,10,1)  [rad]
    turn_r = np.concatenate([np.zeros((N,1,1)), turn_r], 1)  # (N,11,1) 패드

    # 로그 스케일 (곡률·각속도 range가 매우 넓음)
    curvature   = np.log1p(curvature)
    angular_vel = np.log1p(angular_vel)

    return np.concatenate([rel, vp, ap, speed, curvature, angular_vel, turn_r], -1
                          ).astype(np.float32)  # (N,11,13)

# ── scalar feature ─────────────────────────────────────────────
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

    # 기존 21개 feature
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

    # EDA 추가: 곡률·각속도 요약 통계 6개
    cross = np.cross(v[:,:-1], a)                        # (N,9,3)
    cross_mag = np.linalg.norm(cross, axis=-1)           # (N,9)
    sp_mid = np.linalg.norm(v[:,:-1], axis=-1)           # (N,9)
    kappa = cross_mag / (sp_mid**3 + 1e-12)              # 곡률 (N,9)
    omega = cross_mag / (sp_mid**2 + 1e-12)              # 각속도 (N,9)
    v_unit = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cos_t  = (v_unit[:,:-1]*v_unit[:,1:]).sum(-1).clip(-1,1)   # (N,9)
    turn_r = np.arccos(cos_t)                            # (N,9) [rad]

    df['mean_curvature'] = np.log1p(kappa.mean(1))
    df['max_curvature']  = np.log1p(kappa.max(1))
    df['mean_angvel']    = np.log1p(omega.mean(1))
    df['max_angvel']     = np.log1p(omega.max(1))
    df['mean_turnrate']  = turn_r.mean(1)
    df['max_turnrate']   = turn_r.max(1)

    return df.values.astype(np.float32)   # (N, 27) with curvature features

def build_tier3(X):
    d=np.diff(X,axis=1); v=d/DT; sp=np.linalg.norm(v,axis=-1)
    sr=np.stack([sp[:,i:i+3].mean(1) for i in range(8)],1)*DT
    cp=np.concatenate([np.zeros((X.shape[0],1)),
                       np.cumsum(np.linalg.norm(d,axis=-1),1)],1)
    return np.concatenate([sr,cp],1).astype(np.float32)

def build_wavelet(X):
    feats=[]
    for ax in range(3):
        for c in pywt.wavedec(X[:,:,ax], 'db4', level=2, axis=1):
            feats+=[c.mean(1), c.std(1), np.abs(c).max(1)]
    return np.column_stack(feats).astype(np.float32)

# ── 노이즈 캐시 ────────────────────────────────────────────────
def noise_poly2(X):
    V=np.vander(t_obs,3,increasing=False); o=np.zeros(X.shape[0])
    for j in range(3):
        c=np.linalg.lstsq(V,X[:,:,j].T,rcond=None)[0]
        o+=(X[:,:,j]-(V@c).T).std(1)
    return o/3

def noise_savgol(X):
    return (X-savgol_filter(X,5,2,axis=1)).std(1).mean(-1)

def noise_loo_spline(X):
    N,T,_=X.shape; o=np.zeros(N); idx=np.arange(T)
    for i in tqdm(range(N),desc='LOO spline'):
        s=0
        for k in range(1,T-1):
            m=idx!=k
            for j in range(3):
                cs=CubicSpline(t_obs[m],X[i,m,j])
                s+=(X[i,k,j]-cs(t_obs[k]))**2
        o[i]=np.sqrt(s/((T-2)*3))
    return o

CACHE_FILE = os.path.join(CACHE_DIR, 'noise_cache.npz')
if os.path.exists(CACHE_FILE):
    nc=np.load(CACHE_FILE)
    noise_p,noise_s,noise_l=nc['np_'],nc['ns_'],nc['nl_']
    noise_p_te,noise_s_te=nc['np_te'],nc['ns_te']
    print('noise cache loaded')
else:
    print('computing noise features...')
    noise_p=noise_poly2(X_train); noise_s=noise_savgol(X_train)
    noise_l=noise_loo_spline(X_train)
    noise_p_te=noise_poly2(X_test); noise_s_te=noise_savgol(X_test)
    np.savez(CACHE_FILE, np_=noise_p, ns_=noise_s, nl_=noise_l,
             np_te=noise_p_te, ns_te=noise_s_te)
    print(f'saved: {CACHE_FILE}')

# ── feature 빌드 ───────────────────────────────────────────────
scal_tr  = build_scalar(X_train, noise_p, noise_s, noise_l)
scal_te  = build_scalar(X_test,  noise_p_te, noise_s_te)
t3_tr    = build_tier3(X_train); t3_te = build_tier3(X_test)
wav_tr   = build_wavelet(X_train); wav_te = build_wavelet(X_test)

scal_full_tr = np.concatenate([scal_tr, t3_tr, wav_tr], -1)
scal_full_te = np.concatenate([scal_te, t3_te, wav_te], -1)

seq_base_tr  = build_seq_base(X_train);  seq_base_te  = build_seq_base(X_test)
seq_curv_tr  = build_seq_curv(X_train);  seq_curv_te  = build_seq_curv(X_test)

print(f'scal_full {scal_full_tr.shape}')
print(f'seq_base  {seq_base_tr.shape}   seq_curv {seq_curv_tr.shape}')

# ── target ──────────────────────────────────────────────────────
theta_tr = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)

so_best, sp_best = 0.000267, 1.0
kal_tr_best = kalman_predict(X_train, so_best, sp_best)
kal_te_best = kalman_predict(X_test,  so_best, sp_best)
target_main = rotate_xy(y_train - kal_tr_best, theta_tr).astype(np.float32)
aux_F       = rotate_xy(y_train - X_train[:,-1], theta_tr).astype(np.float32)
aux_W       = rotate_xy(y_train - kalman_predict(X_train, 1e-3), theta_tr).astype(np.float32)
print(f'target std (cm): {target_main.std(0)*100}')

# ════════════════════════════════════════════════════════════════
# Loss functions
# ════════════════════════════════════════════════════════════════
def loss_euclid(p, t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p, t, b=0.002):
    return torch.sigmoid(
        (torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_combo(p, t):
    return loss_euclid(p, t) + 0.3*loss_softhit(p, t)

def loss_focal_euclid(p, t, gamma=2.0):
    """EDA 근거: residual kurtosis=28~30 → heavy-tail → focal upweighting 유효
    오차 > 1cm인 hard sample에 (p_miss)^gamma 가중치 부여.
    gradient는 weight를 detach해서 안정적으로 학습.
    """
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)          # (N,)
    p_miss = torch.sigmoid((err - 0.01) / 0.002)        # 1cm 초과 확률
    w = p_miss.pow(gamma).detach()                       # gradient 차단
    w = w / (w.mean() + 1e-8)                            # 정규화
    return (w * err).mean()

def loss_combo_focal(p, t, gamma=2.0):
    return loss_focal_euclid(p, t, gamma) + 0.3*loss_softhit(p, t)

# ════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class GRUFlex(nn.Module):
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
            nn.Linear(gru_out + scal_dim, fc_hidden), act,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden//2), act,
        )
        self.head_main = nn.Linear(fc_hidden//2, 3)
        self.head_F    = nn.Linear(fc_hidden//2, 3)
        self.head_W    = nn.Linear(fc_hidden//2, 3)
        self.clip = main_clip

    def forward(self, seq, scal):
        x = self.gru(seq)[0][:,-1]
        z = self.net(torch.cat([x, scal], 1))
        return (torch.tanh(self.head_main(z)) * self.clip,
                self.head_F(z), self.head_W(z))

# ════════════════════════════════════════════════════════════════
# Training utils
# ════════════════════════════════════════════════════════════════
class WhiteningScaler:
    def fit(self, X):
        self.mu = X.mean(0)
        cov = np.cov(X.T) + np.eye(X.shape[1])*1e-8
        self.L  = np.linalg.cholesky(cov)
        self.Li = np.linalg.inv(self.L)
        return self
    def transform(self, X): return (X - self.mu) @ self.Li.T

def norm_seq(arr, sc):
    N,T,C = arr.shape
    return sc.transform(arr.reshape(-1,C)).astype(np.float32).reshape(N,T,C)

def make_fold_data(tri, vai, seq_arr, scal_arr):
    sc_seq  = WhiteningScaler().fit(seq_arr[tri].reshape(-1, seq_arr.shape[2]))
    sc_scal = StandardScaler().fit(scal_arr[tri])
    return {
        'seq_tr':   norm_seq(seq_arr[tri], sc_seq),
        'scal_tr':  sc_scal.transform(scal_arr[tri]).astype(np.float32),
        'seq_va':   norm_seq(seq_arr[vai], sc_seq),
        'scal_va':  sc_scal.transform(scal_arr[vai]).astype(np.float32),
        'sc_seq': sc_seq, 'sc_scal': sc_scal,
    }

# trial #62 best hyperparams
CFG = dict(lr=0.000989, wd=0.000886, batch=256, lF=0.50, lW=0.25)
N_FOLDS, N_SEEDS, FULL_EP, FULL_ESTOP = 5, 3, 350, 50

def train_fold(model, fd, loss_fn_main, n_ep, estop):
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']

    def T(a): return torch.from_numpy(np.asarray(a, np.float32)).to(device)
    st, sc = T(fd['seq_tr']),  T(fd['scal_tr'])
    sv, scv= T(fd['seq_va']),  T(fd['scal_va'])
    tt  = T(target_main[fd['va_idx']] if 'va_idx' in fd else fd['tgt_tr'])
    af  = T(fd['af_tr']); aw = T(fd['aw_tr'])

    best, bst, no = -1, None, 0
    n = st.shape[0]
    for ep in range(n_ep):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            om, oF, oW = model(st[idx], sc[idx])
            loss = (loss_fn_main(om, tt[idx])
                    + lF * loss_euclid(oF, af[idx])
                    + lW * loss_euclid(oW, aw[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): ov,_,_ = model(sv, scv)
        pred = kal_tr_best[fd['vai']] + inv_rot(ov.cpu().numpy(), theta_tr[fd['vai']])
        rh = r_hit(pred, y_train[fd['vai']])
        if rh > best:
            best = rh
            bst = {k:v.detach().clone() for k,v in model.state_dict().items()}
            no = 0
        else:
            no += 1
        if no >= estop: break
    model.load_state_dict(bst)
    return best, model

# ════════════════════════════════════════════════════════════════
# Ablation 실험 정의
# ════════════════════════════════════════════════════════════════
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

ABLATIONS = [
    # name         use_curv  focal_gamma
    ('A_baseline', False,    0.0),
    ('B_+features',True,     0.0),
    ('C_+focal',   False,    2.0),
    ('D_+both',    True,     2.0),
]

summary = []

for abl_name, use_curv, focal_gamma in ABLATIONS:
    seq_tr = seq_curv_tr if use_curv else seq_base_tr
    seq_te = seq_curv_te if use_curv else seq_base_te
    seq_dim = seq_tr.shape[2]
    scal_dim= scal_full_tr.shape[1]

    loss_fn_main = (loss_combo_focal if focal_gamma > 0
                    else loss_combo)

    print(f'\n{"="*60}')
    print(f'[{abl_name}]  seq_dim={seq_dim}  focal={focal_gamma>0}')
    print(f'{"="*60}')

    oof_rot    = np.zeros((len(X_train), 3))
    test_folds = []
    fold_rh    = []
    t0         = time.time()

    for fi, (tri, vai) in enumerate(kf.split(np.arange(len(X_train)))):
        fd = make_fold_data(tri, vai, seq_tr, scal_full_tr)
        fd['tgt_tr'] = target_main[tri]
        fd['af_tr']  = aux_F[tri]
        fd['aw_tr']  = aux_W[tri]
        fd['vai']    = vai

        te_seq_n  = norm_seq(seq_te, fd['sc_seq'])
        te_scal_n = fd['sc_scal'].transform(scal_full_te).astype(np.float32)
        te_seq_t  = torch.from_numpy(te_seq_n).to(device)
        te_scal_t = torch.from_numpy(te_scal_n).to(device)

        seed_oof, seed_te = [], []
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            m = GRUFlex(seq_dim, scal_dim).to(device)
            _, m = train_fold(m, fd, loss_fn_main, FULL_EP, FULL_ESTOP)
            m.eval()
            sv_t  = torch.from_numpy(fd['seq_va']).to(device)
            sc_t  = torch.from_numpy(fd['scal_va']).to(device)
            with torch.no_grad():
                ov,_,_ = m(sv_t, sc_t)
                te_,_,_= m(te_seq_t, te_scal_t)
            seed_oof.append(ov.cpu().numpy())
            seed_te.append(te_.cpu().numpy())
            torch.cuda.empty_cache(); gc.collect()

        vr  = np.mean(seed_oof, 0); tr_ = np.mean(seed_te, 0)
        oof_rot[vai] = vr; test_folds.append(tr_)
        pv  = kal_tr_best[vai] + inv_rot(vr, theta_tr[vai])
        rh  = r_hit(pv, y_train[vai]); fold_rh.append(rh)
        print(f'  fold {fi+1}: R-Hit={rh:.4f}  ({(time.time()-t0)/60:.1f}min)')

    # OOF 평가
    oof_pred = kal_tr_best + inv_rot(oof_rot, theta_tr)
    oof_rh   = r_hit(oof_pred, y_train)

    # Calibration
    best_cal, best_a = -1, np.ones(3)
    for ax in np.arange(0.85, 1.11, 0.05):
        for ay in np.arange(0.85, 1.06, 0.05):
            for az in np.arange(0.85, 1.11, 0.05):
                a = np.array([ax, ay, az])
                rc = r_hit(kal_tr_best + inv_rot(oof_rot*a, theta_tr), y_train)
                if rc > best_cal: best_cal, best_a = rc, a.copy()

    print(f'  OOF={oof_rh:.4f}  CalOOF={best_cal:.4f}  '
          f'alpha=({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')

    # 테스트 예측 저장
    test_rot  = np.mean(test_folds, 0) * best_a
    test_pred = kal_te_best + inv_rot(test_rot, theta_te)
    ts = time.strftime('%m%d_%H%M')
    fname = f'sub_v3_{abl_name}_OOF{oof_rh:.4f}.csv'
    pd.DataFrame({'id': test_ids,
                  'x': test_pred[:,0], 'y': test_pred[:,1], 'z': test_pred[:,2]}
    ).to_csv(os.path.join(OUT_DIR, fname), index=False)
    print(f'  saved: {fname}')

    summary.append({
        'config': abl_name,
        'use_curv': use_curv,
        'focal': focal_gamma > 0,
        'oof': oof_rh,
        'cal_oof': best_cal,
        'alpha': best_a,
        'time_min': (time.time()-t0)/60,
    })

# ════════════════════════════════════════════════════════════════
# 최종 비교표
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('=== Ablation Results ===')
print(f'{"Config":<18} {"OOF":>7} {"CalOOF":>8} {"+feat":>6} {"+focal":>7} {"time":>6}')
print('-'*60)
v2_baseline = 0.6642  # train_v2 trial#62 참고값
print(f'{"v2_best(ref)":<18} {v2_baseline:>7.4f} {"0.6647":>8} {"N":>6} {"N":>7}')
for r in summary:
    print(f'{r["config"]:<18} {r["oof"]:>7.4f} {r["cal_oof"]:>8.4f} '
          f'{"Y" if r["use_curv"] else "N":>6} '
          f'{"Y" if r["focal"] else "N":>7} '
          f'{r["time_min"]:>5.1f}m')

print('\n=== 개선 기여 분석 ===')
if len(summary) >= 4:
    base  = summary[0]['oof']
    feat  = summary[1]['oof']
    focal = summary[2]['oof']
    both  = summary[3]['oof']
    print(f'  A (baseline)   : {base:.4f}')
    print(f'  B (+features)  : {feat:.4f}  delta={feat-base:+.4f}')
    print(f'  C (+focal)     : {focal:.4f}  delta={focal-base:+.4f}')
    print(f'  D (+both)      : {both:.4f}  delta={both-base:+.4f}')
    interaction = both - base - (feat-base) - (focal-base)
    print(f'  Interaction    : {interaction:+.4f}  (synergy if positive)')
