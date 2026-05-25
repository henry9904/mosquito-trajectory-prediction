# -*- coding: utf-8 -*-
"""
train_v6.py — Rotation Augmentation
  베이스: train_v3.py config D (Mish + BiGRU-1L + curvature + focal_gamma=2)
  추가: 3D 궤적 z축 회전 augmentation (90°/180°/270° → 4x 훈련 데이터)

회전 augmentation 근거:
  - 모기의 절대 방향 preference 없음 (yaw 정규화로 이미 방향 무관 학습 시도 중)
  - 그러나 seq feature는 절대좌표 기반 → 추가 yaw rotation이 실질적 다양성 제공
  - 4x 데이터로 일반화 개선 기대
  - 검증: 원래 X_train (0°) 에서만 OOF 계산 → 공정 비교

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

# ── 회전 augmentation ──────────────────────────────────────────
AUG_DEGS = [0, 90, 180, 270]  # 4x 훈련 데이터 (0° = 원본)
N_AUG = len(AUG_DEGS)

def rotate_traj(X, deg):
    """z축 중심 yaw 회전 (x,y 변환, z 불변)"""
    if deg == 0: return X.copy()
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    Xr = X.copy()
    Xr[:,:,0] = X[:,:,0]*c - X[:,:,1]*s
    Xr[:,:,1] = X[:,:,0]*s + X[:,:,1]*c
    return Xr

def rotate_pts(pts, deg):
    """(N,3) 포인트 yaw 회전"""
    if deg == 0: return pts.copy()
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    out = pts.copy()
    out[:,0] = pts[:,0]*c - pts[:,1]*s
    out[:,1] = pts[:,0]*s + pts[:,1]*c
    return out

# ── Feature 함수들 (v3_D 동일) ─────────────────────────────────
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
    V=np.vander(t_obs,3,increasing=False); o=np.zeros(X.shape[0])
    for j in range(3):
        c=np.linalg.lstsq(V,X[:,:,j].T,rcond=None)[0]
        o+=(X[:,:,j]-(V@c).T).std(1)
    return o/3

def noise_savgol(X):
    return (X-savgol_filter(X,5,2,axis=1)).std(1).mean(-1)

def noise_loo_spline(X):
    N_,T,_=X.shape; o=np.zeros(N_); idx=np.arange(T)
    for i in tqdm(range(N_),desc='LOO spline'):
        s=0
        for k in range(1,T-1):
            m=idx!=k
            for j in range(3):
                cs=CubicSpline(t_obs[m],X[i,m,j])
                s+=(X[i,k,j]-cs(t_obs[k]))**2
        o[i]=np.sqrt(s/((T-2)*3))
    return o

# ── 노이즈 캐시 (회전 불변 → 원본에서 1회 계산) ─────────────────
CACHE_FILE = os.path.join(CACHE_DIR, 'noise_cache.npz')
if os.path.exists(CACHE_FILE):
    nc=np.load(CACHE_FILE)
    noise_p,noise_s,noise_l=nc['np_'],nc['ns_'],nc['nl_']
    noise_p_te,noise_s_te=nc['np_te'],nc['ns_te']
    print('noise cache loaded')
else:
    noise_p=noise_poly2(X_train); noise_s=noise_savgol(X_train)
    noise_l=noise_loo_spline(X_train)
    noise_p_te=noise_poly2(X_test); noise_s_te=noise_savgol(X_test)
    np.savez(CACHE_FILE, np_=noise_p, ns_=noise_s, nl_=noise_l,
             np_te=noise_p_te, ns_te=noise_s_te)

# ── Augmented feature 빌드 ─────────────────────────────────────
so_best, sp_best = 0.000267, 1.0

print('Building augmented features...')
aug_seq   = []  # (N_AUG, N, 11, 13)
aug_scal  = []  # (N_AUG, N, SCAL_DIM)
aug_kal   = []  # (N_AUG, N, 3) — Kalman on rotated trajectory
aug_y     = []  # (N_AUG, N, 3) — rotated y labels
aug_theta = []  # (N_AUG, N)   — yaw angle on rotated trajectory
aug_af    = []  # (N_AUG, N, 3) — aux_F target
aug_aw    = []  # (N_AUG, N, 3) — aux_W target (weak Kalman)
aug_tgt   = []  # (N_AUG, N, 3) — main target (best Kalman residual)

for deg in AUG_DEGS:
    Xr = rotate_traj(X_train, deg)
    yr = rotate_pts(y_train, deg)

    seq_r  = build_seq_curv(Xr)
    scal_r = build_scalar(Xr, noise_p, noise_s, noise_l)
    t3_r   = build_tier3(Xr)
    wav_r  = build_wavelet(Xr)
    scal_full_r = np.concatenate([scal_r, t3_r, wav_r], -1)

    kal_r  = kalman_predict(Xr, so_best, sp_best)
    kal_wk = kalman_predict(Xr, 1e-3, sp_best)
    th_r   = yaw_angle((Xr[:,-1]-Xr[:,-2])/DT)

    tgt_r  = rotate_xy(yr - kal_r,             th_r).astype(np.float32)
    af_r   = rotate_xy(yr - Xr[:,-1],          th_r).astype(np.float32)
    aw_r   = rotate_xy(yr - kal_wk,            th_r).astype(np.float32)

    aug_seq.append(seq_r); aug_scal.append(scal_full_r)
    aug_kal.append(kal_r); aug_y.append(yr); aug_theta.append(th_r)
    aug_tgt.append(tgt_r); aug_af.append(af_r); aug_aw.append(aw_r)
    print(f'  deg={deg:3d}°  seq={seq_r.shape}  scal={scal_full_r.shape}')

# 합치기: (N_AUG*N, ...) — 첫 N개가 원본 (deg=0)
seq_all  = np.concatenate(aug_seq, 0)   # (N_AUG*N, 11, 13)
scal_all = np.concatenate(aug_scal, 0)  # (N_AUG*N, SCAL_DIM)
tgt_all  = np.concatenate(aug_tgt, 0)  # (N_AUG*N, 3)
af_all   = np.concatenate(aug_af, 0)   # (N_AUG*N, 3)
aw_all   = np.concatenate(aug_aw, 0)   # (N_AUG*N, 3)
theta_all= np.concatenate(aug_theta,0) # (N_AUG*N,)
kal_all  = np.concatenate(aug_kal, 0)  # (N_AUG*N, 3)

SEQ_DIM  = seq_all.shape[2]   # 13
SCAL_DIM = scal_all.shape[1]  # 73

# 원본 (deg=0) test features
scal_te  = build_scalar(X_test, noise_p_te, noise_s_te)
t3_te    = build_tier3(X_test)
wav_te   = build_wavelet(X_test)
scal_full_te = np.concatenate([scal_te, t3_te, wav_te], -1)
seq_te   = build_seq_curv(X_test)
kal_te   = kalman_predict(X_test, so_best, sp_best)
theta_te = yaw_angle((X_test[:,-1]-X_test[:,-2])/DT)

print(f'\nAug data: seq {seq_all.shape}  scal {scal_all.shape}  ({N_AUG}x)')
print(f'Test:     seq {seq_te.shape}    scal {scal_full_te.shape}')
print(f'target std (cm): {tgt_all[:N].std(0)*100}')  # orig only

# ── Loss 함수 ──────────────────────────────────────────────────
def loss_euclid(p, t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p, t, b=0.002):
    return torch.sigmoid((torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p, t, gamma=2.0):
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss = torch.sigmoid((err-0.01)/0.002)
    w = p_miss.pow(gamma).detach()
    w = w / (w.mean()+1e-8)
    return (w*err).mean()

def loss_combo_focal(p, t, gamma=2.0):
    return loss_focal_euclid(p,t,gamma) + 0.3*loss_softhit(p,t)

# ── 모델 (v3_D 동일: Mish + BiGRU-1L) ─────────────────────────
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
            nn.Linear(gru_out+scal_dim, fc_hidden), act,
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
        return (torch.tanh(self.head_main(z))*self.clip,
                self.head_F(z), self.head_W(z))

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

# ── Fold data: 원본 검증, augmented 훈련 ─────────────────────────
def make_fold_data(tri_orig, vai_orig):
    """
    tri_orig: 원본 훈련 인덱스 (0..N-1)
    vai_orig: 원본 검증 인덱스 (0..N-1)
    훈련: 4x augmented (deg=0,90,180,270)
    검증: deg=0 원본만
    """
    # augmented training indices: each rotation's copy of tri_orig
    tri_aug = np.concatenate([tri_orig + k*N for k in range(N_AUG)])

    # Whitening: fit on ALL augmented training data
    sc_seq  = WhiteningScaler().fit(seq_all[tri_aug].reshape(-1, SEQ_DIM))
    sc_scal = StandardScaler().fit(scal_all[tri_aug])

    return {
        'seq_tr':  norm_seq(seq_all[tri_aug], sc_seq),
        'scal_tr': sc_scal.transform(scal_all[tri_aug]).astype(np.float32),
        'tgt_tr':  tgt_all[tri_aug],
        'af_tr':   af_all[tri_aug],
        'aw_tr':   aw_all[tri_aug],
        # validation: original only
        'seq_va':  norm_seq(seq_all[vai_orig], sc_seq),
        'scal_va': sc_scal.transform(scal_all[vai_orig]).astype(np.float32),
        'tgt_va':  tgt_all[vai_orig],
        'vai':     vai_orig,
        'sc_seq': sc_seq, 'sc_scal': sc_scal,
    }

# ── Training loop ──────────────────────────────────────────────
CFG = dict(lr=0.000989, wd=0.000886, batch=256, lF=0.50, lW=0.25, focal_gamma=2.0)
N_FOLDS, N_SEEDS, FULL_EP, FULL_ESTOP = 5, 3, 400, 60

def train_fold(model, fd, n_ep, estop):
    opt   = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']
    gamma = CFG['focal_gamma']

    def T(a): return torch.from_numpy(np.asarray(a, np.float32)).to(device)
    st, sc = T(fd['seq_tr']),  T(fd['scal_tr'])
    tt      = T(fd['tgt_tr']); af = T(fd['af_tr']); aw = T(fd['aw_tr'])
    sv, scv = T(fd['seq_va']),  T(fd['scal_va'])

    n_tr = st.shape[0]
    best_rh, best_state, patience = -1.0, None, 0

    for ep in range(n_ep):
        model.train()
        idx = torch.randperm(n_tr)
        ep_loss = 0.0
        for i in range(0, n_tr, bs):
            b = idx[i:i+bs]
            om, oF, oW = model(st[b], sc[b])
            loss = (loss_combo_focal(om, tt[b], gamma)
                    + lF * loss_euclid(oF, af[b])
                    + lW * loss_euclid(oW, aw[b]))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(b)
        sched.step()

        if (ep+1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                ov,_,_ = model(sv, scv)
            ov_np = ov.cpu().numpy()
            vai = fd['vai']
            pv = kal_all[vai] + inv_rot(ov_np, theta_all[vai])
            rh = r_hit(pv, y_train[vai])
            if rh > best_rh:
                best_rh = rh; patience = 0
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= estop // 5: break

    model.load_state_dict(best_state)
    return best_rh, model

# ── 5-fold training ────────────────────────────────────────────
print(f'\n[Config] Mish + BiGRU-1L + focal_gamma={CFG["focal_gamma"]}')
print(f'[Augmentation] {N_AUG}x  ({AUG_DEGS} degrees)')
print(f'[Folds] {N_FOLDS}x{N_SEEDS}  EP={FULL_EP}  ESTOP={FULL_ESTOP}')

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

oof_rot    = np.zeros((N, 3))
test_folds = []
fold_rh    = []
t0         = time.time()

for fi, (tri_orig, vai_orig) in enumerate(kf.split(np.arange(N))):
    fd = make_fold_data(tri_orig, vai_orig)

    # test features with fold's scaler
    te_seq_n  = norm_seq(seq_te, fd['sc_seq'])
    te_scal_n = fd['sc_scal'].transform(scal_full_te).astype(np.float32)
    te_seq_t  = torch.from_numpy(te_seq_n).to(device)
    te_scal_t = torch.from_numpy(te_scal_n).to(device)

    seed_oof, seed_te = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        m = GRUFlex(SEQ_DIM, SCAL_DIM).to(device)
        _, m = train_fold(m, fd, FULL_EP, FULL_ESTOP)
        m.eval()
        sv_t  = torch.from_numpy(fd['seq_va']).to(device)
        scv_t = torch.from_numpy(fd['scal_va']).to(device)
        with torch.no_grad():
            ov,_,_ = m(sv_t, scv_t)
            te_,_,_= m(te_seq_t, te_scal_t)
        seed_oof.append(ov.cpu().numpy())
        seed_te.append(te_.cpu().numpy())
        torch.cuda.empty_cache(); gc.collect()

    vr = np.mean(seed_oof, 0); tr_ = np.mean(seed_te, 0)
    oof_rot[vai_orig] = vr; test_folds.append(tr_)

    vai = vai_orig
    pv = kal_all[vai] + inv_rot(vr, theta_all[vai])
    rh = r_hit(pv, y_train[vai]); fold_rh.append(rh)
    print(f'  fold {fi+1}: R-Hit={rh:.4f}  ({(time.time()-t0)/60:.1f}min)')

# ── OOF 평가 ──────────────────────────────────────────────────
oof_pred = kal_all[:N] + inv_rot(oof_rot, theta_all[:N])
oof_rh   = r_hit(oof_pred, y_train)

best_cal, best_a = -1, np.ones(3)
for ax in np.arange(0.85,1.11,0.05):
    for ay in np.arange(0.85,1.06,0.05):
        for az in np.arange(0.85,1.11,0.05):
            a = np.array([ax,ay,az])
            rc = r_hit(kal_all[:N] + inv_rot(oof_rot*a, theta_all[:N]), y_train)
            if rc > best_cal: best_cal, best_a = rc, a.copy()

print(f'\nOOF={oof_rh:.4f}  CalOOF={best_cal:.4f}  '
      f'alpha=({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')
print(f'v3_D baseline OOF=0.6659')
print(f'Delta = {oof_rh - 0.6659:+.4f}')

# ── 제출 파일 저장 ──────────────────────────────────────────────
test_rot  = np.mean(test_folds, 0) * best_a
test_pred = kal_te + inv_rot(test_rot, theta_te)
ts = time.strftime('%m%d_%H%M')
fname = f'sub_v6_rotaug_OOF{oof_rh:.4f}.csv'
pd.DataFrame({'id': test_ids,
              'x': test_pred[:,0], 'y': test_pred[:,1], 'z': test_pred[:,2]}
).to_csv(os.path.join(OUT_DIR, fname), index=False)
print(f'saved: {fname}')

print(f'\n=== 전체 현황 ===')
print(f'  Kalman only  : OOF 0.5964')
print(f'  v3_D baseline: OOF 0.6659  (LB ~0.6827)')
print(f'  v6 rotaug    : OOF {oof_rh:.4f}  (cal {best_cal:.4f})')
