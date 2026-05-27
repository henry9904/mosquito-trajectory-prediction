# -*- coding: utf-8 -*-
"""
train_v8.py — Transformer + TCN 앙상블
  아키텍처:
    - TransformerPredictor: multi-head self-attention으로 11 시점 전역 패턴 포착
    - TCNPredictor: dilated 1D conv(dilation=1,2,4)로 멀티스케일 시간 패턴 포착
    - 두 모델 예측 평균 → calibration → 제출

  피처 추가 (기존 73 → 100차원):
    - FFT 스펙트럼 18차원: 각 축(x,y,z) rfft 크기 (6개 주파수 × 3축)
    - 다항식 피팅 9차원: 각 축 degree-2 계수 (3개 × 3축)

  Optuna + v3_aug 인사이트 반영:
    gamma=3.5, lF=0.6, lW=0.5, lr=0.0015, noise=1mm, main_clip=0.02
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
    print(f'GPU: {torch.cuda.get_device_name(0)}')

DATA_DIR  = str(pathlib.Path(__file__).parent / 'open')
CACHE_DIR = str(pathlib.Path(__file__).parent / 'cache')
OUT_DIR   = str(pathlib.Path(__file__).parent / 'submissions')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Kalman Filter ─────────────────────────────────────────────
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N, T, _ = X.shape
    F_ = np.array([[1,DT],[0,1]]); Fp = np.array([[1,T_PRED],[0,1]])
    Q  = sigma_proc**2 * np.array([[DT**4/4, DT**3/2],[DT**3/2, DT**2]])
    R  = sigma_obs**2; pred = np.zeros((N,3))
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

so_best, sp_best = 0.000267, 1.0
kal_tr_best = kalman_predict(X_train, so_best, sp_best)
kal_te_best = kalman_predict(X_test,  so_best, sp_best)
print(f'Kalman R-Hit = {r_hit(kal_tr_best, y_train):.4f}')

# ── 피처 빌더 ─────────────────────────────────────────────────
LOG_COLS = ['mean_speed','max_speed','speed_std','mean_acc','max_acc','max_jerk',
            'net_disp','|v_last|','|a_last|','|a_recent|','jerk_last','jerk_recent',
            'noise_poly2','noise_savgol','noise_loo']

def build_seq_curv(X):
    N = X.shape[0]
    rel = X - X[:,-1:]
    v   = np.diff(X, axis=1) / DT
    a   = np.diff(v, axis=1) / DT
    vp  = np.concatenate([np.zeros((N,1,3)), v], 1)
    ap  = np.concatenate([np.zeros((N,2,3)), a], 1)
    speed = np.linalg.norm(vp, axis=-1, keepdims=True)
    cross = np.cross(vp, ap)
    cross_mag = np.linalg.norm(cross, axis=-1, keepdims=True)
    spd_safe = speed + 1e-12
    curvature   = np.log1p(cross_mag / (spd_safe**3))
    angular_vel = np.log1p(cross_mag / (spd_safe**2))
    v_unit = vp / (spd_safe + 1e-12)
    cos_t  = (v_unit[:,:-1] * v_unit[:,1:]).sum(-1, keepdims=True).clip(-1,1)
    turn_r = np.concatenate([np.zeros((N,1,1)), np.arccos(cos_t)], 1)
    return np.concatenate([rel, vp, ap, speed, curvature, angular_vel, turn_r], -1).astype(np.float32)

def build_fft_features(X):
    """FFT magnitude spectrum — 각 축 rfft 크기 (N, 18)"""
    feats = []
    for j in range(3):
        spec = np.abs(np.fft.rfft(X[:,:,j]))  # (N, 6) for T=11
        feats.append(spec)
    return np.column_stack(feats).astype(np.float32)

def build_poly_features(X, deg=2):
    """다항식 피팅 계수 — 각 축 degree-2 (N, 9)
    계수: [quadratic, linear, const] 순 (np.polyfit 반환 순서)
    궤적의 가속도 추세, 속도 추세, 위치 오프셋을 명시적으로 인코딩
    """
    feats = []
    for j in range(3):
        coeffs = np.polyfit(t_obs, X[:,:,j].T, deg=deg)  # (deg+1, N)
        feats.append(coeffs.T)  # (N, deg+1)
    return np.column_stack(feats).astype(np.float32)

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
    for c in LOG_COLS:
        if c in df.columns: df[c]=np.log1p(df[c])
    cross = np.cross(v[:,:-1], a)
    cross_mag = np.linalg.norm(cross, axis=-1)
    sp_mid = np.linalg.norm(v[:,:-1], axis=-1)
    kappa = cross_mag / (sp_mid**3 + 1e-12)
    omega = cross_mag / (sp_mid**2 + 1e-12)
    v_unit2 = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)
    cos_t2  = (v_unit2[:,:-1]*v_unit2[:,1:]).sum(-1).clip(-1,1)
    turn_r2 = np.arccos(cos_t2)
    df['mean_curvature'] = np.log1p(kappa.mean(1))
    df['max_curvature']  = np.log1p(kappa.max(1))
    df['mean_angvel']    = np.log1p(omega.mean(1))
    df['max_angvel']     = np.log1p(omega.max(1))
    df['mean_turnrate']  = turn_r2.mean(1)
    df['max_turnrate']   = turn_r2.max(1)
    return df.values.astype(np.float32)

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

print('피처 빌딩...')
scal_tr  = build_scalar(X_train, noise_p, noise_s, noise_l)
scal_te  = build_scalar(X_test,  noise_p_te, noise_s_te)
t3_tr    = build_tier3(X_train); t3_te = build_tier3(X_test)
wav_tr   = build_wavelet(X_train); wav_te = build_wavelet(X_test)
fft_tr   = build_fft_features(X_train); fft_te = build_fft_features(X_test)
poly_tr  = build_poly_features(X_train); poly_te = build_poly_features(X_test)

scal_full_tr = np.concatenate([scal_tr, t3_tr, wav_tr, fft_tr, poly_tr], -1)
scal_full_te = np.concatenate([scal_te, t3_te, wav_te, fft_te, poly_te], -1)
seq_curv_tr  = build_seq_curv(X_train)
seq_curv_te  = build_seq_curv(X_test)

SEQ_DIM  = seq_curv_tr.shape[2]
SCAL_DIM = scal_full_tr.shape[1]
print(f'seq_dim={SEQ_DIM}  scal_dim={SCAL_DIM}')

theta_tr    = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te    = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)
target_main = rotate_xy(y_train - kal_tr_best, theta_tr).astype(np.float32)
aux_F       = rotate_xy(y_train - X_train[:,-1], theta_tr).astype(np.float32)
aux_W       = rotate_xy(y_train - kalman_predict(X_train, 1e-3), theta_tr).astype(np.float32)

# ── 손실 함수 ─────────────────────────────────────────────────
def loss_euclid(p, t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p, t, b=0.002):
    return torch.sigmoid((torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p, t, gamma=3.5):
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss = torch.sigmoid((err - 0.01) / 0.002)
    w = p_miss.pow(gamma).detach()
    w = w / (w.mean() + 1e-8)
    return (w * err).mean()

def loss_combo_focal(p, t, gamma=3.5):
    return loss_focal_euclid(p, t, gamma) + 0.3*loss_softhit(p, t)

# ── 모델 ─────────────────────────────────────────────────────
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class TransformerPredictor(nn.Module):
    """Self-attention으로 11 시점을 동시에 참조 — GRU의 순차 처리 한계 극복"""
    def __init__(self, seq_dim=SEQ_DIM, scal_dim=SCAL_DIM,
                 d_model=64, nhead=4, num_layers=2,
                 dropout=0.25, main_clip=0.02):
        super().__init__()
        self.proj = nn.Linear(seq_dim, d_model)
        self.pos  = nn.Parameter(torch.randn(1, 11, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        act = Mish()
        self.net = nn.Sequential(
            nn.Linear(d_model + scal_dim, 128), act,
            nn.Dropout(dropout),
            nn.Linear(128, 64), act,
        )
        self.head_main = nn.Linear(64, 3)
        self.head_F    = nn.Linear(64, 3)
        self.head_W    = nn.Linear(64, 3)
        self.clip = main_clip

    def forward(self, seq, scal):
        x = self.proj(seq) + self.pos      # (N, 11, d_model)
        x = self.encoder(x).mean(dim=1)    # (N, d_model)
        z = self.net(torch.cat([x, scal], 1))
        return (torch.tanh(self.head_main(z)) * self.clip,
                self.head_F(z), self.head_W(z))

class TCNBlock(nn.Module):
    """Dilated 1D Conv 블록 — residual connection + BN"""
    def __init__(self, in_ch, out_ch, dilation=1, dropout=0.25):
        super().__init__()
        # padding=dilation: kernel_size=3이면 same-length 출력 보장
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = Mish()
        self.drop  = nn.Dropout(dropout)
        self.res   = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):  # x: (N, C, T)
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.drop(self.act(self.bn2(self.conv2(h))))
        return self.act(h + self.res(x))

class TCNPredictor(nn.Module):
    """3단 dilated TCN (dilation=1,2,4) — 수용 필드 29 > 시퀀스 길이 11"""
    def __init__(self, seq_dim=SEQ_DIM, scal_dim=SCAL_DIM,
                 channels=(64, 128, 128), dropout=0.25, main_clip=0.02):
        super().__init__()
        layers, in_ch = [], seq_dim
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        act = Mish()
        self.net = nn.Sequential(
            nn.Linear(in_ch + scal_dim, 128), act,
            nn.Dropout(dropout),
            nn.Linear(128, 64), act,
        )
        self.head_main = nn.Linear(64, 3)
        self.head_F    = nn.Linear(64, 3)
        self.head_W    = nn.Linear(64, 3)
        self.clip = main_clip

    def forward(self, seq, scal):
        x = self.tcn(seq.transpose(1, 2)).mean(dim=-1)  # (N, channels[-1])
        z = self.net(torch.cat([x, scal], 1))
        return (torch.tanh(self.head_main(z)) * self.clip,
                self.head_F(z), self.head_W(z))

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

# ── 학습 설정 ─────────────────────────────────────────────────
# Optuna v6 인사이트: gamma=3.5, lF=0.6, lW=0.5, lr≈0.0015
# v3_aug 인사이트: noise=1mm가 5-Fold 실전에서 0.2mm보다 우수
CFG = dict(lr=0.0015, wd=0.0005, batch=128,
           lF=0.6, lW=0.5, gamma=3.5,
           noise=0.001, clip=0.02, dropout=0.25)
N_FOLDS, N_SEEDS, FULL_EP, FULL_ESTOP = 5, 5, 400, 60

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
# Transformer / TCN 각각 따로 집계 후 앙상블
oof_rot_tf = np.zeros((len(X_train), 3))
oof_rot_tc = np.zeros((len(X_train), 3))
test_folds_tf, test_folds_tc = [], []
fold_rh_tf, fold_rh_tc, fold_rh_en = [], [], []
t0_all = time.time()

def train_one_model(model_cls, s, st_base, sc_, sv, scv, ste, sce,
                    tm, tf_, tw_, n, CFG):
    torch.manual_seed(s * 100); np.random.seed(s * 100)
    m = model_cls().to(device)
    opt   = torch.optim.AdamW(m.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EP)
    lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']
    gamma, noise_std = CFG['gamma'], CFG['noise']

    best_rh, bst, no_improve = -1, None, 0
    for ep in range(FULL_EP):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            noise = torch.randn_like(st_base[idx]) * noise_std
            om, oF, oW = m(st_base[idx] + noise, sc_[idx])
            loss = (loss_combo_focal(om, tm[idx], gamma)
                    + lF * loss_euclid(oF, tf_[idx])
                    + lW * loss_euclid(oW, tw_[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
        sched.step()

        m.eval()
        with torch.no_grad(): ov, _, _ = m(sv, scv)
        rh = r_hit(ov.cpu().numpy(), None)  # 임시 — 아래서 실제 좌표 변환 후 계산
        # 실제 R-Hit은 fold 루프에서 theta를 알아야 하므로 그냥 raw ov 반환
        # 여기서는 early stop을 위해 체크해야 하므로 임시로 L2 norm 기반 기준 사용
        # → 아래 wrapper에서 처리
        return m, ov  # 마지막 에포크 반환 (early stop 없이 최적 추적은 wrapper에서)

    return m, None  # 미도달 시 (실제로는 아래 wrapper 사용)

def train_model_with_estop(model_cls, s, st_base, sc_, sv, scv, ste, sce,
                            tm, tf_, tw_, n, CFG, vai, theta_tr, kal_tr_best, y_train):
    torch.manual_seed(s * 100); np.random.seed(s * 100)
    m = model_cls().to(device)
    opt   = torch.optim.AdamW(m.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EP)
    lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']
    gamma, noise_std = CFG['gamma'], CFG['noise']

    best_rh, bst, no_improve = -1, None, 0
    for ep in range(FULL_EP):
        m.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            noise = torch.randn_like(st_base[idx]) * noise_std
            om, oF, oW = m(st_base[idx] + noise, sc_[idx])
            loss = (loss_combo_focal(om, tm[idx], gamma)
                    + lF * loss_euclid(oF, tf_[idx])
                    + lW * loss_euclid(oW, tw_[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
            opt.step()
        sched.step()

        m.eval()
        with torch.no_grad(): ov, _, _ = m(sv, scv)
        pred = kal_tr_best[vai] + inv_rot(ov.cpu().numpy(), theta_tr[vai])
        rh = r_hit(pred, y_train[vai])
        if rh > best_rh:
            best_rh = rh
            bst = {k: v.detach().clone() for k, v in m.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= FULL_ESTOP:
            break

    m.load_state_dict(bst)
    m.eval()
    with torch.no_grad():
        ov2, _, _ = m(sv, scv)
        te_, _, _ = m(ste, sce)
    return best_rh, ep+1, ov2.cpu().numpy(), te_.cpu().numpy()

for fi, (tri, vai) in enumerate(kf.split(np.arange(len(X_train)))):
    print(f'\n[Fold {fi+1}/{N_FOLDS}]')

    sc_seq  = WhiteningScaler().fit(seq_curv_tr[tri].reshape(-1, SEQ_DIM))
    sc_scal = StandardScaler().fit(scal_full_tr[tri])

    seq_tr_n  = norm_seq(seq_curv_tr[tri], sc_seq)
    seq_va_n  = norm_seq(seq_curv_tr[vai], sc_seq)
    seq_te_n  = norm_seq(seq_curv_te, sc_seq)
    scal_tr_n = sc_scal.transform(scal_full_tr[tri]).astype(np.float32)
    scal_va_n = sc_scal.transform(scal_full_tr[vai]).astype(np.float32)
    scal_te_n = sc_scal.transform(scal_full_te).astype(np.float32)

    def tt(a): return torch.from_numpy(np.asarray(a, np.float32)).to(device)

    st_base = tt(seq_tr_n); sc_     = tt(scal_tr_n)
    sv      = tt(seq_va_n); scv     = tt(scal_va_n)
    ste     = tt(seq_te_n); sce     = tt(scal_te_n)
    tm      = tt(target_main[tri])
    tf_     = tt(aux_F[tri])
    tw_     = tt(aux_W[tri])
    n       = st_base.shape[0]

    seed_oof_tf, seed_te_tf = [], []
    seed_oof_tc, seed_te_tc = [], []

    for s in range(N_SEEDS):
        # Transformer 학습
        best_rh_tf, ep_tf, ov_tf, te_tf = train_model_with_estop(
            TransformerPredictor, s,
            st_base, sc_, sv, scv, ste, sce,
            tm, tf_, tw_, n, CFG, vai, theta_tr, kal_tr_best, y_train)
        seed_oof_tf.append(ov_tf); seed_te_tf.append(te_tf)
        print(f'  TF  seed {s}  best_rh={best_rh_tf:.4f}  ep={ep_tf}')
        torch.cuda.empty_cache(); gc.collect()

        # TCN 학습
        best_rh_tc, ep_tc, ov_tc, te_tc = train_model_with_estop(
            TCNPredictor, s + 50,
            st_base, sc_, sv, scv, ste, sce,
            tm, tf_, tw_, n, CFG, vai, theta_tr, kal_tr_best, y_train)
        seed_oof_tc.append(ov_tc); seed_te_tc.append(te_tc)
        print(f'  TCN seed {s}  best_rh={best_rh_tc:.4f}  ep={ep_tc}')
        torch.cuda.empty_cache(); gc.collect()

    vr_tf = np.mean(seed_oof_tf, 0)
    vr_tc = np.mean(seed_oof_tc, 0)
    vr_en = (vr_tf + vr_tc) / 2

    oof_rot_tf[vai] = vr_tf
    oof_rot_tc[vai] = vr_tc
    test_folds_tf.append(np.mean(seed_te_tf, 0))
    test_folds_tc.append(np.mean(seed_te_tc, 0))

    pv_tf = kal_tr_best[vai] + inv_rot(vr_tf, theta_tr[vai])
    pv_tc = kal_tr_best[vai] + inv_rot(vr_tc, theta_tr[vai])
    pv_en = kal_tr_best[vai] + inv_rot(vr_en, theta_tr[vai])
    rh_tf = r_hit(pv_tf, y_train[vai])
    rh_tc = r_hit(pv_tc, y_train[vai])
    rh_en = r_hit(pv_en, y_train[vai])
    fold_rh_tf.append(rh_tf); fold_rh_tc.append(rh_tc); fold_rh_en.append(rh_en)
    print(f'  Fold {fi+1}  TF={rh_tf:.4f}  TCN={rh_tc:.4f}  Ensemble={rh_en:.4f}')

# ── 최종 평가 ─────────────────────────────────────────────────
def calibrate(oof_rot, kal_tr_best, theta_tr, y_train, step=0.025):
    best_cal, best_a = -1, np.ones(3)
    for ax in np.arange(0.85, 1.11, step):
        for ay in np.arange(0.85, 1.06, step):
            for az in np.arange(0.85, 1.11, step):
                a = np.array([ax, ay, az])
                rc = r_hit(kal_tr_best + inv_rot(oof_rot * a, theta_tr), y_train)
                if rc > best_cal: best_cal, best_a = rc, a.copy()
    return best_cal, best_a

# 각 모델 OOF
oof_pred_tf = kal_tr_best + inv_rot(oof_rot_tf, theta_tr)
oof_pred_tc = kal_tr_best + inv_rot(oof_rot_tc, theta_tr)
oof_rot_en  = (oof_rot_tf + oof_rot_tc) / 2
oof_pred_en = kal_tr_best + inv_rot(oof_rot_en, theta_tr)

cal_tf, a_tf = calibrate(oof_rot_tf, kal_tr_best, theta_tr, y_train)
cal_tc, a_tc = calibrate(oof_rot_tc, kal_tr_best, theta_tr, y_train)
cal_en, a_en = calibrate(oof_rot_en, kal_tr_best, theta_tr, y_train)

print(f'\n{"="*60}')
print(f'[v8 결과]')
print(f'  Transformer  OOF={r_hit(oof_pred_tf, y_train):.4f}  Cal={cal_tf:.4f}  alpha=({a_tf[0]:.3f},{a_tf[1]:.3f},{a_tf[2]:.3f})')
print(f'  TCN          OOF={r_hit(oof_pred_tc, y_train):.4f}  Cal={cal_tc:.4f}  alpha=({a_tc[0]:.3f},{a_tc[1]:.3f},{a_tc[2]:.3f})')
print(f'  Ensemble     OOF={r_hit(oof_pred_en, y_train):.4f}  Cal={cal_en:.4f}  alpha=({a_en[0]:.3f},{a_en[1]:.3f},{a_en[2]:.3f})')
print(f'  학습시간: {(time.time()-t0_all)/60:.1f}min')
print(f'  기준: v3_aug OOF=0.6669  Cal=0.6676')

# ── 제출 파일 저장 ─────────────────────────────────────────────
test_rot_tf = np.mean(test_folds_tf, 0) * a_tf
test_rot_tc = np.mean(test_folds_tc, 0) * a_tc
test_rot_en = ((np.mean(test_folds_tf, 0) + np.mean(test_folds_tc, 0)) / 2) * a_en

def save_sub(test_pred, name):
    pd.DataFrame({'id': test_ids,
                  'x': test_pred[:,0], 'y': test_pred[:,1], 'z': test_pred[:,2]}
    ).to_csv(os.path.join(OUT_DIR, name), index=False)
    print(f'  저장: {name}')

oof_en_rh = r_hit(oof_pred_en, y_train)
save_sub(kal_te_best + inv_rot(test_rot_tf, theta_te),
         f'sub_v8_tf_OOF{r_hit(oof_pred_tf,y_train):.4f}_Cal{cal_tf:.4f}.csv')
save_sub(kal_te_best + inv_rot(test_rot_tc, theta_te),
         f'sub_v8_tcn_OOF{r_hit(oof_pred_tc,y_train):.4f}_Cal{cal_tc:.4f}.csv')
save_sub(kal_te_best + inv_rot(test_rot_en, theta_te),
         f'sub_v8_ens_OOF{oof_en_rh:.4f}_Cal{cal_en:.4f}.csv')
