# -*- coding: utf-8 -*-
"""
train_v3_pseudo.py — Semi-supervised pseudo-labeling on top of v3_D best config

아이디어:
  - v3_D 테스트 예측을 pseudo-label로 사용
  - Kalman과 GRU가 가깝게 동의하는 신뢰도 높은 샘플만 학습 추가
  - 테스트 분포를 학습에 포함 → OOF보다 LB 개선 기대

기대 효과:
  - LB 0.6780 → 목표 0.6900+
  - train set 10,000 + pseudo 약 6,000 → 16,000 샘플
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

# ── 데이터 로드 ──────────────────────────────────────────────
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

# ── Pseudo-label 로드 ─────────────────────────────────────────
pseudo_candidates = sorted(glob.glob(os.path.join(OUT_DIR, '*v3_D*+both*.csv')))
if not pseudo_candidates:
    pseudo_candidates = sorted(glob.glob(os.path.join(OUT_DIR, '*v3_D*.csv')))
assert pseudo_candidates, 'v3_D 제출 파일이 없습니다. train_v3.py를 먼저 실행하세요.'
pseudo_path = pseudo_candidates[-1]
print(f'Pseudo-label source: {pseudo_path}')
pseudo_df    = pd.read_csv(pseudo_path)
pseudo_df    = pseudo_df.set_index('id').loc[test_ids].reset_index()
y_pseudo_all = pseudo_df[['x','y','z']].values.astype(np.float64)

# ── 칼만 필터 ─────────────────────────────────────────────────
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

so_best, sp_best = 0.000267, 1.0
kalman_train = kalman_predict(X_train)
kalman_test  = kalman_predict(X_test)
kal_tr_best  = kalman_predict(X_train, so_best, sp_best)
kal_te_best  = kalman_predict(X_test,  so_best, sp_best)
print(f'Kalman train R-Hit = {r_hit(kalman_train, y_train):.4f}')

# ── Pseudo-label 신뢰도 필터링 ───────────────────────────────
pseudo_kalman_dist = np.linalg.norm(y_pseudo_all - kal_te_best, axis=1)
CONF_THR = 0.005  # 5mm 이내면 신뢰도 높음
confident_mask = pseudo_kalman_dist <= CONF_THR
n_conf = confident_mask.sum()
print(f'\n[Pseudo-label 필터링]')
print(f'  전체 테스트: {len(y_pseudo_all):,}')
print(f'  신뢰도 높음 (dist≤{CONF_THR*100:.0f}mm): {n_conf:,}  ({n_conf/len(y_pseudo_all)*100:.1f}%)')
print(f'  신뢰도 낮음 (제외):  {(~confident_mask).sum():,}')

X_pseudo     = X_test[confident_mask]
y_pseudo     = y_pseudo_all[confident_mask]
kal_pseudo   = kal_te_best[confident_mask]
print(f'  최종 pseudo 샘플 수: {len(X_pseudo):,}')

# ── 피처 빌딩 ─────────────────────────────────────────────────
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
    spd_safe  = speed + 1e-12
    curvature   = np.log1p(cross_mag / (spd_safe**3))
    angular_vel = np.log1p(cross_mag / (spd_safe**2))
    v_unit = vp / (spd_safe + 1e-12)
    cos_t  = (v_unit[:,:-1] * v_unit[:,1:]).sum(-1, keepdims=True).clip(-1,1)
    turn_r = np.concatenate([np.zeros((N,1,1)), np.arccos(cos_t)], 1)
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
    print(f'saved: {CACHE_FILE}')

# noise for pseudo samples
noise_p_ps = noise_poly2(X_pseudo)
noise_s_ps = noise_savgol(X_pseudo)

# build all features
print('피처 빌딩...')
seq_tr   = build_seq_curv(X_train)
seq_te   = build_seq_curv(X_test)
seq_ps   = build_seq_curv(X_pseudo)

scal_tr  = build_scalar(X_train, noise_p, noise_s, noise_l)
scal_te  = build_scalar(X_test,  noise_p_te, noise_s_te)
scal_ps  = build_scalar(X_pseudo, noise_p_ps, noise_s_ps)

t3_tr = build_tier3(X_train); t3_te = build_tier3(X_test); t3_ps = build_tier3(X_pseudo)
wav_tr= build_wavelet(X_train); wav_te= build_wavelet(X_test); wav_ps= build_wavelet(X_pseudo)

scal_full_tr = np.concatenate([scal_tr, t3_tr, wav_tr], -1)
scal_full_te = np.concatenate([scal_te, t3_te, wav_te], -1)
scal_full_ps = np.concatenate([scal_ps, t3_ps, wav_ps], -1)

SEQ_DIM  = seq_tr.shape[2]
SCAL_DIM = scal_full_tr.shape[1]
print(f'seq_dim={SEQ_DIM}  scal_dim={SCAL_DIM}')

# ── 타깃 ─────────────────────────────────────────────────────
theta_tr = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)
theta_ps = yaw_angle((X_pseudo[:,-1]-X_pseudo[:,-2])/DT)

target_main_tr = rotate_xy(y_train  - kal_tr_best,  theta_tr).astype(np.float32)
target_main_ps = rotate_xy(y_pseudo - kal_pseudo,   theta_ps).astype(np.float32)
aux_F_tr = rotate_xy(y_train  - X_train[:,-1],  theta_tr).astype(np.float32)
aux_F_ps = rotate_xy(y_pseudo - X_pseudo[:,-1], theta_ps).astype(np.float32)
aux_W_tr = rotate_xy(y_train  - kalman_predict(X_train, 1e-3),  theta_tr).astype(np.float32)
aux_W_ps = rotate_xy(y_pseudo - kalman_predict(X_pseudo, 1e-3), theta_ps).astype(np.float32)

print(f'target_main_tr std (cm): {target_main_tr.std(0)*100}')
print(f'target_main_ps std (cm): {target_main_ps.std(0)*100}')

# ── 손실 함수 ─────────────────────────────────────────────────
def loss_euclid(p, t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p, t, b=0.002):
    return torch.sigmoid((torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p, t, gamma=2.0):
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss = torch.sigmoid((err - 0.01) / 0.002)
    w = p_miss.pow(gamma).detach()
    w = w / (w.mean() + 1e-8)
    return (w * err).mean()

def loss_combo_focal(p, t, gamma=2.0):
    return loss_focal_euclid(p, t, gamma) + 0.3*loss_softhit(p, t)

# ── 모델 ─────────────────────────────────────────────────────
class Mish(nn.Module):
    def forward(self, x): return x * torch.tanh(F.softplus(x))

class GRUFlex(nn.Module):
    def __init__(self, seq_dim=SEQ_DIM, scal_dim=SCAL_DIM,
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

# ── 학습 ─────────────────────────────────────────────────────
CFG = dict(lr=0.000989, wd=0.000886, batch=256, lF=0.30, lW=0.30)
N_FOLDS, N_SEEDS, FULL_EP, FULL_ESTOP = 5, 3, 350, 50
PSEUDO_WEIGHT = 0.5   # pseudo sample 손실 가중치

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
oof_rot    = np.zeros((len(X_train), 3))
test_folds = []
fold_rh    = []
t0_all     = time.time()

for fi, (tri, vai) in enumerate(kf.split(np.arange(len(X_train)))):
    print(f'\n[Fold {fi+1}/{N_FOLDS}]')

    # ── 정규화 스케일러 (실제 train fold에만 fit) ─────────────
    sc_seq  = WhiteningScaler().fit(seq_tr[tri].reshape(-1, SEQ_DIM))
    sc_scal = StandardScaler().fit(scal_full_tr[tri])

    # ── 실제 학습 데이터 ─────────────────────────────────────
    seq_tr_n  = norm_seq(seq_tr[tri], sc_seq)
    seq_va_n  = norm_seq(seq_tr[vai], sc_seq)
    scal_tr_n = sc_scal.transform(scal_full_tr[tri]).astype(np.float32)
    scal_va_n = sc_scal.transform(scal_full_tr[vai]).astype(np.float32)

    # ── Pseudo 학습 데이터 ────────────────────────────────────
    seq_ps_n  = norm_seq(seq_ps, sc_seq)
    scal_ps_n = sc_scal.transform(scal_full_ps).astype(np.float32)

    # ── 결합: [real_train | pseudo_test] ─────────────────────
    seq_comb  = np.concatenate([seq_tr_n,  seq_ps_n],  axis=0)
    scal_comb = np.concatenate([scal_tr_n, scal_ps_n], axis=0)
    tgt_main  = np.concatenate([target_main_tr[tri], target_main_ps], axis=0)
    tgt_F     = np.concatenate([aux_F_tr[tri],       aux_F_ps],       axis=0)
    tgt_W     = np.concatenate([aux_W_tr[tri],       aux_W_ps],       axis=0)

    # 손실 가중치: real=1.0, pseudo=PSEUDO_WEIGHT
    n_real = len(tri)
    weights = np.concatenate([np.ones(n_real, np.float32),
                               np.full(len(X_pseudo), PSEUDO_WEIGHT, np.float32)])

    # ── 테스트 피처 ──────────────────────────────────────────
    seq_te_n  = norm_seq(seq_te, sc_seq)
    scal_te_n = sc_scal.transform(scal_full_te).astype(np.float32)

    def tt(a): return torch.from_numpy(np.asarray(a, np.float32)).to(device)

    st  = tt(seq_comb); sc_ = tt(scal_comb)
    sv  = tt(seq_va_n); scv = tt(scal_va_n)
    ste = tt(seq_te_n); sce = tt(scal_te_n)
    tm  = tt(tgt_main); tf  = tt(tgt_F); tw  = tt(tgt_W)
    wt  = tt(weights)

    seed_oof, seed_te = [], []
    for s in range(N_SEEDS):
        torch.manual_seed(s); np.random.seed(s)
        m = GRUFlex().to(device)
        opt   = torch.optim.AdamW(m.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EP)
        lF, lW, bs = CFG['lF'], CFG['lW'], CFG['batch']

        best_rh, bst, no_improve = -1, None, 0
        n_total = st.shape[0]

        for ep in range(FULL_EP):
            m.train()
            perm = torch.randperm(n_total, device=device)
            for i in range(0, n_total, bs):
                idx = perm[i:i+bs]; opt.zero_grad()
                om, oF, oW = m(st[idx], sc_[idx])
                w_b = wt[idx]
                loss_m = (w_b * torch.sqrt(((om - tm[idx])**2).sum(-1)+1e-12)).mean()
                # focal weight on real samples only
                real_mask = idx < n_real
                if real_mask.any():
                    om_r = om[real_mask]; tm_r = tm[idx[real_mask]]
                    err_r = torch.sqrt(((om_r - tm_r)**2).sum(-1)+1e-12)
                    pmiss = torch.sigmoid((err_r - 0.01) / 0.002)
                    fw = pmiss.pow(2.0).detach(); fw = fw/(fw.mean()+1e-8)
                    focal_part = (fw * err_r).mean()
                    loss_m = 0.7*loss_m + 0.3*focal_part
                loss = (loss_m
                        + lF * loss_euclid(oF, tf[idx])
                        + lW * loss_euclid(oW, tw[idx]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                opt.step()
            sched.step()

            m.eval()
            with torch.no_grad(): ov,_,_ = m(sv, scv)
            pred = kal_tr_best[vai] + inv_rot(ov.cpu().numpy(), theta_tr[vai])
            rh = r_hit(pred, y_train[vai])
            if rh > best_rh:
                best_rh = rh
                bst = {k:v.detach().clone() for k,v in m.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= FULL_ESTOP: break

        m.load_state_dict(bst)
        m.eval()
        with torch.no_grad():
            ov2,_,_ = m(sv, scv)
            te_,_,_ = m(ste, sce)
        seed_oof.append(ov2.cpu().numpy())
        seed_te.append(te_.cpu().numpy())
        print(f'  seed {s}  best_rh={best_rh:.4f}  ep={ep+1}')
        torch.cuda.empty_cache(); gc.collect()

    vr = np.mean(seed_oof, 0); tr_ = np.mean(seed_te, 0)
    oof_rot[vai] = vr; test_folds.append(tr_)
    pv  = kal_tr_best[vai] + inv_rot(vr, theta_tr[vai])
    rh  = r_hit(pv, y_train[vai])
    fold_rh.append(rh)
    print(f'  Fold {fi+1} OOF R-Hit: {rh:.4f}')

# ── 최종 평가 + 저장 ─────────────────────────────────────────
oof_pred = kal_tr_best + inv_rot(oof_rot, theta_tr)
oof_rh   = r_hit(oof_pred, y_train)

best_cal, best_a = -1, np.ones(3)
for ax in np.arange(0.85, 1.11, 0.05):
    for ay in np.arange(0.85, 1.06, 0.05):
        for az in np.arange(0.85, 1.11, 0.05):
            a = np.array([ax, ay, az])
            rc = r_hit(kal_tr_best + inv_rot(oof_rot*a, theta_tr), y_train)
            if rc > best_cal: best_cal, best_a = rc, a.copy()

print(f'\n{"="*50}')
print(f'[Pseudo-label 결과]')
print(f'  Pseudo 샘플 수:    {len(X_pseudo):,} ({n_conf/len(y_pseudo_all)*100:.0f}% of test)')
print(f'  OOF R-Hit:         {oof_rh:.4f}')
print(f'  Cal OOF R-Hit:     {best_cal:.4f}')
print(f'  Cal alpha:         ({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')
print(f'  v3_D best 참고:    0.6659')
print(f'  총 학습시간:       {(time.time()-t0_all)/60:.1f}min')

test_rot  = np.mean(test_folds, 0) * best_a
test_pred = kal_te_best + inv_rot(test_rot, theta_te)
fname = f'sub_v3_pseudo_conf{CONF_THR*1000:.0f}mm_OOF{oof_rh:.4f}.csv'
pd.DataFrame({'id': test_ids,
              'x': test_pred[:,0], 'y': test_pred[:,1], 'z': test_pred[:,2]}
).to_csv(os.path.join(OUT_DIR, fname), index=False)
print(f'  저장: {fname}')
