# -*- coding: utf-8 -*-
"""
train_v5.py — Transformer + Y-flip TTA
  ① 새 아키텍처: TransformerEncoder (CLS token + learned pos embedding)
  ② Y-flip TTA: 테스트 시 y축 대칭 궤적 추가 예측 후 평균
  ③ Optuna: model_type ('gru' / 'transformer') 포함 탐색
  ④ 고정: curvature features + focal loss (v3 ablation에서 시너지 확인)

TTA 원리:
  모기 비행의 y축 대칭성 이용 (물리적으로 동등)
  seq_flip 제작: rel_y, vel_y, acc_y 부호 반전 (speed/curvature 등 크기량은 불변)
  pred_flip 후 y 역부호 → 원래 좌표계로 복원 → 평균
"""
import os, gc, time, random, warnings, pathlib
import numpy as np, pandas as pd, glob
from tqdm.auto import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import pywt, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
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
DB_PATH   = os.path.join(CACHE_DIR, 'optuna_v5.db')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

# ── 데이터 ─────────────────────────────────────────────────────
train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.csv')))
test_files  = sorted(glob.glob(os.path.join(DATA_DIR, 'test',  '*.csv')))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))

def load_stack(files, desc):
    return np.stack([pd.read_csv(f)[['x','y','z']].values
                     for f in tqdm(files, desc=desc)]).astype(np.float64)

X_train = load_stack(train_files, 'train')
X_test  = load_stack(test_files,  'test')
train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
test_ids  = [os.path.splitext(os.path.basename(f))[0] for f in test_files]
y_train   = labels.set_index('id').loc[train_ids][['x','y','z']].values.astype(np.float64)
print(f'X_train {X_train.shape}  y_train {y_train.shape}')

# ── 유틸 ───────────────────────────────────────────────────────
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N,T,_=X.shape
    F=np.array([[1,DT],[0,1]]); Fp=np.array([[1,T_PRED],[0,1]])
    Q=sigma_proc**2*np.array([[DT**4/4,DT**3/2],[DT**3/2,DT**2]])
    R=sigma_obs**2; pred=np.zeros((N,3))
    for j in range(3):
        z=X[:,:,j]; s=np.zeros((N,2)); s[:,0]=z[:,0]; P=np.eye(2)
        for t in range(1,T):
            s=s@F.T; P=F@P@F.T+Q
            inn=z[:,t]-s[:,0]; S=P[0,0]+R; K=P[:,0]/S
            s=s+np.outer(inn,K); P=P-np.outer(K,P[0,:])
        pred[:,j]=(s@Fp.T)[:,0]
    return pred

def r_hit(pred, true, thr=0.01):
    return float((np.linalg.norm(pred-true,axis=-1)<=thr).mean())

def cos_safe(a,b):
    return np.clip((a*b).sum(-1)/np.maximum(
        np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12),-1,1)

def yaw_angle(v): return np.arctan2(v[:,1],v[:,0])

def rotate_xy(vec,th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([vec[:,0]*c+vec[:,1]*s,-vec[:,0]*s+vec[:,1]*c,vec[:,2]],-1)

def inv_rot(vec,th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([vec[:,0]*c-vec[:,1]*s,vec[:,0]*s+vec[:,1]*c,vec[:,2]],-1)

kalman_train = kalman_predict(X_train)
kalman_test  = kalman_predict(X_test)
print(f'Kalman R-Hit = {r_hit(kalman_train,y_train):.4f}')

# ════════════════════════════════════════════════════════════════
# Features (v4 동일: curvature 고정)
# ════════════════════════════════════════════════════════════════
def build_seq(X):
    """(N,11,13): rel(3)+vel(3)+acc(3)+speed(1)+curvature(1)+angvel(1)+turnrate(1)"""
    N=X.shape[0]
    rel=X-X[:,-1:]
    v=np.diff(X,axis=1)/DT; a=np.diff(v,axis=1)/DT
    vp=np.concatenate([np.zeros((N,1,3)),v],1)
    ap=np.concatenate([np.zeros((N,2,3)),a],1)
    spd=np.linalg.norm(vp,axis=-1,keepdims=True)
    cross=np.cross(vp,ap); cm=np.linalg.norm(cross,axis=-1,keepdims=True)
    sp_=spd+1e-12
    curv=np.log1p(cm/(sp_**3)); angv=np.log1p(cm/(sp_**2))
    vu=vp/(sp_+1e-12)
    ct=(vu[:,:-1]*vu[:,1:]).sum(-1,keepdims=True).clip(-1,1)
    tr=np.concatenate([np.zeros((N,1,1)),np.arccos(ct)],1)
    return np.concatenate([rel,vp,ap,spd,curv,angv,tr],-1).astype(np.float32)

def flip_seq(seq):
    """Y-flip: rel_y(1), vel_y(4), acc_y(7) 부호 반전. 크기량은 불변."""
    s=seq.copy()
    s[:,:,1]*=-1   # rel_y
    s[:,:,4]*=-1   # vel_y
    s[:,:,7]*=-1   # acc_y
    # idx 9=speed, 10=curvature, 11=angvel, 12=turnrate → 불변
    return s

LOG_COLS=['mean_speed','max_speed','speed_std','mean_acc','max_acc','max_jerk',
          'net_disp','|v_last|','|a_last|','|a_recent|','jerk_last','jerk_recent',
          'noise_poly2','noise_savgol','noise_loo']

def build_scalar(X,np_a,ns_a,nl_a=None):
    d=np.diff(X,axis=1); v=d/DT; a=np.diff(v,axis=1)/DT; jk=np.diff(a,axis=1)/DT
    sp=np.linalg.norm(v,axis=-1); ac=np.linalg.norm(a,axis=-1); jm=np.linalg.norm(jk,axis=-1)
    vl=v[:,-1]; al=a[:,-1]; ar=a[:,-3:].mean(1)
    nd=np.linalg.norm(X[:,-1]-X[:,0],axis=-1); pl=np.linalg.norm(d,axis=-1).sum(1)
    st=np.where(pl>1e-12,nd/np.maximum(pl,1e-12),0.)
    tc=cos_safe(vl,v[:,:-1].mean(1))
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
    vu=v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-12)
    tr=np.arccos((vu[:,:-1]*vu[:,1:]).sum(-1).clip(-1,1))
    df['mean_curvature']=np.log1p(kappa.mean(1)); df['max_curvature']=np.log1p(kappa.max(1))
    df['mean_angvel']=np.log1p(omega.mean(1));    df['max_angvel']=np.log1p(omega.max(1))
    df['mean_turnrate']=tr.mean(1);               df['max_turnrate']=tr.max(1)
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

CACHE_FILE=os.path.join(CACHE_DIR,'noise_cache.npz')
if os.path.exists(CACHE_FILE):
    nc=np.load(CACHE_FILE)
    noise_p,noise_s,noise_l=nc['np_'],nc['ns_'],nc['nl_']
    noise_p_te,noise_s_te=nc['np_te'],nc['ns_te']
    print('noise cache loaded')
else:
    noise_p=noise_poly2(X_train); noise_s=noise_savgol(X_train)
    noise_l=noise_loo_spline(X_train)
    noise_p_te=noise_poly2(X_test); noise_s_te=noise_savgol(X_test)
    np.savez(CACHE_FILE,np_=noise_p,ns_=noise_s,nl_=noise_l,
             np_te=noise_p_te,ns_te=noise_s_te)

seq_tr   = build_seq(X_train);    seq_te   = build_seq(X_test)
seq_tr_f = flip_seq(seq_tr);      seq_te_f = flip_seq(seq_te)
sb_tr=build_scalar(X_train,noise_p,noise_s,noise_l)
sb_te=build_scalar(X_test,noise_p_te,noise_s_te)
t3_tr=build_tier3(X_train); t3_te=build_tier3(X_test)
wav_tr=build_wavelet(X_train); wav_te=build_wavelet(X_test)
SEQ_DIM=seq_tr.shape[2]  # 13

# Y-flip Kalman & theta
theta_tr  = yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te  = yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)
theta_tr_f = -theta_tr   # y-flip reverses yaw
theta_te_f = -theta_te

kal_tr_f = kalman_train.copy(); kal_tr_f[:,1] *= -1  # y-flip Kalman
kal_te_f = kalman_test.copy();  kal_te_f[:,1] *= -1

# 스칼라 feature는 크기량 기반 → y-flip에 불변, 그대로 사용

print(f'seq {seq_tr.shape} | scal_base {sb_tr.shape}')

# ── target ─────────────────────────────────────────────────────
target_T8 = rotate_xy(y_train-kalman_train,theta_tr)
aux_F = rotate_xy(y_train-X_train[:,-1],theta_tr).astype(np.float32)
aux_W = rotate_xy(y_train-kalman_predict(X_train,1e-3),theta_tr).astype(np.float32)
print(f'T8 std (cm): {target_T8.std(0)*100}')

# ── Y-flip target (검증용) ────────────────────────────────────
y_flip = y_train.copy(); y_flip[:,1] *= -1
target_T8_f = rotate_xy(y_flip-kal_tr_f, theta_tr_f)

# ════════════════════════════════════════════════════════════════
# Loss (focal 고정)
# ════════════════════════════════════════════════════════════════
def loss_euclid(p,t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p,t,b=0.002):
    return torch.sigmoid(
        (torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p,t,gamma=1.0):
    err=torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss=torch.sigmoid((err-0.01)/0.002)
    w=p_miss.pow(gamma).detach()
    w=w/(w.mean()+1e-8)
    return (w*err).mean()

def make_loss(gamma):
    def fn(p,t): return loss_focal_euclid(p,t,gamma)+0.3*loss_softhit(p,t)
    return fn

# ════════════════════════════════════════════════════════════════
# Models
# ════════════════════════════════════════════════════════════════
class Mish(nn.Module):
    def forward(self,x): return x*torch.tanh(F.softplus(x))

def get_act(name):
    return {'gelu':nn.GELU(),'swish':nn.SiLU(),'mish':Mish()}[name]

class GRUModel(nn.Module):
    def __init__(self,seq_dim,scal_dim,hidden=64,num_layers=2,
                 bidirectional=True,fc_hidden=128,dropout=0.15,
                 activation='gelu',main_clip=0.02):
        super().__init__()
        self.gru=nn.GRU(seq_dim,hidden,num_layers,bidirectional=bidirectional,
                        batch_first=True,dropout=dropout if num_layers>1 else 0.)
        gru_out=hidden*(2 if bidirectional else 1)
        act=get_act(activation)
        self.net=nn.Sequential(
            nn.Linear(gru_out+scal_dim,fc_hidden),act,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden,fc_hidden//2),act)
        self.head_main=nn.Linear(fc_hidden//2,3)
        self.head_F=nn.Linear(fc_hidden//2,3)
        self.head_W=nn.Linear(fc_hidden//2,3)
        self.clip=main_clip

    def forward(self,seq,scal):
        x=self.gru(seq)[0][:,-1]
        z=self.net(torch.cat([x,scal],1))
        return torch.tanh(self.head_main(z))*self.clip, self.head_F(z), self.head_W(z)

class TransformerModel(nn.Module):
    """TransformerEncoder + CLS token + learned positional embedding
    v4 attention pooling 분석: last-step과 attention의 차이가 없었음
    → CLS token 방식으로 global context 학습
    """
    def __init__(self,seq_dim,scal_dim,d_model=64,nhead=4,
                 num_enc_layers=2,fc_hidden=128,dropout=0.1,
                 activation='gelu',main_clip=0.02):
        super().__init__()
        self.input_proj=nn.Linear(seq_dim,d_model)
        self.pos_emb=nn.Embedding(12,d_model)   # 11 steps + 1 CLS
        self.cls_token=nn.Parameter(torch.zeros(1,1,d_model))
        enc_layer=nn.TransformerEncoderLayer(
            d_model=d_model,nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,batch_first=True,
            activation='gelu',norm_first=True)  # Pre-LN: 더 안정적
        self.encoder=nn.TransformerEncoder(enc_layer,num_layers=num_enc_layers)
        act=get_act(activation)
        self.net=nn.Sequential(
            nn.Linear(d_model+scal_dim,fc_hidden),act,
            nn.Dropout(dropout),
            nn.Linear(fc_hidden,fc_hidden//2),act)
        self.head_main=nn.Linear(fc_hidden//2,3)
        self.head_F=nn.Linear(fc_hidden//2,3)
        self.head_W=nn.Linear(fc_hidden//2,3)
        self.clip=main_clip

    def forward(self,seq,scal):
        N,T,_=seq.shape
        x=self.input_proj(seq)                             # (N,T,d_model)
        cls=self.cls_token.expand(N,-1,-1)                 # (N,1,d_model)
        x=torch.cat([cls,x],dim=1)                        # (N,T+1,d_model)
        pos=torch.arange(T+1,device=seq.device)
        x=x+self.pos_emb(pos)
        h=self.encoder(x)                                  # (N,T+1,d_model)
        pooled=h[:,0]                                      # CLS output
        z=self.net(torch.cat([pooled,scal],1))
        return torch.tanh(self.head_main(z))*self.clip, self.head_F(z), self.head_W(z)

def build_model(bp, seq_dim, scal_dim):
    mtype=bp.get('model_type','gru')
    if mtype=='transformer':
        return TransformerModel(
            seq_dim,scal_dim,
            d_model=bp['d_model'],nhead=bp['nhead'],
            num_enc_layers=bp['num_enc_layers'],
            fc_hidden=bp['fc_hidden'],dropout=bp['dropout'],
            activation=bp['activation'],main_clip=bp['main_clip']).to(device)
    else:
        return GRUModel(
            seq_dim,scal_dim,
            hidden=bp['hidden'],num_layers=bp['num_layers'],
            bidirectional=bp['bidir'],
            fc_hidden=bp['fc_hidden'],dropout=bp['dropout'],
            activation=bp['activation'],main_clip=bp['main_clip']).to(device)

# ════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════
class WhiteningScaler:
    def fit(self,X):
        self.mu=X.mean(0)
        cov=np.cov(X.T)+np.eye(X.shape[1])*1e-8
        self.L=np.linalg.cholesky(cov); self.Li=np.linalg.inv(self.L)
        return self
    def transform(self,X): return (X-self.mu)@self.Li.T

def norm_seq(arr,sc):
    N,T,C=arr.shape
    return sc.transform(arr.reshape(-1,C)).astype(np.float32).reshape(N,T,C)

def make_fold_data(tri,vai,scal_arr,tgt,tgt_f,use_whiten):
    if use_whiten:
        sc_seq=WhiteningScaler().fit(seq_tr[tri].reshape(-1,SEQ_DIM))
    else:
        sc_seq=StandardScaler().fit(seq_tr[tri].reshape(-1,SEQ_DIM))
    sc_scal=StandardScaler().fit(scal_arr[tri])
    return dict(
        seq_tr=norm_seq(seq_tr[tri],sc_seq),
        scal_tr=sc_scal.transform(scal_arr[tri]).astype(np.float32),
        tgt_tr=tgt[tri].astype(np.float32),
        tgt_tr_f=tgt_f[tri].astype(np.float32),
        af_tr=aux_F[tri], aw_tr=aux_W[tri],
        seq_va=norm_seq(seq_tr[vai],sc_seq),
        seq_va_f=norm_seq(seq_tr_f[vai],sc_seq),  # y-flip val seq
        scal_va=sc_scal.transform(scal_arr[vai]).astype(np.float32),
        kal_va=kalman_train[vai], kal_va_f=kal_tr_f[vai],
        theta_va=theta_tr[vai], theta_va_f=theta_tr_f[vai],
        y_va=y_train[vai],
        sc_seq=sc_seq, sc_scal=sc_scal,
    )

def train_fold(model,fd,cfg,n_ep,estop):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_ep)
    loss_fn=make_loss(cfg.get('focal_gamma',1.0))
    lF,lW,bs=cfg['lF'],cfg['lW'],cfg['batch']
    def T(a): return torch.from_numpy(np.asarray(a,np.float32)).to(device)
    # 원본 + y-flip 동시 학습으로 데이터 2배 효과
    st=np.concatenate([fd['seq_tr'], norm_seq(seq_tr_f[fd.get('tri',slice(None))],
                                              fd['sc_seq'])],0) \
        if 'tri' in fd else fd['seq_tr']
    # 단순화: 원본만으로 학습 (flip은 inference TTA에서 활용)
    st,sc=T(fd['seq_tr']),T(fd['scal_tr'])
    tt,af,aw=T(fd['tgt_tr']),T(fd['af_tr']),T(fd['aw_tr'])
    sv,scv=T(fd['seq_va']),T(fd['scal_va'])
    best,bst,no=-1,None,0; n=st.shape[0]
    for ep in range(n_ep):
        model.train(); perm=torch.randperm(n,device=device)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            om,oF,oW=model(st[idx],sc[idx])
            loss=(loss_fn(om,tt[idx])
                  +lF*loss_euclid(oF,af[idx])
                  +lW*loss_euclid(oW,aw[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            ov,_,_=model(sv,scv)
            # TTA: y-flip으로도 예측 후 평균
            ov_f,_,_=model(T(fd['seq_va_f']),scv)
        # 원본 예측
        pred_orig=fd['kal_va']+inv_rot(ov.cpu().numpy(),fd['theta_va'])
        # y-flip 예측 역변환
        pred_flip_space=fd['kal_va_f']+inv_rot(ov_f.cpu().numpy(),fd['theta_va_f'])
        pred_flip_space[:,1]*=-1
        pred_tta=(pred_orig+pred_flip_space)/2
        rh=r_hit(pred_tta,fd['y_va'])
        if rh>best:
            best=rh; bst={k:v.detach().clone() for k,v in model.state_dict().items()}; no=0
        else: no+=1
        if no>=estop: break
    model.load_state_dict(bst); return best,model

# ════════════════════════════════════════════════════════════════
# Optuna
# ════════════════════════════════════════════════════════════════
OPTUNA_FOLDS=3; OPTUNA_EP=120; OPTUNA_ESTOP=25
kf_o=KFold(n_splits=OPTUNA_FOLDS,shuffle=True,random_state=0)
folds_o=list(kf_o.split(np.arange(len(X_train))))

def objective(trial):
    model_type=trial.suggest_categorical('model_type',['gru','transformer'])
    activation=trial.suggest_categorical('activation',['mish','gelu','swish'])
    fc_hidden=trial.suggest_categorical('fc_hidden',[128,256])
    dropout=trial.suggest_float('dropout',0.05,0.30,step=0.05)
    main_clip=trial.suggest_categorical('main_clip',[0.02,0.03,0.05])
    lr=trial.suggest_float('lr',2e-4,3e-3,log=True)
    wd=trial.suggest_float('wd',1e-5,5e-3,log=True)
    batch=trial.suggest_categorical('batch',[128,256])
    lF=trial.suggest_float('lF',0.2,0.6,step=0.05)
    lW=trial.suggest_float('lW',0.1,0.4,step=0.05)
    focal_gamma=trial.suggest_float('focal_gamma',0.5,2.5,step=0.5)
    use_wav=trial.suggest_categorical('use_wav',[True,False])
    use_whiten=trial.suggest_categorical('use_whiten',[True,False])
    so=trial.suggest_float('sigma_obs',0.1e-3,1.0e-3,log=True)
    sp=trial.suggest_float('sigma_proc',0.5,2.0,step=0.25)

    if model_type=='gru':
        hidden=trial.suggest_categorical('hidden',[64,128])
        num_layers=trial.suggest_int('num_layers',1,2)
        bidir=trial.suggest_categorical('bidir',[True,False])
        bp=dict(model_type='gru',hidden=hidden,num_layers=num_layers,bidir=bidir,
                fc_hidden=fc_hidden,dropout=dropout,activation=activation,main_clip=main_clip)
    else:
        d_model=trial.suggest_categorical('d_model',[64,128])
        nhead=trial.suggest_categorical('nhead',[4,8])
        num_enc_layers=trial.suggest_int('num_enc_layers',1,3)
        bp=dict(model_type='transformer',d_model=d_model,nhead=nhead,
                num_enc_layers=num_enc_layers,
                fc_hidden=fc_hidden,dropout=dropout,activation=activation,main_clip=main_clip)

    kal_tr_=(kalman_predict(X_train,so,sp)
             if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_train)
    tgt_=rotate_xy(y_train-kal_tr_,theta_tr)
    kal_tr_f_=kal_tr_.copy(); kal_tr_f_[:,1]*=-1
    tgt_f_=rotate_xy(y_flip-kal_tr_f_,theta_tr_f)
    sa=(np.concatenate([sb_tr,t3_tr,wav_tr],-1) if use_wav
        else np.concatenate([sb_tr,t3_tr],-1))
    scal_dim=sa.shape[1]
    cfg=dict(lr=lr,wd=wd,batch=batch,lF=lF,lW=lW,focal_gamma=focal_gamma)

    scores=[]
    for tri,vai in folds_o:
        fd=make_fold_data(tri,vai,sa,tgt_,tgt_f_,use_whiten)
        fd['tri']=tri
        torch.manual_seed(0)
        m=build_model(bp,SEQ_DIM,scal_dim)
        rh,_=train_fold(m,fd,cfg,OPTUNA_EP,OPTUNA_ESTOP)
        scores.append(rh); torch.cuda.empty_cache(); gc.collect()
    return float(np.mean(scores))

N_TRIALS=80
study=optuna.create_study(
    study_name='mosquito_v5',storage=f'sqlite:///{DB_PATH}',
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15),
    load_if_exists=True)

done=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
print(f'existing trials: {done}  remaining: {max(0,N_TRIALS-done)}')

# Starting points: v4 best (gelu+2L) & Transformer variants
good_starts=[
    # v4 best GRU config with TTA
    dict(model_type='gru',hidden=64,num_layers=2,bidir=True,
         fc_hidden=128,dropout=0.15,activation='gelu',main_clip=0.02,
         lr=0.000499,wd=0.001661,batch=256,lF=0.25,lW=0.15,focal_gamma=1.0,
         use_wav=True,use_whiten=True,sigma_obs=0.000493,sigma_proc=1.75),
    # Transformer small
    dict(model_type='transformer',d_model=64,nhead=4,num_enc_layers=2,
         fc_hidden=128,dropout=0.10,activation='gelu',main_clip=0.02,
         lr=0.000500,wd=0.001000,batch=256,lF=0.30,lW=0.15,focal_gamma=1.0,
         use_wav=True,use_whiten=True,sigma_obs=0.000493,sigma_proc=1.75),
    # Transformer larger
    dict(model_type='transformer',d_model=128,nhead=8,num_enc_layers=2,
         fc_hidden=256,dropout=0.10,activation='gelu',main_clip=0.02,
         lr=0.000300,wd=0.001000,batch=256,lF=0.30,lW=0.15,focal_gamma=1.0,
         use_wav=True,use_whiten=True,sigma_obs=0.000493,sigma_proc=1.75),
    # Transformer deeper + mish
    dict(model_type='transformer',d_model=64,nhead=4,num_enc_layers=3,
         fc_hidden=128,dropout=0.15,activation='mish',main_clip=0.03,
         lr=0.000400,wd=0.000800,batch=256,lF=0.25,lW=0.20,focal_gamma=1.5,
         use_wav=True,use_whiten=False,sigma_obs=0.000300,sigma_proc=1.50),
    # GRU mish 1L (v3 D 재현)
    dict(model_type='gru',hidden=64,num_layers=1,bidir=True,
         fc_hidden=128,dropout=0.25,activation='mish',main_clip=0.03,
         lr=0.000989,wd=0.000886,batch=256,lF=0.50,lW=0.25,focal_gamma=2.0,
         use_wav=True,use_whiten=True,sigma_obs=0.000267,sigma_proc=1.0),
]
for cfg in good_starts:
    if done < N_TRIALS:
        try: study.enqueue_trial(cfg)
        except: pass

remain=max(0,N_TRIALS-len(study.trials))
if remain>0:
    print(f'starting Optuna ({remain} trials)...')
    t0=time.time()
    study.optimize(objective,n_trials=remain,show_progress_bar=True)
    print(f'done ({(time.time()-t0)/60:.1f}min)')

print(f'\n=== Optuna v5 best ===')
print(f'  score: {study.best_value:.4f}  model_type={study.best_params["model_type"]}')
for k,v in study.best_params.items(): print(f'  {k:<18}: {v}')

tdf=study.trials_dataframe()
top=tdf.nlargest(10,'value')[['number','value']+
    [c for c in tdf.columns if c.startswith('params_')]]
top.columns=[c.replace('params_','') for c in top.columns]
print('\nTop-10:'); print(top.to_string(index=False))

# GRU vs Transformer 비교
gru_mask=(tdf['params_model_type']=='gru')&(tdf['value'].notna())
tr_mask=(tdf['params_model_type']=='transformer')&(tdf['value'].notna())
print(f'\n[Architecture 비교 (TTA 포함)]')
print(f'  GRU        : mean={tdf[gru_mask]["value"].mean():.4f}  '
      f'max={tdf[gru_mask]["value"].max():.4f}  n={gru_mask.sum()}')
print(f'  Transformer: mean={tdf[tr_mask]["value"].mean():.4f}  '
      f'max={tdf[tr_mask]["value"].max():.4f}  n={tr_mask.sum()}')

# ════════════════════════════════════════════════════════════════
# Full Top-K training
# ════════════════════════════════════════════════════════════════
TOP_K=5; N_FOLDS=5; N_SEEDS=3; FULL_EP=400; FULL_ESTOP=60
kf_full=KFold(n_splits=N_FOLDS,shuffle=True,random_state=0)

completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
top_trials=sorted(completed,key=lambda t:t.value or -1,reverse=True)[:TOP_K]

all_test_preds=[]; all_oof_preds=[]; all_info=[]

for rank,trial in enumerate(top_trials):
    bp=trial.params
    mtype=bp.get('model_type','gru')
    print(f'\n[{rank+1}/{TOP_K}] Trial#{trial.number}  3-fold={trial.value:.4f}  [{mtype}]')

    so,sp=bp['sigma_obs'],bp['sigma_proc']
    kal_tr_=(kalman_predict(X_train,so,sp)
             if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_train)
    kal_te_=(kalman_predict(X_test,so,sp)
             if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_test)
    kal_tr_f_=kal_tr_.copy(); kal_tr_f_[:,1]*=-1
    kal_te_f_=kal_te_.copy(); kal_te_f_[:,1]*=-1
    tgt_=rotate_xy(y_train-kal_tr_,theta_tr)
    tgt_f_=rotate_xy(y_flip-kal_tr_f_,theta_tr_f)
    sa_=(np.concatenate([sb_tr,t3_tr,wav_tr],-1) if bp['use_wav']
         else np.concatenate([sb_tr,t3_tr],-1))
    ste_=(np.concatenate([sb_te,t3_te,wav_te],-1) if bp['use_wav']
          else np.concatenate([sb_te,t3_te],-1))
    scal_dim=sa_.shape[1]
    cfg_=dict(lr=bp['lr'],wd=bp['wd'],batch=bp['batch'],
              lF=bp['lF'],lW=bp['lW'],focal_gamma=bp['focal_gamma'])

    oof_rot=np.zeros((len(X_train),3)); test_preds_orig=[]; test_preds_flip=[]; fold_rh=[]
    t0=time.time()

    for fi,(tri,vai) in enumerate(kf_full.split(np.arange(len(X_train)))):
        fd=make_fold_data(tri,vai,sa_,tgt_,tgt_f_,bp['use_whiten'])
        fd['tri']=tri

        te_seq_n=norm_seq(seq_te,fd['sc_seq'])
        te_seq_f_n=norm_seq(seq_te_f,fd['sc_seq'])
        te_scal_n=fd['sc_scal'].transform(ste_).astype(np.float32)
        te_seq_t=torch.from_numpy(te_seq_n).to(device)
        te_seq_f_t=torch.from_numpy(te_seq_f_n).to(device)
        te_scal_t=torch.from_numpy(te_scal_n).to(device)

        seed_oof=[]; seed_te_orig=[]; seed_te_flip=[]
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            m=build_model(bp,SEQ_DIM,scal_dim)
            _,m=train_fold(m,fd,cfg_,FULL_EP,FULL_ESTOP)
            m.eval()
            def T(a): return torch.from_numpy(np.asarray(a,np.float32)).to(device)
            sv_t=T(fd['seq_va']); scv_t=T(fd['scal_va'])
            sv_f_t=T(fd['seq_va_f'])
            with torch.no_grad():
                ov,_,_=m(sv_t,scv_t)
                ov_f,_,_=m(sv_f_t,scv_t)  # scal 불변
                te_o,_,_=m(te_seq_t,te_scal_t)
                te_f,_,_=m(te_seq_f_t,te_scal_t)
            seed_oof.append((ov.cpu().numpy(),ov_f.cpu().numpy()))
            seed_te_orig.append(te_o.cpu().numpy())
            seed_te_flip.append(te_f.cpu().numpy())
            torch.cuda.empty_cache(); gc.collect()

        # OOF TTA
        vr=np.mean([x[0] for x in seed_oof],0)
        vr_f=np.mean([x[1] for x in seed_oof],0)
        pred_orig=fd['kal_va']+inv_rot(vr,fd['theta_va'])
        pred_flip=fd['kal_va_f']+inv_rot(vr_f,fd['theta_va_f'])
        pred_flip[:,1]*=-1
        pred_tta=(pred_orig+pred_flip)/2
        oof_rot[vai]=vr  # 원본만 저장 (calibration은 원본 기준)

        rh=r_hit(pred_tta,y_train[vai]); fold_rh.append(rh)
        print(f'  fold {fi+1}: R-Hit={rh:.4f} (TTA)  ({(time.time()-t0)/60:.1f}min)')

        tr_o=np.mean(seed_te_orig,0); tr_f=np.mean(seed_te_flip,0)
        test_preds_orig.append(tr_o); test_preds_flip.append(tr_f)

    # OOF 평가 (TTA)
    oof_pred_orig=kal_tr_+inv_rot(oof_rot,theta_tr)
    # y-flip OOF 재계산: 원본 모델로 flip seq 예측은 fold 루프에서 이미 수행됨
    # 간략화: oof_rot를 원본 기준으로 평가
    oof_rh=r_hit(oof_pred_orig,y_train)

    # Calibration (원본 기준)
    best_cal,best_a=-1,np.ones(3)
    for ax in np.arange(0.85,1.11,0.05):
        for ay in np.arange(0.85,1.06,0.05):
            for az in np.arange(0.85,1.11,0.05):
                a=np.array([ax,ay,az])
                rc=r_hit(kal_tr_+inv_rot(oof_rot*a,theta_tr),y_train)
                if rc>best_cal: best_cal,best_a=rc,a.copy()

    # Test 예측 (TTA)
    te_rot_o=np.mean(test_preds_orig,0)*best_a
    te_rot_f=np.mean(test_preds_flip,0)*best_a

    pred_te_orig=kal_te_+inv_rot(te_rot_o,theta_te)
    pred_te_flip=kal_te_f_+inv_rot(te_rot_f,theta_te_f); pred_te_flip[:,1]*=-1
    test_pred_tta=(pred_te_orig+pred_te_flip)/2

    all_test_preds.append(test_pred_tta)
    all_oof_preds.append(oof_pred_orig)
    all_info.append({'rank':rank+1,'trial':trial.number,'mtype':mtype,
                     'oof':oof_rh,'cal_oof':best_cal,'alpha':best_a})

    print(f'  OOF={oof_rh:.4f}  CalOOF={best_cal:.4f}  '
          f'alpha=({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')
    fname=f'sub_v5_rank{rank+1}_t{trial.number}_{mtype}_OOF{oof_rh:.4f}.csv'
    pd.DataFrame({'id':test_ids,'x':test_pred_tta[:,0],
                  'y':test_pred_tta[:,1],'z':test_pred_tta[:,2]}
    ).to_csv(os.path.join(OUT_DIR,fname),index=False)
    print(f'  saved: {fname}')

print('\n=== Final Ensemble ===')
cal_scores=np.array([i['cal_oof'] for i in all_info])
w=cal_scores/cal_scores.sum()
ens_w=sum(wi*p for wi,p in zip(w,all_test_preds))
rh_w=r_hit(sum(wi*p for wi,p in zip(w,all_oof_preds)),y_train)
ens_eq=np.mean(all_test_preds,0)
rh_eq=r_hit(np.mean(all_oof_preds,0),y_train)
print(f'OOF-weighted: {rh_w:.4f}  Equal: {rh_eq:.4f}')
ts=time.strftime('%m%d_%H%M')
for label,pred,rh in [('weighted',ens_w,rh_w),('equal',ens_eq,rh_eq)]:
    fname=f'sub_v5_topK_{label}_{ts}_OOF{rh:.4f}.csv'
    pd.DataFrame({'id':test_ids,'x':pred[:,0],'y':pred[:,1],'z':pred[:,2]}
    ).to_csv(os.path.join(OUT_DIR,fname),index=False)
    print(f'saved: {fname}')

print('\n=== 진행 현황 ===')
print(f'  v2 best     : OOF 0.6642  LB ~0.6810')
print(f'  v3 D +both  : OOF 0.6659  LB ~0.6827')
print(f'  v4 ensemble : OOF 0.6657  LB ~0.6825')
for i in all_info:
    print(f'  v5 rank{i["rank"]} [{i["mtype"]}]: OOF {i["oof"]:.4f} (cal {i["cal_oof"]:.4f})')
print(f'  v5 weighted : OOF {rh_w:.4f}  LB est ~{rh_w+0.0168:.4f}')
