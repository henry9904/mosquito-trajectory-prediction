# -*- coding: utf-8 -*-
"""
train_v4.py — v3 ablation 결과 반영 + Optuna 재탐색
고정: curvature features + focal loss (둘 다 시너지 확인)
추가: Attention pooling over BiGRU (Optuna 탐색 대상)
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
DB_PATH   = os.path.join(CACHE_DIR, 'optuna_v4.db')
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

# ── 데이터 로드 ────────────────────────────────────────────────
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

# ── 기본 유틸 ──────────────────────────────────────────────────
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N, T, _ = X.shape
    F  = np.array([[1,DT],[0,1]]); Fp = np.array([[1,T_PRED],[0,1]])
    Q  = sigma_proc**2 * np.array([[DT**4/4,DT**3/2],[DT**3/2,DT**2]])
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
    return float((np.linalg.norm(pred-true,axis=-1)<=thr).mean())

def cos_safe(a,b):
    return np.clip((a*b).sum(-1)/np.maximum(
        np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12),-1,1)

def yaw_angle(v): return np.arctan2(v[:,1],v[:,0])

def rotate_xy(vec, th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([vec[:,0]*c+vec[:,1]*s,-vec[:,0]*s+vec[:,1]*c,vec[:,2]],-1)

def inv_rot(vec, th):
    c,s=np.cos(th),np.sin(th)
    return np.stack([vec[:,0]*c-vec[:,1]*s,vec[:,0]*s+vec[:,1]*c,vec[:,2]],-1)

kalman_train = kalman_predict(X_train)
kalman_test  = kalman_predict(X_test)
print(f'Kalman R-Hit = {r_hit(kalman_train,y_train):.4f}')

# ════════════════════════════════════════════════════════════════
# Features — v3 ablation 결과: curvature features 고정 사용
# ════════════════════════════════════════════════════════════════
def build_seq(X):
    """seq (N,11,13): rel(3)+vel(3)+acc(3)+speed(1)+curvature(1)+angvel(1)+turnrate(1)"""
    N = X.shape[0]
    rel = X - X[:,-1:]
    v   = np.diff(X, axis=1) / DT              # (N,10,3)
    a   = np.diff(v, axis=1) / DT              # (N,9,3)
    vp  = np.concatenate([np.zeros((N,1,3)), v], 1)   # (N,11,3)
    ap  = np.concatenate([np.zeros((N,2,3)), a], 1)   # (N,11,3)

    speed = np.linalg.norm(vp, axis=-1, keepdims=True)          # (N,11,1)
    cross     = np.cross(vp, ap)                                  # (N,11,3)
    cross_mag = np.linalg.norm(cross, axis=-1, keepdims=True)    # (N,11,1)
    sp_safe   = speed + 1e-12
    curvature   = np.log1p(cross_mag / (sp_safe**3))             # (N,11,1)
    angular_vel = np.log1p(cross_mag / (sp_safe**2))             # (N,11,1)

    v_unit = vp / (sp_safe + 1e-12)
    cos_t  = (v_unit[:,:-1]*v_unit[:,1:]).sum(-1,keepdims=True).clip(-1,1)
    turn_r = np.concatenate([np.zeros((N,1,1)), np.arccos(cos_t)], 1)  # (N,11,1)

    return np.concatenate([rel,vp,ap,speed,curvature,angular_vel,turn_r],-1
                          ).astype(np.float32)  # (N,11,13)

LOG_COLS=['mean_speed','max_speed','speed_std','mean_acc','max_acc','max_jerk',
          'net_disp','|v_last|','|a_last|','|a_recent|','jerk_last','jerk_recent',
          'noise_poly2','noise_savgol','noise_loo']

def build_scalar(X, np_a, ns_a, nl_a=None):
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
    # 곡률·각속도 요약 (v3 추가)
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

CACHE_FILE = os.path.join(CACHE_DIR,'noise_cache.npz')
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

seq_train=build_seq(X_train);   seq_test=build_seq(X_test)
sb_tr=build_scalar(X_train,noise_p,noise_s,noise_l)
sb_te=build_scalar(X_test,noise_p_te,noise_s_te)
t3_tr=build_tier3(X_train);     t3_te=build_tier3(X_test)
wav_tr=build_wavelet(X_train);  wav_te=build_wavelet(X_test)
SEQ_DIM = seq_train.shape[2]   # 13
print(f'seq {seq_train.shape} | scal_base {sb_tr.shape} | wav {wav_tr.shape}')

# ── target ─────────────────────────────────────────────────────
theta_tr=yaw_angle((X_train[:,-1]-X_train[:,-2])/DT)
theta_te=yaw_angle((X_test[:,-1] -X_test[:,-2]) /DT)
target_T8=rotate_xy(y_train-kalman_train,theta_tr)
aux_F=rotate_xy(y_train-X_train[:,-1],theta_tr).astype(np.float32)
aux_W=rotate_xy(y_train-kalman_predict(X_train,1e-3),theta_tr).astype(np.float32)
print(f'T8 std (cm): {target_T8.std(0)*100}')

# ════════════════════════════════════════════════════════════════
# Loss — v3 ablation 결과: focal loss 고정 사용 (시너지 확인)
# ════════════════════════════════════════════════════════════════
def loss_euclid(p,t):
    return torch.sqrt(((p-t)**2).sum(-1)+1e-12).mean()

def loss_softhit(p,t,b=0.002):
    return torch.sigmoid(
        (torch.sqrt(((p-t)**2).sum(-1)+1e-12)-0.01)/b).mean()

def loss_focal_euclid(p, t, gamma=2.0):
    err = torch.sqrt(((p-t)**2).sum(-1)+1e-12)
    p_miss = torch.sigmoid((err-0.01)/0.002)
    w = p_miss.pow(gamma).detach()
    w = w / (w.mean()+1e-8)
    return (w*err).mean()

def make_loss(gamma):
    def fn(p, t):
        return loss_focal_euclid(p,t,gamma) + 0.3*loss_softhit(p,t)
    return fn

# ════════════════════════════════════════════════════════════════
# Model — Attention pooling 추가
# ════════════════════════════════════════════════════════════════
class Mish(nn.Module):
    def forward(self,x): return x*torch.tanh(F.softplus(x))

def get_act(name):
    return {'gelu':nn.GELU(),'swish':nn.SiLU(),'mish':Mish()}[name]

class AttentionPool(nn.Module):
    """BiGRU 전 timestep에 attention 가중합
    EDA: 단순 last-step보다 전체 시퀀스 보는 것이 급기동 감지에 유리"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):   # h: (N, T, D)
        w = torch.softmax(self.attn_w(h), dim=1)  # (N, T, 1)
        return (w * h).sum(1)                       # (N, D)

class GRUv4(nn.Module):
    def __init__(self, seq_dim, scal_dim,
                 hidden=64, num_layers=1, bidirectional=True,
                 fc_hidden=128, dropout=0.25, activation='mish',
                 main_clip=0.03, use_attention=True):
        super().__init__()
        self.gru = nn.GRU(seq_dim, hidden, num_layers,
                          bidirectional=bidirectional, batch_first=True,
                          dropout=dropout if num_layers>1 else 0.)
        gru_out = hidden*(2 if bidirectional else 1)

        self.use_attention = use_attention
        if use_attention:
            self.attn_pool = AttentionPool(gru_out)

        act = get_act(activation)
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
        h = self.gru(seq)[0]                       # (N,T,gru_out)
        if self.use_attention:
            x = self.attn_pool(h)                  # (N,gru_out)
        else:
            x = h[:,-1]                            # last step only
        z = self.net(torch.cat([x,scal],1))
        return (torch.tanh(self.head_main(z))*self.clip,
                self.head_F(z), self.head_W(z))

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

def make_fold_data(tri,vai,scal_arr,tgt,use_whiten):
    if use_whiten:
        sc_seq=WhiteningScaler().fit(seq_train[tri].reshape(-1,SEQ_DIM))
    else:
        sc_seq=StandardScaler().fit(seq_train[tri].reshape(-1,SEQ_DIM))
    sc_scal=StandardScaler().fit(scal_arr[tri])
    return dict(
        seq_tr=norm_seq(seq_train[tri],sc_seq),
        scal_tr=sc_scal.transform(scal_arr[tri]).astype(np.float32),
        tgt_tr=tgt[tri].astype(np.float32),
        af_tr=aux_F[tri], aw_tr=aux_W[tri],
        seq_va=norm_seq(seq_train[vai],sc_seq),
        scal_va=sc_scal.transform(scal_arr[vai]).astype(np.float32),
        kal_va=kalman_train[vai], theta_va=theta_tr[vai], y_va=y_train[vai],
        sc_seq=sc_seq, sc_scal=sc_scal,
    )

def train_fold(model, fd, cfg, n_ep=200, estop=30):
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_ep)
    loss_fn=make_loss(cfg.get('focal_gamma',2.0))
    lF,lW,bs=cfg['lF'],cfg['lW'],cfg['batch']
    def T(a): return torch.from_numpy(np.asarray(a,np.float32)).to(device)
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
        with torch.no_grad(): ov,_,_=model(sv,scv)
        pred=fd['kal_va']+inv_rot(ov.cpu().numpy(),fd['theta_va'])
        rh=r_hit(pred,fd['y_va'])
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
    hidden       =trial.suggest_categorical('hidden',[64,128])
    num_layers   =trial.suggest_int('num_layers',1,2)
    bidir        =trial.suggest_categorical('bidir',[True,False])
    fc_hidden    =trial.suggest_categorical('fc_hidden',[128,256])
    dropout      =trial.suggest_float('dropout',0.05,0.35,step=0.05)
    activation   =trial.suggest_categorical('activation',['mish','swish','gelu'])
    main_clip    =trial.suggest_categorical('main_clip',[0.02,0.03,0.05])
    lr           =trial.suggest_float('lr',3e-4,3e-3,log=True)
    wd           =trial.suggest_float('wd',1e-5,5e-3,log=True)
    batch        =trial.suggest_categorical('batch',[128,256])
    lF           =trial.suggest_float('lF',0.2,0.6,step=0.05)
    lW           =trial.suggest_float('lW',0.1,0.5,step=0.05)
    focal_gamma  =trial.suggest_float('focal_gamma',1.0,3.0,step=0.5)
    use_attention=trial.suggest_categorical('use_attention',[True,False])
    use_wav      =trial.suggest_categorical('use_wav',[True,False])
    use_whiten   =trial.suggest_categorical('use_whiten',[True,False])
    so           =trial.suggest_float('sigma_obs',0.1e-3,1.0e-3,log=True)
    sp           =trial.suggest_float('sigma_proc',0.5,2.0,step=0.25)

    kal_tr=(kalman_predict(X_train,so,sp)
            if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_train)
    tgt=rotate_xy(y_train-kal_tr,theta_tr)
    sa=(np.concatenate([sb_tr,t3_tr,wav_tr],-1) if use_wav
        else np.concatenate([sb_tr,t3_tr],-1))
    scal_dim=sa.shape[1]
    cfg=dict(lr=lr,wd=wd,batch=batch,lF=lF,lW=lW,focal_gamma=focal_gamma)

    scores=[]
    for tri,vai in folds_o:
        fd=make_fold_data(tri,vai,sa,tgt,use_whiten)
        torch.manual_seed(0)
        m=GRUv4(SEQ_DIM,scal_dim,hidden,num_layers,bidir,
                fc_hidden,dropout,activation,main_clip,use_attention).to(device)
        rh,_=train_fold(m,fd,cfg,OPTUNA_EP,OPTUNA_ESTOP)
        scores.append(rh); torch.cuda.empty_cache(); gc.collect()
    return float(np.mean(scores))

N_TRIALS=80
study=optuna.create_study(
    study_name='mosquito_v4', storage=f'sqlite:///{DB_PATH}',
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15),
    load_if_exists=True,
)

done=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
print(f'existing trials: {done}  remaining: {max(0,N_TRIALS-done)}')

# v3 D (+both) 최고 파라미터 + attention 변형을 starting point로
good_starts=[
    # v3 D 기반 (trial#62 params) + attention=True
    dict(hidden=64,num_layers=1,bidir=True,fc_hidden=128,dropout=0.25,
         activation='mish',main_clip=0.03,lr=0.000989,wd=0.000886,
         batch=256,lF=0.50,lW=0.25,focal_gamma=2.0,use_attention=True,
         use_wav=True,use_whiten=True,sigma_obs=0.000267,sigma_proc=1.0),
    # attention=False (기존 방식) baseline
    dict(hidden=64,num_layers=1,bidir=True,fc_hidden=128,dropout=0.25,
         activation='mish',main_clip=0.03,lr=0.000989,wd=0.000886,
         batch=256,lF=0.50,lW=0.25,focal_gamma=2.0,use_attention=False,
         use_wav=True,use_whiten=True,sigma_obs=0.000267,sigma_proc=1.0),
    # focal_gamma 탐색
    dict(hidden=64,num_layers=1,bidir=True,fc_hidden=128,dropout=0.20,
         activation='mish',main_clip=0.03,lr=0.000760,wd=0.000187,
         batch=256,lF=0.40,lW=0.35,focal_gamma=1.5,use_attention=True,
         use_wav=True,use_whiten=False,sigma_obs=0.000311,sigma_proc=1.0),
    # larger model + attention
    dict(hidden=128,num_layers=1,bidir=True,fc_hidden=256,dropout=0.20,
         activation='mish',main_clip=0.02,lr=0.000700,wd=0.000200,
         batch=256,lF=0.40,lW=0.30,focal_gamma=2.0,use_attention=True,
         use_wav=True,use_whiten=False,sigma_obs=0.000300,sigma_proc=1.0),
    # gamma=3 강하게
    dict(hidden=64,num_layers=1,bidir=True,fc_hidden=128,dropout=0.15,
         activation='mish',main_clip=0.03,lr=0.000800,wd=0.000300,
         batch=256,lF=0.45,lW=0.25,focal_gamma=3.0,use_attention=True,
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

print(f'\n=== Optuna v4 best ===')
print(f'  score : {study.best_value:.4f}')
for k,v in study.best_params.items(): print(f'  {k:<16}: {v}')

tdf=study.trials_dataframe()
top=tdf.nlargest(10,'value')[['number','value']+
    [c for c in tdf.columns if c.startswith('params_')]]
top.columns=[c.replace('params_','') for c in top.columns]
print('\nTop-10 trials:'); print(top.to_string(index=False))

# attention 효과 집계
att_mask=(tdf['params_use_attention']==True) & (tdf['value'].notna())
noatt_mask=(tdf['params_use_attention']==False) & (tdf['value'].notna())
print(f'\n[Attention 효과]')
print(f'  use_attention=True  : mean={tdf[att_mask]["value"].mean():.4f}  '
      f'n={att_mask.sum()}')
print(f'  use_attention=False : mean={tdf[noatt_mask]["value"].mean():.4f}  '
      f'n={noatt_mask.sum()}')

# ════════════════════════════════════════════════════════════════
# Full training — Top-K
# ════════════════════════════════════════════════════════════════
TOP_K=5; N_FOLDS=5; N_SEEDS=3; FULL_EP=400; FULL_ESTOP=60
kf_full=KFold(n_splits=N_FOLDS,shuffle=True,random_state=0)

completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
top_trials=sorted(completed,key=lambda t:t.value or -1,reverse=True)[:TOP_K]

all_test_preds=[]; all_oof_preds=[]; all_info=[]

for rank,trial in enumerate(top_trials):
    bp=trial.params
    print(f'\n[{rank+1}/{TOP_K}] Trial#{trial.number}  3-fold={trial.value:.4f}')
    print(f'  act={bp["activation"]} attn={bp["use_attention"]} '
          f'bidir={bp["bidir"]} hidden={bp["hidden"]} fc={bp["fc_hidden"]} '
          f'gamma={bp["focal_gamma"]:.1f}')

    so,sp=bp['sigma_obs'],bp['sigma_proc']
    kal_tr_=(kalman_predict(X_train,so,sp)
             if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_train)
    kal_te_=(kalman_predict(X_test,so,sp)
             if (abs(so-0.3e-3)>1e-6 or sp!=1.) else kalman_test)
    tgt_=rotate_xy(y_train-kal_tr_,theta_tr)
    sa_=(np.concatenate([sb_tr,t3_tr,wav_tr],-1) if bp['use_wav']
         else np.concatenate([sb_tr,t3_tr],-1))
    ste_=(np.concatenate([sb_te,t3_te,wav_te],-1) if bp['use_wav']
          else np.concatenate([sb_te,t3_te],-1))
    scal_dim=sa_.shape[1]
    cfg_=dict(lr=bp['lr'],wd=bp['wd'],batch=bp['batch'],
              lF=bp['lF'],lW=bp['lW'],focal_gamma=bp['focal_gamma'])

    oof_rot=np.zeros((len(X_train),3)); test_folds=[]; fold_rh=[]
    t0=time.time()

    for fi,(tri,vai) in enumerate(kf_full.split(np.arange(len(X_train)))):
        fd=make_fold_data(tri,vai,sa_,tgt_,bp['use_whiten'])
        te_seq_n=norm_seq(seq_test,fd['sc_seq'])
        te_scal_n=fd['sc_scal'].transform(ste_).astype(np.float32)
        te_seq_t=torch.from_numpy(te_seq_n).to(device)
        te_scal_t=torch.from_numpy(te_scal_n).to(device)

        seed_oof,seed_te=[],[]
        for s in range(N_SEEDS):
            torch.manual_seed(s); np.random.seed(s)
            m=GRUv4(SEQ_DIM,scal_dim,bp['hidden'],bp['num_layers'],
                    bp['bidir'],bp['fc_hidden'],bp['dropout'],
                    bp['activation'],bp['main_clip'],
                    bp['use_attention']).to(device)
            _,m=train_fold(m,fd,cfg_,FULL_EP,FULL_ESTOP)
            m.eval()
            sv_t=torch.from_numpy(fd['seq_va']).to(device)
            sc_t=torch.from_numpy(fd['scal_va']).to(device)
            with torch.no_grad():
                ov,_,_=m(sv_t,sc_t); te_,_,_=m(te_seq_t,te_scal_t)
            seed_oof.append(ov.cpu().numpy())
            seed_te.append(te_.cpu().numpy())
            torch.cuda.empty_cache(); gc.collect()

        vr=np.mean(seed_oof,0); tr_=np.mean(seed_te,0)
        oof_rot[vai]=vr; test_folds.append(tr_)
        pv=kal_tr_[vai]+inv_rot(vr,theta_tr[vai])
        rh=r_hit(pv,y_train[vai]); fold_rh.append(rh)
        print(f'  fold {fi+1}: R-Hit={rh:.4f}  ({(time.time()-t0)/60:.1f}min)')

    oof_pred=kal_tr_+inv_rot(oof_rot,theta_tr)
    oof_rh=r_hit(oof_pred,y_train)
    test_rot=np.mean(test_folds,0)

    best_cal,best_a=-1,np.ones(3)
    for ax in np.arange(0.85,1.11,0.05):
        for ay in np.arange(0.85,1.06,0.05):
            for az in np.arange(0.85,1.11,0.05):
                a=np.array([ax,ay,az])
                rc=r_hit(kal_tr_+inv_rot(oof_rot*a,theta_tr),y_train)
                if rc>best_cal: best_cal,best_a=rc,a.copy()

    test_cal=test_rot*best_a
    test_pred_=kal_te_+inv_rot(test_cal,theta_te)
    all_test_preds.append(test_pred_)
    all_oof_preds.append(oof_pred)
    all_info.append({'rank':rank+1,'trial':trial.number,
                     'oof':oof_rh,'cal_oof':best_cal,'alpha':best_a,
                     'attn':bp['use_attention'],'gamma':bp['focal_gamma']})
    print(f'  OOF={oof_rh:.4f}  CalOOF={best_cal:.4f}  '
          f'alpha=({best_a[0]:.2f},{best_a[1]:.2f},{best_a[2]:.2f})')
    fname=f'sub_v4_rank{rank+1}_t{trial.number}_OOF{oof_rh:.4f}.csv'
    pd.DataFrame({'id':test_ids,'x':test_pred_[:,0],
                  'y':test_pred_[:,1],'z':test_pred_[:,2]}
    ).to_csv(os.path.join(OUT_DIR,fname),index=False)
    print(f'  saved: {fname}')

# ── 앙상블 ─────────────────────────────────────────────────────
print('\n=== Final Ensemble ===')
cal_scores=np.array([i['cal_oof'] for i in all_info])
w_oof=cal_scores/cal_scores.sum()
ens_w=sum(w*p for w,p in zip(w_oof,all_test_preds))
rh_w=r_hit(sum(w*p for w,p in zip(w_oof,all_oof_preds)),y_train)
ens_eq=np.mean(all_test_preds,0)
rh_eq=r_hit(np.mean(all_oof_preds,0),y_train)
print(f'OOF-weighted: {rh_w:.4f}  Equal: {rh_eq:.4f}')

ts=time.strftime('%m%d_%H%M')
for label,pred,rh in [('weighted',ens_w,rh_w),('equal',ens_eq,rh_eq)]:
    fname=f'sub_v4_topK_{label}_{ts}_OOF{rh:.4f}.csv'
    pd.DataFrame({'id':test_ids,'x':pred[:,0],'y':pred[:,1],'z':pred[:,2]}
    ).to_csv(os.path.join(OUT_DIR,fname),index=False)
    print(f'saved: {fname}')

print('\n=== Summary ===')
print(f'  Kalman baseline : 0.5964')
print(f'  v2_best ref     : OOF 0.6642 -> LB ~0.6810 (est)')
print(f'  v3_D (+both)    : OOF 0.6659 -> LB ~0.6827 (est)')
for i in all_info:
    attn='attn' if i['attn'] else 'last'
    print(f'  v4 rank{i["rank"]} t{i["trial"]} [{attn} gamma={i["gamma"]:.1f}]: '
          f'OOF {i["oof"]:.4f} (cal {i["cal_oof"]:.4f})')
print(f'  v4 TopK weighted: {rh_w:.4f}')
print(f'  v4 TopK equal   : {rh_eq:.4f}')
