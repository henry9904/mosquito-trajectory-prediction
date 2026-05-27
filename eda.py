# -*- coding: utf-8 -*-
"""
Mosquito Trajectory EDA
목표: 왜 특정 모델/파라미터가 더 잘 되는지 데이터에서 근거 찾기
"""
import os, glob, pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = str(pathlib.Path(__file__).parent / 'open')
OUT_DIR  = str(pathlib.Path(__file__).parent / 'eda_out')
os.makedirs(OUT_DIR, exist_ok=True)

DT     = 0.040
T_PRED = 0.080

# ── 데이터 로드 ─────────────────────────────────────────────────
print('Loading data...')
train_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train', '*.csv')))
test_files  = sorted(glob.glob(os.path.join(DATA_DIR, 'test',  '*.csv')))
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'))

def load_stack(files):
    arrs = []
    for f in files:
        df = pd.read_csv(f)
        if 'timestep_ms' in df.columns:
            arrs.append(df[['x','y','z']].values)
        else:
            arrs.append(df.values)
    return np.stack(arrs).astype(np.float64)

X_train = load_stack(train_files)
X_test  = load_stack(test_files)
train_ids = [os.path.splitext(os.path.basename(f))[0] for f in train_files]
y_train = labels.set_index('id').loc[train_ids][['x','y','z']].values.astype(np.float64)
N = len(X_train)
print(f'X_train: {X_train.shape}  X_test: {X_test.shape}')

# ── 핵심 물리량 계산 ────────────────────────────────────────────
d  = np.diff(X_train, axis=1)            # (N,10,3) displacement
v  = d / DT                               # (N,10,3) velocity  [m/s]
a  = np.diff(v, axis=1) / DT             # (N,9,3)  acceleration
jk = np.diff(a, axis=1) / DT             # (N,8,3)  jerk

sp   = np.linalg.norm(v,  axis=-1)       # (N,10) speed
ac   = np.linalg.norm(a,  axis=-1)       # (N,9)  |acc|
jm   = np.linalg.norm(jk, axis=-1)       # (N,8)  |jerk|

mean_sp  = sp.mean(1)
max_sp   = sp.max(1)
mean_ac  = ac.mean(1)
max_ac   = ac.max(1)
max_jk   = jm.max(1)

# 방향 변화 (각속도 대리 지표)
v_unit = v / (sp[...,None] + 1e-12)
cos_turn = (v_unit[:,:-1] * v_unit[:,1:]).sum(-1).clip(-1,1)  # (N,9)
mean_turn = np.degrees(np.arccos(cos_turn.clip(-1,1))).mean(1)  # deg per step
max_turn  = np.degrees(np.arccos(cos_turn.clip(-1,1))).max(1)

# 직진성: net_disp / path_length
net_d   = np.linalg.norm(X_train[:,-1] - X_train[:,0], axis=-1)
path_l  = np.linalg.norm(d, axis=-1).sum(1)
straight= np.where(path_l > 1e-9, net_d / path_l, 1.0)

# Kalman 예측 (단순 등속도)
def kalman_predict(X, sigma_obs=0.3e-3, sigma_proc=1.0):
    N2, T, _ = X.shape
    F  = np.array([[1,DT],[0,1]])
    Fp = np.array([[1,T_PRED],[0,1]])
    Q  = sigma_proc**2 * np.array([[DT**4/4, DT**3/2],[DT**3/2, DT**2]])
    R  = sigma_obs**2
    pred = np.zeros((N2,3))
    for j in range(3):
        z = X[:,:,j]
        s = np.zeros((N2,2)); s[:,0] = z[:,0]
        P = np.eye(2)
        for t in range(1, T):
            s = s @ F.T
            P = F @ P @ F.T + Q
            inn = z[:,t] - s[:,0]
            S = P[0,0] + R
            K = P[:,0] / S
            s = s + np.outer(inn, K)
            P = P - np.outer(K, P[0,:])
        pred[:,j] = (s @ Fp.T)[:,0]
    return pred

kal_pred = kalman_predict(X_train)
kal_err  = np.linalg.norm(kal_pred - y_train, axis=-1)   # Kalman 오차 (m)
kal_hit  = (kal_err <= 0.01).astype(float)                # 1cm 이내 적중

print(f'Kalman R-Hit: {kal_hit.mean():.4f}')
print(f'Kalman err (median): {np.median(kal_err)*100:.3f} cm')


# ════════════════════════════════════════════════════════════════
# 1. 속도·가속도·급기동 분포
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Trajectory Physics Distribution (Train, N=10000)', fontsize=14)

def hist_ax(ax, data, title, xlabel, color='steelblue', bins=80):
    ax.hist(data, bins=bins, color=color, edgecolor='none', alpha=0.8)
    ax.axvline(np.median(data), color='red', lw=1.5, label=f'median={np.median(data):.3f}')
    ax.axvline(np.percentile(data,95), color='orange', lw=1.5, ls='--',
               label=f'P95={np.percentile(data,95):.3f}')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.legend(fontsize=8)

hist_ax(axes[0,0], mean_sp,  'Mean Speed [m/s]',       'm/s')
hist_ax(axes[0,1], max_sp,   'Max Speed [m/s]',        'm/s', 'darkorange')
hist_ax(axes[0,2], mean_ac,  'Mean |Accel| [m/s²]',    'm/s²', 'seagreen')
hist_ax(axes[1,0], max_ac,   'Max |Accel| [m/s²]',     'm/s²', 'firebrick')
hist_ax(axes[1,1], mean_turn,'Mean Turn Angle [deg/step]','deg', 'purple')
hist_ax(axes[1,2], straight, 'Straightness (net/path)', '', 'teal')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '01_physics_dist.png'), dpi=120)
plt.close()
print('saved: 01_physics_dist.png')


# ════════════════════════════════════════════════════════════════
# 2. Kalman 오차 vs 물리량: 어떤 상황에서 Kalman이 틀리나?
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Kalman Error vs Physics Features', fontsize=14)

pairs = [
    (mean_sp,  'Mean Speed [m/s]',        'steelblue'),
    (max_ac,   'Max Accel [m/s²]',        'firebrick'),
    (mean_turn,'Mean Turn [deg]',          'purple'),
    (straight, 'Straightness',             'teal'),
    (max_jk,   'Max Jerk [m/s³]',         'darkorange'),
    (max_sp,   'Max Speed [m/s]',         'seagreen'),
]

for ax, (feat, xlabel, c) in zip(axes.flat, pairs):
    # 100분위로 bin해서 평균 오차 플롯
    pcts = np.percentile(feat, np.linspace(0,100,21))
    bins = np.digitize(feat, pcts[1:-1])
    bin_centers, bin_means = [], []
    for b in range(20):
        m = feat[bins==b]
        e = kal_err[bins==b]
        if len(e):
            bin_centers.append(m.mean())
            bin_means.append(e.mean()*100)
    ax.bar(range(len(bin_means)), bin_means, color=c, alpha=0.7)
    ax.set_xlabel(f'{xlabel} (low→high quantile)')
    ax.set_ylabel('Mean Kalman Error (cm)')
    ax.set_title(xlabel)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '02_kalman_err_vs_physics.png'), dpi=120)
plt.close()
print('saved: 02_kalman_err_vs_physics.png')


# ════════════════════════════════════════════════════════════════
# 3. 잔차(Residual) 분포 분석: 모델이 배워야 할 것
# ════════════════════════════════════════════════════════════════
def yaw_angle(v_vec): return np.arctan2(v_vec[:,1], v_vec[:,0])
def rotate_xy(vec, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([vec[:,0]*c + vec[:,1]*s,
                    -vec[:,0]*s + vec[:,1]*c,
                     vec[:,2]], axis=-1)

vl       = (X_train[:,-1] - X_train[:,-2]) / DT
theta    = yaw_angle(vl)
residual = y_train - kal_pred                          # Kalman 잔차 (world frame)
res_rot  = rotate_xy(residual, theta)                  # 회전 좌표계 잔차

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Kalman Residual Distribution (Rotated Body Frame)', fontsize=14)

for i, (ax, lbl) in enumerate(zip(axes[0], ['Forward (x)', 'Left (y)', 'Up (z)'])):
    ax.hist(res_rot[:,i]*100, bins=100, color=['steelblue','seagreen','firebrick'][i],
            alpha=0.8, edgecolor='none')
    mu, sig = res_rot[:,i].mean()*100, res_rot[:,i].std()*100
    ax.axvline(mu, color='red', lw=1.5, label=f'μ={mu:.2f}cm')
    ax.axvline(mu+sig, color='orange', ls='--', lw=1, label=f'σ={sig:.2f}cm')
    ax.axvline(mu-sig, color='orange', ls='--', lw=1)
    ax.set_xlabel(f'Residual {lbl} (cm)'); ax.legend(fontsize=8)
    ax.set_title(f'Body-frame {lbl} residual')

# 잔차 크기 vs Kalman 히트 여부
res_mag = np.linalg.norm(res_rot, axis=-1)
axes[1,0].hist(res_mag[kal_hit==1]*100, bins=60, alpha=0.7, label='Kalman Hit', color='green')
axes[1,0].hist(res_mag[kal_hit==0]*100, bins=60, alpha=0.7, label='Kalman Miss', color='red')
axes[1,0].set_xlabel('Residual magnitude (cm)')
axes[1,0].set_title('Residual size: hit vs miss')
axes[1,0].legend()

# 잔차 x-y 산점도 (체축)
axes[1,1].hexbin(res_rot[:,0]*100, res_rot[:,1]*100, gridsize=60,
                 cmap='Blues', mincnt=1)
axes[1,1].set_xlabel('Forward residual (cm)')
axes[1,1].set_ylabel('Left residual (cm)')
axes[1,1].set_title('Body-frame residual scatter (x-y)')
circle = plt.Circle((0,0), 1.0, fill=False, color='red', lw=2, label='1cm')
axes[1,1].add_patch(circle)
axes[1,1].set_aspect('equal'); axes[1,1].legend()

# 잔차 정규성 검정 (Q-Q plot)
from scipy.stats import probplot
probplot(res_rot[:,0]*100, plot=axes[1,2])
axes[1,2].set_title('Q-Q plot (Forward residual) — Normal?')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '03_residual_dist.png'), dpi=120)
plt.close()
print('saved: 03_residual_dist.png')


# ════════════════════════════════════════════════════════════════
# 4. 궤적 유형 클러스터링: 어떤 비행 패턴이 있나?
# ════════════════════════════════════════════════════════════════
feat_cluster = np.column_stack([
    (mean_sp - mean_sp.mean()) / mean_sp.std(),
    (max_ac  - max_ac.mean())  / max_ac.std(),
    (mean_turn - mean_turn.mean()) / mean_turn.std(),
    (straight  - straight.mean())  / straight.std(),
])

pca = PCA(n_components=2)
feat_2d = pca.fit_transform(feat_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels_k = kmeans.fit_predict(feat_cluster)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Trajectory Clustering (K=5)', fontsize=13)

colors_k = ['steelblue','firebrick','seagreen','purple','darkorange']
for k in range(5):
    mask = labels_k == k
    axes[0].scatter(feat_2d[mask,0], feat_2d[mask,1],
                    s=5, alpha=0.4, c=colors_k[k], label=f'C{k}(n={mask.sum()})')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
axes[0].set_title('PCA of trajectory features')
axes[0].legend(markerscale=3, fontsize=8)

# 클러스터별 Kalman 오차
kal_by_cluster = [kal_err[labels_k==k]*100 for k in range(5)]
axes[1].boxplot(kal_by_cluster, labels=[f'C{k}' for k in range(5)],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
axes[1].set_ylabel('Kalman Error (cm)')
axes[1].set_title('Kalman Error by Cluster')
axes[1].axhline(1.0, color='red', ls='--', lw=1.5, label='1cm hit threshold')
axes[1].legend()

# 클러스터별 평균 물리량
cluster_stats = pd.DataFrame({
    'mean_speed': [mean_sp[labels_k==k].mean() for k in range(5)],
    'max_accel':  [max_ac[labels_k==k].mean() for k in range(5)],
    'mean_turn':  [mean_turn[labels_k==k].mean() for k in range(5)],
    'straightness':[straight[labels_k==k].mean() for k in range(5)],
    'kal_hit':    [kal_hit[labels_k==k].mean() for k in range(5)],
}, index=[f'C{k}' for k in range(5)])

print('\n=== Cluster Statistics ===')
print(cluster_stats.to_string())

x_pos = np.arange(5)
w = 0.2
for i, (col, c) in enumerate(zip(['mean_speed','max_accel','mean_turn','kal_hit'],
                                  ['steelblue','firebrick','purple','seagreen'])):
    axes[2].bar(x_pos + i*w,
                cluster_stats[col] / cluster_stats[col].max(),
                width=w, label=col, color=c, alpha=0.8)
axes[2].set_xticks(x_pos + 1.5*w)
axes[2].set_xticklabels([f'C{k}' for k in range(5)])
axes[2].set_ylabel('Normalized value')
axes[2].set_title('Cluster profiles (normalized)')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '04_clustering.png'), dpi=120)
plt.close()
print('saved: 04_clustering.png')


# ════════════════════════════════════════════════════════════════
# 5. BiGRU 필요성 분석: 과거 시점 정보가 미래 예측에 도움이 되나?
#    → 초반(t=0~3) 정보와 후반(t=7~10) 정보의 예측력 비교
# ════════════════════════════════════════════════════════════════
print('\n=== BiGRU 필요성: 시간축 예측력 분석 ===')

# 각 시점의 속도로 단순 extrapolation 오차 비교
extrap_errors = []
for t in range(10):
    v_t = (X_train[:,t+1] - X_train[:,t]) / DT if t < 10 else \
          (X_train[:,10] - X_train[:,9]) / DT
    pred_t = X_train[:,t] + v_t * (T_PRED + (10 - t) * DT)
    err_t  = np.linalg.norm(pred_t - y_train, axis=-1)
    extrap_errors.append(err_t.mean() * 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('BiGRU Necessity Analysis', fontsize=13)

axes[0].plot(range(10), extrap_errors, 'o-', color='steelblue', lw=2)
axes[0].set_xlabel('Time step (0=earliest, 9=latest)')
axes[0].set_ylabel('Mean extrapolation error (cm)')
axes[0].set_title('Single-step velocity extrapolation error\nby observation time step')
axes[0].axvline(9, color='red', ls='--', lw=1.5, label='Last step (t=0ms)')
axes[0].legend()

# 초기 속도 방향 vs 최종 속도 방향 일치도
v_early = (X_train[:,2] - X_train[:,0]) / (2*DT)
v_late  = (X_train[:,10] - X_train[:,9]) / DT

cos_sim = (v_early * v_late).sum(-1) / (
    np.linalg.norm(v_early, axis=-1) * np.linalg.norm(v_late, axis=-1) + 1e-12)
cos_sim = cos_sim.clip(-1, 1)

axes[1].hist(np.degrees(np.arccos(cos_sim)), bins=80, color='purple', alpha=0.8)
axes[1].set_xlabel('Angle between early & late velocity (deg)')
axes[1].set_ylabel('Count')
axes[1].set_title('Direction change across observation window\n'
                  '(0°=straight, 90°=turn, 180°=reverse)')
consistent = (np.abs(np.degrees(np.arccos(cos_sim))) < 30).mean()
axes[1].axvline(30, color='red', ls='--', lw=1.5, label=f'<30° (straight): {consistent:.1%}')
axes[1].legend()

# 초기 정보 유용성: early velocity로 예측하면 얼마나 잘 되나?
pred_early = X_train[:,-1] + v_early * T_PRED
pred_late  = X_train[:,-1] + v_late  * T_PRED
err_early = np.linalg.norm(pred_early - y_train, axis=-1)
err_late  = np.linalg.norm(pred_late  - y_train, axis=-1)
improvement = err_early - err_late  # 양수 = late가 더 좋음

axes[2].scatter(err_late*100, improvement*100, s=5, alpha=0.2, c='steelblue')
axes[2].axhline(0, color='red', lw=1.5, label='No improvement from early info')
axes[2].set_xlabel('Late velocity prediction error (cm)')
axes[2].set_ylabel('Improvement using early vs late (cm)\n(positive = early info hurts)')
axes[2].set_title('Does early trajectory context help?\n'
                  f'(Cases where early info helps: {(improvement<0).mean():.1%})')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '05_bigru_necessity.png'), dpi=120)
plt.close()
print('saved: 05_bigru_necessity.png')


# ════════════════════════════════════════════════════════════════
# 6. Activation function 적합성 분석: Mish vs GELU vs Swish
#    → 잔차 분포의 특성으로 유추
# ════════════════════════════════════════════════════════════════
print('\n=== Activation Function 분석: 잔차 분포 꼬리 ===')

res_mag_cm = np.linalg.norm(residual, axis=-1) * 100
kurtosis   = stats.kurtosis(res_rot[:,0])
skewness   = stats.skew(res_rot[:,0])
print(f'Forward residual: kurtosis={kurtosis:.2f}  skewness={skewness:.2f}')
print(f'  (>3 kurtosis → heavy tail → smooth activation 유리)')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Residual distribution shape → Activation function suitability', fontsize=12)

# 로그스케일 히스토그램 (꼬리 분포 확인)
for i, (ax, lbl, c) in enumerate(zip(axes, ['Forward','Left','Up'],
                                       ['steelblue','seagreen','firebrick'])):
    vals = res_rot[:,i]*100
    ax.hist(vals, bins=150, color=c, alpha=0.8, edgecolor='none', density=True)
    # 정규분포 피팅
    mu_fit, sig_fit = stats.norm.fit(vals)
    x_fit = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(x_fit, stats.norm.pdf(x_fit, mu_fit, sig_fit), 'r--', lw=2, label='Normal fit')
    ax.set_yscale('log')
    ax.set_xlabel(f'{lbl} residual (cm)'); ax.set_ylabel('Density (log)')
    kt = stats.kurtosis(vals)
    sk = stats.skew(vals)
    ax.set_title(f'{lbl}  kurt={kt:.1f}  skew={sk:.2f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '06_residual_tails.png'), dpi=120)
plt.close()
print('saved: 06_residual_tails.png')


# ════════════════════════════════════════════════════════════════
# 7. 특징 상관분석: 어떤 feature가 Kalman 오차와 가장 연관?
# ════════════════════════════════════════════════════════════════
feature_df = pd.DataFrame({
    'mean_speed':  mean_sp,
    'max_speed':   max_sp,
    'mean_accel':  mean_ac,
    'max_accel':   max_ac,
    'max_jerk':    max_jk,
    'mean_turn':   mean_turn,
    'max_turn':    max_turn,
    'straightness':straight,
    'path_length': path_l,
    'net_disp':    net_d,
    'kalman_err':  kal_err,
    'kal_hit':     kal_hit,
})

corr = feature_df.corr()['kalman_err'].drop('kalman_err').sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors_c = ['firebrick' if c > 0 else 'steelblue' for c in corr]
corr.plot(kind='barh', ax=ax, color=colors_c)
ax.axvline(0, color='black', lw=0.8)
ax.set_title('Feature correlation with Kalman Error\n(red=positive, blue=negative)')
ax.set_xlabel('Pearson r')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, '07_feature_correlation.png'), dpi=120)
plt.close()
print('saved: 07_feature_correlation.png')


# ════════════════════════════════════════════════════════════════
# 8. 텍스트 요약 리포트
# ════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('=== EDA SUMMARY ===')
print('='*60)

print(f'\n[속도 분포]')
print(f'  Mean speed: {mean_sp.mean():.3f} m/s  (median {np.median(mean_sp):.3f})')
print(f'  Max speed : {max_sp.mean():.3f} m/s  (P95 {np.percentile(max_sp,95):.3f})')
print(f'  → 속도 범위: {mean_sp.min():.3f} ~ {mean_sp.max():.3f} m/s')

print(f'\n[가속도 분포]')
print(f'  Mean |accel|: {mean_ac.mean():.2f} m/s²  (P95 {np.percentile(mean_ac,95):.2f})')
print(f'  Max  |accel|: {max_ac.mean():.2f} m/s²  (P99 {np.percentile(max_ac,99):.2f})')

print(f'\n[방향 변화]')
print(f'  Mean turn/step: {mean_turn.mean():.1f}°  (P95 {np.percentile(mean_turn,95):.1f}°)')
print(f'  Max  turn/step: {max_turn.mean():.1f}°   (P99 {np.percentile(max_turn,99):.1f}°)')
straight_pct = (mean_turn < 10).mean()
print(f'  직선비행 (<10°/step): {straight_pct:.1%}')
maneuver_pct = (max_turn > 60).mean()
print(f'  급기동 (>60° turn)  : {maneuver_pct:.1%}')

print(f'\n[Kalman 성능]')
print(f'  R-Hit@1cm: {kal_hit.mean():.4f}')
print(f'  Median err: {np.median(kal_err)*100:.3f} cm')
print(f'  P90 err   : {np.percentile(kal_err,90)*100:.3f} cm')

print(f'\n[잔차 통계 (body frame)]')
for i, ax_name in enumerate(['forward','left','up']):
    print(f'  {ax_name}: μ={res_rot[:,i].mean()*100:.3f}cm  '
          f'σ={res_rot[:,i].std()*100:.3f}cm  '
          f'kurt={stats.kurtosis(res_rot[:,i]):.2f}')

print(f'\n[BiGRU 필요성]')
print(f'  방향 일관성 (<30° 변화): {consistent:.1%}')
print(f'  초기 정보가 도움이 되는 케이스: {(improvement<0).mean():.1%}')
print(f'  최초~최후 시점 외삽 오차 차이: {extrap_errors[0]:.2f} vs {extrap_errors[9]:.2f} cm')

print(f'\n[출력 파일]')
for f in sorted(os.listdir(OUT_DIR)):
    print(f'  eda_out/{f}')
print('\nDONE.')
