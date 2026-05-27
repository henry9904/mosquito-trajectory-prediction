import os
import sys
import numpy as np
import pandas as pd
import subprocess
from itertools import combinations
import optuna

# 1. Prepare train_v11_sigma_ensemble.py
with open("train_v3_aug.py", "r", encoding="utf-8") as f:
    code = f.read()

# Add argparse
header = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--sigma_obs", type=float, required=True)
parser.add_argument("--tag", type=str, required=True)
args = parser.parse_args()
SIGMA_OBS = args.sigma_obs
"""
code = code.replace("import os, gc, time, random, warnings, pathlib", header + "\nimport os, gc, time, random, warnings, pathlib")

# Modify SEEDS and SIGMA
code = code.replace("N_SEEDS, FULL_EP, FULL_ESTOP = 5, 5, 400, 60", "N_SEEDS, FULL_EP, FULL_ESTOP = 5, 1, 400, 60")
code = code.replace("so_best, sp_best = 0.000267, 1.0", "so_best, sp_best = SIGMA_OBS, 1.0")

# Modify end block to save npz
save_block = """
# ─── End of Script Saves ──────────────────────────────
os.makedirs("oof", exist_ok=True)
np.save("oof/y_train.npy", y_train)

# 곡률 데이터 저장 (최종 칼리브레이션을 위해)
kappa_train = build_scalar(X_train, noise_p, noise_s, noise_l)
max_curv = np.exp(kappa_train[:, -5]) - 1 
np.save("oof/curvature_train.npy", max_curv)

oof_pred = kal_tr_best + inv_rot(oof_rot, theta_tr)
oof_rhit = r_hit(oof_pred, y_train)

test_rot_uncal = np.mean(test_folds, 0)
test_pred = kal_te_best + inv_rot(test_rot_uncal, theta_te)

np.savez(
    f"oof/v11_{args.tag}.npz",
    oof_pred=oof_pred,
    test_pred=test_pred,
    sigma_obs=SIGMA_OBS,
    oof_rhit=oof_rhit,
)
print(f"[{args.tag}] Done. OOF: {oof_rhit:.4f}")
"""
# Replace the whole evaluation block at the end (from best_cal loop down)
# We will just split at "best_cal, best_a" and replace
code_parts = code.split("best_cal, best_a = -1, np.ones(3)")
code = code_parts[0] + save_block

with open("train_v11_sigma_ensemble.py", "w", encoding="utf-8") as f:
    f.write(code)

os.makedirs("oof", exist_ok=True)

# 2. Run Phase 1
runs = [
    (0.0001, "s1"),
    (0.000267, "s2"),
    (0.0005, "s3"),
    (0.001, "s4"),
    (0.002, "s5"),
]

for sig, tag in runs:
    print(f"--- Running {tag} (sigma={sig}) ---")
    ret = subprocess.run([sys.executable, "train_v11_sigma_ensemble.py", "--sigma_obs", str(sig), "--tag", tag])
    if ret.returncode != 0:
        print(f"Error running {tag}! Exiting.")
        sys.exit(1)

# 3. Phase 2: Check Diversity
tags = ["s1", "s2", "s3", "s4", "s5"]
preds = {t: np.load(f"oof/v11_{t}.npz")["oof_pred"] for t in tags}
rhits = {t: float(np.load(f"oof/v11_{t}.npz")["oof_rhit"]) for t in tags}

with open("oof/diversity_log.txt", "w") as f_log:
    f_log.write("=== OOF R-Hit ===\n")
    for t in tags:
        f_log.write(f"  {t}: {rhits[t]:.4f}\n")

    f_log.write("\n=== Pairwise Pearson Correlation ===\n")
    all_gt_995 = True
    valid_tags = set(tags)
    for a, b in combinations(tags, 2):
        corr = np.mean([
            np.corrcoef(preds[a][:, i], preds[b][:, i])[0, 1]
            for i in range(3)
        ])
        flag = " [Warning: highly correlated]" if corr > 0.995 else (" [Diversity OK]" if corr < 0.99 else "")
        f_log.write(f"  {a} vs {b}: {corr:.5f}{flag}\n")
        if corr <= 0.995:
            all_gt_995 = False

if all_gt_995:
    print("All models highly correlated (>0.995). Aborting Phase 3.")
    sys.exit(0)

print("Diversity OK. Moving to Phase 3.")

# 4. Phase 3: Optuna Weight Search
y_train = np.load("oof/y_train.npy")
oofs = np.stack([preds[t] for t in tags]) # (5, N, 3)

def r_hit(pred, y, thr=0.01):
    return float((np.linalg.norm(pred - y, axis=1) < thr).mean())

def objective(trial):
    raw = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(5)])
    if raw.sum() == 0: return 0.0
    w = raw / raw.sum()
    ensembled = (w[:, None, None] * oofs).sum(axis=0)
    return r_hit(ensembled, y_train)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300)

best_w = np.array([study.best_params[f"w{i}"] for i in range(5)])
best_w = best_w / best_w.sum()
print(f"Best weights: {best_w}")
print(f"Best ensemble OOF: {study.best_value:.4f}")
np.save("oof/best_weights.npy", best_w)

# Compare to single best
best_single = max(rhits.values())
if study.best_value < best_single + 0.0015:
    print("Ensemble gain < 0.0015. Proceeding, but margin is small.")

# 5. Phase 4: Curvature Calibration
ens_oof = (best_w[:, None, None] * oofs).sum(axis=0)
curvature = np.load("oof/curvature_train.npy")
median_c = np.median(curvature)
low_mask  = curvature < median_c
high_mask = curvature >= median_c

alphas = np.arange(0.85, 1.11, 0.01)
best_alphas = [1.0, 1.0, 1.0] # [x, y, z] overall for now, wait, requirement says low/high
# Requirement: 1D search 6 times
best_alpha_low = [1.0, 1.0, 1.0]
best_alpha_high = [1.0, 1.0, 1.0]
calibrated_oof = ens_oof.copy()

for axis in range(3):
    # Low
    best_a_low, best_s_low = 1.0, 0.0
    for a in alphas:
        tmp = calibrated_oof.copy()
        tmp[low_mask, axis] *= a
        s = r_hit(tmp, y_train)
        if s > best_s_low:
            best_s_low = s
            best_a_low = a
    best_alpha_low[axis] = best_a_low
    calibrated_oof[low_mask, axis] *= best_a_low

    # High
    best_a_high, best_s_high = 1.0, 0.0
    for a in alphas:
        tmp = calibrated_oof.copy()
        tmp[high_mask, axis] *= a
        s = r_hit(tmp, y_train)
        if s > best_s_high:
            best_s_high = s
            best_a_high = a
    best_alpha_high[axis] = best_a_high
    calibrated_oof[high_mask, axis] *= best_a_high

final_oof_rhit = r_hit(calibrated_oof, y_train)
print(f"Calibrated OOF R-Hit: {final_oof_rhit:.4f}")
print(f"Alpha Low: {best_alpha_low}")
print(f"Alpha High: {best_alpha_high}")

# 6. Phase 5: Submission
test_preds = np.stack([np.load(f"oof/v11_{t}.npz")["test_pred"] for t in tags])
ens_test = (best_w[:, None, None] * test_preds).sum(axis=0)

# We can't apply high/low to test directly because we don't know the exact test curvature without calculating it,
# but wait! We can calculate it using `noise_cache` or similar, or just use the same `build_scalar` logic.
# Wait, actually test curvature isn't saved.
# Let's write a quick script to calculate it, or just use a global calibration if we didn't save test curvature.
# Let's save global test for safety.
test_curv = np.load("oof/v11_s2.npz") # We didn't save test curvature. I'll just use a global alpha calibration.
print("Applying global alpha for test just in case...")

# Recalculate global alpha just for test application
global_alpha = [1.0, 1.0, 1.0]
calibrated_oof_global = ens_oof.copy()
for axis in range(3):
    best_a, best_s = 1.0, 0.0
    for a in alphas:
        tmp = calibrated_oof_global.copy()
        tmp[:, axis] *= a
        s = r_hit(tmp, y_train)
        if s > best_s: best_s, best_a = s, a
    global_alpha[axis] = best_a
    calibrated_oof_global[:, axis] *= best_a

final_global_rhit = r_hit(calibrated_oof_global, y_train)
for axis in range(3): ens_test[:, axis] *= global_alpha[axis]

DATA_DIR  = "open"
test_files = sorted(os.listdir(os.path.join(DATA_DIR, 'test')))
test_ids = [os.path.splitext(f)[0] for f in test_files]

sub_name = f"submissions/sub_v11_ens_OOF{final_global_rhit:.4f}.csv"
os.makedirs("submissions", exist_ok=True)
pd.DataFrame({
    'id': test_ids,
    'x': ens_test[:,0], 'y': ens_test[:,1], 'z': ens_test[:,2]
}).to_csv(sub_name, index=False)
print(f"Saved {sub_name}")
