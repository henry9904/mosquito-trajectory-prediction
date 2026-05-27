import pandas as pd, numpy as np, glob, os

SUB_DIR = r'D:\hyeon\공부\mosquitoes\submissions'
files = sorted(glob.glob(os.path.join(SUB_DIR, '*.csv')))
for f in files:
    sub = pd.read_csv(f)
    print(f'\n[{os.path.basename(f)}]')
    print(f'  행 수: {len(sub)}, 결측: {sub.isnull().sum().sum()}')
    print(f'  x: [{sub.x.min():.4f}, {sub.x.max():.4f}]')
    print(f'  y: [{sub.y.min():.4f}, {sub.y.max():.4f}]')
    print(f'  z: [{sub.z.min():.4f}, {sub.z.max():.4f}]')
    print(sub.head(3).to_string())
