import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
s = optuna.load_study(study_name='v6_fine', storage='sqlite:///cache/optuna_v6.db')
trials = [t for t in s.trials if t.state.name == 'COMPLETE']
print(f'Completed: {len(trials)}/50  Best: {s.best_value:.4f}')
top5 = sorted(trials, key=lambda t: t.value, reverse=True)[:5]
for i, t in enumerate(top5):
    ns = t.params['noise_std'] * 1000
    gm = t.params['focal_gamma']
    lr = t.params['lr']
    lf = t.params['lF']
    lw = t.params['lW']
    dr = t.params['dropout']
    mc = t.params['main_clip']
    print(f'  {i+1}. {t.value:.4f}  noise={ns:.1f}mm  gamma={gm}  lr={lr:.5f}  lF={lf}  lW={lw}  dropout={dr}  clip={mc}')
