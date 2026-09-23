# -*- coding: utf-8 -*-
"""EXP-20260923-P2-01: Classical Scaling Law Estimation on B1 (Pythia Trajectories).

Implements:
  - M2-EQ01 baseline fit on B1 (1176 points, 8 models)
  - Training interval comparison (all, D>=1B, D>=5B) per P2-04
  - Leave-One-Model-Out Cross Validation (LOMOCV) per P2-03
  - Bootstrap 95% Confidence Intervals (B=1000)
  - Figure fig_p2_classical_scaling.pdf + table table_p2_scaling_parameters.csv

Run:
  & "D:\Program Files\CondaEnvs\AIOPS\python.exe" src/problem02/exp01_baseline_scaling.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from p2_common import (
    classical_scaling_law, save_figure_with_caption,
    DIR_B, RES_P2, OUT_FIGS, OUT_TABS, SEED
)

t0 = time.time()
print("=" * 70)
print("EXP-20260923-P2-01: Classical Scaling Law Fitting on B1")
print("=" * 70)

# 1. Load B1 Dataset
b1_path = os.path.join(DIR_B, "pythia_training_log_existing.csv")
df_b1 = pd.read_csv(b1_path)
print(f"[1] Loaded B1: {b1_path}, shape = {df_b1.shape}")
models = sorted(df_b1['N_params_B'].unique())
print(f"    8 Model scales (Billion params): {models}")

# 2. Fit Classical Scaling Law across 3 Data Cutoffs
bounds = ([0.5, 1e-4, 1e-4, 1e-4, 1e-4], [3.0, 10.0, 2.0, 10.0, 2.0])
p0 = [1.69, 0.35, 0.34, 1.24, 0.28]

cutoffs = [
    ("all", 0.0),
    ("D>=1B", 1.0),
    ("D>=5B", 5.0)
]

cutoff_results = {}
for name, d_min in cutoffs:
    sub = df_b1[df_b1['D_tokens_B'] >= d_min].copy()
    X = (sub['N_params_B'].values, sub['D_tokens_B'].values)
    y = sub['val_loss'].values
    popt, pcov = curve_fit(
        lambda X, E, A, a, B, b: classical_scaling_law(X[0], X[1], E, A, a, B, b),
        X, y, p0=p0, bounds=bounds, method='trf', maxfev=20000
    )
    y_pred = classical_scaling_law(X[0], X[1], *popt)
    r2 = float(r2_score(y, y_pred))
    rmse = float(root_mean_squared_error(y, y_pred))
    mae = float(mean_absolute_error(y, y_pred))
    mape = float(np.mean(np.abs((y - y_pred) / y)) * 100)
    perr = np.sqrt(np.diag(pcov))

    cutoff_results[name] = {
        'n_samples': len(sub),
        'E': float(popt[0]), 'E_se': float(perr[0]),
        'A': float(popt[1]), 'A_se': float(perr[1]),
        'alpha': float(popt[2]), 'alpha_se': float(perr[2]),
        'B': float(popt[3]), 'B_se': float(perr[3]),
        'beta': float(popt[4]), 'beta_se': float(perr[4]),
        'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE_pct': mape
    }
    print(f"[2] Cutoff {name} (n={len(sub)}): E={popt[0]:.4f}, A={popt[1]:.4f}, a={popt[2]:.4f}, B={popt[3]:.4f}, b={popt[4]:.4f} | R2={r2:.5f}, RMSE={rmse:.5f}")

# Adopt all-data parameters as baseline (or D>=1B, both are identical to 4 decimals)
base_params = cutoff_results['all']
E_fit, A_fit, a_fit, B_fit, b_fit = base_params['E'], base_params['A'], base_params['alpha'], base_params['B'], base_params['beta']

# 3. Leave-One-Model-Out Cross Validation (LOMOCV)
print("\n[3] Running Leave-One-Model-Out Cross Validation (8 folds)...")
loocv_records = []
all_y_true, all_y_pred = [], []

for left_out_model in models:
    train_mask = df_b1['N_params_B'] != left_out_model
    test_mask = df_b1['N_params_B'] == left_out_model

    tr = df_b1[train_mask]
    te = df_b1[test_mask]

    X_tr = (tr['N_params_B'].values, tr['D_tokens_B'].values)
    y_tr = tr['val_loss'].values
    X_te = (te['N_params_B'].values, te['D_tokens_B'].values)
    y_te = te['val_loss'].values

    popt_cv, _ = curve_fit(
        lambda X, E, A, a, B, b: classical_scaling_law(X[0], X[1], E, A, a, B, b),
        X_tr, y_tr, p0=p0, bounds=bounds, method='trf', maxfev=20000
    )
    pred_te = classical_scaling_law(X_te[0], X_te[1], *popt_cv)

    rmse_fold = float(root_mean_squared_error(y_te, pred_te))
    mae_fold = float(mean_absolute_error(y_te, pred_te))
    r2_fold = float(r2_score(y_te, pred_te))

    all_y_true.extend(y_te)
    all_y_pred.extend(pred_te)

    loocv_records.append({
        'left_out_model_B': float(left_out_model),
        'n_test': len(te),
        'RMSE': rmse_fold,
        'MAE': mae_fold,
        'R2': r2_fold,
        'fitted_E': float(popt_cv[0]),
        'fitted_alpha': float(popt_cv[2]),
        'fitted_beta': float(popt_cv[4])
    })
    print(f"    Left out N={left_out_model:8.4f}B: RMSE={rmse_fold:.5f}, MAE={mae_fold:.5f}, R2={r2_fold:.5f}")

loocv_overall_rmse = float(root_mean_squared_error(all_y_true, all_y_pred))
loocv_overall_r2 = float(r2_score(all_y_true, all_y_pred))
print(f"    ==> Overall LOOCV: RMSE = {loocv_overall_rmse:.5f}, R2 = {loocv_overall_r2:.5f}")

# 4. Bootstrap 95% Confidence Intervals (B=1000)
print("\n[4] Running Trajectory-Stratified Bootstrap (B=1000)...")
B_BOOT = 1000
boot_params = []
rng = np.random.default_rng(SEED)

for b in range(B_BOOT):
    # Resample models with replacement to respect trajectory dependence
    sampled_models = rng.choice(models, size=len(models), replace=True)
    boot_dfs = [df_b1[df_b1['N_params_B'] == m] for m in sampled_models]
    boot_df = pd.concat(boot_dfs, ignore_index=True)

    X_b = (boot_df['N_params_B'].values, boot_df['D_tokens_B'].values)
    y_b = boot_df['val_loss'].values

    try:
        popt_b, _ = curve_fit(
            lambda X, E, A, a, B, b: classical_scaling_law(X[0], X[1], E, A, a, B, b),
            X_b, y_b, p0=p0, bounds=bounds, method='trf', maxfev=5000
        )
        boot_params.append(popt_b)
    except Exception:
        continue

boot_params = np.array(boot_params)
param_names = ['E', 'A', 'alpha', 'B', 'beta']
ci_95 = {}
for idx, name in enumerate(param_names):
    vals = boot_params[:, idx]
    ci_95[name] = {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'p2_5': float(np.percentile(vals, 2.5)),
        'p97_5': float(np.percentile(vals, 97.5))
    }
    print(f"    Bootstrap {name:5s}: mean={ci_95[name]['mean']:.4f}, 95% CI = [{ci_95[name]['p2_5']:.4f}, {ci_95[name]['p97_5']:.4f}]")

# 5. Save Output Tables & Summary JSON
summary = {
    'experiment_id': 'EXP-20260923-P2-01',
    'date': '2026-09-23',
    'dataset': 'B1: pythia_training_log_existing.csv',
    'n_total_samples': len(df_b1),
    'n_models': len(models),
    'cutoffs': cutoff_results,
    'baseline_fit': {
        'E': E_fit, 'A': A_fit, 'alpha': a_fit, 'B': B_fit, 'beta': b_fit,
        'R2': base_params['R2'], 'RMSE': base_params['RMSE']
    },
    'loocv': {
        'overall_rmse': loocv_overall_rmse,
        'overall_r2': loocv_overall_r2,
        'folds': loocv_records
    },
    'bootstrap_95ci': ci_95,
    'runtime_sec': round(time.time() - t0, 2)
}

with open(os.path.join(RES_P2, "exp01_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Format Table table_p2_scaling_parameters.csv
tab_rows = []
for idx, p in enumerate(param_names):
    tab_rows.append({
        'parameter': p,
        'estimate': round(base_params[p], 5),
        'std_error': round(base_params[p + '_se'], 5),
        'ci_95_lower': round(ci_95[p]['p2_5'], 5),
        'ci_95_upper': round(ci_95[p]['p97_5'], 5),
        'unit': 'Nats' if p == 'E' else ('dimensionless' if 'alpha' in p or 'beta' in p else 'Nats*unit^pow'),
        'description': '不可约损失下限 E' if p == 'E' else (
            '模型规模系数 A' if p == 'A' else (
                '模型规模幂指数 alpha' if p == 'alpha' else (
                    '数据量系数 B' if p == 'B' else '数据量幂指数 beta'
                )
            )
        )
    })
tab_df = pd.DataFrame(tab_rows)
tab_path = os.path.join(OUT_TABS, "table_p2_scaling_parameters.csv")
tab_df.to_csv(tab_path, index=False, encoding="utf-8-sig")
print(f"[5] Saved parameter table to: {tab_path}")

# 6. Generate High-DPI Figure fig_p2_classical_scaling.pdf
print("\n[6] Generating publication figure fig_p2_classical_scaling.pdf...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))

# Panel 1: Trajectory fits on log-scale
colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(models)))
D_grid = np.geomspace(0.1, 320, 200)

for idx, m in enumerate(models):
    sub = df_b1[df_b1['N_params_B'] == m]
    ax1.scatter(sub['D_tokens_B'], sub['val_loss'], color=colors[idx], alpha=0.45, s=12, label=f'{m:.3f}B' if m < 1 else f'{m:.1f}B')
    L_curve = classical_scaling_law(m, D_grid, E_fit, A_fit, a_fit, B_fit, b_fit)
    ax1.plot(D_grid, L_curve, color=colors[idx], linestyle='-', linewidth=1.8)

ax1.set_xscale('log')
ax1.set_xlabel("Training Tokens D (Billion Tokens, Log Scale)", labelpad=6)
ax1.set_ylabel("Validation Cross-Entropy Loss (Nats)", labelpad=6)
ax1.set_xlim(0.08, 350)
ax1.set_ylim(1.95, 5.35)  # Extended upper limit so curves never touch the legend
ax1.legend(title="Model Scale (N)", ncol=4, loc="upper right", framealpha=0.92, fontsize=8.5, columnspacing=1.0)

# Panel 2: Predicted vs Actual Scatter across All Models with Residual Density
y_all = df_b1['val_loss'].values
y_pred_all = classical_scaling_law(df_b1['N_params_B'].values, df_b1['D_tokens_B'].values, E_fit, A_fit, a_fit, B_fit, b_fit)
residuals = y_all - y_pred_all

sc = ax2.scatter(y_pred_all, y_all, c=np.abs(residuals), cmap='plasma', alpha=0.7, s=16)
cbar = plt.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label("Absolute Error |Residual| (Nats)", labelpad=6)

lims = [1.95, 4.85]
ax2.plot(lims, lims, 'k--', linewidth=1.4, alpha=0.8,
         label="Ideal y = x")
ax2.set_xlim(lims)
ax2.set_ylim(lims)
ax2.set_xlabel("Predicted Loss L_pred (Nats)", labelpad=6)
ax2.set_ylabel("Observed Loss L_obs (Nats)", labelpad=6)
ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.8)

fig.tight_layout()

fig_path = os.path.join(OUT_FIGS, "fig_p2_classical_scaling.pdf")
caption = """# 图注：Pythia 经典双变量标度律拟合与留一模型泛化检验 (fig_p2_classical_scaling)

- **数据来源**：附件 B1 (`pythia_training_log_existing.csv`)，包含 8 种参数规格（0.0705B 至 11.9658B）、共计 1,176 个连续训练检查点观测。
- **左图说明**：各模型规模下验证损失随累计 Token 数量 $D$ 的轨迹；散点为检查点观测，实线为基于全体 B1 数据拟合的标度律曲线。横轴采用对数刻度。
- **右图说明**：全体 B1 检查点的预测损失与观测损失散点图；对角虚线为参考线 $y=x$，颜色表示绝对残差大小。
- **说明**：参数估计、留一模型轨迹验证和残差指标见 `EXP-20260923-P2-01` 记录及配套结果表。
"""

fig1_meta = {
    "figure_id": "fig_p2_classical_scaling",
    "experiment_id": "EXP-20260923-P2-01",
    "problem": "problem-02",
    "source_data": "Attachment B1 (pythia_training_log_existing.csv, 1,176 points)",
    "script": "src/problem02/exp01_baseline_scaling.py",
    "code_version": "v1.1",
    "format": "pdf/png",
    "dpi": 300,
    "width_mm": 280,
    "height_mm": 115,
    "title_in_figure": False,
    "caption": "See companion fig_p2_classical_scaling.pdf.caption.md",
    "panels": [
        {
            "panel": "(a) Left",
            "description": "Validation Loss vs Training Tokens D across 8 Pythia model scales (0.0705B to 11.9658B, n=1,176)",
            "x_axis": "Training Tokens D (Billion Tokens, Log Scale)",
            "y_axis": "Validation Cross-Entropy Loss (Nats)",
            "curves": "Chinchilla Power Law Fit vs Observed Pythia checkpoints across 8 model scales"
        },
        {
            "panel": "(b) Right",
            "description": "Predicted Loss vs Observed Loss across all 1,176 Pythia checkpoints with Absolute Error colormap and ideal y = x line",
            "x_axis": "Predicted Loss L_pred (Nats)",
            "y_axis": "Observed Loss L_obs (Nats)",
            "curves": "Scatter points colored by absolute error with diagonal reference line y = x"
        }
    ],
    "parameters": {
        "E": round(E_fit, 4),
        "A": round(A_fit, 4),
        "alpha": round(a_fit, 4),
        "B": round(B_fit, 4),
        "beta": round(b_fit, 4)
    },
    "metrics": {
        "n_samples": len(df_b1),
        "R2": round(base_params['R2'], 5),
        "RMSE": round(base_params['RMSE'], 5),
        "LOOCV_R2": round(loocv_overall_r2, 5),
        "LOOCV_RMSE": round(loocv_overall_rmse, 5)
    }
}

save_figure_with_caption(fig, fig_path, caption, metadata_dict=fig1_meta)
plt.close(fig)

print(f"\n[Done] EXP-P2-01 completed in {time.time() - t0:.2f} s.")
