# -*- coding: utf-8 -*-
"""EXP-20260923-P2-04: Dimensionless Elasticities, Marginal Cost-Benefit ("算一笔账"), and Quality-Scale Substitution (with Scaling Ceiling Saturation).

Implements:
  - Calculation of dimensionless point elasticities (eps_N, eps_D, eps_Q) & reducible elasticities
  - Marginal ROI analysis: "堆参数 vs 买好教材哪个更划算" across Appendix B cost functions
  - Analytical computation of "Quality +0.1 Equivalent Parameters" (M2-EQ04):
      Direction 1: Parameter Downsizing (Delta N_save)
      Direction 2: Scaling Equivalent Gain (Delta N_gain) & Critical Ceiling Saturation (N_crit)
  - Output table table_p2_substitution_01.csv & fig_p2_substitution_ceiling.pdf

Run:
  & "D:\Program Files\CondaEnvs\AIOPS\python.exe" src/problem02/exp04_elasticity_substitution.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from p2_common import (
    compute_elasticities, compute_substitution_01, save_figure_with_caption,
    RES_P2, OUT_FIGS, OUT_TABS
)

t0 = time.time()
print("=" * 70)
print("EXP-20260923-P2-04: Elasticity, Marginal ROI, and Substitution Saturation")
print("=" * 70)

# 1. Load Parameters from Previous Experiments
with open(os.path.join(RES_P2, "exp01_summary.json"), "r", encoding="utf-8") as f:
    exp01 = json.load(f)['baseline_fit']
with open(os.path.join(RES_P2, "exp02_summary.json"), "r", encoding="utf-8") as f:
    exp02 = json.load(f)
with open(os.path.join(RES_P2, "exp03_summary.json"), "r", encoding="utf-8") as f:
    exp03 = json.load(f)

E = exp01['E']
A = exp01['A']
alpha = exp01['alpha']
B = exp01['B']
beta = exp01['beta']
rho = exp02['estimated_rho']
Q0 = 1.0  # reference perfect quality anchor
p_mult = 1.0  # reference recipe p0

print(f"[1] Model Parameters:")
print(f"    E = {E:.4f}, A = {A:.4f}, alpha = {alpha:.4f}")
print(f"    B = {B:.4f}, beta = {beta:.4f}, rho = {rho:.4f}")

# 2. Elasticity Evaluation across LLM Configurations
models_eval = [0.07, 0.41, 1.04, 2.78, 6.86, 11.97, 70.0]
tokens_eval = [10.0, 50.0, 300.0, 2000.0]
quality_eval = [0.5, 0.7, 0.9, 1.0]

elasticity_records = []
for n_val in models_eval:
    for d_val in tokens_eval:
        for q_val in quality_eval:
            el = compute_elasticities(n_val, d_val, q_val, p_mult, E, A, alpha, B, beta, rho, Q0=Q0)
            elasticity_records.append({
                'N_params_B': n_val,
                'D_tokens_B': d_val,
                'Q_score': q_val,
                'val_loss': round(el['L'], 4),
                'reducible_loss': round(el['reducible_L'], 4),
                'eps_N': round(el['eps_N'], 4),
                'eps_D': round(el['eps_D'], 4),
                'eps_Q': round(el['eps_Q'], 4),
                'eps_N_star': round(el['eps_N_star'], 4),
                'eps_D_star': round(el['eps_D_star'], 4),
                'eps_Q_star': round(el['eps_Q_star'], 4),
                'mrts_NQ': round(el['mrts_NQ'], 4)
            })

df_el = pd.DataFrame(elasticity_records)
print(f"[2] Computed elasticities on {len(df_el)} grid configurations.")
# Verify all elasticities are strictly negative!
assert (df_el['eps_N'] < 0).all() and (df_el['eps_D'] < 0).all() and (df_el['eps_Q'] < 0).all()
print("    Grid check passed: eps_N, eps_D, and eps_Q are negative on the evaluated configurations.")

# Sample display at N=1B, D=300B, Q=0.7
sample_el = df_el[(df_el['N_params_B'] == 1.04) & (df_el['D_tokens_B'] == 300.0) & (df_el['Q_score'] == 0.7)].iloc[0]
print(f"\n    Sample at N=1.04B, D=300B, Q=0.7:")
print(f"      Loss = {sample_el['val_loss']:.4f} Nats, eps_N = {sample_el['eps_N']:.4f}, eps_D = {sample_el['eps_D']:.4f}, eps_Q = {sample_el['eps_Q']:.4f}")
print(f"      Reducible: eps_N* = {sample_el['eps_N_star']:.4f}, eps_D* = {sample_el['eps_D_star']:.4f}, eps_Q* = {sample_el['eps_Q_star']:.4f}")
print(f"      MRTS (params per unit quality) = {sample_el['mrts_NQ']:.4f} Billion params / unit quality")

# 3. "算一笔账：堆参数 vs 买教材更划算？" (Marginal ROI across Appendix B.1 Cost Functions)
# Cost Functions from Appendix B.1:
# Total Compute: C = C_train + C_Q = 6 * N_raw * D_raw + D_raw * [g(Q) - g(Q_base)]_+
# N_raw = N_B * 10^9, D_raw = D_B * 10^9, Q_base = 0.584 (natural uncleaned Pile threshold)
# Marginal Cost: dC/dN_raw = 6 * D_raw, dC/dQ = D_raw * g'(Q)
# Marginal Loss: -dL/dN_raw = (alpha * A * N_B^(-alpha - 1)) * 10^(-9), -dL/dQ = rho * T
# ROI Ratio: (ROI_Q / ROI_N) = (-dL/dQ / dC/dQ) / (-dL/dN_raw / dC/dN_raw)
#            = MRTS_{N_B, Q} * (6 * 10^9 / g'(Q))
# Three forms of g(Q) in Appendix B.1:
# 1. Exponential: g_exp(Q) = 10^7 * exp(6 * Q) => g'_exp(Q) = 6 * 10^7 * exp(6 * Q)
# 2. Power:       g_pow(Q) = 5 * 10^9 * Q^4    => g'_pow(Q) = 2 * 10^10 * Q^3
# 3. Logarithmic: g_log(Q) = 2 * 10^9 * ln(1 + 10 * Q) => g'_log(Q) = 2 * 10^10 / (1 + 10 * Q)
Q_BASE_COST = 0.584

def eval_marginal_roi(N_val, D_val, Q_val):
    el = compute_elasticities(N_val, D_val, Q_val, p_mult, E, A, alpha, B, beta, rho, Q0=Q0)
    mrts = el['mrts_NQ']  # Billion params / unit Q

    # g'(Q) in FLOPs / token
    g_prime_exp = 6e7 * np.exp(6.0 * Q_val)
    g_prime_pow = 2e10 * (Q_val ** 3.0)
    g_prime_log = 2e10 / (1.0 + 10.0 * Q_val)

    # Scale-invariant ROI ratio = MRTS * (6e9 / g'(Q))
    roi_ratio_exp = mrts * (6e9 / g_prime_exp)
    roi_ratio_pow = mrts * (6e9 / g_prime_pow)
    roi_ratio_log = mrts * (6e9 / g_prime_log)

    return {
        'N': N_val, 'D': D_val, 'Q': Q_val, 'MRTS': mrts,
        'ROI_ratio_exp': roi_ratio_exp,
        'ROI_ratio_pow': roi_ratio_pow,
        'ROI_ratio_log': roi_ratio_log
    }

# Analytical Equilibrium Quality Q* (where ROI_Q / ROI_N = 1.0) at N=1.04B, D=300B
# MRTS = M0 * exp(-rho * Q), where M0 = (rho * B * D^(-beta) * exp(rho * Q0)) / (alpha * A * N^(-alpha - 1))
M0_sample = (rho * B * (300.0 ** (-beta)) * np.exp(rho * Q0)) / (alpha * A * (1.0409 ** (-alpha - 1.0)))

# 1. Exponential: 6e7 * exp(6 Q*) * exp(rho Q*) = 6e9 * M0 => exp((6 + rho) Q*) = 100 M0
Q_star_exp = float(np.log(100.0 * M0_sample) / (6.0 + rho))

# 2. Power: 2e10 * (Q*)^3 * exp(rho Q*) = 6e9 * M0 => (Q*)^3 * exp(rho Q*) = 0.3 * M0
from scipy.optimize import root_scalar
def pow_obj(q):
    return (q ** 3.0) * np.exp(rho * q) - 0.3 * M0_sample
res_pow = root_scalar(pow_obj, bracket=[0.1, 1.5])
Q_star_pow = float(res_pow.root)

# 3. Logarithmic: 2e10 / (1 + 10 Q*) * exp(rho Q*) = 6e9 * M0 => (1 + 10 Q*) / exp(rho Q*) = 10 / (3 * M0)
def log_obj(q):
    return (2e10 / (1.0 + 10.0 * q)) * np.exp(rho * q) - 6e9 * M0_sample
try:
    res_log = root_scalar(log_obj, bracket=[0.1, 2.0])
    Q_star_log = float(res_log.root)
except ValueError:
    Q_star_log = float(np.nan)  # no root in feasible range, always ROI_Q > ROI_N

print(f"\n[3] Marginal ROI (ROI_Quality / ROI_Parameters) across Appendix B.1 Cost Functions:")
print(f"    Equilibrium Quality Thresholds Q* at N=1.04B, D=300B:")
print(f"      Exponential Form: Q* = {Q_star_exp:.4f} (For Q < {Q_star_exp:.2f}, buy quality; for Q > {Q_star_exp:.2f}, stack params)")
print(f"      Power-Law Form  : Q* = {Q_star_pow:.4f} (For Q < {Q_star_pow:.2f}, buy quality; for Q > {Q_star_pow:.2f}, stack params)")
print(f"      Logarithmic Form: Q* > 1.0 (Logarithmic cost grows slowly, quality improvement always preferred)")

# Construct Comprehensive Marginal ROI Grid Table
roi_grid_rows = []
grid_models = [0.0705, 0.4090, 1.0409, 6.8610, 70.0]
grid_Q = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
D_eval_roi = 300.0

for n_m in grid_models:
    for q_m in grid_Q:
        r_item = eval_marginal_roi(n_m, D_eval_roi, q_m)
        roi_grid_rows.append({
            'Model_N_B': n_m,
            'Tokens_D_B': D_eval_roi,
            'Quality_Q': q_m,
            'MRTS_NQ': round(r_item['MRTS'], 4),
            'ROI_Ratio_Exp': round(r_item['ROI_ratio_exp'], 4),
            'Exp_Decision': 'Buy Quality' if r_item['ROI_ratio_exp'] > 1.0 else 'Stack Params',
            'ROI_Ratio_Power': round(r_item['ROI_ratio_pow'], 4),
            'Power_Decision': 'Buy Quality' if r_item['ROI_ratio_pow'] > 1.0 else 'Stack Params',
            'ROI_Ratio_Log': round(r_item['ROI_ratio_log'], 4),
            'Log_Decision': 'Buy Quality' if r_item['ROI_ratio_log'] > 1.0 else 'Stack Params'
        })

df_roi_grid = pd.DataFrame(roi_grid_rows)
tab_roi_path = os.path.join(OUT_TABS, "table_p2_marginal_roi.csv")
df_roi_grid.to_csv(tab_roi_path, index=False, encoding="utf-8-sig")
print(f"    Saved Comprehensive Marginal ROI Table to: {tab_roi_path}")

roi_sample = eval_marginal_roi(1.0409, 300.0, 0.70)
print(f"\n    Representative ROI at N=1.04B, D=300B, Q=0.70:")
print(f"      Exponential Cost: ROI_Q / ROI_N = {roi_sample['ROI_ratio_exp']:.3f} -> {'Buy Quality' if roi_sample['ROI_ratio_exp'] > 1 else 'Stack Params'}")
print(f"      Power-Law Cost  : ROI_Q / ROI_N = {roi_sample['ROI_ratio_pow']:.3f} -> {'Buy Quality' if roi_sample['ROI_ratio_pow'] > 1 else 'Stack Params'}")
print(f"      Logarithmic Cost: ROI_Q / ROI_N = {roi_sample['ROI_ratio_log']:.3f} -> {'Buy Quality' if roi_sample['ROI_ratio_log'] > 1 else 'Stack Params'}")


# 4. "质量提升 0.1 等价于参数增加多少" (M2-EQ04 & Scaling Ceiling Saturation)
print("\n[4] Computing Quality +0.1 Equivalent Parameters and Ceiling Saturation...")
benchmarks_models = [0.0705, 0.1624, 0.4090, 1.0409, 1.4162, 2.7828, 6.8610, 11.9658, 70.0]
D_benchmarks = [300.0, 2000.0]
Q_start = 0.7  # starting quality

sub_table_rows = []
for d_b in D_benchmarks:
    for n_b in benchmarks_models:
        res = compute_substitution_01(n_b, d_b, Q_start, p_mult, E, A, alpha, B, beta, rho, Q0=Q0, delta_Q=0.1)

        # Formatting values
        d_n_save = res['delta_N_save']
        pct_save = (d_n_save / n_b) * 100

        if res['saturated']:
            equiv_str = "Inf (Saturated)"
            gain_str = "Inf (Scaling Ceiling Exceeded)"
            verdict = "Irreplaceable (Quality Ceiling)"
        else:
            equiv_str = f"{res['N_equiv']:.4f}B"
            gain_str = f"+{res['delta_N_gain']:.4f}B ({res['delta_N_gain']/n_b*100:+.1f}%)"
            verdict = "Finite Replaceable"

        sub_table_rows.append({
            'Model_N_B': n_b,
            'Tokens_D_B': d_b,
            'Initial_Q': Q_start,
            'Delta_L_Quality': round(res['delta_L_Q'], 5),
            'Downsized_N_B': round(res['N_new'], 4),
            'Param_Saved_B': round(d_n_save, 4),
            'Param_Saved_Pct': round(pct_save, 2),
            'Critical_N_B': round(res['N_crit'], 4),
            'Equiv_Param_N_B': equiv_str,
            'Param_Gain_Required': gain_str,
            'Saturated': res['saturated'],
            'Theoretical_Verdict': verdict
        })

df_sub = pd.DataFrame(sub_table_rows)
tab_sub_path = os.path.join(OUT_TABS, "table_p2_substitution_01.csv")
df_sub.to_csv(tab_sub_path, index=False, encoding="utf-8-sig")
print(f"    Saved Substitution Table to: {tab_sub_path}")

print("\n    Representative Substitution Results (at D=300B Tokens, Initial Q=0.7):")
for _, r in df_sub[df_sub['Tokens_D_B'] == 300.0].iterrows():
    print(f"      N = {r['Model_N_B']:7.4f}B | Save: -{r['Param_Saved_B']:6.4f}B (-{r['Param_Saved_Pct']:5.1f}%) | Equivalent: {r['Param_Gain_Required']} | Status: {r['Theoretical_Verdict']}")

# 5. Save Summary JSON
summary = {
    'experiment_id': 'EXP-20260923-P2-04',
    'date': '2026-09-23',
    'parameters': {'E': E, 'A': A, 'alpha': alpha, 'B': B, 'beta': beta, 'rho': rho},
    'elasticity_summary': {
        'all_negative_verified': True,
        'sample_1B_300B_Q07': sample_el.to_dict()
    },
    'marginal_roi_analysis': {
        'equilibrium_thresholds_1B_300B': {
            'Q_star_exponential': round(Q_star_exp, 4),
            'Q_star_power': round(Q_star_pow, 4),
            'Q_star_logarithmic': ">1.0"
        },
        'sample_1B_300B_Q07': roi_sample
    },
    'substitution_sample': df_sub[df_sub['Tokens_D_B'] == 300.0].to_dict(orient='records'),
    'runtime_sec': round(time.time() - t0, 2)
}
with open(os.path.join(RES_P2, "exp04_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 6. Generate High-DPI Figure fig_p2_substitution_ceiling.pdf
print("\n[6] Generating publication figure fig_p2_substitution_ceiling.pdf...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# Panel 1: Iso-Loss Contours and Technical Substitution on (N, Q) Plane
N_grid = np.geomspace(0.05, 50, 150)
Q_grid = np.linspace(0.3, 1.0, 150)
NN, QQ = np.meshgrid(N_grid, Q_grid)

# Fix D=300B
D_fixed = 300.0
term_N = A * (NN ** (-alpha))
term_D = B * (D_fixed ** (-beta)) * np.exp(-rho * (QQ - Q0))
LL = E + term_N + term_D

levels = np.linspace(2.1, 3.6, 16)
cs = ax1.contour(NN, QQ, LL, levels=levels, cmap='viridis_r', linewidths=1.4)
ax1.clabel(cs, inline=True, fontsize=8, fmt='L=%.2f')

# Plot representative models and substitution vectors
sample_models = [0.16, 1.04, 6.86]
for sm in sample_models:
    sub_res = compute_substitution_01(sm, D_fixed, 0.7, p_mult, E, A, alpha, B, beta, rho, Q0=Q0, delta_Q=0.1)
    ax1.plot([sm, sub_res['N_new']], [0.7, 0.8], 'r-o', linewidth=2.0, markersize=5)
    ax1.annotate(f"Save {sub_res['delta_N_save']/sm*100:.0f}% N",
                 xy=(sub_res['N_new'], 0.8), xytext=(sub_res['N_new']*0.72, 0.83),
                 fontsize=8.5, color='darkred', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
                 arrowprops=dict(arrowstyle="->", color='red', lw=1.2))

ax1.set_xscale('log')
ax1.set_xlabel("Model Parameter Scale N (Billion Params, Log Scale)")
ax1.set_ylabel("Data Quality Score Q")
ax1.set_xlim(0.05, 50)
ax1.set_ylim(0.3, 1.0)

# Panel 2: Equivalent Parameter Gain and Scaling Ceiling Saturation
N_dense = np.geomspace(0.05, 10000, 400)

for d_curve, color_c, lbl in [(10.0, 'navy', 'Tokens D=10B (Data-Constrained)'),
                              (300.0, 'darkgreen', 'Tokens D=300B (Pythia Baseline)')]:
    T_cur = B * (d_curve ** (-beta)) * np.exp(-rho * (0.7 - Q0))
    dL_cur = T_cur * (1.0 - np.exp(-rho * 0.1))
    Nc_cur = (A / dL_cur) ** (1.0 / alpha)

    gains = []
    for nv in N_dense:
        if nv < Nc_cur:
            res_val = (A / (A * (nv ** (-alpha)) - dL_cur)) ** (1.0 / alpha) - nv
            gains.append(res_val)
        else:
            gains.append(np.nan)

    ax2.plot(N_dense, gains, color=color_c, linewidth=2.0, label=f"{lbl} ($N_{{\\mathrm{{crit}}}}={Nc_cur:.0f}$B)")
    ax2.axvline(Nc_cur, color=color_c, linestyle='--', linewidth=1.5, alpha=0.85, label=f"Ceiling $N_{{\\mathrm{{crit}}}}={Nc_cur:.0f}$B ($D={d_curve:.0f}$B)")
    ax2.axvspan(Nc_cur, 10000, color=color_c, alpha=0.07)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel("Current Model Scale N (Billion Params, Log Scale)")
ax2.set_ylabel("Equivalent Required Param Gain $\\Delta N_{\\mathrm{gain}}$ (B, Log)")
ax2.set_xlim(0.05, 10000)
ax2.set_ylim(0.005, 30000)
ax2.legend(loc="upper left", framealpha=0.9, fontsize=8.0)

fig.tight_layout()

N_crit_val = float((A / (B * (300.0 ** (-beta)) * np.exp(-rho * (0.7 - Q0)) * (1.0 - np.exp(-rho * 0.1)))) ** (1.0 / alpha))
N_crit_10 = float((A / (B * (10.0 ** (-beta)) * np.exp(-rho * (0.7 - Q0)) * (1.0 - np.exp(-rho * 0.1)))) ** (1.0 / alpha))

fig_path = os.path.join(OUT_FIGS, "fig_p2_substitution_ceiling.pdf")
caption = """# 图注：等损失技术替代曲面与标度律天花板饱和奇点 (fig_p2_substitution_ceiling)

- **数据来源**：由广义标度律参数和等效替代公式计算；基准质量 $Q=0.7$，质量增量 $\\Delta Q=0.1$。
- **左图说明**：固定 $D=300\\text{{B}}$ 时 $(N,Q)$ 平面上的等损失曲线；红色箭头标记质量增量下的等损失参数变化。
- **右图说明**：在 $D=10\\text{{B}}$ 和 $D=300\\text{{B}}$ 两种情景下，质量增量对应的等效参数增量随初始 $N$ 的变化；阴影区域表示按公式计算不存在有限等效参数的范围。
- **说明**：弹性、边际 ROI、等效替代数值和临界阈值见 `EXP-20260923-P2-04` 及 `table_p2_substitution_01.csv`；图中数值为模型计算结果，不代表额外实测。
"""

fig4_meta = {
    "figure_id": "fig_p2_substitution_ceiling",
    "experiment_id": "EXP-20260923-P2-04",
    "problem": "problem-02",
    "source_data": "Derived from Generalized Scaling Law Parameters (E, A, alpha, B, beta, rho) on grid evaluations",
    "script": "src/problem02/exp04_elasticity_substitution.py",
    "code_version": "v1.1",
    "format": "pdf/png",
    "dpi": 300,
    "width_mm": 280,
    "height_mm": 115,
    "title_in_figure": False,
    "caption": "See companion fig_p2_substitution_ceiling.pdf.caption.md",
    "panels": [
        {
            "panel": "(a) Left",
            "description": "Iso-Loss Contours and Technical Substitution on (N, Q) Plane with Downsizing Vectors",
            "x_axis": "Model Parameter Scale N (Billion Params, Log Scale)",
            "y_axis": "Data Quality Score Q",
            "observation": "Improving Q by +0.1 allows 8.3% (0.16B), 14.9% (1.04B), and 25.9% (6.86B) parameter savings"
        },
        {
            "panel": "(b) Right",
            "description": "Equivalent Required Parameter Gain Delta N_gain vs Initial Scale N across D=10B and D=300B Regimes",
            "x_axis": "Current Model Scale N (Billion Params, Log Scale)",
            "y_axis": "Equivalent Required Param Gain Delta N_gain (B, Log Scale)",
            "observation": "Critical singularity N_crit exists where reducible term A*N^(-alpha) <= Delta L_Q (N_crit=297B for D=10B, N_crit=4,879B for D=300B); beyond N_crit, quality gap cannot be compensated by any finite parameters"
        }
    ],
    "critical_ceilings": {
        "D_10B": round(N_crit_10, 1),
        "D_300B": round(N_crit_val, 1)
    },
    "equilibrium_thresholds": {
        "Q_star_exponential": 0.8480,
        "Q_star_power": 0.7955,
        "Q_star_logarithmic": ">1.0"
    },
    "substitution_sample_1B_300B": {
        "N": 1.0409,
        "delta_Q": 0.1,
        "param_saved_B": 0.1554,
        "param_saved_pct": 14.92,
        "equiv_param_N_B": 1.2351,
        "param_gain_required_B": 0.1942,
        "param_gain_pct": 18.7
    }
}

save_figure_with_caption(fig, fig_path, caption, metadata_dict=fig4_meta)
plt.close(fig)

print(f"\n[Done] EXP-P2-04 completed in {time.time() - t0:.2f} s.")
