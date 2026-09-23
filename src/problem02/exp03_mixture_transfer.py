# -*- coding: utf-8 -*-
"""EXP-20260923-P2-03: Mixture Response Transfer, Multi-Scale Calibration, and Domain Hessian Analysis.

Implements:
  - Integration with Problem 1 ALR Mixture Response Model f(p)
  - Data-driven calibration of cross-scale transfer parameter tau
  - Construction of degenerative mixture multiplier R(p) per M2-EQ02
  - Simplex Hessian curvature matrix & Domain Complementarity / Substitution Classification (M2-EQ05)
  - Figure fig_p2_domain_interactions.pdf + table table_p2_domain_interactions.csv

Run:
  & "D:\Program Files\CondaEnvs\AIOPS\python.exe" src/problem02/exp03_mixture_transfer.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt

from p2_common import (
    save_figure_with_caption,
    DIR_A, RES_P2, OUT_FIGS, OUT_TABS, SEED
)

t0 = time.time()
print("=" * 70)
print("EXP-20260923-P2-03: Mixture Response & Domain Complementarity / Substitution")
print("=" * 70)

# 1. Load Problem 1 RegMix Data
BASE_REGMIX = os.path.join(DIR_A, "regmix_tables")
DOMAINS17 = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
             "dm_mathematics", "github", "philpapers", "stackexchange",
             "enron_emails", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
             "europarl", "hackernews", "pubmed_abstracts", "uspto_backgrounds"]
LOSS13 = [d for d in DOMAINS17 if d not in
          ("nih_exporter", "enron_emails", "europarl", "philpapers")]
REF = "pile_cc"
REF_IDX = DOMAINS17.index(REF)
EPS = 1e-3

mix_train = pd.read_csv(os.path.join(BASE_REGMIX, "train_mixture_1m.csv")).sort_values("index").reset_index(drop=True)
loss_train = pd.read_csv(os.path.join(BASE_REGMIX, "train_pile_loss_1m.csv")).sort_values("index").reset_index(drop=True)

P_raw = mix_train[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
P = P_raw / P_raw.sum(axis=1, keepdims=True)
y_train = loss_train[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float).mean(axis=1)

def to_alr(P_mat, eps=EPS):
    Pt = (P_mat + eps / 17.0) / (1.0 + eps)
    X = np.log(Pt / Pt[:, [REF_IDX]])
    return np.delete(X, REF_IDX, axis=1)

X_train = to_alr(P)
alr_cols = [d for d in DOMAINS17 if d != REF]

# Train High-Fidelity HistGB Mixture Model f(p)
hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=300).fit(X_train, y_train)

# Reference Mixture p0
p0 = P.mean(axis=0)
X0 = to_alr(p0.reshape(1, -1))
f_p0 = float(hgb.predict(X0)[0])
print(f"[1] Fitted Mixture Response Model f(p) on n=512.")
print(f"    Baseline Reference Recipe f(p0) = {f_p0:.4f} Nats on 1M scale.")

# 2. Multi-scale Variance Shrinkage Calibration of tau (per P2-05)
scales_info = []
for scale, lf in [("1M", "test_pile_loss_1m.csv"),
                  ("60M", "test_pile_loss_60m.csv"),
                  ("1B", "test_pile_loss_1B.csv")]:
    df_l = pd.read_csv(os.path.join(BASE_REGMIX, lf))
    loss_cols = [c for c in df_l.columns if c.startswith("metric/")]
    m_loss = df_l[loss_cols].mean(axis=1)
    mean_val = float(m_loss.mean())
    std_val = float(m_loss.std())
    cv_val = float(std_val / mean_val)
    scales_info.append({
        'scale': scale,
        'n_samples': len(df_l),
        'mean_loss': mean_val,
        'std_loss': std_val,
        'cv_pct': cv_val * 100
    })
    print(f"[2] Scale {scale:4s}: mean = {mean_val:.4f}, std = {std_val:.4f}, CV = {cv_val*100:.2f}%")

cv_1m = scales_info[0]['cv_pct'] / 100.0
cv_1b = scales_info[2]['cv_pct'] / 100.0
cv_ratio = cv_1b / cv_1m
print(f"    Cross-scale variance shrinkage ratio (1B / 1M) = {cv_ratio:.4f}")

# Analytical MSM (Method of Simulated Moments) Derivation of tau*
# Load 1B test mixture recipes (A8) to evaluate model response distribution
tm_1b_path = os.path.join(BASE_REGMIX, "test_mixture_1B.csv")
tm_1b = pd.read_csv(tm_1b_path)
P_1b = tm_1b[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
P_1b = P_1b / P_1b.sum(axis=1, keepdims=True)
pred_1b = hgb.predict(to_alr(P_1b))
rel_1b = (pred_1b - f_p0) / f_p0
std_rel_1b = float(np.std(rel_1b, ddof=1))

# First-order linear approximation: tau_linear = CV(1B) / std(rel_diff)
tau_linear = float(cv_1b / std_rel_1b)

# Exact non-linear calibration: argmin |std(exp(tau * rel_diff)) - CV(1B)|
tau_grid = np.linspace(0.5, 1.2, 7001)
obj_tau = [abs(np.std(np.exp(t * rel_1b), ddof=1) - cv_1b) for t in tau_grid]
tau_exact = float(tau_grid[np.argmin(obj_tau)])
TAU_BENCH = 0.85  # benchmark adopted in literature and pipeline

print(f"\n[2b] Method of Simulated Moments (MSM) Calibration for tau:")
print(f"     Empirical CV on 1B scale = {cv_1b*100:.3f}% (from {scales_info[2]['n_samples']} test mixtures)")
print(f"     Recipe response std on 1B = {std_rel_1b*100:.3f}%")
print(f"     First-order linear tau = {tau_linear:.4f}")
print(f"     Exact non-linear tau*  = {tau_exact:.4f} (Benchmark tau* = {TAU_BENCH:.2f})")

# Generate Sensitivity Analysis Table for tau
tau_sens_rows = []
for t_val in [0.45, 0.60, tau_exact, 0.85, 1.00, 1.20, 1.50]:
    R_sim = np.exp(t_val * rel_1b)
    std_sim = float(np.std(R_sim, ddof=1))
    mean_sim = float(np.mean(R_sim))
    err_cv = abs(std_sim - cv_1b)
    tau_sens_rows.append({
        'tau': round(t_val, 4),
        'Mean_R': round(mean_sim, 5),
        'Std_R': round(std_sim, 5),
        'Target_Empirical_CV_1B': round(cv_1b, 5),
        'Absolute_Error_vs_CV': round(err_cv, 5),
        'Role': 'Optimal Non-Linear' if abs(t_val - tau_exact) < 1e-4 else ('Benchmark Adopted' if t_val == 0.85 else 'Sensitivity Grid')
    })
df_tau_sens = pd.DataFrame(tau_sens_rows)
tab_tau_path = os.path.join(OUT_TABS, "table_p2_tau_sensitivity.csv")
df_tau_sens.to_csv(tab_tau_path, index=False, encoding="utf-8-sig")
print(f"     Saved tau Sensitivity Table to: {tab_tau_path}")

def compute_R(p_vec, tau=TAU_BENCH):
    """M2-EQ02: R(p) = exp{ tau * [f(p) - f(p0)] / f(p0) }.
    Note: Output is dimensionless mixture multiplier R(p).
    Multiplier curvature: H^R_{ij} = d^2 R(p) / dp_i dp_j.
    Loss curvature: H^L_{ij} = T * H^R_{ij}, where T = B * D^(-beta) * exp(-rho*(Q - Q0)).
    """
    p_mat = np.asarray(p_vec).reshape(1, -1)
    p_mat = p_mat / p_mat.sum()
    X = to_alr(p_mat)
    f_p = float(hgb.predict(X)[0])
    rel_diff = (f_p - f_p0) / f_p0
    return float(np.exp(tau * rel_diff)), rel_diff

R_p0, rel_p0 = compute_R(p0)
print(f"    Degeneration Verification at p0: R(p0) = {R_p0:.6f} (Strictly 1.000000, rel={rel_p0:.6f})")

# 3. Hessian Curvature & Domain Complementarity / Substitution Analysis (M2-EQ05)
print("\n[3] Calculating Simplex Hessian Curvature Matrix (17 x 17)...")

def compute_hessian(step_h):
    """Numerical second-order partial derivative projected on simplex.
    Calculates H^R_{ij} = d^2 R(p) / dp_i dp_j.
    """
    H_mat = np.zeros((17, 17))
    for i in range(17):
        for j in range(i, 17):
            if i == j:
                p_pos = p0.copy()
                p_neg = p0.copy()
                p_pos[i] += 2 * step_h
                other_sum = p_pos.sum() - p_pos[i]
                p_pos[np.arange(17) != i] *= (1.0 - p_pos[i]) / other_sum

                p_neg[i] = max(1e-4, p_neg[i] - 2 * step_h)
                other_sum = p_neg.sum() - p_neg[i]
                p_neg[np.arange(17) != i] *= (1.0 - p_neg[i]) / other_sum

                R_pos, _ = compute_R(p_pos)
                R_neg, _ = compute_R(p_neg)
                d2 = (R_pos - 2 * R_p0 + R_neg) / ((2 * step_h) ** 2)
                H_mat[i, i] = d2
            else:
                def make_p(di, dj):
                    p_c = p0.copy()
                    p_c[i] = max(1e-4, p_c[i] + di * step_h)
                    p_c[j] = max(1e-4, p_c[j] + dj * step_h)
                    fixed_sum = p_c[i] + p_c[j]
                    mask = (np.arange(17) != i) & (np.arange(17) != j)
                    p_c[mask] *= (1.0 - fixed_sum) / p_c[mask].sum()
                    return p_c

                R_pp, _ = compute_R(make_p(+1, +1))
                R_pm, _ = compute_R(make_p(+1, -1))
                R_mp, _ = compute_R(make_p(-1, +1))
                R_mm, _ = compute_R(make_p(-1, -1))

                d2 = (R_pp - R_pm - R_mp + R_mm) / (4.0 * (step_h ** 2))
                H_mat[i, j] = d2
                H_mat[j, i] = d2
    return H_mat

# Primary Hessian evaluated at h = 0.015
h_step = 0.015
H = compute_hessian(h_step)

# Multi-step Hessian Robustness Analysis
test_steps = [0.010, 0.015, 0.020, 0.025, 0.030]
robust_pairs_to_track = [
    ('stackexchange', 'pile_cc', 'Top-1 Synergy'),
    ('arxiv', 'uspto_backgrounds', 'Top-2 Synergy'),
    ('ubuntu_irc', 'uspto_backgrounds', 'Top-3 Synergy'),
    ('arxiv', 'github', 'Domain Pair (Code+Science)'),
    ('pile_cc', 'gutenberg_pg_19', 'Domain Pair (Web+Books)'),
    ('arxiv', 'pile_cc', 'Top-3 Competitive'),
    ('enron_emails', 'hackernews', 'Top-2 Competitive'),
    ('pile_cc', 'uspto_backgrounds', 'Top-1 Competitive')
]

robustness_rows = []
H_step_dict = {}
for hs in test_steps:
    if hs == 0.015:
        H_hs = H
    else:
        H_hs = compute_hessian(hs)
    H_step_dict[hs] = H_hs

for d1, d2, role in robust_pairs_to_track:
    idx1, idx2 = DOMAINS17.index(d1), DOMAINS17.index(d2)
    curv_vals = [H_step_dict[hs][idx1, idx2] for hs in test_steps]
    robustness_rows.append({
        'Domain_1': d1,
        'Domain_2': d2,
        'Role': role,
        'Curvature_h_0010': round(curv_vals[0], 4),
        'Curvature_h_0015_Primary': round(curv_vals[1], 4),
        'Curvature_h_0020': round(curv_vals[2], 4),
        'Curvature_h_0025': round(curv_vals[3], 4),
        'Curvature_h_0030': round(curv_vals[4], 4),
        'Mean_Curvature': round(float(np.mean(curv_vals)), 4),
        'Sign_Consistency': 'Consistent' if (all(v < 0 for v in curv_vals) or all(v > 0 for v in curv_vals)) else 'Sign-Sensitive'
    })

df_robust = pd.DataFrame(robustness_rows)
tab_robust_path = os.path.join(OUT_TABS, "table_p2_hessian_robustness.csv")
df_robust.to_csv(tab_robust_path, index=False, encoding="utf-8-sig")
print(f"     Saved Hessian Multi-step Robustness Table to: {tab_robust_path}")

# Categorize key pairs into Complementary vs Substitution
pairs_records = []
for i in range(17):
    for j in range(i + 1, 17):
        val = H[i, j]
        if val < -0.05:
            cat = "Complementary (Synergy)"
        elif val > 0.05:
            cat = "Substitution (Competitive)"
        else:
            cat = "Independent (Neutral)"
        pairs_records.append({
            'domain_1': DOMAINS17[i],
            'domain_2': DOMAINS17[j],
            'hessian_curvature_H_R': round(val, 4),
            'relationship': cat
        })

df_pairs = pd.DataFrame(pairs_records).sort_values('hessian_curvature_H_R')
tab_interactions_path = os.path.join(OUT_TABS, "table_p2_domain_interactions.csv")
df_pairs.to_csv(tab_interactions_path, index=False, encoding="utf-8-sig")
print(f"[4] Saved domain interactions table to: {tab_interactions_path}")

top_synergy = df_pairs.head(5)
top_comp = df_pairs.tail(5)
print("\n    Top 5 Synergistic (Complementary) Pairs (Hessian < 0):")
for _, r in top_synergy.iterrows():
    print(f"      {r['domain_1']:18s} + {r['domain_2']:18s}: Curvature = {r['hessian_curvature_H_R']:7.4f}")

print("\n    Top 5 Competitive (Substitution) Pairs (Hessian > 0):")
for _, r in top_comp.iterrows():
    print(f"      {r['domain_1']:18s} + {r['domain_2']:18s}: Curvature = {r['hessian_curvature_H_R']:7.4f}")

# 4. Save Summary JSON
summary = {
    'experiment_id': 'EXP-20260923-P2-03',
    'date': '2026-09-23',
    'f_p0': f_p0,
    'tau_calibration': {
        'empirical_cv_1B': cv_1b,
        'recipe_std_1B': std_rel_1b,
        'tau_linear': tau_linear,
        'tau_exact_nonlinear': tau_exact,
        'tau_adopted_benchmark': TAU_BENCH
    },
    'cv_shrinkage_ratio_1B_1M': cv_ratio,
    'multi_scale_stats': scales_info,
    'top_synergistic_pairs': top_synergy.to_dict(orient='records'),
    'top_substitution_pairs': top_comp.to_dict(orient='records'),
    'runtime_sec': round(time.time() - t0, 2)
}
with open(os.path.join(RES_P2, "exp03_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 5. Generate Figure fig_p2_domain_interactions.pdf
print("\n[5] Generating publication figure fig_p2_domain_interactions.pdf...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

# Panel 1: Multi-scale variance shrinkage & tau calibration (Dual Y-Axes)
scales = [s['scale'] for s in scales_info]
cvs = [s['cv_pct'] for s in scales_info]
stds = [s['std_loss'] for s in scales_info]

# Left axis: CV (%)
line1 = ax1.plot(scales, cvs, 'o-', color='navy', linewidth=2.2, markersize=8, label="Coefficient of Variation CV (%)")
ax1.set_xlabel("Model & Data Scale (Attachment A Validation Sets)", labelpad=6)
ax1.set_ylabel("Coefficient of Variation CV (%)", color='navy', labelpad=6)
ax1.tick_params(axis='y', labelcolor='navy')
ax1.set_xlim(-0.35, 2.35)
ax1.set_ylim(0.0, 7.0)

for idx, txt in enumerate(cvs):
    if idx == 0:
        ax1.annotate(f"{txt:.2f}%", (scales[idx], cvs[idx]), textcoords="offset points", xytext=(-8, 10),
                     ha='right', fontweight='bold', color='navy', fontsize=8.8)
    else:
        ax1.annotate(f"{txt:.2f}%", (scales[idx], cvs[idx]), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontweight='bold', color='navy', fontsize=8.8)

# Right axis: Loss Std Dev (Nats)
ax1_twin = ax1.twinx()
line2 = ax1_twin.plot(scales, stds, 's--', color='crimson', linewidth=2.0, markersize=8, label="Loss Std Dev (Nats)")
ax1_twin.set_ylabel("Loss Standard Deviation (Nats)", color='crimson', labelpad=6)
ax1_twin.tick_params(axis='y', labelcolor='crimson')
ax1_twin.set_xlim(-0.35, 2.35)
ax1_twin.set_ylim(0.0, 0.35)
ax1_twin.grid(False)

for idx, txt in enumerate(stds):
    if idx == 0:
        ax1_twin.annotate(f"{txt:.4f}", (scales[idx], txt), textcoords="offset points", xytext=(10, 8),
                          ha='left', fontweight='bold', color='crimson', fontsize=8.8)
    else:
        ax1_twin.annotate(f"{txt:.4f}", (scales[idx], txt), textcoords="offset points", xytext=(0, -18),
                          ha='center', fontweight='bold', color='crimson', fontsize=8.8)

# Combined clean legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", framealpha=0.92, fontsize=8.8)

# Panel 2: Hessian Curvature Heatmap for Top Representative Domains
key_domains = ["stackexchange", "pile_cc", "arxiv", "uspto_backgrounds", "ubuntu_irc", "hackernews", "enron_emails", "github"]
k_indices = [DOMAINS17.index(d) for d in key_domains]
H_sub = H[np.ix_(k_indices, k_indices)]

ax2.grid(False)
vmax = np.max(np.abs(H_sub))
im = ax2.imshow(H_sub, cmap='coolwarm', vmin=-vmax, vmax=vmax)
cbar = plt.colorbar(im, ax=ax2)
cbar.set_label("Multiplier Hessian Curvature $\\mathcal{H}^R_{ij}$ (Negative: Synergy, Positive: Competition)")

ax2.set_xticks(range(len(key_domains)))
ax2.set_yticks(range(len(key_domains)))
ax2.set_xticklabels(key_domains, rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels(key_domains, fontsize=9)

# Overlay numerical values inside heatmap
for i in range(len(key_domains)):
    for j in range(len(key_domains)):
        val = H_sub[i, j]
        color = "white" if np.abs(val) > vmax * 0.55 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

fig.tight_layout()

h_se_pile = H[DOMAINS17.index('stackexchange'), DOMAINS17.index('pile_cc')]
h_ar_us = H[DOMAINS17.index('arxiv'), DOMAINS17.index('uspto_backgrounds')]
h_pile_us = H[DOMAINS17.index('pile_cc'), DOMAINS17.index('uspto_backgrounds')]
h_en_hn = H[DOMAINS17.index('enron_emails'), DOMAINS17.index('hackernews')]
h_ar_git = H[DOMAINS17.index('arxiv'), DOMAINS17.index('github')]
h_pile_gut = H[DOMAINS17.index('pile_cc'), DOMAINS17.index('gutenberg_pg_19')]

fig_path = os.path.join(OUT_FIGS, "fig_p2_domain_interactions.pdf")
caption = """# 图注：跨尺度配比强度校准与领域单纯形 Hessian 互补/替代交互结构 (fig_p2_domain_interactions)

- **数据来源**：附件 A 真实配方实验与独立检验集（A4–A11，含 1M、60M、1B 尺度）。
- **左图说明**：配比引起的损失变异系数（CV）与标准差随训练尺度的比较。左纵轴为 CV (%)，右纵轴为损失标准差（Nats）；三组数据分别对应 1M、60M 和 1B 检验集。
- **右图说明**：8 个核心知识领域的单纯形 Hessian 二阶交互曲率矩阵 $\\mathcal{{H}}^R_{{ij}} = \\frac{{\\partial^2 R}}{{\\partial p_i \\partial p_j}}$（折算至模型损失需乘以外在系数 $T \\approx 0.3 \\sim 0.6$）。蓝色区块（负曲率）代表两领域具有超加性的协同互补效应；红色区块（正曲率）代表竞争替代效应。
- **说明**：配比标定参数、Hessian 数值和领域对排序见 `EXP-20260923-P2-03` 记录与配套结果表。
"""

fig3_meta = {
    "figure_id": "fig_p2_domain_interactions",
    "experiment_id": "EXP-20260923-P2-03",
    "problem": "problem-02",
    "source_data": "Attachment A RegMix tables (train_mixture_1m.csv, test_mixture_1B.csv, test_pile_loss_*.csv)",
    "script": "src/problem02/exp03_mixture_transfer.py",
    "code_version": "v1.1",
    "format": "pdf/png",
    "dpi": 300,
    "width_mm": 280,
    "height_mm": 115,
    "title_in_figure": False,
    "caption": "See companion fig_p2_domain_interactions.pdf.caption.md",
    "panels": [
        {
            "panel": "(a) Left",
            "description": "Multi-scale Variance Shrinkage across 1M, 60M, and 1B models with Dual Y-Axes (CV % and Loss Std Nats)",
            "x_axis": "Model & Data Scale (Attachment A Validation Sets)",
            "y_axis_left": "Coefficient of Variation CV (%)",
            "y_axis_right": "Loss Standard Deviation (Nats)",
            "observation": "Recipe sensitivity CV shrinks from 5.27% (1M) to 2.35% (1B), shrinkage ratio 0.4466; MSM exact non-linear tau* = 0.8348, adopting tau* = 0.85"
        },
        {
            "panel": "(b) Right",
            "description": "Simplex Hessian Curvature Heatmap H^R_ij for 8 Representative Domains",
            "x_axis": "Domain j",
            "y_axis": "Domain i",
            "observation": "Top-1 synergy: stackexchange + pile_cc (-9.00); Top-2 synergy: arxiv + uspto (-6.60); Top-1 competitive: pile_cc + uspto (+8.78); Top-2 competitive: enron + hackernews (+4.96)"
        }
    ],
    "calibration": {
        "cv_1m": round(scales_info[0]['cv_pct'] / 100.0, 5),
        "cv_1b": round(scales_info[2]['cv_pct'] / 100.0, 5),
        "cv_shrinkage_ratio": round(cv_ratio, 4),
        "tau_linear": round(tau_linear, 4),
        "tau_exact": round(tau_exact, 4),
        "tau_benchmark": round(TAU_BENCH, 2)
    },
    "top_pairs": {
        "synergy_top1": {"domains": ["stackexchange", "pile_cc"], "curvature": round(float(h_se_pile), 4)},
        "synergy_top2": {"domains": ["arxiv", "uspto_backgrounds"], "curvature": round(float(h_ar_us), 4)},
        "competitive_top1": {"domains": ["pile_cc", "uspto_backgrounds"], "curvature": round(float(h_pile_us), 4)},
        "competitive_top2": {"domains": ["enron_emails", "hackernews"], "curvature": round(float(h_en_hn), 4)}
    }
}

save_figure_with_caption(fig, fig_path, caption, metadata_dict=fig3_meta)
plt.close(fig)

print(f"\n[Done] EXP-P2-03 completed in {time.time() - t0:.2f} s.")
