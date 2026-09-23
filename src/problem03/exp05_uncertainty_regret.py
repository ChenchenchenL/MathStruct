"""
exp05_uncertainty_regret.py - EXP-305: Hypothetical Parameter Perturbation & Numerical Reference-Gap Analysis
Stress-tests the sensitivity of resource allocation under hypothetical parameter perturbations around nominal values
(A, alpha, B, beta, rho) via 1000 independent Gaussian perturbation draws using the marginal standard errors
reported in Problem 2. This is a sensitivity scenario test, not joint posterior covariance propagation.
Evaluates:
  - Realized loss under parameter perturbation scenarios L(x_nom; theta^(m))
  - Numerical reference loss for the first 100 scenarios per budget
  - Policy-to-reference gaps on that subsample (local solver, no global verification)
  - Scenario loss quantiles over all 1000 perturbations.
Generates:
  - result/tables/problem03/tab_p3_regret_analysis.csv
  - result/figures/problem03/fig_p3_regret_robustness.pdf (.png, .caption.md, .json)
  - result/problem03/exp05_summary.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from p3_common import (
    solve_optimal_allocation_2d,
    compute_loss,
    compute_optimal_d,
    BUDGET_TIERS_EFLOPS,
    P2_PARAM_NOMINAL,
    P2_PARAM_STD
)

DIR_ROOT = "d:/project/MathStruct"
DIR_RES = os.path.join(DIR_ROOT, "result/problem03")
DIR_TABS = os.path.join(DIR_ROOT, "result/tables/problem03")
DIR_FIGS = os.path.join(DIR_ROOT, "result/figures/problem03")

def run_exp05():
    print("=" * 80)
    print("EXP-305: Parameter Scenario Sensitivity & Reference-Gap Analysis")
    print("=" * 80)

    np.random.seed(42)
    n_samples = 1000

    # Draw 1000 parameter perturbation vectors theta = [A, alpha, B, beta, rho]
    # Independent perturbations use the marginal standard errors reported in Problem 2.
    theta_samples = []
    while len(theta_samples) < n_samples:
        draw = np.random.normal(P2_PARAM_NOMINAL, P2_PARAM_STD)
        if np.all(draw > 0.05):
            theta_samples.append(draw)
    theta_samples = np.array(theta_samples)

    tiers = ['low', 'mid', 'high']
    records = []
    summary_dict = {}

    for tier in tiers:
        c_bar = BUDGET_TIERS_EFLOPS[tier]
        c_flops = c_bar * 1e18
        summary_dict[tier] = {}

        # 1. Nominal optimal policy
        sol_nom = solve_optimal_allocation_2d(c_bar, cost_type='exp', L_ctx=2048, verify_global=False)
        n_nom, d_nom, q_nom = sol_nom['opt_n_B'], sol_nom['opt_d_B'], sol_nom['opt_Q']
        loss_nom_ideal = sol_nom['opt_loss']

        # 2. Conservative policy: Allocate 15% more data tokens as buffer (hedge against quality deficit)
        n_cons = n_nom * 0.90
        d_cons = compute_optimal_d(n_cons, q_nom, c_bar, L_ctx=2048, cost_type='exp')

        # 3. Evaluate across 1000 parameter realizations
        t0 = time.time()
        nom_losses = []
        cons_losses = []
        reference_losses = []
        reference_gaps_nom = []
        reference_gaps_cons = []

        # Solve numerical reference allocations only for the first 100 scenarios.
        n_reference_evals = 100
        for i in range(n_samples):
            theta_i = theta_samples[i]

            # Loss under nominal policy
            l_nom = compute_loss(n_nom, d_nom, q_nom, custom_params=theta_i)
            nom_losses.append(l_nom)

            # Loss under conservative policy
            l_cons = compute_loss(n_cons, d_cons, q_nom, custom_params=theta_i)
            cons_losses.append(l_cons)

            # Numerical reference allocation from the local multistart solver.
            if i < n_reference_evals:
                sol_reference = solve_optimal_allocation_2d(
                    c_bar,
                    cost_type='exp',
                    L_ctx=2048,
                    custom_params=theta_i,
                    verify_global=False
                )
                l_reference = sol_reference['opt_loss']
                reference_losses.append(l_reference)
                reference_gaps_nom.append(l_nom - l_reference)
                reference_gaps_cons.append(l_cons - l_reference)

        nom_losses = np.array(nom_losses)
        cons_losses = np.array(cons_losses)
        reference_losses = np.array(reference_losses)
        reference_gaps_nom = np.array(reference_gaps_nom)
        reference_gaps_cons = np.array(reference_gaps_cons)

        # Regret statistics
        mean_reg_nom = float(np.mean(reference_gaps_nom))
        p95_reg_nom = float(np.percentile(reference_gaps_nom, 95))
        max_reg_nom = float(np.max(reference_gaps_nom))

        mean_reg_cons = float(np.mean(reference_gaps_cons))
        p95_reg_cons = float(np.percentile(reference_gaps_cons, 95))
        max_reg_cons = float(np.max(reference_gaps_cons))

        row_nom = {
            'budget_tier': tier,
            'budget_FLOPs': f"{c_flops:.1e}",
            'policy': 'Nominal_Pareto_Optimal',
            'N_B': round(n_nom, 4),
            'D_B': round(d_nom, 4),
            'Q': round(q_nom, 4),
            'expected_loss': round(float(np.mean(nom_losses)), 4),
            'loss_scenario_q025_q975': f"[{np.percentile(nom_losses, 2.5):.4f}, {np.percentile(nom_losses, 97.5):.4f}]",
            'numerical_reference_scenarios': len(reference_losses),
            'mean_reference_gap_nats': round(mean_reg_nom, 5),
            'p95_reference_gap_nats': round(p95_reg_nom, 5),
            'sample_max_reference_gap_nats': round(max_reg_nom, 5),
            'mean_reference_gap_pct': round((mean_reg_nom / float(np.mean(reference_losses))) * 100, 3)
        }
        row_cons = {
            'budget_tier': tier,
            'budget_FLOPs': f"{c_flops:.1e}",
            'policy': 'Conservative_Data_Heavy',
            'N_B': round(n_cons, 4),
            'D_B': round(d_cons, 4),
            'Q': round(q_nom, 4),
            'expected_loss': round(float(np.mean(cons_losses)), 4),
            'loss_scenario_q025_q975': f"[{np.percentile(cons_losses, 2.5):.4f}, {np.percentile(cons_losses, 97.5):.4f}]",
            'numerical_reference_scenarios': len(reference_losses),
            'mean_reference_gap_nats': round(mean_reg_cons, 5),
            'p95_reference_gap_nats': round(p95_reg_cons, 5),
            'sample_max_reference_gap_nats': round(max_reg_cons, 5),
            'mean_reference_gap_pct': round((mean_reg_cons / float(np.mean(reference_losses))) * 100, 3)
        }
        records.extend([row_nom, row_cons])

        summary_dict[tier] = {
            'nom_policy': {'N_B': n_nom, 'D_B': d_nom, 'Q': q_nom},
            'cons_policy': {'N_B': n_cons, 'D_B': d_cons, 'Q': q_nom},
            'scenario_count': n_samples,
            'numerical_reference_count': len(reference_losses),
            'nom_loss_mean': float(np.mean(nom_losses)),
            'nom_loss_std': float(np.std(nom_losses)),
            'reference_gap_nom': {'mean': mean_reg_nom, 'p95': p95_reg_nom, 'sample_max': max_reg_nom, 'values': reference_gaps_nom.tolist()},
            'reference_gap_cons': {'mean': mean_reg_cons, 'p95': p95_reg_cons, 'sample_max': max_reg_cons, 'values': reference_gaps_cons.tolist()},
            'nom_losses': nom_losses.tolist(),
            'numerical_reference_losses': reference_losses.tolist()
        }

        print(f"[{tier.upper():4s}] Nom reference gap: Mean={mean_reg_nom:.5f}, P95={p95_reg_nom:.5f}, Max={max_reg_nom:.5f} Nats "
              f"| Cons reference gap: Mean={mean_reg_cons:.5f}, P95={p95_reg_cons:.5f}, Max={max_reg_cons:.5f} Nats")

    # Save CSV table
    df_out = pd.DataFrame(records)
    csv_path = os.path.join(DIR_TABS, "tab_p3_regret_analysis.csv")
    df_out.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[+] Saved regret analysis table to {csv_path}")

    # Save summary JSON
    json_path = os.path.join(DIR_RES, "exp05_summary.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved experiment summary JSON to {json_path}")

    # Plot Publication Figure
    plot_regret_robustness(summary_dict)

def plot_regret_robustness(summary_dict):
    plt.rcParams.update({
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10.5,
        'axes.labelsize': 11.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'figure.dpi': 300
    })

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))

    # -------------------------------------------------------------
    # Subplot 1: Loss Uncertainty Distributions across Tiers
    # -------------------------------------------------------------
    ax1 = axes[0]
    tiers = ['low', 'mid', 'high']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    for idx, t in enumerate(tiers):
        losses = summary_dict[t]['nom_losses']
        ax1.hist(losses, bins=35, density=True, alpha=0.55, color=colors[idx], edgecolor='black',
                 label=f'{t.capitalize()} Budget ($\mu={np.mean(losses):.2f}$)')

    ax1.set_xlabel('Realized Validation Loss $L$ (Nats)')
    ax1.set_ylabel('Probability Density')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------
    # Subplot 2: Regret Distribution (Nominal vs Conservative) - Mid Budget
    # -------------------------------------------------------------
    ax2 = axes[1]
    reg_nom = np.array(summary_dict['mid']['reference_gap_nom']['values'])
    reg_cons = np.array(summary_dict['mid']['reference_gap_cons']['values'])

    ax2.hist(reg_nom, bins=25, alpha=0.6, color='#4575b4', edgecolor='black',
             label=f'Nominal Policy (Max={np.max(reg_nom):.4f})')
    ax2.hist(reg_cons, bins=25, alpha=0.4, color='#f46d43', edgecolor='black',
             label=f'Conservative Policy (Max={np.max(reg_cons):.4f})')

    ax2.axvline(np.percentile(reg_nom, 95), color='#4575b4', linestyle='--', linewidth=1.8, label=r'Nominal 95th Percentile')
    ax2.axvline(np.percentile(reg_cons, 95), color='#f46d43', linestyle='--', linewidth=1.8, label=r'Conservative 95th Percentile')

    ax2.set_xlabel('Gap to Numerical Reference (Nats)')
    ax2.set_ylabel('Sample Frequency')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', framealpha=0.9)

    # -------------------------------------------------------------
    # Subplot 3: Reference-gap summaries across budget tiers
    # -------------------------------------------------------------
    ax3 = axes[2]
    x_pos = np.arange(len(tiers))
    width = 0.30

    mean_noms = [summary_dict[t]['reference_gap_nom']['mean'] * 1000 for t in tiers] # in mNats
    max_noms = [summary_dict[t]['reference_gap_nom']['sample_max'] * 1000 for t in tiers]
    mean_cons = [summary_dict[t]['reference_gap_cons']['mean'] * 1000 for t in tiers]
    max_cons = [summary_dict[t]['reference_gap_cons']['sample_max'] * 1000 for t in tiers]

    ax3.bar(x_pos - width/2, mean_noms, width, color='#4575b4', alpha=0.85, edgecolor='black', label='Nominal (Mean)')
    ax3.bar(x_pos + width/2, mean_cons, width, color='#f46d43', alpha=0.85, edgecolor='black', label='Conservative (Mean)')

    # Plot error bars to the observed subsample maximum.
    ax3.errorbar(x_pos - width/2, mean_noms, yerr=[[0]*len(tiers), np.array(max_noms)-np.array(mean_noms)],
                 fmt='none', ecolor='black', capsize=4, label='Subsample Maximum')
    ax3.errorbar(x_pos + width/2, mean_cons, yerr=[[0]*len(tiers), np.array(max_cons)-np.array(mean_cons)],
                 fmt='none', ecolor='black', capsize=4)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([r'Low ($10^{19}$)', r'Mid ($10^{22}$)', r'High ($10^{24}$)'])
    ax3.set_xlabel('Compute Budget Tier')
    ax3.set_ylabel('Reference Gap (mNats, $10^{-3}$ Nats)')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax3.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()

    # Save figure
    fig_pdf = os.path.join(DIR_FIGS, "fig_p3_regret_robustness.pdf")
    fig_png = os.path.join(DIR_FIGS, "fig_p3_regret_robustness.png")
    plt.savefig(fig_pdf, bbox_inches='tight')
    plt.savefig(fig_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved figure to {fig_pdf} and {fig_png}")

    # Generate companion .caption.md
    caption_path = os.path.join(DIR_FIGS, "fig_p3_regret_robustness.caption.md")
    caption_text = """# 图说明：假设参数扰动情景分析与数值参考差距 (fig_p3_regret_robustness)

### 图面要素说明
- **左图（扰动情景下的损失分布）**：按问题二五个标度律参数的边际标准误独立抽样 1000 组参数情景，展示名义配置损失分布的经验 2.5%–97.5% 分位区间；这不是参数置信区间或联合后验分布。
- **中图（中档预算相对数值参考解的差距）**：仅展示前 100 个扰动情景。参考解由多起点局部优化得到且未作全局核验；虚线表示该子样本的 95% 分位数。
- **右图（各预算档位的相对数值参考差距）**：柱表示前 100 个情景的平均差距，误差棒上端表示该子样本最大值，不是全局最坏风险界。

### 核心实验发现
1. 100 个参考情景下，名义策略的平均差距在低、中、高预算分别为 0.00208、0.00013 和 0.00010 Nats；相应样本最大值为 0.01150、0.00070 和 0.00052 Nats。
2. 数据偏重策略在该子样本各预算下的平均差距均高于名义策略。现有结果不支持将其称为更稳健方案；情景抽样和数值参考解的范围也不足以推出通用工程规则。
"""
    with open(caption_path, 'w', encoding='utf-8') as f:
        f.write(caption_text)
    print(f"[+] Saved caption to {caption_path}")

    # Generate companion .json metadata
    meta_path = os.path.join(DIR_FIGS, "fig_p3_regret_robustness.json")
    meta_dict = {
        "figure_id": "fig_p3_regret_robustness",
        "experiment_id": "EXP-305",
        "date": "2026-09-23",
        "scenario_count": len(summary_dict['low']['nom_losses']),
        "numerical_reference_count_per_budget": summary_dict['low']['numerical_reference_count'],
        "sample_max_reference_gap_mNats": {
            "low": round(summary_dict['low']['reference_gap_nom']['sample_max'] * 1000, 3),
            "mid": round(summary_dict['mid']['reference_gap_nom']['sample_max'] * 1000, 3),
            "high": round(summary_dict['high']['reference_gap_nom']['sample_max'] * 1000, 3)
        }
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_dict, f, indent=2)
    print(f"[+] Saved metadata JSON to {meta_path}")

if __name__ == '__main__':
    run_exp05()
