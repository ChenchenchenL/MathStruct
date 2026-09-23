# -*- coding: utf-8 -*-
"""Common utilities, mathematical definitions, and plotting functions for Problem 2.
Python environment: AIOPS (D:\Program Files\CondaEnvs\AIOPS\python.exe).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Global Constants & Paths
SEED = 20260923
np.random.seed(SEED)

BASE_DATA = r"d:\project\MathStruct\data\real_attachments"
DIR_B = os.path.join(BASE_DATA, "B_scaling_laws")
DIR_A = os.path.join(BASE_DATA, "A_data_value")
RES_P1 = r"d:\project\MathStruct\result"
RES_P2 = r"d:\project\MathStruct\result\problem02"
OUT_FIGS = os.path.join(r"d:\project\MathStruct\result\figures", "problem02")
OUT_TABS = os.path.join(r"d:\project\MathStruct\result\tables", "problem02")

os.makedirs(RES_P2, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)
os.makedirs(OUT_TABS, exist_ok=True)

# Matplotlib Publication Styling (per result/AGENTS.md: clean, no in-plot title, high DPI)
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


def classical_scaling_law(N, D, E, A, alpha, B, beta):
    """M2-EQ01 baseline: L(N, D) = E + A * N^(-alpha) + B * D^(-beta)."""
    return E + A * (np.asarray(N, dtype=float) ** (-alpha)) + B * (np.asarray(D, dtype=float) ** (-beta))


def generalized_scaling_law(N, D, Q, p_mult, E, A, alpha, B, beta, rho, Q0=0.584):
    """M2-EQ01 generalized:
    L(N, D, Q, p) = E + A*N^(-alpha) + B*D^(-beta) * exp(-rho*(Q - Q0)) * R(p).
    """
    term_N = A * (np.asarray(N, dtype=float) ** (-alpha))
    term_D = B * (np.asarray(D, dtype=float) ** (-beta)) * np.exp(-rho * (np.asarray(Q, dtype=float) - Q0)) * np.asarray(p_mult, dtype=float)
    return E + term_N + term_D


def compute_elasticities(N, D, Q, p_mult, E, A, alpha, B, beta, rho, Q0=0.584):
    """Compute dimensionless point elasticities on L and reducible loss (L - E)."""
    term_N = A * (N ** (-alpha))
    term_D = B * (D ** (-beta)) * np.exp(-rho * (Q - Q0)) * p_mult
    L = E + term_N + term_D
    reducible_L = term_N + term_D

    eps_N = -alpha * term_N / L
    eps_D = -beta * term_D / L
    eps_Q = -rho * Q * term_D / L

    eps_N_star = -alpha * term_N / reducible_L
    eps_D_star = -beta * term_D / reducible_L
    eps_Q_star = -rho * Q * term_D / reducible_L

    # Marginal rate of technical substitution MRTS = - dN / dQ |_L=const
    # = (rho * term_D) / (alpha * A * N^(-alpha - 1))
    mrts_NQ = (rho * term_D) / (alpha * A * (N ** (-alpha - 1)))

    return {
        'L': L,
        'reducible_L': reducible_L,
        'eps_N': eps_N,
        'eps_D': eps_D,
        'eps_Q': eps_Q,
        'eps_N_star': eps_N_star,
        'eps_D_star': eps_D_star,
        'eps_Q_star': eps_Q_star,
        'mrts_NQ': mrts_NQ
    }


def compute_substitution_01(N, D, Q, p_mult, E, A, alpha, B, beta, rho, Q0=0.584, delta_Q=0.1):
    """Compute quality-scale substitution for +0.1 quality improvement (M2-EQ04).
    Direction 1: Parameter Downsizing Equivalence (Delta N_save = N - N_new).
    Direction 2: Scaling Equivalent Gain (Delta N_gain = N_equiv - N).
    """
    T = B * (D ** (-beta)) * np.exp(-rho * (Q - Q0)) * p_mult
    delta_L_Q = T * (1.0 - np.exp(-rho * delta_Q))

    # Direction 1: Downsizing
    inner_save = (N ** (-alpha)) + (T / A) * (1.0 - np.exp(-rho * delta_Q))
    N_new = inner_save ** (-1.0 / alpha)
    delta_N_save = N - N_new

    # Direction 2: Equivalent expansion & Ceiling Saturation check
    margin = (N ** (-alpha)) - (T / A) * (1.0 - np.exp(-rho * delta_Q))
    N_crit = (A / (T * (1.0 - np.exp(-rho * delta_Q)))) ** (1.0 / alpha)

    if margin > 1e-7:
        N_equiv = margin ** (-1.0 / alpha)
        delta_N_gain = N_equiv - N
        saturated = False
    else:
        N_equiv = np.inf
        delta_N_gain = np.inf
        saturated = True

    return {
        'N': N,
        'D': D,
        'Q': Q,
        'delta_Q': delta_Q,
        'delta_L_Q': delta_L_Q,
        'N_new': N_new,
        'delta_N_save': delta_N_save,
        'N_crit': N_crit,
        'margin': margin,
        'N_equiv': N_equiv,
        'delta_N_gain': delta_N_gain,
        'saturated': saturated
    }


def save_figure_with_caption(fig, fig_path, caption_text, metadata_dict=None, dpi=300):
    """Save high-DPI figure, companion .caption.md file, and .json metadata per result/AGENTS.md."""
    fig.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    png_path = fig_path.replace('.pdf', '.png')
    if fig_path.endswith('.pdf'):
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')

    caption_path = fig_path + ".caption.md"
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write(caption_text.strip() + "\n")

    if metadata_dict is not None:
        json_path = fig_path.replace('.pdf', '.json')
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        print(f"Saved: {fig_path}, {caption_path} & {json_path}")
    else:
        print(f"Saved: {fig_path} & {caption_path}")
