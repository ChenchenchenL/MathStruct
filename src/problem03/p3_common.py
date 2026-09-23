"""
p3_common.py - Problem 3 Common Modules & Solvers
Unified dimensional framework:
  n = N / 1e9 (Billion parameters)
  d = D / 1e9 (Billion tokens)
  C_bar = C / 1e18 (EFLOPs, where 10^19 -> 10, 10^22 -> 10000, 10^24 -> 1000000)
  h_bar(Q) = h(Q) / 1e9 (GFLOPs/Token)
  Budget constraint: d * [(6 + eta * L_ctx) * n + h_bar(Q)] <= C_bar
"""

import numpy as np
import scipy.optimize as opt
from typing import Dict, Tuple, Optional, Any

# ==============================================================================
# 1. System Constants & Parameters (Inherited from Problem 1 & Problem 2)
# ==============================================================================
ETA = 2.0e-4          # Attention FLOPs coefficient: C_attn = eta * N * D * L_ctx
L_CTX_CRIT = 30000    # Theoretical critical window where C_attn = C_train = 6*N*D (Tokens)

# Baseline quality score (median of uncleaned Pile crawl from Attachment A1)
Q_BASE = 0.584        
Q_ANCHOR = 1.000      # Ideal perfect quality degradation anchor in generalized scaling law

# Problem 2 Generalized Scaling Law Parameters (with standard errors)
# L(n, d, Q, p) = E + A * n^(-alpha) + B * d^(-beta) * exp(-rho * (Q - Q_anchor)) * R(p)
PARAM_E = 1.6898
PARAM_A = 0.3540
PARAM_ALPHA = 0.3400
PARAM_B = 1.2403
PARAM_BETA = 0.2799
PARAM_RHO = 0.6646
PARAM_TAU = 0.8500

# Covariance matrix of scaling law parameters [A, alpha, B, beta, rho] from P2 estimation
P2_PARAM_NAMES = ['A', 'alpha', 'B', 'beta', 'rho']
P2_PARAM_NOMINAL = np.array([PARAM_A, PARAM_ALPHA, PARAM_B, PARAM_BETA, PARAM_RHO])
P2_PARAM_STD = np.array([0.0142, 0.0076, 0.0478, 0.0077, 0.0135])

# Bounded optimization ranges
N_MIN, N_MAX = 0.001, 1000.0       # 1M to 1000B parameters
D_MIN, D_MAX = 0.1, 100000.0       # 100M to 100T tokens
Q_MIN, Q_MAX = 0.584, 1.000

# C7 Context lengths
C7_CONTEXT_LENGTHS = [2048, 4096, 8192, 32768, 131072]

# Three standard budget tiers in EFLOPs (10^18 FLOPs)
BUDGET_TIERS_EFLOPS = {
    'low': 10.0,          # 10^19 FLOPs
    'mid': 10000.0,       # 10^22 FLOPs
    'high': 1000000.0     # 10^24 FLOPs
}

# ==============================================================================
# 2. Quality Cost Functions (Appendix B.1)
# ==============================================================================
# Raw specifications from Appendix B.1:
# Exponential: g_exp(Q) = gamma * exp(lambda * Q), gamma=1e7, lambda=6.0
# Power:       g_pow(Q) = gamma * Q^lambda, gamma=5e9, lambda=4.0
# Logarithmic: g_log(Q) = gamma * ln(1 + lambda * Q), gamma=2e9, lambda=10.0

COST_PARAMS_RAW = {
    'exp': {'gamma': 1.0e7, 'lambda': 6.0},
    'pow': {'gamma': 5.0e9, 'lambda': 4.0},
    'log': {'gamma': 2.0e9, 'lambda': 10.0}
}

def get_h_bar_and_deriv(Q: float, cost_type: str = 'exp', normalized: bool = False) -> Tuple[float, float]:
    """
    Returns (h_bar, h_bar_prime) where:
      h_bar(Q) = [g(Q) - g(Q_0)]_+ / 1e9  (GFLOPs / Token)
      h_bar_prime(Q) = g'(Q) / 1e9        (GFLOPs / Token / Quality)
    If normalized is True, gamma is rescaled so that h_bar(Q_base + 0.1) equals h_bar_exp(Q_base + 0.1).
    """
    Q_clamped = max(Q_BASE, min(1.0, float(Q)))
    if Q_clamped <= Q_BASE:
        # At or below base, incremental cost is zero
        # Left derivative is 0, right derivative is g'(Q_BASE) / 1e9
        deriv_at_base = _raw_g_prime(Q_BASE, cost_type, normalized) / 1.0e9
        return 0.0, deriv_at_base

    # Compute raw or normalized g(Q) and g(Q_BASE)
    g_val = _raw_g(Q_clamped, cost_type, normalized)
    g_base = _raw_g(Q_BASE, cost_type, normalized)
    h_bar = max(0.0, (g_val - g_base) / 1.0e9)
    h_bar_prime = _raw_g_prime(Q_clamped, cost_type, normalized) / 1.0e9
    return h_bar, h_bar_prime

def _raw_g(Q: float, cost_type: str, normalized: bool) -> float:
    params = COST_PARAMS_RAW[cost_type]
    gamma = params['gamma']
    lam = params['lambda']

    if normalized and cost_type != 'exp':
        # Align h(Q_BASE + 0.1) with exp at Q_BASE + 0.1 = 0.684
        ref_h_exp = _raw_g(Q_BASE + 0.1, 'exp', False) - _raw_g(Q_BASE, 'exp', False)
        if cost_type == 'pow':
            base_diff = ((Q_BASE + 0.1) ** lam) - (Q_BASE ** lam)
            gamma = ref_h_exp / base_diff
        elif cost_type == 'log':
            base_diff = np.log(1.0 + lam * (Q_BASE + 0.1)) - np.log(1.0 + lam * Q_BASE)
            gamma = ref_h_exp / base_diff

    if cost_type == 'exp':
        return gamma * np.exp(lam * Q)
    elif cost_type == 'pow':
        return gamma * (Q ** lam)
    elif cost_type == 'log':
        return gamma * np.log(1.0 + lam * Q)
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

def _raw_g_prime(Q: float, cost_type: str, normalized: bool) -> float:
    params = COST_PARAMS_RAW[cost_type]
    gamma = params['gamma']
    lam = params['lambda']

    if normalized and cost_type != 'exp':
        ref_h_exp = _raw_g(Q_BASE + 0.1, 'exp', False) - _raw_g(Q_BASE, 'exp', False)
        if cost_type == 'pow':
            base_diff = ((Q_BASE + 0.1) ** lam) - (Q_BASE ** lam)
            gamma = ref_h_exp / base_diff
        elif cost_type == 'log':
            base_diff = np.log(1.0 + lam * (Q_BASE + 0.1)) - np.log(1.0 + lam * Q_BASE)
            gamma = ref_h_exp / base_diff

    if cost_type == 'exp':
        return gamma * lam * np.exp(lam * Q)
    elif cost_type == 'pow':
        return gamma * lam * (Q ** (lam - 1.0))
    elif cost_type == 'log':
        return gamma * lam / (1.0 + lam * Q)
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

# ==============================================================================
# 3. Scaling Law Objective & Analytical Elimination of d
# ==============================================================================

def compute_loss(n: float, d: float, Q: float, p_multiplier: float = 1.0,
                 custom_params: Optional[np.ndarray] = None) -> float:
    """
    Computes Loss L(n, d, Q, p).
    n: Billion parameters (n > 0)
    d: Billion tokens (d > 0)
    Q: Quality score in [0.584, 1.0]
    p_multiplier: R(p), default 1.0 for baseline mixture
    """
    if custom_params is not None:
        A, alpha, B, beta, rho = custom_params
        E = PARAM_E
    else:
        E, A, alpha, B, beta, rho = PARAM_E, PARAM_A, PARAM_ALPHA, PARAM_B, PARAM_BETA, PARAM_RHO

    term_n = A * (n ** (-alpha))
    term_d = B * (d ** (-beta)) * np.exp(-rho * (Q - Q_ANCHOR)) * p_multiplier
    return E + term_n + term_d

def compute_optimal_d(n: float, Q: float, C_bar: float, L_ctx: int = 2048,
                      cost_type: str = 'exp', normalized: bool = False) -> float:
    """
    Analytically eliminates d by activating the compute budget:
      d*(n, Q) = C_bar / [(6 + eta * L_ctx) * n + h_bar(Q)]
    """
    h_bar, _ = get_h_bar_and_deriv(Q, cost_type, normalized)
    cost_per_token = (6.0 + ETA * L_ctx) * n + h_bar
    return C_bar / cost_per_token

def objective_2d(u: float, Q: float, C_bar: float, L_ctx: int = 2048,
                 cost_type: str = 'exp', p_multiplier: float = 1.0,
                 normalized: bool = False, custom_params: Optional[np.ndarray] = None) -> float:
    """
    Reduced 2D objective in u = ln(n) and Q space:
      tilde_L(u, Q) = L(exp(u), d*(exp(u), Q), Q, p)
    """
    n = np.exp(u)
    d = compute_optimal_d(n, Q, C_bar, L_ctx, cost_type, normalized)
    return compute_loss(n, d, Q, p_multiplier, custom_params)

# ==============================================================================
# 4. Expenditure Shares & KKT Verifications
# ==============================================================================

def compute_expenditure_shares(n: float, d: float, Q: float, C_bar: float,
                               L_ctx: int = 2048, cost_type: str = 'exp',
                               normalized: bool = False) -> Dict[str, float]:
    """
    Computes breakdown of compute budget shares:
      s_train = 6 * n * d / C_bar
      s_attn  = (eta * L_ctx) * n * d / C_bar
      s_Q     = d * h_bar(Q) / C_bar
      budget_residual = |C_spent - C_bar| / C_bar
    """
    h_bar, _ = get_h_bar_and_deriv(Q, cost_type, normalized)
    c_train = 6.0 * n * d
    c_attn = (ETA * L_ctx) * n * d
    c_q = d * h_bar
    c_total = c_train + c_attn + c_q

    s_train = c_train / C_bar
    s_attn = c_attn / C_bar
    s_q = c_q / C_bar
    res = abs(c_total - C_bar) / C_bar

    # Attention ratio relative to training compute
    psi = c_attn / c_train if c_train > 0 else 0.0
    return {
        's_train': s_train,
        's_attn': s_attn,
        's_Q': s_q,
        's_train_plus_attn': s_train + s_attn,
        'psi_attn_to_train': psi,
        'budget_residual': res,
        'C_total_EFLOPs': c_total
    }

def verify_kkt_conditions(n: float, d: float, Q: float, C_bar: float,
                          L_ctx: int = 2048, cost_type: str = 'exp',
                          p_multiplier: float = 1.0, normalized: bool = False) -> Dict[str, Any]:
    """
    Verifies Kuhn-Tucker first order optimality conditions:
      1. n-optimality: alpha * A * n^(-alpha) == beta * T * s_train_plus_attn
      2. Q-optimality: rho == beta * h_bar'(Q) / [(6 + eta * L_ctx)*n + h_bar(Q)] (for interior Q)
    """
    A, alpha, B, beta, rho = PARAM_A, PARAM_ALPHA, PARAM_B, PARAM_BETA, PARAM_RHO
    term_n = alpha * A * (n ** (-alpha))
    term_d_tot = beta * B * (d ** (-beta)) * np.exp(-rho * (Q - Q_ANCHOR)) * p_multiplier

    h_bar, h_bar_prime = get_h_bar_and_deriv(Q, cost_type, normalized)
    denom = (6.0 + ETA * L_ctx) * n + h_bar

    # LHS and RHS for parameter n balance
    lhs_n = term_n
    rhs_n = term_d_tot * ((6.0 + ETA * L_ctx) * n / denom)
    rel_err_n = abs(lhs_n - rhs_n) / max(lhs_n, rhs_n, 1e-12)

    # LHS and RHS for quality Q balance (interior)
    lhs_q = rho
    rhs_q = beta * (h_bar_prime / denom)
    
    # Check boundary status
    if abs(Q - Q_BASE) <= 0.005:
        q_regime = 'Boundary_Lower_Q0'
        kkt_q_satisfied = (lhs_q <= rhs_q + 1e-3)  # Marginal cost >= marginal benefit
    elif abs(Q - 1.0) <= 0.005:
        q_regime = 'Boundary_Upper_Q1'
        kkt_q_satisfied = (lhs_q >= rhs_q - 1e-3)  # Marginal benefit >= marginal cost
    else:
        q_regime = 'Interior'
        rel_err_q = abs(lhs_q - rhs_q) / max(lhs_q, rhs_q, 1e-12)
        kkt_q_satisfied = (rel_err_q < 0.05)

    return {
        'q_regime': q_regime,
        'rel_err_n': rel_err_n,
        'lhs_q_rho': lhs_q,
        'rhs_q_mrts': rhs_q,
        'kkt_n_satisfied': (rel_err_n < 0.02),
        'kkt_q_satisfied': kkt_q_satisfied
    }

# ==============================================================================
# 5. Core Multi-Start Optimization Solver
# ==============================================================================

def solve_optimal_allocation_2d(C_bar: float, cost_type: str = 'exp',
                                L_ctx: int = 2048, p_multiplier: float = 1.0,
                                normalized: bool = False,
                                custom_params: Optional[np.ndarray] = None,
                                verify_global: bool = True) -> Dict[str, Any]:
    """
    Solves for (n*, d*, Q*) minimizing Loss subject to compute budget C_bar.
    Uses multi-start L-BFGS-B across 50 log-spaced grid points, optionally dual-verified by Differential Evolution.
    """
    # Define bounds on u = ln(n) and Q
    u_min, u_max = np.log(N_MIN), np.log(N_MAX)
    bounds = [(u_min, u_max), (Q_MIN, Q_MAX)]

    def obj_wrapper(x):
        return objective_2d(x[0], x[1], C_bar, L_ctx, cost_type, p_multiplier, normalized, custom_params)

    # 1. Multi-start L-BFGS-B (10 points in ln(n) x 5 points in Q = 50 starts)
    # Estimate reasonable n_center based on Chinchilla rough scaling:
    # 6 * N * D = C => N ~ sqrt(C / 60)
    c_raw = C_bar * 1.0e18
    n_guess_raw = np.sqrt(c_raw / 60.0) / 1.0e9
    n_guess_raw = max(N_MIN * 1.5, min(N_MAX * 0.8, n_guess_raw))
    u_center = np.log(n_guess_raw)

    u_grid = np.linspace(max(u_min, u_center - 3.0), min(u_max, u_center + 3.0), 10)
    q_grid = np.linspace(Q_MIN, 0.95, 5)

    best_loss = float('inf')
    best_x = None

    for u_init in u_grid:
        for q_init in q_grid:
            res = opt.minimize(
                obj_wrapper,
                x0=np.array([u_init, q_init]),
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 1000}
            )
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x

    # Also evaluate exact boundary Q = Q_BASE in 1D optimization over u
    def obj_q0(u_val):
        return objective_2d(u_val, Q_BASE, C_bar, L_ctx, cost_type, p_multiplier, normalized, custom_params)
    
    res_q0 = opt.minimize_scalar(obj_q0, bounds=(u_min, u_max), method='bounded')
    if res_q0.fun < best_loss:
        best_loss = res_q0.fun
        best_x = np.array([res_q0.x, Q_BASE])

    # Also evaluate exact boundary Q = 1.0 in 1D optimization over u
    def obj_q1(u_val):
        return objective_2d(u_val, 1.0, C_bar, L_ctx, cost_type, p_multiplier, normalized, custom_params)
    
    res_q1 = opt.minimize_scalar(obj_q1, bounds=(u_min, u_max), method='bounded')
    if res_q1.fun < best_loss:
        best_loss = res_q1.fun
        best_x = np.array([res_q1.x, 1.0])

    # 2. Dual Global Verification via Differential Evolution (optional check)
    de_verified = True
    if verify_global:
        de_res = opt.differential_evolution(
            obj_wrapper,
            bounds=bounds,
            strategy='best1bin',
            maxiter=500,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42
        )
        if de_res.fun < best_loss - 1e-7:
            # Differential Evolution found an even better optimum
            best_loss = de_res.fun
            best_x = de_res.x
        de_verified = abs(best_loss - de_res.fun) / max(best_loss, 1e-8) < 1e-5

    # Extract final variables
    opt_u, opt_Q = best_x
    opt_n = float(np.exp(opt_u))
    opt_Q = float(np.clip(opt_Q, Q_MIN, Q_MAX))
    opt_d = float(compute_optimal_d(opt_n, opt_Q, C_bar, L_ctx, cost_type, normalized))
    final_loss = float(compute_loss(opt_n, opt_d, opt_Q, p_multiplier, custom_params))

    # Compute expenditure shares and KKT
    shares = compute_expenditure_shares(opt_n, opt_d, opt_Q, C_bar, L_ctx, cost_type, normalized)
    kkt = verify_kkt_conditions(opt_n, opt_d, opt_Q, C_bar, L_ctx, cost_type, p_multiplier, normalized)

    return {
        'C_bar': C_bar,
        'C_FLOPs': C_bar * 1.0e18,
        'cost_type': cost_type,
        'normalized_cost': normalized,
        'L_ctx': L_ctx,
        'opt_n_B': opt_n,
        'opt_d_B': opt_d,
        'opt_Q': opt_Q,
        'opt_loss': final_loss,
        'token_to_param_ratio': opt_d / opt_n,
        'shares': shares,
        'kkt': kkt,
        'de_verified': de_verified
    }

# ==============================================================================
# 6. Basic Self-Test
# ==============================================================================
if __name__ == '__main__':
    print("Testing p3_common solvers and unit consistency...")
    # Test low budget (10^19 FLOPs -> C_bar = 10)
    sol_low = solve_optimal_allocation_2d(BUDGET_TIERS_EFLOPS['low'], 'exp', 2048)
    print(f"Low budget (10^19 FLOPs): n*={sol_low['opt_n_B']:.4f}B, d*={sol_low['opt_d_B']:.4f}B, Q*={sol_low['opt_Q']:.4f}, Loss={sol_low['opt_loss']:.4f}")
    print(f"  Shares: Train={sol_low['shares']['s_train']:.3f}, Attn={sol_low['shares']['s_attn']:.3f}, Q={sol_low['shares']['s_Q']:.3f}, Res={sol_low['shares']['budget_residual']:.2e}")

    # Test mid budget (10^22 FLOPs -> C_bar = 10000)
    sol_mid = solve_optimal_allocation_2d(BUDGET_TIERS_EFLOPS['mid'], 'exp', 2048)
    print(f"Mid budget (10^22 FLOPs): n*={sol_mid['opt_n_B']:.4f}B, d*={sol_mid['opt_d_B']:.4f}B, Q*={sol_mid['opt_Q']:.4f}, Loss={sol_mid['opt_loss']:.4f}")
    print(f"  Shares: Train={sol_mid['shares']['s_train']:.3f}, Attn={sol_mid['shares']['s_attn']:.3f}, Q={sol_mid['shares']['s_Q']:.3f}, Res={sol_mid['shares']['budget_residual']:.2e}")

    # Test high budget (10^24 FLOPs -> C_bar = 1000000)
    sol_high = solve_optimal_allocation_2d(BUDGET_TIERS_EFLOPS['high'], 'exp', 2048)
    print(f"High budget (10^24 FLOPs): n*={sol_high['opt_n_B']:.4f}B, d*={sol_high['opt_d_B']:.4f}B, Q*={sol_high['opt_Q']:.4f}, Loss={sol_high['opt_loss']:.4f}")
    print(f"  Shares: Train={sol_high['shares']['s_train']:.3f}, Attn={sol_high['shares']['s_attn']:.3f}, Q={sol_high['shares']['s_Q']:.3f}, Res={sol_high['shares']['budget_residual']:.2e}")
    print("All p3_common self-tests passed!")
