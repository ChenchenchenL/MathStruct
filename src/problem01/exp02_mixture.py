# -*- coding: utf-8 -*-
"""EXP-20260923-02: Problem-1 mixture-regression baseline on A4/A5 (1M).

Implements model.md M1-v3: EQ14 target, EQ15 pseudo-count smoothing,
EQ16 ALR (ref=pile_cc), EQ17 OLS vs HistGB (5-fold CV inside A4/A5 only),
EQ22/23 transfer-effect matrix + bootstrap CI, and protocol-v2 item 3
(no-calibration Spearman on est tables A12/A13, A14/A15).

Run: python exp02_mixture.py
"""
import os
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

SEED = 20260923
BASE = r"D:\project\MathStruct\data\real_attachments\A_data_value\regmix_tables"
OUT = r"D:\project\MathStruct\result\problem01\exp02_mixture"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

DOMAINS17 = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
             "dm_mathematics", "github", "philpapers", "stackexchange",
             "enron_emails", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
             "europarl", "hackernews", "pubmed_abstracts", "uspto_backgrounds"]
LOSS13 = [d for d in DOMAINS17 if d not in
          ("nih_exporter", "enron_emails", "europarl", "philpapers")]
REF = "pile_cc"
EPS = 1e-3

mix = pd.read_csv(os.path.join(BASE, "train_mixture_1m.csv"))
loss = pd.read_csv(os.path.join(BASE, "train_pile_loss_1m.csv"))
mix = mix.sort_values("index").reset_index(drop=True)
loss = loss.sort_values("index").reset_index(drop=True)
assert (mix["index"].values == loss["index"].values).all()

P = mix[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
row_sums = P.sum(axis=1, keepdims=True)
print(f"[1] A4 rows={len(P)}  row-sum min/max = {row_sums.min():.6f}/{row_sums.max():.6f} (AS-08)")
P = P / row_sums  # AS-08 renormalization

y = loss[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float).mean(axis=1)
print(f"[2] target y (13-domain mean val_loss): mean={y.mean():.4f} std={y.std():.4f} min={y.min():.4f} max={y.max():.4f}")

# EQ15 pseudo-count smoothing (keeps sum 1), EQ16 ALR with ref=pile_cc
# NOTE: drop the ref column so X columns align 1:1 with alr_cols.
Pt = (P + EPS / 17.0) / (1.0 + EPS)
ref_col = DOMAINS17.index(REF)
X = np.log(Pt / Pt[:, [ref_col]])
X = np.delete(X, ref_col, axis=1)
alr_cols = [d for d in DOMAINS17 if d != REF]
print(f"[3] ALR matrix: {X.shape}, ref={REF}, eps={EPS}")

# ---- model selection inside A4/A5 only (5-fold CV) -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


def cv_scores(model_fn):
    rmse, r2 = [], []
    for tr, te in kf.split(X):
        m = model_fn()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        rmse.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        ss = 1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[tr].mean()) ** 2)
        r2.append(float(ss))
    return float(np.mean(rmse)), float(np.std(rmse)), float(np.mean(r2))


base_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
rmse_l, sd_l, r2_l = cv_scores(lambda: LinearRegression())
rmse_h, sd_h, r2_h = cv_scores(
    lambda: HistGradientBoostingRegressor(random_state=SEED, max_iter=300))
print(f"[4] 5-fold CV inside A4/A5 (protocol v2 item 1):")
print(f"    mean-baseline RMSE        = {base_rmse:.4f}")
print(f"    OLS(ALR)     RMSE={rmse_l:.4f}±{sd_l:.4f}  R2={r2_l:.4f}")
print(f"    HistGB       RMSE={rmse_h:.4f}±{sd_h:.4f}  R2={r2_h:.4f}")

# ---- fit OLS on all 512, transfer matrix EQ22/23 ----------------------------
ols = LinearRegression().fit(X, y)
beta = dict(zip(alr_cols, ols.coef_))
r2_full = float(ols.score(X, y))
print(f"[5] full-fit OLS R2={r2_full:.4f} (data-note claimed 0.93 -> verify)")

p0 = P.mean(axis=0)  # mean mixture as baseline p0
idx13 = {d: DOMAINS17.index(d) for d in LOSS13}
FRAC = 0.5  # proportional transfer: move FRAC of a's share to b (m = FRAC*p_a, always inside [0, p_a])
print("[5b] mean mixture p0 (x100) =", {d: round(float(p0[i]) * 100, 3) for i, d in enumerate(DOMAINS17)})


def delta_general(a, b, p_start, model_beta, cols, frac=FRAC):
    """EQ22 transfer effect, PROPORTIONAL route: m = frac * p_a (guaranteed
    in [0, p_a] for every domain). Recompute ALR features at p + m(e_b - e_a),
    take the linear-model difference. Valid for any (a, b) incl. ref."""
    p = p_start.copy()
    ia, ib = DOMAINS17.index(a), DOMAINS17.index(b)
    m_use = frac * p[ia]
    p[ia] -= m_use
    p[ib] += m_use
    pt = (p + EPS / 17.0) / (1.0 + EPS)
    p1t = (p_start + EPS / 17.0) / (1.0 + EPS)
    diff17 = np.log(pt / pt[ref_col]) - np.log(p1t / p1t[ref_col])  # 17-dim; ref entry = 0
    return float(sum(model_beta[d] * diff17[DOMAINS17.index(d)] for d in cols))


M = pd.DataFrame(index=LOSS13, columns=LOSS13, dtype=float)
for a in LOSS13:
    for b in LOSS13:
        if a == b:
            M.loc[a, b] = 0.0
        else:
            M.loc[a, b] = delta_general(a, b, p0, beta, alr_cols)
print(f"[6] transfer-effect matrix dy for proportional transfer r={FRAC} from p0 (rows: reduce, cols: increase):")
print(M.round(4).to_string())

stack = []
for a in LOSS13:
    for b in LOSS13:
        if a != b:
            stack.append((a, b, float(M.loc[a, b])))
stack.sort(key=lambda t: t[2])
print("[6b] top-5 negative (reduce a, increase b lowers loss) & top-5 positive:")
for a, b, v in stack[:5]:
    print(f"    reduce {a:>18s} -> increase {b:<18s} dy={v:+.4f}")
for a, b, v in stack[-5:]:
    print(f"    reduce {a:>18s} -> increase {b:<18s} dy={v:+.4f}")

# bootstrap CI for the 5 strongest pairs
rng = np.random.default_rng(SEED)
strong = stack[:5] + stack[-5:]
B = 200
boot = {(k[0], k[1]): [] for k in strong}
for _ in range(B):
    idx = rng.integers(0, len(X), len(X))
    ols_b = LinearRegression().fit(X[idx], y[idx])
    bb = dict(zip(alr_cols, ols_b.coef_))
    for (a, b, _) in strong:
        boot[(a, b)].append(delta_general(a, b, p0, bb, alr_cols))
print(f"[6c] bootstrap 95% CI (B={B}) for strongest pairs:")
for (a, b, v) in strong:
    lo, hi = np.percentile(boot[(a, b)], [2.5, 97.5])
    print(f"    {a:>18s} -> {b:<18s} dy={v:+.4f}  CI[{lo:+.4f}, {hi:+.4f}]{' *sig' if lo > 0 or hi < 0 else ''}")

# ---- protocol v2 item 3: no-calibration Spearman on est tables --------------
print("[7] est-table no-calibration rank check (A12/A13 10B; A14/A15 70B):")
for tag, mfile, lfile in (("10B", "est_mixture_10b.csv", "est_pile_loss_10b.csv"),
                          ("70B", "est_mixture_70b.csv", "est_pile_loss_70b.csv")):
    me = pd.read_csv(os.path.join(BASE, mfile)).sort_values("index").reset_index(drop=True)
    le = pd.read_csv(os.path.join(BASE, lfile)).sort_values("index").reset_index(drop=True)
    if not (me["index"].values == le["index"].values).all():
        me = me.set_index("index").loc[le["index"]].reset_index()
    Pe = me[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
    Pe = Pe / Pe.sum(axis=1, keepdims=True)
    Pte = (Pe + EPS / 17.0) / (1.0 + EPS)
    Xe = np.log(Pte / Pte[:, [ref_col]])
    Xe = np.delete(Xe, ref_col, axis=1)
    pred = ols.predict(Xe)
    ye = le[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float).mean(axis=1)
    rho, _ = spearmanr(pred, ye)
    print(f"    {tag}: n={len(ye)}  Spearman(pred, est-mean-loss) = {rho:.4f}  (est=extrapolated, rank-only per AS-06)")

# ---- sensitivity: eps and ref ----------------------------------------------
print("[8] sensitivity (CV R2, 16-col ALR consistent with main run):")
for eps_try in (1e-4, 1e-3, 1e-2):
    Ptt = (P + eps_try / 17.0) / (1.0 + eps_try)
    Xt = np.log(Ptt / Ptt[:, [ref_col]])
    Xt = np.delete(Xt, ref_col, axis=1)
    r = []
    for tr, te in kf.split(Xt):
        mm = LinearRegression().fit(Xt[tr], y[tr])
        pred = mm.predict(Xt[te])
        r.append(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[tr].mean()) ** 2))
    print(f"    eps={eps_try:.0e}: R2={np.mean(r):.4f}")
for ref_try in ("pile_cc", "wikipedia_en", "arxiv"):
    Ptt = (P + EPS / 17.0) / (1.0 + EPS)
    rc = DOMAINS17.index(ref_try)
    Xt = np.log(Ptt / Ptt[:, [rc]])
    Xt = np.delete(Xt, rc, axis=1)
    r = []
    for tr, te in kf.split(Xt):
        mm = LinearRegression().fit(Xt[tr], y[tr])
        pred = mm.predict(Xt[te])
        r.append(1 - np.sum((y[te] - pred) ** 2) / np.sum((y[te] - y[tr].mean()) ** 2))
    print(f"    ref={ref_try}: R2={np.mean(r):.4f}")

M.to_csv(os.path.join(OUT, "transfer_matrix_m05.csv"))
with open(os.path.join(OUT, "exp02_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED, "runtime_s": round(time.time() - t0, 1),
        "rowsum_minmax": [float(row_sums.min()), float(row_sums.max())],
        "y_stats": {"mean": float(y.mean()), "std": float(y.std())},
        "cv": {"baseline_rmse": base_rmse, "ols_rmse": rmse_l, "ols_r2": r2_l,
               "histgb_rmse": rmse_h, "histgb_r2": r2_h},
        "full_ols_r2": r2_full,
        "top_transfers": stack[:5],
        "worst_transfers": stack[-5:],
        "est_spearman": "see stdout",
    }, f, ensure_ascii=False, indent=2)
print(f"[done] {time.time()-t0:.1f}s -> {OUT}")
