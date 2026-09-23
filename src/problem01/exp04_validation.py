# -*- coding: utf-8 -*-
"""EXP-20260923-04: full test-set validation on A6-A11 + remaining mixture items.

Protocol v2 coverage:
  item 2: A6/A7 (1M, 256 pairs) independent same-scale test  -> RMSE/R2/Spearman
  item 3: A8-A11 (60M/1B) NO-calibration Spearman
  item 4: A8-A11 split-half calibration RMSE (odd index fit intercept, even eval)
Extras: coefficient drift across scales (AS-05), HistGB transfer check (EQ23b),
3-domain combination case (EQ23c), AS-09 (with/without 4 no-loss columns),
AS-04 (target weighting sensitivity), bootstrap B=1000.

Run: python exp04_validation.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

SEED = 20260923
BASE = r"D:\project\MathStruct\data\real_attachments\A_data_value\regmix_tables"
OUT = r"D:\project\MathStruct\result\problem01\exp04_validation"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

DOMAINS17 = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
             "dm_mathematics", "github", "philpapers", "stackexchange",
             "enron_emails", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
             "europarl", "hackernews", "pubmed_abstracts", "uspto_backgrounds"]
LOSS13 = [d for d in DOMAINS17 if d not in
          ("nih_exporter", "enron_emails", "europarl", "philpapers")]
REF = "pile_cc"
REF_C = DOMAINS17.index(REF)
EPS = 1e-3


def alr(P, eps=EPS):
    Pt = (P + eps / 17.0) / (1.0 + eps)
    X = np.log(Pt / Pt[:, [REF_C]])
    return np.delete(X, REF_C, axis=1)


def load_pair(mfile, lfile):
    me = pd.read_csv(os.path.join(BASE, mfile)).sort_values("index").reset_index(drop=True)
    le = pd.read_csv(os.path.join(BASE, lfile)).sort_values("index").reset_index(drop=True)
    if not (me["index"].values == le["index"].values).all():
        me = me.set_index("index").loc[le["index"]].reset_index()
    P = me[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
    P = P / P.sum(axis=1, keepdims=True)                      # AS-08
    L = le[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float)
    return P, L


print("[1] loading train (A4/A5) and tests (A6-A11) ...", flush=True)
P_tr, L_tr = load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
X_tr, y_tr = alr(P_tr), L_tr.mean(axis=1)
tests = {}
for tag, mf, lf in (("1M", "test_mixture_1m.csv", "test_pile_loss_1m.csv"),
                    ("60M", "test_mixture_60m.csv", "test_pile_loss_60m.csv"),
                    ("1B", "test_mixture_1B.csv", "test_pile_loss_1B.csv")):
    P_t, L_t = load_pair(mf, lf)
    tests[tag] = (P_t, L_t, alr(P_t), L_t.mean(axis=1))
    print(f"    {tag}: n={len(P_t)}")

ols = LinearRegression().fit(X_tr, y_tr)
hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=300).fit(X_tr, y_tr)
base_rmse = float(np.sqrt(np.mean((y_tr - y_tr.mean()) ** 2)))
print(f"[2] models fit on A4/A5 (n={len(y_tr)}); train-mean-baseline RMSE={base_rmse:.4f}")

# ---- item 2: A6/A7 independent same-scale test ------------------------------
P_te, L_te, X_te, y_te = tests["1M"]
for name, mdl in (("OLS", ols), ("HistGB", hgb)):
    pred = mdl.predict(X_te)
    rmse = float(np.sqrt(np.mean((y_te - pred) ** 2)))
    r2_tr = float(1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_tr.mean()) ** 2))
    r2_te = float(1 - np.sum((y_te - pred) ** 2) / np.sum((y_te - y_te.mean()) ** 2))
    rho, _ = spearmanr(pred, y_te)
    print(f"[3] item2 A6/A7 (1M, n={len(y_te)}) {name}: RMSE={rmse:.4f} "
          f"(vs train-mean-baseline {base_rmse:.4f}, drop {100*(1-rmse/base_rmse):.1f}%) "
          f"R2(train-mean)={r2_tr:.4f} R2(sklearn)={r2_te:.4f} Spearman={rho:.4f}")

# ---- item 3: no-calibration Spearman on 60M/1B ------------------------------
print("[4] item3 no-calibration Spearman (protocol v2 item 3):")
for tag in ("60M", "1B"):
    P_t, L_t, X_t, y_t = tests[tag]
    for name, mdl in (("OLS", ols), ("HistGB", hgb)):
        pred = mdl.predict(X_t)
        rho, _ = spearmanr(pred, y_t)
        print(f"    {tag} n={len(y_t)} {name}: Spearman={rho:.4f}")

# ---- item 4: split-half calibration RMSE (labelled post-calibration) -------
print("[5] item4 split-half calibrated RMSE (odd=calibrate intercept, even=evaluate):")
for tag in ("60M", "1B"):
    P_t, L_t, X_t, y_t = tests[tag]
    odd, even = np.arange(len(y_t)) % 2 == 1, np.arange(len(y_t)) % 2 == 0
    for name, mdl in (("OLS", ols), ("HistGB", hgb)):
        pred = mdl.predict(X_t)
        alpha = float(np.mean(y_t[odd] - pred[odd]))     # scale-specific intercept
        rmse = float(np.sqrt(np.mean((y_t[even] - (pred[even] + alpha)) ** 2)))
        rmse_nc = float(np.sqrt(np.mean((y_t[even] - pred[even]) ** 2)))
        print(f"    {tag} {name}: calibrated RMSE(even)={rmse:.4f}  vs no-calib {rmse_nc:.4f}  "
              f"alpha={alpha:+.4f}  [post-calibration analysis, not independent test]")

# ---- AS-05: coefficient drift across scales ---------------------------------
print("[6] AS-05 coefficient drift (OLS beta per scale, drift analysis only):")
b1m = pd.Series(ols.coef_, index=[d for d in DOMAINS17 if d != REF])
drift = {}
for tag in ("60M", "1B"):
    P_t, L_t, X_t, y_t = tests[tag]
    b_s = pd.Series(LinearRegression().fit(X_t, y_t).coef_,
                    index=[d for d in DOMAINS17 if d != REF])
    r, _ = spearmanr(b1m, b_s)
    drift[tag] = {"spearman_vs_1M": round(float(r), 4),
                  "top_moves": (b_s - b1m).abs().sort_values(ascending=False).head(5).round(4).to_dict()}
    print(f"    {tag}: Spearman(beta_{tag}, beta_1M)={r:.4f}  biggest moves: {drift[tag]['top_moves']}")

# ---- EQ23b: HistGB numeric transfer check (top-10 linear pairs) -------------
print("[7] EQ23b HistGB numeric transfer check (r=0.5 from p0):")
p0 = P_tr.mean(axis=0)
idx17 = {d: i for i, d in enumerate(DOMAINS17)}
alr_cols = [d for d in DOMAINS17 if d != REF]
beta = dict(zip(alr_cols, ols.coef_))


def delta_general(a, b, p_start, frac=0.5):
    p = p_start.copy()
    ia, ib = idx17[a], idx17[b]
    m_use = frac * p[ia]
    p[ia] -= m_use
    p[ib] += m_use
    x2 = alr(p[None, :])[0]
    x1 = alr(p_start[None, :])[0]
    return float(sum(beta[d] * (x2[i] - x1[i]) for i, d in enumerate(alr_cols)))


def delta_histgb(a, b, p_start, frac=0.5):
    p = p_start.copy()
    ia, ib = idx17[a], idx17[b]
    m_use = frac * p[ia]
    p[ia] -= m_use
    p[ib] += m_use
    return float(hgb.predict(alr(p[None, :]))[0] - hgb.predict(alr(p_start[None, :]))[0])


pairs = [("pubmed_central", "ubuntu_irc"), ("pile_cc", "ubuntu_irc"),
         ("arxiv", "ubuntu_irc"), ("github", "ubuntu_irc"),
         ("freelaw", "ubuntu_irc"), ("ubuntu_irc", "uspto_backgrounds"),
         ("ubuntu_irc", "gutenberg_pg_19"), ("stackexchange", "hackernews"),
         ("dm_mathematics", "hackernews"), ("ubuntu_irc", "hackernews")]
print(f"    {'pair':^42s} {'linear':>9s} {'HistGB':>9s}  agree?")
for a, b in pairs:
    dl = delta_general(a, b, p0)
    dh = delta_histgb(a, b, p0)
    print(f"    {a:>19s} -> {b:<20s} {dl:+9.4f} {dh:+9.4f}  {'yes' if dl * dh > 0 else 'NO'}")

# ---- EQ23c: 3-domain combination case ---------------------------------------
print("[8] EQ23c three-domain combination (from p0: pile_cc +0.05, github +0.05, arxiv -0.10):")
p = p0.copy()
p[idx17["pile_cc"]] += 0.05
p[idx17["github"]] += 0.05
p[idx17["arxiv"]] -= 0.10
assert p.min() >= 0 and abs(p.sum() - 1) < 1e-9, f"p range violated: min={p.min()}"
comb_ols = float(ols.predict(alr(p[None, :]))[0] - ols.predict(alr(p0[None, :]))[0])
comb_hgb = float(hgb.predict(alr(p[None, :]))[0] - hgb.predict(alr(p0[None, :]))[0])
single_ols = (delta_general("pile_cc", "arxiv", p0) + delta_general("github", "arxiv", p0))
single_hgb = (delta_histgb("pile_cc", "arxiv", p0) + delta_histgb("github", "arxiv", p0))
print(f"    combination dy: OLS={comb_ols:+.4f}  HistGB={comb_hgb:+.4f}")
print(f"    sum of two single 50%-transfers into arxiv: OLS={single_ols:+.4f}  HistGB={single_hgb:+.4f}")
print(f"    note: single transfers above each move 50% of source share; combination uses fixed +0.05/-0.10 masses (illustrative).")

# ---- AS-09: with/without the 4 no-loss ratio columns ------------------------
print("[9] AS-09: CV with vs without the 4 no-loss ratio columns:")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
alr_all = alr(P_tr)
col_pos = {d: i for i, d in enumerate([d for d in DOMAINS17 if d != REF])}
X17 = alr_all
X13f = alr_all[:, [col_pos[d] for d in LOSS13 if d != REF]]
for tag, XX in (("17-col (keep 4 no-loss cols)", X17), ("13-col (drop them)", X13f)):
    r = []
    for tr, te in kf.split(XX):
        mm = LinearRegression().fit(XX[tr], y_tr[tr])
        pred = mm.predict(XX[te])
        r.append(1 - np.sum((y_tr[te] - pred) ** 2) / np.sum((y_tr[te] - y_tr[tr].mean()) ** 2))
    print(f"    {tag}: CV R2={np.mean(r):.4f}")

# ---- AS-04: target weighting sensitivity ------------------------------------
print("[10] AS-04 target weighting sensitivity (5-fold CV R2, OLS):")
p13 = P_tr[:, [idx17[d] for d in LOSS13]]
p13n = p13 / p13.sum(axis=1, keepdims=True)
y_w = (L_tr * p13n).sum(axis=1)          # mixture-weighted within 13 loss domains
for tag, yy in (("equal(13-domain mean)", y_tr), ("mixture-weighted", y_w)):
    r = []
    for tr, te in kf.split(X_tr):
        mm = LinearRegression().fit(X_tr[tr], yy[tr])
        pred = mm.predict(X_tr[te])
        r.append(1 - np.sum((yy[te] - pred) ** 2) / np.sum((yy[te] - yy[tr].mean()) ** 2))
    print(f"    {tag}: CV R2={np.mean(r):.4f}")
# per-domain multi-output check: mean of per-domain CV R2
per = []
for k, d in enumerate(LOSS13):
    yy = L_tr[:, k]
    r = []
    for tr, te in kf.split(X_tr):
        mm = LinearRegression().fit(X_tr[tr], yy[tr])
        pred = mm.predict(X_tr[te])
        r.append(1 - np.sum((yy[te] - pred) ** 2) / np.sum((yy[te] - yy[tr].mean()) ** 2))
    per.append(np.mean(r))
print(f"    per-domain multi-output: mean CV R2={np.mean(per):.4f} (min={np.min(per):.4f}, max={np.max(per):.4f})")

# ---- bootstrap B=1000 for top-10 pairs (linear model) -----------------------
print("[11] bootstrap 95% CI (B=1000) for top-10 pairs (linear):")
rng = np.random.default_rng(SEED)
boot = {k: [] for k in pairs}
for _ in range(1000):
    idx = rng.integers(0, len(X_tr), len(X_tr))
    bb = dict(zip(alr_cols, LinearRegression().fit(X_tr[idx], y_tr[idx]).coef_))
    for (a, b) in pairs:
        p = p0.copy()
        ia, ib = idx17[a], idx17[b]
        m_use = 0.5 * p[ia]
        p[ia] -= m_use
        p[ib] += m_use
        x2 = alr(p[None, :])[0]
        x1 = alr(p0[None, :])[0]
        boot[(a, b)].append(float(sum(bb[d] * (x2[i] - x1[i]) for i, d in enumerate(alr_cols))))
ci_rows = []
for (a, b) in pairs:
    v = delta_general(a, b, p0)
    lo, hi = np.percentile(boot[(a, b)], [2.5, 97.5])
    ci_rows.append({"a": a, "b": b, "dy": v, "lo": float(lo), "hi": float(hi),
                    "sig": bool(lo > 0 or hi < 0)})
    print(f"    {a:>19s} -> {b:<20s} dy={v:+.4f} CI[{lo:+.4f},{hi:+.4f}]{' *' if lo > 0 or hi < 0 else ''}")
pd.DataFrame(ci_rows).to_csv(os.path.join(OUT, "bootstrap_ci_B1000.csv"), index=False)

with open(os.path.join(OUT, "exp04_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED, "runtime_s": round(time.time() - t0, 1),
        "item2_1M": {"baseline_rmse": base_rmse},
        "drift": drift,
        "combo_case": {"ols": comb_ols, "hgb": comb_hgb,
                       "single_sum_ols": single_ols, "single_sum_hgb": single_hgb},
        "as09": "see stdout", "as04": "see stdout",
    }, f, ensure_ascii=False, indent=2)
print(f"[done] {time.time()-t0:.1f}s -> {OUT}")
