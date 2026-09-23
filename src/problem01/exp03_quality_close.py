# -*- coding: utf-8 -*-
"""EXP-20260923-03: close quality-chain items.
(1) dsir length-confound control -> residualize on log word count -> rerun EQ21;
(2) qur_re / professionalism 3-protocol sensitivity (drop / keep / flip) on Q_d ordering;
(3) modernbert_* expected-level vs argmax rank agreement (EQ10 dual protocol);
(4) lambda sweep for conflict penalty (tau = P95) + conflict spot-check material export.

Run: python exp03_quality_close.py
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from p1_common import (FEATURES, BASE5, read_split, fit_quantile_maps,
                       frozen_rank)

SEED = 20260923
BASE = r"D:\project\MathStruct\data\real_attachments\A_data_value"
OUT = r"D:\project\MathStruct\result\problem01\exp03_quality_close"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

print("[1] reading A1 (with content for spot-check material) ...", flush=True)
a1 = read_split(BASE + r"\slimpajama_quality_signal_sample.jsonl.xz", "A1",
                domain_field="_source_domain", keep_content=True)
print(f"    rows={len(a1)}", flush=True)

maps = fit_quantile_maps(a1, FEATURES)
rn = frozen_rank(a1, FEATURES, maps)

# ---------------------------------------------------------------- (1) dsir
print("[2] AS-03: dsir length confound -> residualize on log word_count ...")
wc = a1["rps_doc_word_count"].to_numpy(dtype=float)
logwc = np.log1p(wc)
dsir_res = {}
for f in ("dsir_books", "dsir_math", "dsir_wiki"):
    v = a1[f].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    A = np.vstack([np.ones(ok.sum()), logwc[ok]]).T
    coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    resid = np.full_like(v, np.nan)
    resid[ok] = v[ok] - (coef[0] + coef[1] * logwc[ok])
    r_raw, _ = spearmanr(v[ok], logwc[ok])
    dsir_res[f] = {"slope_per_log_word": float(coef[1]), "spearman_with_logwc": float(r_raw)}
    rn[f] = resid  # replace with residuals; re-rank below
    print(f"    {f}: slope={coef[1]:.1f} per log-word, spearman(dsir, log wc)={r_raw:.3f}")
# re-rank residualized dsir on the same frozen-grid style (fit new maps on A1)
for f in ("dsir_books", "dsir_math", "dsir_wiki"):
    v = rn[f].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    q = np.quantile(v[ok], np.linspace(0, 1, 1001))
    rn.loc[ok, f] = np.interp(v[ok], q, np.linspace(0, 1, 1001))

print("[3] EQ21 rerun after dsir residualization (7-domain, rho_min=0.2, need>=4):")
base_score = rn[BASE5].mean(axis=1)
DOMAINS = sorted(a1["domain"].unique())
CANDIDATES = [f for f in FEATURES if f not in BASE5]
rows = []
for f in CANDIDATES:
    rhos = {}
    for d in DOMAINS:
        m = (a1["domain"] == d) & rn[f].notna() & base_score.notna()
        if m.sum() > 30:
            r, _ = spearmanr(rn.loc[m, f], base_score[m])
            if not np.isnan(r):
                rhos[d] = float(r)
    n_pos = sum(1 for r in rhos.values() if r >= 0.2)
    n_neg = sum(1 for r in rhos.values() if r <= -0.2)
    verdict = "positive" if n_pos >= 4 else ("negative" if n_neg >= 4 else "indeterminate")
    rows.append({"feature": f, "n_pos": n_pos, "n_neg": n_neg, "verdict": verdict,
                 "rhos": {k: round(v, 3) for k, v in rhos.items()}})
dir_df = pd.DataFrame(rows)
print(dir_df[["feature", "n_pos", "n_neg", "verdict"]].to_string(index=False))
dir_df.to_csv(os.path.join(OUT, "eq21_rerun_after_dsir_residual.csv"), index=False)

# ------------------------------------------- (2) 3-protocol sensitivity
print("[4] 3-protocol sensitivity for qur_re / professionalism (drop/keep/flip):")
def build_q(feature, mode):
    S = [f for f in FEATURES if f not in ("qur_re", "modernbert_professionalism")]
    pos = list(S)
    if feature in ("qur_re",):
        extra = "qur_re"
    else:
        extra = "modernbert_professionalism"
    if mode == "keep":
        pos = pos + [extra]
    elif mode == "flip":
        pos = pos + [extra]
    X = rn[pos].copy()
    if mode == "flip":
        X[extra] = 1.0 - X[extra]
    # negative-verdict features flipped (from EXP-01; dsir now residualized - keep their rerun verdicts below)
    neg = {"dsir_books", "dsir_math", "dsir_wiki", "rps_doc_frac_unique_words",
           "rps_doc_frac_no_alph_words"}
    neg_now = {r["feature"] for r in rows if r["verdict"] == "negative"}
    flip = (neg - {"dsir_books", "dsir_math", "dsir_wiki"}) | (neg_now & {"dsir_books", "dsir_math", "dsir_wiki"})
    for f in flip:
        if f in X.columns:
            X[f] = 1.0 - X[f]
    return X.mean(axis=1)

ref_order = pd.Series(rn[BASE5].mean(axis=1)).groupby(a1["domain"]).median().sort_values()
sens = {}
for feat in ("qur_re", "modernbert_professionalism"):
    qds = {}
    for mode in ("drop", "keep", "flip"):
        q = build_q(feat, mode)
        qds[mode] = q.groupby(a1["domain"]).median()
    mat = pd.DataFrame(qds)
    rr = {}
    for m1 in ("drop", "keep", "flip"):
        for m2 in ("drop", "keep", "flip"):
            if m1 < m2:
                r, _ = spearmanr(mat[m1], mat[m2])
                rr[f"{m1}-vs-{m2}"] = round(float(r), 4)
    sens[feat] = {"domain_order": mat.round(4).to_dict(orient="index"), "spearman_between": rr}
    print(f"    {feat}: order-agreement {rr}")

# --------------------------------- (3) EQ10 expected-level vs argmax agreement
print("[5] EQ10 dual protocol: expected level vs argmax (sample-level Spearman):")
argmax_rows = {}
for f in ("modernbert_cleanliness", "modernbert_readability",
          "modernbert_reasoning", "modernbert_professionalism"):
    # raw logits needed -> re-read quickly from a small A1 slice? we stored expected only.
    # argmax equals expected level monotone transformation in most cases; approximate
    # agreement via rank correlation between expected level (computed) and its
    # integer-rounded value (argmax-like). For rigor we recompute argmax on the fly.
    pass  # replaced below by direct recomputation during read (see exp03b note)

# Direct recomputation: stream A1 once more, computing argmax alongside expected.
import lzma
raw_arg = {f: [] for f in ("modernbert_cleanliness", "modernbert_readability",
                           "modernbert_reasoning", "modernbert_professionalism")}
raw_exp = {f: [] for f in raw_arg}
with lzma.open(BASE + r"\slimpajama_quality_signal_sample.jsonl.xz", "rt", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for f in raw_arg:
            v = rec.get(f)
            if isinstance(v, list) and len(v) == 6 and all(isinstance(t, (int, float)) for t in v):
                z = np.asarray(v, dtype=float)
                a = int(np.argmax(z))
                z = z - z.max()
                e = np.exp(z) / np.exp(z).sum()
                raw_arg[f].append(a)
                raw_exp[f].append(float((np.arange(6) * e).sum()))
            else:
                raw_arg[f].append(np.nan)
                raw_exp[f].append(np.nan)
agree = {}
for f in raw_arg:
    a = np.asarray(raw_arg[f], dtype=float)
    e = np.asarray(raw_exp[f], dtype=float)
    ok = ~np.isnan(a)
    r, _ = spearmanr(a[ok], e[ok])
    exact = float(np.mean(np.round(e[ok]) == a[ok]))
    agree[f] = {"spearman": round(float(r), 4), "round_match_rate": round(exact, 4)}
    print(f"    {f}: spearman={r:.4f}, round(exp)==argmax rate={exact:.4f}")

# ------------------------------------- (4) lambda sweep + spot-check material
print("[6] conflict lambda sweep (tau = P95) and spot-check material export:")
rep = rn[["rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"]].mean(axis=1)
noalph = rn["rps_doc_frac_no_alph_words"]
edu, flu = rn["fineweb_edu"], rn["fluency"]
clean, reas = rn["modernbert_cleanliness"], rn["modernbert_reasoning"]
# ad strength: adfree was indeterminate -> still used for C1 (mechanism separate)
ad_strength = 1.0 - rn["adfree"]
c1 = np.sqrt(np.clip(edu, 0, 1) * np.clip(ad_strength, 0, 1))
c2 = np.sqrt(np.clip(edu, 0, 1) * np.clip(np.maximum(rep, noalph), 0, 1))
c3 = np.sqrt(np.clip(flu, 0, 1) * np.clip(1 - clean, 0, 1))
c4 = np.sqrt(np.clip(reas, 0, 1) * np.clip(rep, 0, 1))
c_tilde = pd.concat([c1, c2, c3, c4], axis=1).max(axis=1)
which = pd.concat([c1, c2, c3, c4], axis=1).idxmax(axis=1)
tau = float(np.nanquantile(c_tilde, 0.95))
h = np.clip((c_tilde - tau) / (1 - tau), 0, 1)

# base Q with selected set from rerun verdicts
pos_now = [r["feature"] for r in rows if r["verdict"] == "positive"]
neg_now = [r["feature"] for r in rows if r["verdict"] == "negative"]
S = BASE5 + pos_now + neg_now
X = rn[S].copy()
for f in neg_now:
    X[f] = 1.0 - X[f]
q_base = X.mean(axis=1)

lam_scan = {}
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    q = q_base * (1 - lam * h)
    qd = pd.Series(q).groupby(a1["domain"]).median()
    r_vs_lam0, _ = spearmanr(qd, pd.Series(q_base).groupby(a1["domain"]).median())
    lam_scan[lam] = {"domain_order_spearman_vs_lam0": round(float(r_vs_lam0), 4),
                     "q_drop_flagged_mean": round(float(np.mean((q_base - q)[h > 0])), 4),
                     "q_min": round(float(np.nanmin(q)), 4)}
    print(f"    lambda={lam}: domain-order rho vs lam0={r_vs_lam0:.4f}, "
          f"mean Q drop on flagged={lam_scan[lam]['q_drop_flagged_mean']:.4f}, Q_min={lam_scan[lam]['q_min']:.4f}")

# spot-check material: top-50 conflict samples (C1-heavy) with content preview
top = c_tilde.sort_values(ascending=False).head(50).index
mat = pd.DataFrame({
    "sid": a1.loc[top, "sid"], "domain": a1.loc[top, "domain"],
    "c_tilde": c_tilde[top].round(4), "dominant": which[top],
    "edu": edu[top].round(4), "ad_strength": ad_strength[top].round(4),
    "rep": rep[top].round(4), "noalph": noalph[top].round(4),
    "content_head": a1.loc[top, "content"].astype(str).str.slice(0, 200).str.replace("\n", " "),
})
mat.to_csv(os.path.join(OUT, "conflict_spotcheck_top50.csv"), index=False)
print(f"    exported top-50 conflict samples for manual review -> conflict_spotcheck_top50.csv")
print(f"    composition of top-50: {which[top].value_counts().to_dict()}, domains: {a1.loc[top,'domain'].value_counts().to_dict()}")

with open(os.path.join(OUT, "exp03_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED, "runtime_s": round(time.time() - t0, 1),
        "dsir_residual": dsir_res,
        "eq21_rerun": rows,
        "three_protocol_sensitivity": sens,
        "eq10_expected_vs_argmax": agree,
        "tau_p95": round(tau, 4),
        "lambda_scan": {str(k): v for k, v in lam_scan.items()},
        "selected_set_rerun": S,
    }, f, ensure_ascii=False, indent=2)
print(f"[done] {time.time()-t0:.1f}s -> {OUT}")
