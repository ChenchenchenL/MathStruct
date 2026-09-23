# -*- coding: utf-8 -*-
"""EXP-20260923-01: Problem-1 quality chain baseline on A1 + A2/A3 dual-protocol.

Implements model.md M1-v3: EQ07-11 list compression, EQ21 direction validation,
EQ19' selected-set scoring (equal vs entropy weights), EQ18/19 conflict degree
and threshold candidates, EQ20 domain-level aggregation, and the dual-protocol
(Protocol A full records / Protocol B dedup) for A2/A3.

Run: python exp01_quality.py
Outputs: result/problem01/exp01_quality/ (JSON + markdown summary)
"""
import json
import lzma
import glob
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20260923
BASE = r"D:\project\MathStruct\data\real_attachments\A_data_value"
OUT = r"D:\project\MathStruct\result\problem01\exp01_quality"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
t0 = time.time()

# ---------------------------------------------------------------- field maps
LIST_FIELDS = {
    "fineweb_edu": 1, "fluency_en": 2, "ad_en": 2, "qurater": 4,
    "modernbert_cleanliness": 6, "modernbert_readability": 6,
    "modernbert_reasoning": 6, "modernbert_professionalism": 6,
}
SCALAR_FIELDS = [
    "dsir_books", "dsir_math", "dsir_wiki",
    "rps_doc_word_count", "rps_doc_num_sentences", "rps_doc_unigram_entropy",
    "rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
ALL_22 = list(LIST_FIELDS) + SCALAR_FIELDS


def safe_comp(record):
    """Compress one JSON record to the 22-dim vector per M1-EQ07..11."""
    out = {}
    for f in SCALAR_FIELDS:
        v = record.get(f)
        out[f] = float(v) if isinstance(v, (int, float)) else np.nan
    # single-value
    v = record.get("fineweb_edu")
    out["fineweb_edu"] = float(v[0]) if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], (int, float)) else np.nan
    # logit pairs
    for f, key in (("fluency_en", "fluency"), ("ad_en", "adfree")):
        v = record.get(f)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(t, (int, float)) for t in v):
            out[key] = float(v[1]) - float(v[0])  # EQ08/09: position2 - position1
        else:
            out[key] = np.nan
    # 6-level PRRC -> softmax expected level / 5  (EQ10)
    for f in ("modernbert_cleanliness", "modernbert_readability",
              "modernbert_reasoning", "modernbert_professionalism"):
        v = record.get(f)
        if isinstance(v, list) and len(v) == 6 and all(isinstance(t, (int, float)) for t in v):
            z = np.asarray(v, dtype=float)
            z = z - z.max()
            e = np.exp(z) / np.exp(z).sum()
            out[f] = float((np.arange(6) * e).sum() / 5.0)  # in [0,1]
        else:
            out[f] = np.nan
    # qurater: keep 4 raw dims (direction validation per-dim, EQ21)
    v = record.get("qurater")
    if isinstance(v, list) and len(v) == 4 and all(isinstance(t, (int, float)) for t in v):
        for j, name in enumerate(["qur_ws", "qur_re", "qur_ft", "qur_ev"]):
            out[name] = float(v[j])
    else:
        for name in ("qur_ws", "qur_re", "qur_ft", "qur_ev"):
            out[name] = np.nan
    return out


def read_split(path, domain, domain_field=None, id_set=None):
    """domain_field: if given, per-record domain label is taken from that JSON
    field (A1 carries _source_domain); otherwise the passed domain is used
    (A2/A3 carry no domain field, inferred from filename)."""
    rows, ids = [], []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = safe_comp(rec)
            row["sid"] = rec.get("id", "")
            row["domain"] = str(rec.get(domain_field, domain)) if domain_field else domain
            rows.append(row)
            ids.append(rec.get("id", ""))
    df = pd.DataFrame(rows)
    return df, set(ids)


print("[1] reading A1 ...", flush=True)
a1, ids_a1 = read_split(BASE + r"\slimpajama_quality_signal_sample.jsonl.xz", "A1-sample",
                        domain_field="_source_domain")
print(f"    A1 rows={len(a1)}  domains={a1['domain'].value_counts().to_dict()}", flush=True)

# qurater dim names join the frame; logit pairs renamed fluency/adfree
FEATURES = ALL_22 + ["qur_ws", "qur_re", "qur_ft", "qur_ev"]
FEATURES = [f for f in FEATURES if f not in ("qurater", "fluency_en", "ad_en")]
FEATURES += ["fluency", "adfree"]

missing = a1[FEATURES].isna().mean().sort_values(ascending=False)
print("[1b] per-feature missingness (top 5):")
print(missing.head(5).to_string())

# ------------------------------------------------- frozen rank-normalization
# Quantile (rank) normalization computed ON A1 and frozen; extended sets are
# mapped through A1's empirical quantile function (AS-11 scale freezing).
print("[2] fitting frozen quantile maps on A1 ...", flush=True)
quant_maps = {}
for f in FEATURES:
    v = a1[f].to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    qs = np.quantile(v, np.linspace(0, 1, 1001))
    quant_maps[f] = qs


def frozen_rank(df, features):
    out = pd.DataFrame(index=df.index)
    for f in features:
        v = df[f].to_numpy(dtype=float)
        qs = quant_maps[f]
        r = np.interp(v, qs, np.linspace(0, 1, 1001))
        out[f] = r  # NaN preserved by interp? no -> mask
        out.loc[df[f].isna(), f] = np.nan
    return out


rn_a1 = frozen_rank(a1, FEATURES)
print(f"    done at {time.time()-t0:.1f}s", flush=True)

# ------------------------------------------------------------- EQ21 baseline
BASE5 = ["fineweb_edu", "fluency", "modernbert_reasoning",
         "modernbert_readability", "modernbert_cleanliness"]
base_score = rn_a1[BASE5].mean(axis=1)  # semantic-positive composite

DOMAINS = sorted(a1["domain"].unique())
print("[3] EQ21 direction validation (per-domain Spearman vs base):", flush=True)
CANDIDATES = [f for f in FEATURES if f not in BASE5]
rho_min, need = 0.2, 4
dir_rows = []
for f in CANDIDATES:
    rhos = {}
    for d in DOMAINS:
        m = (a1["domain"] == d) & rn_a1[f].notna() & base_score.notna()
        if m.sum() > 30:
            r, _ = spearmanr(rn_a1.loc[m, f], base_score[m])
            rhos[d] = float(r)
    n_pos = sum(1 for r in rhos.values() if r >= rho_min)
    n_neg = sum(1 for r in rhos.values() if r <= -rho_min)
    verdict = "positive" if n_pos >= need else ("negative" if n_neg >= need else "indeterminate")
    dir_rows.append({"feature": f, "n_domains": len(rhos), "n_pos": n_pos,
                     "n_neg": n_neg, "verdict": verdict,
                     "rhos": {k: round(v, 3) for k, v in rhos.items()}})
dir_df = pd.DataFrame(dir_rows)
print(dir_df[["feature", "n_domains", "n_pos", "n_neg", "verdict"]].to_string(index=False))

selected = [f for f in CANDIDATES if dir_df.set_index("feature").loc[f, "verdict"] == "positive"]
# negative ones get flipped into the selected set (x -> 1-x)
neg_selected = [f for f in CANDIDATES if dir_df.set_index("feature").loc[f, "verdict"] == "negative"]
S = BASE5 + selected + neg_selected
print(f"[3b] selected set |S|={len(S)} (base5={len(BASE5)}, +pos={len(selected)}, +neg_flipped={len(neg_selected)})")

# -------------------------------------------------------------- EQ19' scores
X = rn_a1.copy()
for f in neg_selected:
    X[f] = 1.0 - X[f]
S_mat = X[S]
# equal weights
w_eq = np.full(len(S), 1.0 / len(S))
q_equal = S_mat.to_numpy() @ w_eq
# entropy weights (on non-nan rows via column-wise fill with column mean 0.5)
P = S_mat.fillna(0.5).to_numpy()
P = np.clip(P, 1e-12, 1)
Pn = P / P.sum(axis=0, keepdims=True)
ej = -(Pn * np.log(Pn)).sum(axis=0) / np.log(len(Pn))
wj = (1 - ej) / (1 - ej).sum()
q_entropy = S_mat.fillna(0.5).to_numpy() @ wj
mask_ok = S_mat.notna().all(axis=1)
rho_w, _ = spearmanr(q_equal[mask_ok], q_entropy[mask_ok])
print(f"[4] weight schemes: equal vs entropy sample-level Spearman = {rho_w:.4f}")

# --------------------------------------------- EQ19 conflict + threshold scan
rep = rn_a1[["rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"]].mean(axis=1)
ad_strength = 1.0 - rn_a1["adfree"]
edu = rn_a1["fineweb_edu"]
flu = rn_a1["fluency"]
clean = rn_a1["modernbert_cleanliness"]
reas = rn_a1["modernbert_reasoning"]
noalph = rn_a1["rps_doc_frac_no_alph_words"]
c1 = np.sqrt(np.clip(edu, 0, 1) * np.clip(ad_strength, 0, 1))
c2 = np.sqrt(np.clip(edu, 0, 1) * np.clip(np.maximum(rep, noalph), 0, 1))
c3 = np.sqrt(np.clip(flu, 0, 1) * np.clip(1 - clean, 0, 1))
c4 = np.sqrt(np.clip(reas, 0, 1) * np.clip(rep, 0, 1))
c_tilde = pd.concat([c1, c2, c3, c4], axis=1).max(axis=1)
conf_parts = pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4})
which = conf_parts.idxmax(axis=1)

qs = {p: float(np.nanquantile(c_tilde, p)) for p in (0.50, 0.90, 0.95, 0.975)}
scan = {}
for name, tau in (("P90", qs[0.90]), ("P95", qs[0.95]), ("P97.5", qs[0.975])):
    flagged = c_tilde > tau
    c1_dom = (which == "c1") & flagged
    scan[name] = {
        "tau": round(tau, 4),
        "overall_rate": round(float(flagged.mean()), 4),
        "c1_share": round(float(c1_dom.sum()) / max(int(flagged.sum()), 1), 4),
        "ad_strength_mean_flagged_c1": round(float(ad_strength[c1_dom].mean()), 4) if c1_dom.sum() else None,
        "ad_strength_mean_all": round(float(ad_strength.mean()), 4),
        "edu_mean_flagged_c1": round(float(edu[c1_dom].mean()), 4) if c1_dom.sum() else None,
        "edu_mean_all": round(float(edu.mean()), 4),
    }
print("[5] conflict threshold scan (proxy spot-check: flagged C1 vs population):")
for k, v in scan.items():
    print("   ", k, v)

# --------------------------------------------------- EQ20 domain-level Q (A1)
dom_q = pd.DataFrame({"Q_equal": q_equal, "Q_entropy": q_entropy, "domain": a1["domain"].values})
dom_summary = dom_q.groupby("domain").agg(Q_eq_med=("Q_equal", "median"),
                                          Q_eq_iqr=("Q_equal", lambda s: float(np.nanquantile(s, .75) - np.nanquantile(s, .25))),
                                          Q_ent_med=("Q_entropy", "median"),
                                          n=("Q_equal", "size"))
print("[6] A1 domain-level Q (equal / entropy):")
print(dom_summary.round(4).to_string())

# ------------------------------------------ A2/A3 dual protocol (EQ: Prot A/B)
print("[7] reading A2 (arxiv) & A3 (github) ...", flush=True)
a2, ids_a2 = read_split(glob.glob(BASE + r"\slimpajama_quality_extended\arxiv_*.jsonl.xz")[0], "arxiv")
a3, ids_a3 = read_split(glob.glob(BASE + r"\slimpajama_quality_extended\github_*.jsonl.xz")[0], "github")
ov_a2 = len(ids_a1 & ids_a2)
ov_a3 = len(ids_a1 & ids_a3)
print(f"    A2 rows={len(a2)}  overlap with A1={ov_a2}")
print(f"    A3 rows={len(a3)}  overlap with A1={ov_a3}")

ext = pd.concat([a2, a3], ignore_index=True)
ext_rn = frozen_rank(ext, FEATURES)
for f in neg_selected:
    ext_rn[f] = 1.0 - ext_rn[f]
ext["Q_full"] = ext_rn[S].fillna(0.5).to_numpy() @ np.array([w_eq[S.index(f)] if False else 1 / len(S) for f in S])
ext["Q_full"] = ext_rn[S].mean(axis=1)  # equal-weight on selected set (mean == weighted with re-normalized equal weights)
mask_ext = ext_rn[S].notna().all(axis=1)
ext["Q_full"] = np.where(mask_ext, ext_rn[S].mean(axis=1), np.nan)
ext["dedup"] = ~ext["sid"].isin(ids_a1)

protA = ext.groupby("domain")["Q_full"].agg(["median", "size"])
protB = ext[ext["dedup"]].groupby("domain")["Q_full"].agg(["median", "size"])
print("[7b] Protocol A (full) vs Protocol B (dedup) domain-level Q median:")
cmp = protA.rename(columns={"median": "Q_A", "size": "n_A"}).join(
    protB.rename(columns={"median": "Q_B", "size": "n_B"}), how="outer")
print(cmp.round(4).to_string())
# A1 reference domains for comparison
a1_arxiv = dom_q[dom_q.domain == "arxiv"]["Q_equal"].median() if "arxiv" in dom_summary.index else np.nan
print("    note: A1 'arxiv' domain Q_eq_med =", round(float(dom_summary.loc['arxiv', 'Q_eq_med']), 4) if 'arxiv' in dom_summary.index else 'n/a')

# ----------------------------------------------------------------- save all
dir_df.to_csv(os.path.join(OUT, "eq21_direction_table.csv"), index=False)
dom_summary.reset_index().to_csv(os.path.join(OUT, "a1_domain_q.csv"), index=False)
cmp.reset_index().to_csv(os.path.join(OUT, "a2a3_dual_protocol.csv"), index=False)
with open(os.path.join(OUT, "exp01_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED, "runtime_s": round(time.time() - t0, 1),
        "a1_rows": len(a1), "a2_rows": len(a2), "a3_rows": len(a3),
        "overlap_a1_a2": ov_a2, "overlap_a1_a3": ov_a3,
        "missingness_top5": missing.head(5).round(4).to_dict(),
        "selected_set_size": len(S), "selected_set": S,
        "neg_flipped": neg_selected,
        "weights_spearman_equal_vs_entropy": round(float(rho_w), 4),
        "entropy_weights": {f: round(float(w), 4) for f, w in zip(S, wj)},
        "conflict_quantiles": {k: round(v, 4) for k, v in qs.items()},
        "threshold_scan": scan,
        "direction_table": dir_rows,
        "a1_domain_q": dom_summary.round(4).to_dict(orient="index"),
        "dual_protocol": cmp.round(4).to_dict(orient="index"),
    }, f, ensure_ascii=False, indent=2)
print(f"[done] {time.time()-t0:.1f}s -> {OUT}")
