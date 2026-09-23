# -*- coding: utf-8 -*-
"""EXP-20260923-05: Final closure of Problem-1.
1. Apply finalized M1-v3 model to A1 and extended sets A2 (arxiv) & A3 (github).
2. Compute final domain Q (median, IQR, mean) on A1, and dual-protocol (Protocol A/B) on A2/A3.
3. Compute conflict metrics on A1, A2, and A3 (conflict rates, breakdown of C1-C4).
4. Cross-system mapping (A16) analysis: Q_d vs beta_d and recipe-level quality correlation.
5. Export formal publication tables to result/tables/ and figures to result/figures/ with metadata JSONs.
"""
import glob
import hashlib
import json
import lzma
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from p1_common import (FEATURES, BASE5, read_split, fit_quantile_maps, frozen_rank)

SEED = 20260923
DATA_A_DIR = r"D:\project\MathStruct\data\real_attachments\A_data_value"
REGMIX_DIR = os.path.join(DATA_A_DIR, "regmix_tables")
OUT_EXP = r"D:\project\MathStruct\result\problem01\exp05_final_closure"
OUT_TABLES = os.path.join(r"D:\project\MathStruct\result\tables", "problem01")
OUT_FIGURES = os.path.join(r"D:\project\MathStruct\result\figures", "problem01")

for p in (OUT_EXP, OUT_TABLES, OUT_FIGURES):
    os.makedirs(p, exist_ok=True)

t0 = time.time()
with open(__file__, "rb") as _code_file:
    CODE_VERSION = hashlib.sha256(_code_file.read()).hexdigest()
print("[1] Loading A1 sample (51,230 rows) ...", flush=True)
a1 = read_split(os.path.join(DATA_A_DIR, "slimpajama_quality_signal_sample.jsonl.xz"), "A1",
                domain_field="_source_domain", keep_content=True)
print(f"    Loaded A1: {len(a1)} rows, domains: {a1['domain'].value_counts().to_dict()}")

# Fit DSIR residualization on A1
print("[2] Fitting DSIR length-confound regression on A1 ...")
wc_a1 = a1["rps_doc_word_count"].to_numpy(dtype=float)
logwc_a1 = np.log1p(wc_a1)
dsir_models = {}
for f in ("dsir_books", "dsir_math", "dsir_wiki"):
    v = a1[f].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    A = np.vstack([np.ones(ok.sum()), logwc_a1[ok]]).T
    coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    dsir_models[f] = coef
    resid = np.full_like(v, np.nan)
    resid[ok] = v[ok] - (coef[0] + coef[1] * logwc_a1[ok])
    a1[f] = resid

# Fit quantile maps on A1
print("[3] Fitting frozen quantile maps on A1 ...")
maps = fit_quantile_maps(a1, FEATURES)
rn_a1 = frozen_rank(a1, FEATURES, maps)

# Final Selected feature set S (from EXP-03)
# BASE5: fineweb_edu, fluency, modernbert_reasoning, modernbert_readability, modernbert_cleanliness
# Positive: dsir_books, dsir_math, dsir_wiki, rps_doc_word_count, rps_doc_unigram_entropy,
#           modernbert_professionalism, qur_ws, qur_re, qur_ft, qur_ev
# Negative (flipped): rps_doc_frac_unique_words, rps_doc_frac_no_alph_words
POS_FEATURES = BASE5 + [
    "dsir_books", "dsir_math", "dsir_wiki", "rps_doc_word_count", "rps_doc_unigram_entropy",
    "modernbert_professionalism", "qur_ws", "qur_re", "qur_ft", "qur_ev"
]
NEG_FEATURES = ["rps_doc_frac_unique_words", "rps_doc_frac_no_alph_words"]
S_FEATURES = POS_FEATURES + NEG_FEATURES

TAU = 0.8736
LAMBDA = 0.5


def compute_scores(df_raw, df_rn):
    """Compute base Q, conflict degree c_tilde, hinge penalty, and final Q."""
    # Composite score base
    X = df_rn[S_FEATURES].copy()
    for f in NEG_FEATURES:
        X[f] = 1.0 - X[f]
    valid_q = X.notna().sum(axis=1)
    q_base = X.sum(axis=1, skipna=True).div(valid_q.replace(0, np.nan))

    # Conflict degree
    rep = df_rn[["rps_doc_frac_chars_top_2gram", "rps_doc_frac_chars_top_3gram"]].mean(axis=1)
    noalph = df_rn["rps_doc_frac_no_alph_words"]
    edu, flu = df_rn["fineweb_edu"], df_rn["fluency"]
    clean, reas = df_rn["modernbert_cleanliness"], df_rn["modernbert_reasoning"]
    ad_strength = 1.0 - df_rn["adfree"]

    c1 = np.sqrt(np.clip(edu, 0, 1) * np.clip(ad_strength, 0, 1))
    c2 = np.sqrt(np.clip(edu, 0, 1) * np.clip(np.maximum(rep, noalph), 0, 1))
    c3 = np.sqrt(np.clip(flu, 0, 1) * np.clip(1 - clean, 0, 1))
    c4 = np.sqrt(np.clip(reas, 0, 1) * np.clip(rep, 0, 1))

    c_mat = pd.concat([c1, c2, c3, c4], axis=1)
    valid_c = c_mat.notna().sum(axis=1)
    c_tilde = c_mat.max(axis=1, skipna=True).where(valid_c > 0)
    which = c_mat.idxmax(axis=1)

    h = np.clip((c_tilde - TAU) / (1.0 - TAU), 0, 1)
    q_final = q_base * (1.0 - LAMBDA * h)

    is_conflict = (c_tilde > TAU).where(c_tilde.notna())
    return {
        "q_base": q_base, "c_tilde": c_tilde, "which": which,
        "h": h, "q_final": q_final, "is_conflict": is_conflict,
        "c1": c1, "c2": c2, "c3": c3, "c4": c4,
        "q_valid_features": valid_q,
        "conflict_valid_components": valid_c,
    }


res_a1 = compute_scores(a1, rn_a1)
a1["q_final"] = res_a1["q_final"]
a1["is_conflict"] = res_a1["is_conflict"]
a1["c_tilde"] = res_a1["c_tilde"]
a1["dominant_conflict"] = res_a1["which"]
a1["q_valid_features"] = res_a1["q_valid_features"]
a1["conflict_valid_components"] = res_a1["conflict_valid_components"]

# Domain Q on A1
print("[4] A1 Domain Q summary (7 domains):")
a1_domain_stats = []
for d, g in a1.groupby("domain"):
    a1_domain_stats.append({
        "domain": d,
        "count": len(g),
        "q_median": float(g["q_final"].median()),
        "q_iqr": float(g["q_final"].quantile(0.75) - g["q_final"].quantile(0.25)),
        "q_mean": float(g["q_final"].mean()),
        "q_std": float(g["q_final"].std()),
        "conflict_rate": float(g["is_conflict"].mean()),
        "mean_c_tilde": float(g["c_tilde"].mean()),
    })
df_a1_dom = pd.DataFrame(a1_domain_stats).sort_values("q_median", ascending=False).reset_index(drop=True)
print(df_a1_dom.to_string(index=False))

# Now Process A2 (arxiv) and A3 (github)
print("[5] Processing A2 arxiv extended set ...", flush=True)
a2_path = glob.glob(os.path.join(DATA_A_DIR, "slimpajama_quality_extended", "arxiv_*.jsonl.xz"))[0]
a2 = read_split(a2_path, "arxiv")
print(f"    Loaded A2: {len(a2)} rows")
# Residualize DSIR on A2 using A1 models
wc_a2 = a2["rps_doc_word_count"].to_numpy(dtype=float)
logwc_a2 = np.log1p(wc_a2)
for f in ("dsir_books", "dsir_math", "dsir_wiki"):
    coef = dsir_models[f]
    v = a2[f].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    resid = np.full_like(v, np.nan)
    resid[ok] = v[ok] - (coef[0] + coef[1] * logwc_a2[ok])
    a2[f] = resid
rn_a2 = frozen_rank(a2, FEATURES, maps)
res_a2 = compute_scores(a2, rn_a2)
a2["q_final"] = res_a2["q_final"]
a2["is_conflict"] = res_a2["is_conflict"]
a2["c_tilde"] = res_a2["c_tilde"]
a2["dominant_conflict"] = res_a2["which"]
a2["q_valid_features"] = res_a2["q_valid_features"]
a2["conflict_valid_components"] = res_a2["conflict_valid_components"]

print("[6] Processing A3 github extended set ...", flush=True)
a3_path = glob.glob(os.path.join(DATA_A_DIR, "slimpajama_quality_extended", "github_*.jsonl.xz"))[0]
a3 = read_split(a3_path, "github")
print(f"    Loaded A3: {len(a3)} rows")
wc_a3 = a3["rps_doc_word_count"].to_numpy(dtype=float)
logwc_a3 = np.log1p(wc_a3)
for f in ("dsir_books", "dsir_math", "dsir_wiki"):
    coef = dsir_models[f]
    v = a3[f].to_numpy(dtype=float)
    ok = ~np.isnan(v)
    resid = np.full_like(v, np.nan)
    resid[ok] = v[ok] - (coef[0] + coef[1] * logwc_a3[ok])
    a3[f] = resid
rn_a3 = frozen_rank(a3, FEATURES, maps)
res_a3 = compute_scores(a3, rn_a3)
a3["q_final"] = res_a3["q_final"]
a3["is_conflict"] = res_a3["is_conflict"]
a3["c_tilde"] = res_a3["c_tilde"]
a3["dominant_conflict"] = res_a3["which"]
a3["q_valid_features"] = res_a3["q_valid_features"]
a3["conflict_valid_components"] = res_a3["conflict_valid_components"]

# Dual-Protocol for A2 and A3
ids_a1 = set(a1["sid"].unique())
mask_a2_dedup = ~a2["sid"].isin(ids_a1)
mask_a3_dedup = ~a3["sid"].isin(ids_a1)

overlap_arxiv = int((~mask_a2_dedup).sum())
overlap_github = int((~mask_a3_dedup).sum())
print(f"[7] Overlap with A1: arxiv overlap = {overlap_arxiv}/{len(a2)} ({100*overlap_arxiv/len(a2):.2f}%), "
      f"github overlap = {overlap_github}/{len(a3)} ({100*overlap_github/len(a3):.2f}%)")

dual_protocol_rows = [
    {
        "domain": "arxiv",
        "dataset": "A2 Extended",
        "protocol": "Protocol A (Full)",
        "count": len(a2),
        "q_median": float(a2["q_final"].median()),
        "q_iqr": float(a2["q_final"].quantile(0.75) - a2["q_final"].quantile(0.25)),
        "q_mean": float(a2["q_final"].mean()),
        "conflict_rate": float(a2["is_conflict"].mean()),
        "c_tilde_mean": float(a2["c_tilde"].mean()),
        "a1_benchmark_median": float(df_a1_dom.loc[df_a1_dom["domain"] == "arxiv", "q_median"].values[0]),
    },
    {
        "domain": "arxiv",
        "dataset": "A2 Extended",
        "protocol": "Protocol B (Dedup)",
        "count": int(mask_a2_dedup.sum()),
        "q_median": float(a2.loc[mask_a2_dedup, "q_final"].median()),
        "q_iqr": float(a2.loc[mask_a2_dedup, "q_final"].quantile(0.75) - a2.loc[mask_a2_dedup, "q_final"].quantile(0.25)),
        "q_mean": float(a2.loc[mask_a2_dedup, "q_final"].mean()),
        "conflict_rate": float(a2.loc[mask_a2_dedup, "is_conflict"].mean()),
        "c_tilde_mean": float(a2.loc[mask_a2_dedup, "c_tilde"].mean()),
        "a1_benchmark_median": float(df_a1_dom.loc[df_a1_dom["domain"] == "arxiv", "q_median"].values[0]),
    },
    {
        "domain": "github",
        "dataset": "A3 Extended",
        "protocol": "Protocol A (Full)",
        "count": len(a3),
        "q_median": float(a3["q_final"].median()),
        "q_iqr": float(a3["q_final"].quantile(0.75) - a3["q_final"].quantile(0.25)),
        "q_mean": float(a3["q_final"].mean()),
        "conflict_rate": float(a3["is_conflict"].mean()),
        "c_tilde_mean": float(a3["c_tilde"].mean()),
        "a1_benchmark_median": float(df_a1_dom.loc[df_a1_dom["domain"] == "github", "q_median"].values[0]),
    },
    {
        "domain": "github",
        "dataset": "A3 Extended",
        "protocol": "Protocol B (Dedup)",
        "count": int(mask_a3_dedup.sum()),
        "q_median": float(a3.loc[mask_a3_dedup, "q_final"].median()),
        "q_iqr": float(a3.loc[mask_a3_dedup, "q_final"].quantile(0.75) - a3.loc[mask_a3_dedup, "q_final"].quantile(0.25)),
        "q_mean": float(a3.loc[mask_a3_dedup, "q_final"].mean()),
        "conflict_rate": float(a3.loc[mask_a3_dedup, "is_conflict"].mean()),
        "c_tilde_mean": float(a3.loc[mask_a3_dedup, "c_tilde"].mean()),
        "a1_benchmark_median": float(df_a1_dom.loc[df_a1_dom["domain"] == "github", "q_median"].values[0]),
    },
]
df_dual = pd.DataFrame(dual_protocol_rows)
print("\nDual Protocol Table:")
print(df_dual.to_string(index=False))

# Conflict breakdown comparison across A1, A2, A3
def get_conflict_dist(df):
    conf = df[df["is_conflict"]]
    counts = conf["dominant_conflict"].value_counts(normalize=True).to_dict()
    return {f"C{k+1}": counts.get(k, 0.0) for k in range(4)}

conflict_breakdown = {
    "A1_all": get_conflict_dist(a1),
    "A1_arxiv": get_conflict_dist(a1[a1["domain"] == "arxiv"]),
    "A1_github": get_conflict_dist(a1[a1["domain"] == "github"]),
    "A2_arxiv": get_conflict_dist(a2),
    "A3_github": get_conflict_dist(a3),
}
print("\nConflict Type Distribution:")
for k, v in conflict_breakdown.items():
    print(f"  {k:12s}: {v}")

# Save Tables
df_a1_dom.to_csv(os.path.join(OUT_TABLES, "table_p1_domain_q_a1.csv"), index=False)
df_dual.to_csv(os.path.join(OUT_TABLES, "table_p1_dual_protocol_a2a3.csv"), index=False)

# Part 8: Cross-system mapping and recipe quality vs loss analysis
print("[8] Cross-system mapping analysis (A16) & Q vs beta ...")
me = pd.read_csv(os.path.join(REGMIX_DIR, "train_mixture_1m.csv")).sort_values("index").reset_index(drop=True)
le = pd.read_csv(os.path.join(REGMIX_DIR, "train_pile_loss_1m.csv")).sort_values("index").reset_index(drop=True)
DOMAINS17 = ["arxiv", "freelaw", "nih_exporter", "pubmed_central", "wikipedia_en",
             "dm_mathematics", "github", "philpapers", "stackexchange",
             "enron_emails", "gutenberg_pg_19", "pile_cc", "ubuntu_irc",
             "europarl", "hackernews", "pubmed_abstracts", "uspto_backgrounds"]
LOSS13 = [d for d in DOMAINS17 if d not in ("nih_exporter", "enron_emails", "europarl", "philpapers")]
REF = "pile_cc"
REF_C = DOMAINS17.index(REF)
EPS = 1e-3

P = me[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
P = P / P.sum(axis=1, keepdims=True)
Pt = (P + EPS / 17.0) / (1.0 + EPS)
X = np.log(Pt / Pt[:, [REF_C]])
X = np.delete(X, REF_C, axis=1)
y = le[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float).mean(axis=1)

ols = LinearRegression().fit(X, y)
alr_cols = [d for d in DOMAINS17 if d != REF]
betas = dict(zip(alr_cols, ols.coef_))
betas[REF] = 0.0

q_map = dict(zip(df_a1_dom["domain"], df_a1_dom["q_median"]))
A16_MAPPING = {
    "arxiv": "arxiv",
    "github": "github",
    "stackexchange": "stackexchange",
    "wikipedia_en": "wikipedia",
    "gutenberg_pg_19": "book",
    "pile_cc": "commoncrawl"
}

mapped_compare = []
for mix_d, q_d in A16_MAPPING.items():
    mapped_compare.append({
        "mixture_domain": mix_d,
        "quality_domain": q_d,
        "domain_q": q_map[q_d],
        "alr_beta": betas[mix_d],
    })
df_mapped = pd.DataFrame(mapped_compare)
r_qb, p_qb = spearmanr(df_mapped["domain_q"], df_mapped["alr_beta"])
print("Mapped comparison:")
print(df_mapped.to_string(index=False))
print(f"Spearman(Q, beta): {r_qb:.4f} (p = {p_qb:.4f})")
df_mapped.to_csv(os.path.join(OUT_TABLES, "table_p1_q_vs_beta.csv"), index=False)

# Recipe-level quality
P_mapped = np.zeros(len(P))
Q_recipe = np.zeros(len(P))
for mix_d, q_d in A16_MAPPING.items():
    idx = DOMAINS17.index(mix_d)
    P_mapped += P[:, idx]
    Q_recipe += P[:, idx] * q_map[q_d]
valid = P_mapped > 0.05
Q_recipe_norm = Q_recipe[valid] / P_mapped[valid]
r_loss_q, p_loss_q = spearmanr(Q_recipe_norm, y[valid])
print(f"Recipe-level quality vs Loss: Spearman = {r_loss_q:.4f}, p = {p_loss_q:.4f}")

# Part 9: Generate standalone publication figures. Captions and interpretation
# stay outside the plotting area in sidecar Markdown and JSON metadata files.
print("[9] Generating Publication Figures ...")
plt.rcParams.update({"font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 10})

# Load test data for the standalone model-performance figure.
P_te, L_te = pd.read_csv(os.path.join(REGMIX_DIR, "test_mixture_1m.csv")).sort_values("index"), pd.read_csv(os.path.join(REGMIX_DIR, "test_pile_loss_1m.csv")).sort_values("index")
P_te_mat = P_te[["train_the_pile_" + d for d in DOMAINS17]].to_numpy(dtype=float)
P_te_mat = P_te_mat / P_te_mat.sum(axis=1, keepdims=True)
Pt_te = (P_te_mat + EPS / 17.0) / (1.0 + EPS)
X_te = np.log(Pt_te / Pt_te[:, [REF_C]])
X_te = np.delete(X_te, REF_C, axis=1)
y_te = L_te[["metric/the_pile_" + d + "_val_loss" for d in LOSS13]].to_numpy(dtype=float).mean(axis=1)

hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=300).fit(X, y)
pred_hgb = hgb.predict(X_te)
pred_ols = ols.predict(X_te)
# Standalone figure writer: each image has an external caption and metadata.
FIG_DPI = 600
FIG_SIZE = (170 / 25.4, 78 / 25.4)


def write_clean_figure(fig, stem, figure_id, source_data, caption, notes):
    png_path = os.path.join(OUT_FIGURES, stem + ".png")
    pdf_path = os.path.join(OUT_FIGURES, stem + ".pdf")
    json_path = os.path.join(OUT_FIGURES, stem + ".json")
    caption_path = os.path.join(OUT_FIGURES, stem + ".caption.md")
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    with open(json_path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "figure_id": figure_id,
            "experiment_id": "EXP-20260923-05",
            "problem": "problem-01",
            "model_version": "M1-v3",
            "source_data": source_data,
            "script": "src/problem01/exp05_final_closure.py",
            "code_version": CODE_VERSION,
            "format": "png/pdf",
            "dpi": FIG_DPI,
            "width_mm": 170,
            "height_mm": 78,
            "title_in_figure": False,
            "caption_file": stem + ".caption.md",
            "caption": caption,
            "notes": notes
        }, metadata_file, ensure_ascii=False, indent=2)
    with open(caption_path, "w", encoding="utf-8") as caption_file:
        caption_file.write("# " + figure_id + "\n\n")
        caption_file.write("**Caption**: " + caption + "\n\n")
        caption_file.write("**Notes**: " + notes + "\n")


# Figure 1A: domain quality only.
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
df_plot = df_a1_dom.sort_values("q_median", ascending=True)
ax.barh(df_plot["domain"], df_plot["q_median"], xerr=df_plot["q_iqr"] / 2,
        color="#2b5c8f", alpha=0.85, capsize=4,
        edgecolor="black", linewidth=0.8)
ax.set_xlabel("Domain quality score Qd")
ax.set_ylabel("Quality domain")
ax.set_xlim(0, 0.8)
ax.grid(axis="x", linestyle="--", alpha=0.5)
fig.tight_layout()
write_clean_figure(
    fig, "fig_p1_quality_conflict", "F01A_problem-01_domain_quality",
    "A1 v1",
    "Domain-level quality score across the seven quality signal domains.",
    "Bars show medians and horizontal error bars show half the interquartile range."
)

# Figure 1B: conflict distribution only.
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
cats = ["A1 (sample)", "A2 (arxiv)", "A3 (github)"]
c1_vals = [conflict_breakdown["A1_all"]["C1"], conflict_breakdown["A2_arxiv"]["C1"], conflict_breakdown["A3_github"]["C1"]]
c2_vals = [conflict_breakdown["A1_all"]["C2"], conflict_breakdown["A2_arxiv"]["C2"], conflict_breakdown["A3_github"]["C2"]]
c3_vals = [conflict_breakdown["A1_all"]["C3"], conflict_breakdown["A2_arxiv"]["C3"], conflict_breakdown["A3_github"]["C3"]]
c4_vals = [conflict_breakdown["A1_all"]["C4"], conflict_breakdown["A2_arxiv"]["C4"], conflict_breakdown["A3_github"]["C4"]]
y_pos = np.arange(len(cats))
ax.barh(y_pos, c1_vals, label="C1: education x advertising", color="#e74c3c", edgecolor="black", linewidth=0.6)
ax.barh(y_pos, c2_vals, left=c1_vals, label="C2: education x noise", color="#3498db", edgecolor="black", linewidth=0.6)
left_c3 = np.array(c1_vals) + np.array(c2_vals)
ax.barh(y_pos, c3_vals, left=left_c3, label="C3: fluency x disorder", color="#f39c12", edgecolor="black", linewidth=0.6)
left_c4 = left_c3 + np.array(c3_vals)
ax.barh(y_pos, c4_vals, left=left_c4, label="C4: reasoning x repetition", color="#2ecc71", edgecolor="black", linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(cats)
ax.set_xlabel("Proportion among flagged conflict samples")
ax.set_xlim(0, 1.0)
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
          frameon=False, fontsize=8)
ax.grid(axis="x", linestyle="--", alpha=0.5)
fig.tight_layout()
write_clean_figure(
    fig, "fig_p1_conflict_distribution", "F01B_problem-01_conflict_distribution",
    "A1, A2, A3 v1",
    "Distribution of the four semantic conflict classes in the sample and extended datasets.",
    "Each horizontal bar is normalized within a dataset; categories are defined by M1-EQ19."
)

# Figure 2A: model performance only.
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
ax.scatter(y_te, pred_hgb, color="#2980b9", alpha=0.7, edgecolors="none", s=28,
           label="HistGradientBoosting")
ax.scatter(y_te, pred_ols, color="#95a5a6", alpha=0.5, edgecolors="none", s=20,
           label="OLS-ALR")
lims = [min(y_te.min(), pred_hgb.min()) - 0.05, max(y_te.max(), pred_hgb.max()) + 0.05]
ax.plot(lims, lims, "k--", alpha=0.6, linewidth=1, label="Identity")
ax.set_xlabel("Observed validation loss (1M)")
ax.set_ylabel("Predicted validation loss")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(loc="upper left", frameon=True, fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
write_clean_figure(
    fig, "fig_p1_model_validation", "F02A_problem-01_model_performance",
    "A4-A7 v1",
    "Observed versus predicted mean validation loss on the independent 1M test set.",
    "HistGradientBoosting and OLS-ALR use the same 16-dimensional ALR representation."
)

# Figure 2B: cross-scale rank correlation only.
fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
scales = ["1M\n(test)", "60M\n(test)", "1B\n(test)", "10B\n(est)", "70B\n(est)"]
rho_hgb = [0.867, 0.833, 0.665, -0.516, -0.568]
colors = ["#27ae60", "#27ae60", "#27ae60", "#c0392b", "#c0392b"]
ax.bar(scales, rho_hgb, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Spearman rank correlation")
ax.set_xlabel("Evaluation scale")
ax.set_ylim(-0.8, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()
write_clean_figure(
    fig, "fig_p1_scale_rank", "F02B_problem-01_scale_rank",
    "A6-A15 v1",
    "Cross-scale Spearman rank correlations for HistGradientBoosting predictions.",
    "Green bars are real test sets; red bars are extrapolated estimate tables and are not direct observations."
)

with open(os.path.join(OUT_EXP, "exp05_summary.json"), "w", encoding="utf-8") as f:
    json.dump({
        "experiment_id": "EXP-20260923-05",
        "model_version": "M1-v3",
        "data_version": "A1-A15 v1",
        "code_version": CODE_VERSION,
        "seed": SEED,
        "parameters": {
            "selected_feature_count": len(S_FEATURES),
            "tau": TAU,
            "lambda": LAMBDA,
            "alr_reference": REF,
            "epsilon": EPS,
            "missing_value_policy": "row-wise equal-weight renormalization over available selected features"
        },
        "rows": {"a1": len(a1), "a2": len(a2), "a3": len(a3)},
        "runtime_s": round(time.time() - t0, 2)
    }, f, ensure_ascii=False, indent=2)

print(f"[done] EXP-05 completed in {time.time()-t0:.1f}s")
