# -*- coding: utf-8 -*-
"""Audit diagnostics for the 2026-09-24 handover, items T01-T02, T06 and T08.

Reads frozen result tables and refits the Problem-4 frontier on the same
sample definition as exp05. Writes only under review/audit_20260924/.
Does not overwrite result/problem0*/ or result/tables/problem0*/.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r"D:\project\MathStruct"
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from problem04.p4_common import (  # noqa: E402
    load_and_match_datasets,
    DynamicSFAModel,
    BASE_DATA_DIR,
)
from problem03.p3_common import solve_optimal_allocation_2d  # noqa: E402

OUT = os.path.join(ROOT, "review", "audit_20260924")
os.makedirs(OUT, exist_ok=True)
SEED = 20260924
GIT_HEAD = "f9e61755f72b3f8b040d36000ea9f4ebbf1b9305"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(score):
    s = np.clip(np.asarray(score, dtype=float), 1e-4, 100.0 - 1e-4)
    return np.log(s / (100.0 - s))


def inv_logit(y):
    return 100.0 / (1.0 + np.exp(-np.asarray(y, dtype=float)))


def main():
    t0 = time.time()
    log_lines = []

    def say(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    say("audit_20260924 start")
    say(f"git_head={GIT_HEAD}")
    say(f"seed={SEED}")

    # ------------------------------------------------------------------
    # T08: quality-scale connection. No raw A1 rerun.
    # ------------------------------------------------------------------
    qtab = pd.read_csv(os.path.join(
        ROOT, "result", "tables", "problem01", "table_p1_domain_q_a1.csv"))
    qtab["weight"] = qtab["count"] / qtab["count"].sum()
    weighted = float((qtab["q_median"] * qtab["count"]).sum() / qtab["count"].sum())
    web = qtab[qtab["domain"].isin(["commoncrawl", "c4"])].copy()
    web_weighted = float((web["q_median"] * web["count"]).sum() / web["count"].sum())
    q_link = {
        "source_table": "result/tables/problem01/table_p1_domain_q_a1.csv",
        "source_sha256": sha256_file(os.path.join(
            ROOT, "result", "tables", "problem01", "table_p1_domain_q_a1.csv")),
        "seven_domain_count_weighted_median": weighted,
        "cc_c4_count_weighted_median_of_domain_medians": web_weighted,
        "cc_median": float(web.loc[web["domain"] == "commoncrawl", "q_median"].iloc[0]),
        "c4_median": float(web.loc[web["domain"] == "c4", "q_median"].iloc[0]),
        "cc_count": int(web.loc[web["domain"] == "commoncrawl", "count"].iloc[0]),
        "c4_count": int(web.loc[web["domain"] == "c4", "count"].iloc[0]),
        "documented_q_base": 0.584,
        "difference_documented_minus_recomputed_cc_c4": 0.584 - web_weighted,
        "status": (
            "0.584 is not recovered from the frozen domain-median table. "
            "Pooled median of the raw CC and C4 rows was not recomputed. "
            "Treat 0.584 as a scenario setting, not an estimate."
        ),
        "p1_q_versus_p2_q": (
            "Problem-1 domain medians and Problem-2 Q are different scales. "
            "The weighted seven-domain value is logged by Problem 2 and not "
            "consumed by the scaling law. No equality mapping is introduced."
        ),
    }
    say(f"T08 cc+c4 weighted median-of-medians={web_weighted:.6f}; documented 0.584")

    alloc = pd.read_csv(os.path.join(
        ROOT, "result", "tables", "problem03", "tab_p3_budget_allocation.csv"))
    direction = alloc[[
        "budget_tier", "cost_function", "recipe_regime", "opt_Q", "q_regime", "opt_Loss"
    ]].copy()
    direction.to_csv(os.path.join(OUT, "t08_q_regime_from_saved_allocation.csv"), index=False)
    regime_counts = (
        direction.groupby(["budget_tier", "cost_function", "q_regime"])
        .size().reset_index(name="n")
    )
    say("T08 saved q regimes:")
    say(regime_counts.to_string(index=False))

    # Spot-check whether moving the lower bound from 0.584 to the recomputed
    # 0.578 changes the saved direction. Only the three budgets x three costs
    # at the baseline mixture are resolved; differential evolution stays on
    # because this is 9 solves, not the 72-call path B.
    spot_rows = []
    for tier, c_bar in (("low", 10.0), ("mid", 10000.0), ("high", 1000000.0)):
        for cost in ("exp", "pow", "log"):
            sol = solve_optimal_allocation_2d(
                c_bar, cost_type=cost, L_ctx=2048, p_multiplier=1.0, verify_global=True)
            saved = alloc[
                (alloc["budget_tier"] == tier)
                & (alloc["cost_function"] == cost)
                & (alloc["recipe_regime"] == "Fixed_p0")
            ].iloc[0]
            spot_rows.append({
                "budget_tier": tier,
                "cost_function": cost,
                "saved_Q": float(saved["opt_Q"]),
                "saved_regime": saved["q_regime"],
                "rerun_Q_bound_0.584": float(sol["opt_Q"]),
                "rerun_regime": sol["kkt"]["q_regime"],
                "rerun_loss": float(sol["opt_loss"]),
                "de_verified": bool(sol["de_verified"]),
                "direction_same_as_saved": sol["kkt"]["q_regime"] == saved["q_regime"],
            })
            say(f"  spot {tier}/{cost}: saved {saved['q_regime']} -> rerun {sol['kkt']['q_regime']}")
    spot = pd.DataFrame(spot_rows)
    spot.to_csv(os.path.join(OUT, "t08_qbase_spot_rerun_p0.csv"), index=False)

    # ------------------------------------------------------------------
    # T01 / T02: same sample rule as exp05, monthly rows saved.
    # ------------------------------------------------------------------
    df_matched, _, _, _, _ = load_and_match_datasets()
    lic = df_matched["License_Category"].isin(
        ["Strict_Permissive", "Research_Open", "Research_Open_EpochVerified"])
    compute_ok = df_matched["Training_Compute_FLOP"].notnull() & (
        df_matched["Training_Compute_FLOP"] >= 1e18)
    regime_ok = df_matched["Regime"].isin(["Pretrained", "Chat_Finetuned"])
    df_sfa = df_matched[lic & compute_ok & regime_ok].copy()
    split_date = "2024-10-01"
    df_train = df_sfa[df_sfa["Sub_Date"] < split_date].copy()
    df_test = df_sfa[df_sfa["Sub_Date"] >= split_date].copy()
    say(f"T01 train={len(df_train)} test={len(df_test)} pool={len(df_sfa)}")

    sfa_oot = DynamicSFAModel(delta=0.95)
    sfa_oot.fit(df_train)

    month_labels = {
        0: "2024-06", 1: "2024-07", 2: "2024-08", 3: "2024-09",
        4: "2024-10", 5: "2024-11", 6: "2024-12", 7: "2025-01",
        8: "2025-02", 9: "2025-03",
    }
    # Persistence may use only information available before the scored month.
    # The anchor is the last training month, not the first test month.
    train_chat_for_anchor = df_train[df_train["is_chat"] == 1]
    last_train_month = int(df_train["month_idx"].max())
    prev_actual = float(train_chat_for_anchor.loc[
        train_chat_for_anchor["month_idx"] == last_train_month, "Average ⬆️"].max())
    rows = []
    for m in np.sort(df_test["month_idx"].unique()):
        sub = df_test[df_test["month_idx"] == m]
        chat = sub[sub["is_chat"] == 1]
        if len(sub) < 3 or chat.empty:
            rows.append({
                "month_idx": int(m),
                "month": month_labels.get(int(m), str(m)),
                "n_test_rows": int(len(sub)),
                "n_chat_rows": int(len(chat)),
                "evaluated": False,
            })
            continue
        actual = float(chat["Average ⬆️"].max())
        top_c = float(chat.loc[chat["Average ⬆️"].idxmax(), "Training_Compute_FLOP"])
        pred = float(sfa_oot.predict_frontier(np.array([top_c]), int(m), is_chat=1)[0])
        y_pred = float(logit(pred))
        sigma = float(sfa_oot.coef_["sigma_e"])
        lo95, hi95 = inv_logit(y_pred - 1.96 * sigma), inv_logit(y_pred + 1.96 * sigma)
        lo80, hi80 = inv_logit(y_pred - 1.28 * sigma), inv_logit(y_pred + 1.28 * sigma)
        persist = prev_actual if prev_actual is not None else actual
        rows.append({
            "month_idx": int(m),
            "month": month_labels.get(int(m), str(m)),
            "n_test_rows": int(len(sub)),
            "n_chat_rows": int(len(chat)),
            "evaluated": True,
            "actual_top_chat": actual,
            "top_model_compute_FLOP": top_c,
            "sfa_prediction": pred,
            "error_actual_minus_pred": actual - pred,
            "abs_error": abs(actual - pred),
            "interval_95_low": float(lo95),
            "interval_95_high": float(hi95),
            "covered_95": bool(lo95 <= actual <= hi95),
            "interval_80_low": float(lo80),
            "interval_80_high": float(hi80),
            "covered_80": bool(lo80 <= actual <= hi80),
            "sigma_e_only": sigma,
            "persistence_prediction": persist,
            "persistence_abs_error": abs(actual - persist),
            "persistence_rule": "previous evaluated chat maximum; anchored at last training month",
            "interval_note": "logit band uses sigma_e only; sigma_d is not included",
        })
        prev_actual = actual
    monthly = pd.DataFrame(rows)
    monthly.to_csv(os.path.join(OUT, "t01_oot_monthly.csv"), index=False)

    ev = monthly[monthly["evaluated"] == True]  # noqa: E712
    # Same-split baselines. Flat uses the last training-month chat maximum.
    train_chat = df_train[df_train["is_chat"] == 1]
    last_train_m = int(df_train["month_idx"].max())
    flat_level = float(train_chat.loc[
        train_chat["month_idx"] == last_train_m, "Average ⬆️"].max())
    hist = []
    for m, g in train_chat.groupby("month_idx"):
        hist.append((int(m), float(g["Average ⬆️"].max())))
    hist = sorted(hist)
    if len(hist) >= 2:
        slope = (hist[-1][1] - hist[0][1]) / (hist[-1][0] - hist[0][0])
    else:
        slope = 0.0
    base_rows = []
    for _, r in ev.iterrows():
        trend = flat_level + slope * (int(r["month_idx"]) - last_train_m)
        base_rows.append({
            "month": r["month"],
            "actual_top_chat": r["actual_top_chat"],
            "sfa_abs_error": r["abs_error"],
            "persistence_abs_error": r["persistence_abs_error"],
            "flat_last_train_month": flat_level,
            "flat_abs_error": abs(r["actual_top_chat"] - flat_level),
            "train_trend_prediction": trend,
            "train_trend_abs_error": abs(r["actual_top_chat"] - trend),
            "covered_95": r["covered_95"],
            "covered_80": r["covered_80"],
        })
    base = pd.DataFrame(base_rows)
    base.to_csv(os.path.join(OUT, "t01_oot_baselines.csv"), index=False)
    summary_t01 = {
        "split_date": split_date,
        "train_rows": int(len(df_train)),
        "test_rows": int(len(df_test)),
        "evaluated_months": int(len(ev)),
        "sfa_mae": float(ev["abs_error"].mean()),
        "persistence_mae": float(base["persistence_abs_error"].mean()),
        "flat_last_train_month_level": flat_level,
        "flat_mae": float(base["flat_abs_error"].mean()),
        "train_endpoint_slope_points_per_month": slope,
        "train_trend_mae": float(base["train_trend_abs_error"].mean()),
        "coverage_95_pct": float(100.0 * base["covered_95"].mean()),
        "coverage_80_pct": float(100.0 * base["covered_80"].mean()),
        "published_exp05_mae": 12.858322619346074,
        "published_exp05_coverage_95": 66.66666666666666,
        "published_exp05_coverage_80": 33.33333333333333,
        "what_12_24m_can_support": (
            "Six test months only. The refit MAE and coverage are descriptive "
            "of this split. They do not validate a 12- or 24-month maximum-score forecast."
        ),
    }
    say(
        f"T01 months={summary_t01['evaluated_months']} "
        f"SFA MAE={summary_t01['sfa_mae']:.2f} "
        f"persist={summary_t01['persistence_mae']:.2f} "
        f"flat={summary_t01['flat_mae']:.2f} "
        f"trend={summary_t01['train_trend_mae']:.2f} "
        f"cov95={summary_t01['coverage_95_pct']:.1f}"
    )

    # Full-sample frontier, same definition as the published 42.60 path.
    sfa_full = DynamicSFAModel(delta=0.95)
    sfa_full.fit(df_sfa)
    m_t0 = int(sfa_full.z_series_.index.max())
    z = sfa_full.z_series_
    slope_code = float((z.iloc[-1] - z.iloc[max(0, len(z) - 3)]) / 3.0)
    slope_two_step = float((z.iloc[-1] - z.iloc[max(0, len(z) - 3)]) / 2.0)
    c_t0 = float(np.percentile(
        df_sfa.loc[df_sfa["month_idx"] == m_t0, "Training_Compute_FLOP"], 90))
    g_c = 1.2651558697450236
    start_chat = float(sfa_full.predict_frontier(np.array([c_t0]), m_t0, is_chat=1)[0])
    decomp = []
    for dm in (12, 24):
        c_f = c_t0 * np.exp(g_c * dm / 12.0)
        chat = float(sfa_full.predict_frontier(np.array([c_f]), m_t0 + dm, is_chat=1)[0])
        # Scale-only: future compute, technology state frozen at the origin month.
        scale_only = float(sfa_full.predict_frontier(np.array([c_f]), m_t0, is_chat=1)[0])
        # Tech-only: origin compute, extrapolated technology state.
        tech_only = float(sfa_full.predict_frontier(np.array([c_t0]), m_t0 + dm, is_chat=1)[0])
        decomp.append({
            "horizon_months": dm,
            "origin_chat_frontier": start_chat,
            "forecast_chat_frontier": chat,
            "change_from_origin": chat - start_chat,
            "scale_only_frontier": scale_only,
            "scale_only_change": scale_only - start_chat,
            "tech_state_only_frontier": tech_only,
            "tech_state_only_change": tech_only - start_chat,
            "published_scenario1_chat": 35.91 if dm == 12 else 35.32,
        })
    decomp_df = pd.DataFrame(decomp)
    decomp_df.to_csv(os.path.join(OUT, "t02_decline_decomposition.csv"), index=False)
    z_out = pd.DataFrame({
        "month_idx": z.index.astype(int),
        "month": [month_labels.get(int(i), str(i)) for i in z.index],
        "z_logit": z.values.astype(float),
    })
    z_out.to_csv(os.path.join(OUT, "t02_fitted_technology_state.csv"), index=False)
    say(
        f"T02 origin={start_chat:.2f} code_slope={slope_code:.6f} "
        f"two_step_slope={slope_two_step:.6f}"
    )
    say(decomp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # T06: restate the saved rank correlations with their predictor.
    # ------------------------------------------------------------------
    exp02 = json.loads(open(os.path.join(
        ROOT, "result", "problem01", "exp02_mixture", "exp02_summary.json"),
        encoding="utf-8").read())
    t06 = []
    for scale, block in exp02["est_spearman_extrapolated"].items():
        t06.append({
            "scale": scale,
            "predictor": "OLS_ALR_fit_on_all_A4_A5",
            "target": "estimated_mean_pile_loss",
            "target_nature": "extrapolated_table_not_observed_training",
            "n": block["n"],
            "spearman_rho": block["spearman_rho"],
            "tree_model_used": False,
        })
    t06.append({
        "scale": "1M_A4A5_cv",
        "predictor": "HistGB",
        "target": "observed_mean_pile_loss",
        "target_nature": "cross_validated_on_training_table",
        "n": 512,
        "spearman_rho": None,
        "r2": exp02["cv"]["histgb_r2"],
        "tree_model_used": True,
    })
    pd.DataFrame(t06).to_csv(os.path.join(OUT, "t06_rank_by_predictor.csv"), index=False)

    c6 = pd.read_csv(os.path.join(BASE_DATA_DIR, "loss_benchmark_bridge_expanded.csv"))
    t05 = {
        "c6_n": int(len(c6)),
        "c6_val_loss_min": float(c6["Val_Loss"].min()),
        "c6_val_loss_max": float(c6["Val_Loss"].max()),
        "path_b_loss_min_saved": 1.798,
        "path_b_loss_max_saved": 1.8494,
        "path_b_inside_c6_range": True,
        "path_b_budget_definition": (
            "C4 training FLOPs divided by 1e18, then passed to the Problem-3 "
            "budget that also prices attention and quality. Not a like-for-like "
            "training-FLOP constraint."
        ),
        "multiplier_0.9602": (
            "Hard-coded scenario value. No script in this audit recomputes it "
            "from a frozen recipe."
        ),
    }

    payload = {
        "audit_id": "AUDIT-20260924-T01-T10",
        "git_head": GIT_HEAD,
        "seed": SEED,
        "runtime_s": round(time.time() - t0, 2),
        "did_not_overwrite_original_experiments": True,
        "t01": summary_t01,
        "t02": {
            "definition_kept": (
                "SFA conditional-expectation frontier at the month-90th-percentile "
                "compute, chat regime. Not the maximum of models that still exist, "
                "and not the maximum among models first released that month."
            ),
            "origin_compute_FLOP_p90": c_t0,
            "origin_chat_frontier": start_chat,
            "published_origin": 42.6,
            "technology_slope_as_coded_per_month": slope_code,
            "technology_slope_if_divided_by_two": slope_two_step,
            "slope_bug": (
                "predict_frontier divides a two-month technology change by 3. "
                "The audit does not patch it; the published decline uses the coded divisor."
            ),
            "decomposition": decomp,
            "sfa_sigma_e": float(sfa_full.coef_["sigma_e"]),
            "sfa_sigma_d": float(sfa_full.coef_["sigma_d"]),
            "sfa_b": float(sfa_full.coef_["b"]),
        },
        "t05": t05,
        "t06": t06,
        "t08": q_link,
        "t08_direction_unchanged_in_9_spot_solves": bool(spot["direction_same_as_saved"].all()),
    }
    with open(os.path.join(OUT, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    say(f"wrote {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
