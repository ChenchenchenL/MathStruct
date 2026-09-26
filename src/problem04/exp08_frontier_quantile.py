"""EXP-408: upper-quantile capability surface and C3 comparability audit.

Model M4-v2.1. Reuse EXP-407's frozen C1/C2/C4 sample, matching, record,
compute scenario, and mean-model definitions. C3 is audited but never fitted.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

import exp07_frontier_v2 as base


ROOT = base.ROOT
DATA = base.DATA
TABLES = base.TABLES
FIGURES = base.FIGURES
OUT = ROOT / "result" / "problem04" / "v21"
EXPERIMENT_ID = "EXP-408"
MODEL_VERSION = "M4-v2.1"
DATA_VERSION = "D4V2-C1-C2-C3-C4"
TAU = 0.90
EPSILON = 0.03
SEED = base.SEED
BOOTSTRAP_DRAWS = 200
SCENARIOS = (("momentum", 1.0), ("halved", 0.5), ("stagnation", 0.0))


def design(matched: pd.DataFrame, use_family: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """M4-v2.1 design, with the same identity weights and dummy rules as EXP-407."""
    data = matched.loc[matched["match_level"] != "unmatched"].copy()
    data = data.dropna(subset=["compute_flop", "score", "month_idx"])
    if len(data) < 30 or data["month_idx"].nunique() < 3:
        raise ValueError("Too few time-consistent matched models for quantile fitting")
    cluster = "bootstrap_cluster" if "bootstrap_cluster" in data else "c4_row"
    counts = data.groupby(cluster)[cluster].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    score = np.clip(data["score"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)
    y = np.log(score / (1.0 - score))
    x_base = np.column_stack((
        np.ones(len(data)),
        np.log(data["compute_flop"].to_numpy(dtype=float) / 1e18),
        data["month_idx"].to_numpy(dtype=float),
        (data["regime"] == "Chat_Finetuned").to_numpy(dtype=float),
    ))
    family_names: list[str] = []
    if use_family:
        independent = data.groupby("family")["c4_row"].nunique()
        permitted = set(independent[independent >= 3].index)
        labels = np.array([f if f in permitted else "Other" for f in data["family"]])
        family_names = sorted(set(labels) - {"Other"})
        if "Other" not in labels and family_names:
            family_names = family_names[1:]
        dummies = np.column_stack([(labels == f).astype(float) for f in family_names]) if family_names else np.empty((len(data), 0))
    else:
        dummies = np.empty((len(data), 0))
    return data, np.column_stack((x_base, dummies)), y, weights, family_names


def smooth_pinball(theta: np.ndarray, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                   penalty: np.ndarray) -> tuple[float, np.ndarray]:
    """M4-v2.1 objective and exact gradient; all responses are logit units."""
    residual = y - x @ theta
    scaled = residual / EPSILON
    loss = (TAU - 1.0) * residual + EPSILON * np.logaddexp(0.0, scaled)
    value = float(np.dot(weights, loss) + 0.5 * np.dot(penalty, theta * theta))
    gradient = -(x.T @ (weights * ((TAU - 1.0) + expit(scaled)))) + penalty * theta
    return value, gradient


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, quantile * cumulative[-1], side="left")])


def fit_quantile(matched: pd.DataFrame, use_family: bool = True) -> dict:
    data, x, y, weights, family_names = design(matched, use_family)
    penalty = np.array([0.0, 0.1, 0.1, 0.1] + [5.0] * len(family_names))
    initial = np.zeros(x.shape[1])
    initial[0] = weighted_quantile(y, weights, TAU)
    bounds = [(None, None)] * x.shape[1]
    bounds[1] = (0.0, None)
    result = minimize(
        smooth_pinball, initial, args=(x, y, weights, penalty), jac=True,
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-9, "maxls": 50},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Quantile optimization failed: {result.message}")
    coefficient = result.x
    family_effect = x[:, 4:] @ coefficient[4:] if family_names else np.zeros(len(data))
    return {
        "coef": coefficient,
        "family_names": family_names,
        "mean_family_effect": float(np.average(family_effect, weights=weights)),
        "n_rows": len(data),
        "n_compute_identities": int(data["c4_row"].nunique()),
        "solver_success": bool(result.success),
        "solver_message": str(result.message),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "weighted_training_pinball": pinball_metric(y, x @ coefficient, weights)[0],
    }


def pinball_metric(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    residual = y - prediction
    loss = np.where(residual >= 0, TAU * residual, (TAU - 1.0) * residual)
    return float(np.average(loss, weights=weights)), float(np.average(y <= prediction, weights=weights))


def unseen_family_prediction(model: dict, test: pd.DataFrame) -> np.ndarray:
    """Held-out families have no fitted dummy: their effect is exactly zero."""
    score = np.clip(test["score"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)
    if not np.isfinite(score).all():
        raise ValueError("Invalid held-out scores")
    x = np.column_stack((
        np.ones(len(test)),
        np.log(test["compute_flop"].to_numpy(dtype=float) / 1e18),
        test["month_idx"].to_numpy(dtype=float),
        (test["regime"] == "Chat_Finetuned").to_numpy(dtype=float),
    ))
    return x @ model["coef"][:4]


def family_holdout(audit: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = audit.loc[audit["match_level"] != "unmatched"].dropna(subset=["compute_flop", "score", "month_idx"])
    rows = []
    all_y, all_mean, all_quantile, all_weight = [], [], [], []
    for family in sorted(data["family"].unique()):
        test = data.loc[data["family"] == family]
        test_ids = set(test["c4_row"])
        training = data.loc[~data["c4_row"].isin(test_ids)]
        if set(training["c4_row"]) & test_ids:
            raise AssertionError("C4 identity leaked into a family holdout fold")
        mean_model = base.fit_capability(training)
        quantile_model = fit_quantile(training)
        y = np.log(np.clip(test["score"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)
                   / (1 - np.clip(test["score"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)))
        weights = 1.0 / test.groupby("c4_row")["c4_row"].transform("size").to_numpy(dtype=float)
        mean_prediction = unseen_family_prediction(mean_model, test)
        quantile_prediction = unseen_family_prediction(quantile_model, test)
        mean_loss, mean_coverage = pinball_metric(y, mean_prediction, weights)
        quantile_loss, quantile_coverage = pinball_metric(y, quantile_prediction, weights)
        rows.append({
            "held_out_family": family, "n_test_rows": len(test),
            "n_test_c4_identities": len(test_ids), "n_train_rows": len(training),
            "n_train_c4_identities": int(training["c4_row"].nunique()),
            "mean_pinball_logit": mean_loss, "quantile_pinball_logit": quantile_loss,
            "mean_coverage": mean_coverage, "quantile_coverage": quantile_coverage,
        })
        all_y.append(y)
        all_mean.append(mean_prediction)
        all_quantile.append(quantile_prediction)
        all_weight.append(weights)
    y = np.concatenate(all_y)
    mean_prediction = np.concatenate(all_mean)
    quantile_prediction = np.concatenate(all_quantile)
    weights = np.concatenate(all_weight)
    mean_loss, mean_coverage = pinball_metric(y, mean_prediction, weights)
    quantile_loss, quantile_coverage = pinball_metric(y, quantile_prediction, weights)
    return pd.DataFrame(rows), {
        "n_families": len(rows), "n_test_rows": len(y),
        "mean_pinball_logit": mean_loss, "quantile_pinball_logit": quantile_loss,
        "mean_coverage": mean_coverage, "quantile_coverage": quantile_coverage,
        "quantile_improves_pinball": bool(quantile_loss < mean_loss),
    }


def c3_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    c3 = pd.read_csv(DATA / "leaderboard_extended_timeseries.csv")
    benchmarks = ["IFEval", "BBH", "MATH_Lvl5", "GPQA", "MUSR", "MMLU_PRO"]
    for column in benchmarks + ["Average"]:
        c3[column] = pd.to_numeric(c3[column], errors="coerce")
    c3["n_zero"] = c3[benchmarks].eq(0).sum(axis=1)
    c3["n_missing"] = c3[benchmarks].isna().sum(axis=1)
    c3["all_six_nonzero"] = c3[benchmarks].gt(0).all(axis=1)
    c3["six_mean"] = c3[benchmarks].mean(axis=1)
    c3["average_minus_six_mean"] = c3["Average"] - c3["six_mean"]
    grouped = c3.groupby(["Year", "Source"], dropna=False)
    summary = grouped.agg(
        n_rows=("Model", "size"), rows_with_any_zero=("n_zero", lambda x: int((x > 0).sum())),
        rows_with_any_missing=("n_missing", lambda x: int((x > 0).sum())),
        all_six_nonzero_rows=("all_six_nonzero", "sum"),
        six_score_zero_cells=("n_zero", "sum"), six_score_missing_cells=("n_missing", "sum"),
    ).reset_index()
    for benchmark in benchmarks:
        counts = grouped[benchmark].agg(zero_cells=lambda x: int(x.eq(0).sum()),
                                         missing_cells=lambda x: int(x.isna().sum())).reset_index()
        summary = summary.merge(counts.rename(columns={
            "zero_cells": f"{benchmark}_zero_cells", "missing_cells": f"{benchmark}_missing_cells"
        }), on=["Year", "Source"], validate="one_to_one")
    historical = c3.loc[c3["Source"] != "Open LLM Leaderboard", [
        "Model", "Year", "Source", "Average", *benchmarks, "n_zero", "n_missing",
        "all_six_nonzero", "six_mean", "average_minus_six_mean"
    ]].copy()
    return summary, historical, {
        "c3_rows": len(c3), "historical_source_rows": len(historical),
        "historical_rows_with_zero": int((historical["n_zero"] > 0).sum()),
        "historical_rows_with_missing": int((historical["n_missing"] > 0).sum()),
        "c3_used_in_fit": False,
        "zero_interpretation": "possible_unassessed_placeholder_not_proven_true_zero",
    }


def rolling_backtest(monthly: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = monthly.set_index("month_idx")["cumulative_best"]
    rows = []
    for horizon in (1, 3):
        for target in range(base.SPLIT_MONTH, 10):
            origin = target - horizon
            if origin < base.SPLIT_MONTH - 1:
                continue
            history = records.loc[:origin].to_numpy(dtype=float)
            current = float(history[-1])
            mean_gain = float(np.diff(history).mean())
            training = audit.loc[audit["month_idx"] <= origin]
            if (training["month_idx"] > origin).any():
                raise AssertionError("Rolling training used future submissions")
            mean_model = base.fit_capability(training)
            quantile_model = fit_quantile(training)
            actual = float(records.loc[target])
            row = {
                "horizon_months": horizon, "origin_month_idx": origin,
                "target_month_idx": target, "target_month": monthly.loc[target, "month"],
                "actual_cumulative_best": actual, "persistence": current,
                "mean_gain": float(min(100, current + horizon * mean_gain)),
                "mean_conditional": base.conditional_record(mean_model, training, origin, current, horizon),
                "quantile_conditional": base.conditional_record(quantile_model, training, origin, current, horizon),
                "training_model_rows": quantile_model["n_rows"],
                "record_updated": bool(actual > current + 1e-10),
            }
            for method in ("persistence", "mean_gain", "mean_conditional", "quantile_conditional"):
                row[f"{method}_abs_error"] = abs(actual - row[method])
            rows.append(row)
    backtest = pd.DataFrame(rows)
    summary = backtest.groupby("horizon_months").agg(
        n_origins=("actual_cumulative_best", "size"),
        n_record_updates=("record_updated", "sum"),
        persistence_mae=("persistence_abs_error", "mean"),
        mean_gain_mae=("mean_gain_abs_error", "mean"),
        mean_conditional_mae=("mean_conditional_abs_error", "mean"),
        quantile_conditional_mae=("quantile_conditional_abs_error", "mean"),
    ).reset_index()
    one_month = summary.loc[summary["horizon_months"] == 1].iloc[0]
    methods = ("persistence", "mean_gain", "mean_conditional", "quantile_conditional")
    selected = min(methods, key=lambda name: (one_month[f"{name}_mae"], methods.index(name)))
    return backtest, summary, {
        "selected_method": selected, "selection_horizon_months": 1,
        "selected_mae_points": float(one_month[f"{selected}_mae"]),
        "quantile_beats_persistence": bool(one_month["quantile_conditional_mae"] < one_month["persistence_mae"]),
    }


def bootstrap_sensitivity(audit: pd.DataFrame, origin: int, record: float) -> tuple[dict, dict]:
    matched = audit.loc[audit["match_level"] != "unmatched"].copy()
    identities = matched["c4_row"].dropna().unique()
    groups = {identity: matched.loc[matched["c4_row"] == identity] for identity in identities}
    rng = np.random.default_rng(SEED)
    predictions = {(horizon, name): [] for horizon in (12, 24) for name, _ in SCENARIOS}
    failed = 0
    for _ in range(BOOTSTRAP_DRAWS):
        selected = rng.choice(identities, size=len(identities), replace=True)
        pieces = []
        for draw_id, identity in enumerate(selected):
            piece = groups[identity].copy()
            piece["bootstrap_cluster"] = draw_id
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            model = fit_quantile(sample)
            for horizon in (12, 24):
                for name, ratio in SCENARIOS:
                    predictions[(horizon, name)].append(
                        base.conditional_record(model, audit, origin, record, horizon, ratio)
                    )
        except (ValueError, RuntimeError):
            failed += 1
    successful = BOOTSTRAP_DRAWS - failed
    if successful < 0.8 * BOOTSTRAP_DRAWS:
        raise RuntimeError("Fewer than 80% of C4 identity bootstrap fits succeeded")
    sensitivity = {
        key: {"parameter_sensitivity_p10": float(np.percentile(values, 10)),
              "parameter_sensitivity_p90": float(np.percentile(values, 90))}
        for key, values in predictions.items()
    }
    return sensitivity, {"requested": BOOTSTRAP_DRAWS, "successful": successful, "failed": failed,
                         "seed": SEED, "coverage_calibrated": False}


def future_scenarios(monthly: pd.DataFrame, audit: pd.DataFrame, quantile_model: dict,
                     mean_model: dict, selection: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    origin = int(monthly["month_idx"].max())
    record = float(monthly["cumulative_best"].iloc[-1])
    average_gain = float(monthly["record_gain"].iloc[1:].mean())
    compute, annual_growth, compute_monthly = base.compute_growth(audit)
    sensitivity, bootstrap = bootstrap_sensitivity(audit, origin, record)
    rows = []
    for horizon in (12, 24):
        for name, ratio in SCENARIOS:
            forecasts = {
                "persistence": record,
                "mean_gain": float(min(100, record + horizon * average_gain)),
                "mean_conditional": base.conditional_record(mean_model, audit, origin, record, horizon, ratio),
                "quantile_conditional": base.conditional_record(quantile_model, audit, origin, record, horizon, ratio),
            }
            rows.append({
                "horizon_months": horizon, "scenario": name,
                "annual_log_compute_growth": annual_growth * ratio,
                "origin_compute_p90_flop": compute,
                "future_compute_p90_flop": compute * math.exp(annual_growth * ratio * horizon / 12),
                "current_record_score": record,
                "selected_method": selection["selected_method"],
                "main_point_forecast": forecasts[selection["selected_method"]],
                **{f"{key}_score": value for key, value in forecasts.items()},
                **sensitivity[(horizon, name)],
                "bootstrap_successful": bootstrap["successful"],
                "coverage_calibrated": False,
                "conditional_status": "unvalidated_12_24_month_sensitivity",
            })
    return pd.DataFrame(rows), compute_monthly, bootstrap


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "experiment_id", EXPERIMENT_ID)
    output.insert(1, "model_version", MODEL_VERSION)
    output.insert(2, "data_version", DATA_VERSION)
    return output


def make_figure(monthly: pd.DataFrame, future: pd.DataFrame, hashes: dict,
                script_hash: str, base_hash: str) -> list[Path]:
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.plot(monthly["month_idx"], monthly["cumulative_best"], color="#17614f", marker="o",
            label="Observed cumulative record")
    x_future = np.array([9, 21, 33])
    colors = {"momentum": "#b43f38", "halved": "#bd7a24", "stagnation": "#4473a8"}
    for name, group in future.groupby("scenario"):
        ordered = group.sort_values("horizon_months")
        values = np.r_[monthly["cumulative_best"].iloc[-1], ordered["quantile_conditional_score"].to_numpy()]
        ax.plot(x_future, values, color=colors[name], linestyle="--", marker="s",
                label=f"{name} condition")
    selected = future.loc[future["scenario"] == "momentum"].sort_values("horizon_months")
    ax.plot(x_future, np.r_[monthly["cumulative_best"].iloc[-1], selected["main_point_forecast"].to_numpy()],
            color="#30343b", linewidth=1.7, linestyle=":", label="Backtest-selected forecast")
    ax.axvline(9, color="#777777", linewidth=0.8)
    ax.set(xlabel="Months since 2024-06", ylabel="Leaderboard average score (0-100)",
           xlim=(0, 33), ylim=(0, 100))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    pdf = FIGURES / "exp408_frontier_quantile.pdf"
    png = FIGURES / "exp408_frontier_quantile.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    metadata = {
        "figure_id": "EXP408-F01", "experiment_id": EXPERIMENT_ID,
        "problem": "problem-04", "model_version": MODEL_VERSION, "data_version": DATA_VERSION,
        "source_data_sha256": hashes, "script": "src/problem04/exp08_frontier_quantile.py",
        "script_sha256": script_hash, "shared_source_sha256": base_hash,
        "formats": ["pdf", "png"], "dpi_png": 300, "width_mm": 180, "height_mm": 91,
        "title_in_figure": False, "axes": {"x": "months since 2024-06", "y": "score out of 100"},
        "notes": "Dashed: unvalidated conditional sensitivities. Dotted: backtest-selected forecast. No predictive coverage claim.",
    }
    meta = FIGURES / "exp408_frontier_quantile.json"
    meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    caption = FIGURES / "exp408_frontier_quantile.pdf.caption.md"
    caption.write_text(
        "# EXP-408 cumulative record and conditional upper-quantile surface\n\n"
        "Observed record uses eligible C1 submissions. Dashed curves vary compute growth while holding the fitted M4-v2.1 "
        "upper-quantile relationship fixed; they are unvalidated 12/24-month sensitivities. Dotted is the forecast "
        "selected by six one-month rolling tests. Input D4V2-C1/C2/C4; C3 audit only. No coverage claim.\n",
        encoding="utf-8",
    )
    return [pdf, png, meta, caption]


def synthetic_validation() -> dict:
    x = np.column_stack((np.ones(5), np.arange(5, dtype=float)))
    y = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    weights = np.ones(5)
    penalty = np.array([0.0, 0.1])
    theta = np.array([1.1, 0.2])
    _, analytical = smooth_pinball(theta, x, y, weights, penalty)
    numerical = np.array([
        (smooth_pinball(theta + np.eye(2)[j] * 1e-6, x, y, weights, penalty)[0]
         - smooth_pinball(theta - np.eye(2)[j] * 1e-6, x, y, weights, penalty)[0]) / 2e-6
        for j in range(2)
    ])
    gradient_error = float(np.max(np.abs(analytical - numerical)))
    if gradient_error > 1e-6:
        raise AssertionError(f"Pinball analytical gradient failed: {gradient_error}")
    intercept_x = np.ones((5, 1))
    result = minimize(smooth_pinball, np.array([2.0]), args=(intercept_x, np.arange(5.0), weights,
                       np.array([0.0])), jac=True, method="L-BFGS-B")
    if not result.success or result.x[0] <= 2.0:
        raise AssertionError("Upper quantile did not rise above the median")
    slope = minimize(smooth_pinball, np.array([1.0, 0.0]), args=(x, y, weights, penalty),
                     jac=True, method="L-BFGS-B", bounds=[(None, None), (0.0, None)])
    if not slope.success or slope.x[1] < -1e-12 or slope.x[1] > 1e-7:
        raise AssertionError("Nonnegative compute coefficient bound failed")
    records = np.maximum.accumulate([8.0, 7.0, 10.0, 9.0])
    if np.any(np.diff(records) < 0):
        raise AssertionError("Cumulative record is not monotone")
    return {"gradient_max_error": gradient_error, "synthetic_quantile_intercept": float(result.x[0]),
            "synthetic_bound_slope": float(slope.x[1]), "record_monotone": True}


def run(validate_only: bool = False) -> None:
    hashes = base.verify_data()
    primary, filter_counts = base.load_primary()
    audit = base.match_compute(primary, base.load_c4())
    monthly = base.monthly_record(primary)
    checks = synthetic_validation()
    base.small_validation(primary, audit, monthly)
    early_model = fit_quantile(audit.loc[audit["month_idx"] < base.SPLIT_MONTH])
    early_record = float(monthly.loc[monthly["month_idx"] == base.SPLIT_MONTH - 1, "cumulative_best"].iloc[0])
    early_forecast = base.conditional_record(early_model, audit.loc[audit["month_idx"] < base.SPLIT_MONTH],
                                             base.SPLIT_MONTH - 1, early_record, 1)
    if not early_record <= early_forecast <= 100:
        raise AssertionError("Upper-quantile conditional forecast violates record bounds")
    checks["early_forecast_bound_check"] = early_forecast
    if validate_only:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "validation": checks}, ensure_ascii=False, indent=2))
        return

    c3_summary, c3_historical, c3_counts = c3_audit()
    quantile_model = fit_quantile(audit)
    mean_model = base.fit_capability(audit)
    holdout, holdout_summary = family_holdout(audit)
    historical = base.decompose(quantile_model, audit)
    backtest, backtest_summary, selection = rolling_backtest(monthly, audit)
    future, compute_monthly, bootstrap = future_scenarios(monthly, audit, quantile_model, mean_model, selection)
    if (future["main_point_forecast"] < future["current_record_score"] - 1e-12).any():
        raise AssertionError("A forecast fell below the observed record")
    if (audit.loc[audit["match_level"] != "unmatched", "c4_pub_date"]
        > audit.loc[audit["match_level"] != "unmatched", "date"]).any():
        raise AssertionError("Accepted compute metadata postdates submission")

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frames = {
        TABLES / "exp408_match_audit.csv": audit,
        TABLES / "exp408_monthly_record.csv": monthly,
        TABLES / "exp408_c3_year_source_audit.csv": c3_summary,
        TABLES / "exp408_c3_historical_rows.csv": c3_historical,
        TABLES / "exp408_family_holdout.csv": holdout,
        TABLES / "exp408_rolling_backtest.csv": backtest,
        TABLES / "exp408_backtest_summary.csv": backtest_summary,
        TABLES / "exp408_scenario_forecast.csv": future,
        TABLES / "exp408_compute_monthly.csv": compute_monthly,
    }
    for path, frame in frames.items():
        add_metadata(frame).to_csv(path, index=False, encoding="utf-8")
    script_hash = base.sha256(Path(__file__))
    base_hash = base.sha256(Path(base.__file__))
    figure_paths = make_figure(monthly, future, hashes, script_hash, base_hash)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    summary = {
        "experiment_id": EXPERIMENT_ID, "model_version": MODEL_VERSION, "data_version": DATA_VERSION,
        "script_sha256": script_hash, "shared_exp407_source_sha256": base_hash,
        "git_head": git_head, "seed": SEED, "filter_counts": filter_counts,
        "match_counts": audit["match_level"].value_counts().to_dict(),
        "future_dated_accepted_matches": 0,
        "c3_audit": c3_counts,
        "validation": checks,
        "quantile_model": {
            "tau": TAU, "smooth_epsilon_logit": EPSILON,
            "coefficients": {name: float(value) for name, value in zip(
                ["intercept", "log_compute", "month", "chat"] + quantile_model["family_names"],
                quantile_model["coef"])},
            "n_rows": quantile_model["n_rows"],
            "n_compute_identities": quantile_model["n_compute_identities"],
            "weighted_training_pinball_logit": quantile_model["weighted_training_pinball"],
            "iterations": quantile_model["iterations"], "solver_message": quantile_model["solver_message"],
        },
        "family_holdout": holdout_summary,
        "historical_decomposition": historical,
        "backtest": backtest_summary.to_dict("records"),
        "selection": selection,
        "bootstrap": bootstrap,
        "forecast_status": "conditional_sensitivity_not_12_24_month_validated",
        "c3_used_in_fit": False,
    }
    summary_path = OUT / "exp408_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = [*frames, *figure_paths, summary_path]
    manifest = {
        "experiment_id": EXPERIMENT_ID, "model_version": MODEL_VERSION, "data_version": DATA_VERSION,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__, "matplotlib": matplotlib.__version__},
        "data_sha256": hashes, "script_sha256": script_hash,
        "shared_exp407_source_sha256": base_hash, "git_head": git_head,
        "outputs": {str(path.relative_to(ROOT)): base.sha256(path) for path in outputs},
    }
    (OUT / "exp408_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID, "family_holdout": holdout_summary,
        "backtest": summary["backtest"], "selection": selection,
        "bootstrap": bootstrap, "forecast_status": summary["forecast_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Run synthetic and early-window checks without writing results")
    run(validate_only=parser.parse_args().validate_only)
