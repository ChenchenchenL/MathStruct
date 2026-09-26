"""EXP-407: time-consistent Problem 4 frontier reconstruction (M4-v2).

The record forecast uses all eligible C1 submissions. C4 compute is matched
only for the separate conditional capability and decomposition analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.stats import theilslopes


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "real_attachments" / "C_efficiency_evolution"
TABLES = ROOT / "result" / "tables" / "problem04"
FIGURES = ROOT / "result" / "figures" / "problem04"
OUT = ROOT / "result" / "problem04" / "v2"
EXPERIMENT_ID = "EXP-407"
MODEL_VERSION = "M4-v2"
DATA_VERSION = "D4V2-C1-C2-C4"
SEED = 20260925
SPLIT_MONTH = 4  # 2024-10, with 2024-06 as month zero.
T0 = pd.Timestamp("2025-03-13")
EXPECTED_HASHES = {
    "leaderboard_cleaned.csv": "199891128629f24f83ed2f51211eee65a82a13f45275e79a8227d5dacc7c2846",
    "leaderboard_enhanced.csv": "d8ae5b1e00f36bf17adb0c3fbf88c60edd54298625d2db1aa744b2b001202799",
    "leaderboard_extended_timeseries.csv": "ab2b5715b2945ebdeeabfb18e8d040f7c68d8c76bd02c01e446959218bfcd184",
    "epoch_all_ai_models.csv": "0b98a01bcb8d96958745b6fbe505416c38f719948d9029b38ff6233425c0e1c1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data() -> dict[str, str]:
    actual = {name: sha256(DATA / name) for name in EXPECTED_HASHES}
    for name, expected in EXPECTED_HASHES.items():
        if actual[name] != expected:
            raise ValueError(f"Data version changed: {name}: {actual[name]}")
    return actual


def normalized_name(value: object) -> str:
    name = str(value).split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]", "", name)


def model_family(name: object) -> str:
    value = str(name).lower()
    for key, label in (
        ("qwen", "Qwen"), ("llama", "Llama"), ("gemma", "Gemma"),
        ("deepseek", "DeepSeek"), ("mistral", "Mistral"), ("mixtral", "Mistral"),
        ("pythia", "Pythia"), ("phi", "Phi"), ("yi-", "Yi"),
        ("internlm", "InternLM"), ("glm", "GLM"), ("bloom", "BLOOM"),
    ):
        if key in value:
            return label
    return "Other"


def license_category(license_name: object, epoch_open: object) -> str:
    name = str(license_name).lower().strip() if pd.notna(license_name) else ""
    if "-nc" in name or "non-commercial" in name or "noncommercial" in name:
        return "noncommercial_excluded"
    if any(name.startswith(prefix) for prefix in ("apache-2.0", "mit", "bsd")):
        return "strict_permissive"
    if any(token in name for token in ("llama", "gemma", "qwen", "cc-by", "openrail", "gpl", "agpl")):
        return "research_open"
    if str(epoch_open).strip().lower() == "yes":
        return "epoch_open_verified"
    return "unknown_excluded"


def regime_category(value: object) -> str:
    name = str(value).lower()
    if "pretrained" in name and "continuously" not in name:
        return "Pretrained"
    if any(term in name for term in ("chat", "fine-tuned", "continuously")):
        return "Chat_Finetuned"
    return "Other"


def load_primary() -> tuple[pd.DataFrame, dict[str, int]]:
    c1 = pd.read_csv(DATA / "leaderboard_cleaned.csv")
    c2 = pd.read_csv(DATA / "leaderboard_enhanced.csv")
    if len(c1) != len(c2) or not c1[["Model", "Submission Date"]].equals(c2[["Model", "Submission Date"]]):
        raise ValueError("C1/C2 model and submission-date rows are not aligned")
    frame = c1.copy()
    frame["c1_row"] = np.arange(len(frame))
    frame["epoch_date"] = pd.to_datetime(c2["Epoch_AI_Publication_Date"], errors="coerce")
    frame["epoch_org"] = c2["Epoch_AI_Organization"]
    frame["license"] = [
        license_category(a, b)
        for a, b in zip(frame["Hub License"], c2["Epoch_AI_Open_Weights"])
    ]
    frame["regime"] = frame["Type"].map(regime_category)
    frame["date"] = pd.to_datetime(frame["Submission Date"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["Average ⬆️"], errors="coerce")
    frame["name_key"] = frame["Model"].map(normalized_name)
    frame["family"] = frame["Model"].map(model_family)
    before = len(frame)
    valid = (
        frame["date"].between("2024-06-01", T0)
        & frame["score"].between(0, 100)
        & frame["regime"].isin(["Pretrained", "Chat_Finetuned"])
        & frame["license"].isin(["strict_permissive", "research_open", "epoch_open_verified"])
    )
    frame = frame.loc[valid].copy()
    eligible = len(frame)
    frame = frame.sort_values(["Model", "date", "score"], ascending=[True, True, False])
    frame = frame.drop_duplicates("Model", keep="first")
    frame["month_idx"] = (frame["date"].dt.year - 2024) * 12 + frame["date"].dt.month - 6
    frame["month"] = frame["date"].dt.to_period("M").astype(str)
    if frame.empty or frame["month_idx"].min() != 0 or frame["month_idx"].max() != 9:
        raise ValueError("Unexpected C1 observation window")
    return frame.sort_values(["date", "Model"]).reset_index(drop=True), {
        "c1_rows": before, "eligible_rows": eligible, "unique_model_rows": len(frame),
        "duplicate_rows_removed": eligible - len(frame),
    }


def load_c4() -> pd.DataFrame:
    c4 = pd.read_csv(DATA / "epoch_all_ai_models.csv", low_memory=False)
    c4["c4_row"] = np.arange(len(c4))
    c4["pub_date"] = pd.to_datetime(c4["Publication date"], errors="coerce")
    c4["compute_flop"] = pd.to_numeric(c4["Training compute (FLOP)"], errors="coerce")
    c4["tokens"] = pd.to_numeric(c4["Training dataset size (total)"], errors="coerce")
    c4["name_key"] = c4["Model"].map(normalized_name)
    valid = c4["pub_date"].between("2018-01-01", T0) & c4["compute_flop"].between(1e18, 1e28)
    return c4.loc[valid].copy()


def match_compute(primary: pd.DataFrame, c4: pd.DataFrame) -> pd.DataFrame:
    by_name = {key: group for key, group in c4.groupby("name_key") if key}
    by_org_date = {
        key: group for key, group in c4.groupby(["Organization", "pub_date"])
    }
    records = []
    for row in primary.itertuples(index=False):
        chosen = None
        level = "unmatched"
        reason = "no_unique_match"
        official = None
        if pd.notna(row.epoch_date) and pd.notna(row.epoch_org):
            official = by_org_date.get((row.epoch_org, row.epoch_date))
            if row.epoch_date > row.date:
                reason = "official_date_after_submission"
            elif official is not None:
                candidates = official.loc[official["pub_date"] <= row.date]
                if len(candidates) == 1:
                    chosen = candidates.iloc[0]
                    level = "official_unique"
                elif len(candidates) > 1:
                    exact = candidates.loc[candidates["name_key"] == row.name_key]
                    if len(exact) == 1:
                        chosen = exact.iloc[0]
                        level = "official_name_unique"
                    else:
                        reason = "official_ambiguous"
        if chosen is None and reason != "official_ambiguous":
            candidates = by_name.get(row.name_key)
            if candidates is not None:
                candidates = candidates.loc[candidates["pub_date"] <= row.date]
                if len(candidates) == 1:
                    chosen = candidates.iloc[0]
                    level = "name_unique"
                elif len(candidates) > 1:
                    reason = "name_ambiguous"
                elif reason == "no_unique_match":
                    reason = "name_date_or_compute_invalid"
        record = {
            "c1_row": row.c1_row, "Model": row.Model, "date": row.date,
            "month_idx": row.month_idx, "score": row.score, "regime": row.regime,
            "family": row.family, "license": row.license, "match_level": level,
            "reject_reason": "" if chosen is not None else reason,
            "c4_row": int(chosen["c4_row"]) if chosen is not None else pd.NA,
            "c4_model": chosen["Model"] if chosen is not None else pd.NA,
            "c4_pub_date": chosen["pub_date"] if chosen is not None else pd.NaT,
            "compute_flop": float(chosen["compute_flop"]) if chosen is not None else np.nan,
            "tokens": float(chosen["tokens"]) if chosen is not None and pd.notna(chosen["tokens"]) else np.nan,
        }
        records.append(record)
    audit = pd.DataFrame(records)
    accepted = audit["match_level"] != "unmatched"
    if (audit.loc[accepted, "c4_pub_date"] > audit.loc[accepted, "date"]).any():
        raise AssertionError("A future C4 model entered the matched sample")
    return audit


def monthly_record(primary: pd.DataFrame) -> pd.DataFrame:
    months = np.arange(0, 10)
    grouped = primary.groupby("month_idx")
    monthly = grouped.agg(
        month=("month", "first"), n_models=("Model", "size"),
        monthly_best=("score", "max"),
        n_chat=("regime", lambda s: int((s == "Chat_Finetuned").sum())),
        n_pretrained=("regime", lambda s: int((s == "Pretrained").sum())),
    ).reindex(months)
    if monthly["monthly_best"].isna().any():
        raise ValueError("A month has no eligible model; no interpolation was preregistered")
    monthly["cumulative_best"] = monthly["monthly_best"].cummax()
    monthly["record_gain"] = monthly["cumulative_best"].diff().fillna(0.0)
    monthly = monthly.reset_index().rename(columns={"month_idx": "month_idx"})
    if (monthly["record_gain"] < -1e-12).any():
        raise AssertionError("Cumulative record decreased")
    return monthly


def fit_capability(matched: pd.DataFrame, use_family: bool = True) -> dict:
    data = matched.loc[matched["match_level"] != "unmatched"].copy()
    data = data.dropna(subset=["compute_flop", "score", "month_idx"])
    if len(data) < 30 or data["month_idx"].nunique() < 3:
        raise ValueError("Too few reliable matched models to fit M4V2-EQ02")
    cluster_col = "bootstrap_cluster" if "bootstrap_cluster" in data else "c4_row"
    counts = data.groupby(cluster_col)[cluster_col].transform("size").astype(float)
    weights = 1.0 / counts.to_numpy()
    score = np.clip(data["score"].to_numpy(dtype=float) / 100.0, 1e-4, 1 - 1e-4)
    y = np.log(score / (1 - score))
    base = np.column_stack([
        np.ones(len(data)),
        np.log(data["compute_flop"].to_numpy(dtype=float) / 1e18),
        data["month_idx"].to_numpy(dtype=float),
        (data["regime"] == "Chat_Finetuned").to_numpy(dtype=float),
    ])
    family_names = []
    labels = np.full(len(data), "Other", dtype=object)
    if use_family:
        family_counts = data.groupby("family")["c4_row"].nunique()
        permitted = set(family_counts[family_counts >= 3].index)
        labels = np.array([name if name in permitted else "Other" for name in data["family"]])
        family_names = sorted(set(labels) - {"Other"})
        if "Other" not in labels and family_names:
            family_names = family_names[1:]
    dummies = np.column_stack([(labels == name).astype(float) for name in family_names]) if family_names else np.empty((len(data), 0))
    x = np.column_stack([base, dummies])
    penalty = np.array([0.0, 0.1, 0.1, 0.1] + [5.0] * len(family_names))
    root_weight = np.sqrt(weights)
    a = np.vstack([x * root_weight[:, None], np.diag(np.sqrt(penalty))])
    target = np.concatenate([y * root_weight, np.zeros(len(penalty))])
    lower = np.full(x.shape[1], -np.inf)
    upper = np.full(x.shape[1], np.inf)
    lower[1] = 0.0
    solution = lsq_linear(a, target, bounds=(lower, upper), tol=1e-10, max_iter=1000, lsq_solver="exact")
    if not solution.success:
        raise RuntimeError(f"Capability model did not converge: {solution.message}")
    coef = solution.x
    family_effect = dummies @ coef[4:] if family_names else np.zeros(len(data))
    fitted = x @ coef
    weighted_mean = float(np.average(y, weights=weights))
    r2 = 1 - float(np.sum(weights * (y - fitted) ** 2) / np.sum(weights * (y - weighted_mean) ** 2))
    return {
        "coef": coef,
        "family_names": family_names,
        "mean_family_effect": float(np.average(family_effect, weights=weights)),
        "n_rows": len(data),
        "n_compute_identities": int(data["c4_row"].nunique()),
        "weighted_r2_logit": r2,
        "solver_status": int(solution.status),
        "solver_message": str(solution.message),
    }


def predict_score(model: dict, compute_flop: float, month_idx: int, is_chat: int = 1) -> float:
    coef = model["coef"]
    y = (
        coef[0] + coef[1] * math.log(compute_flop / 1e18)
        + coef[2] * month_idx + coef[3] * is_chat + model["mean_family_effect"]
    )
    return float(100.0 / (1.0 + math.exp(-float(np.clip(y, -30, 30)))))


def compute_growth(matched: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    data = matched.loc[matched["match_level"] != "unmatched"]
    monthly = data.groupby("month_idx")["compute_flop"].agg(
        n="size", p90=lambda values: float(np.percentile(values, 90))
    ).reset_index()
    monthly = monthly.loc[monthly["n"] >= 3].copy()
    if monthly.empty:
        raise ValueError("No month has three time-consistent compute matches")
    if len(monthly) >= 4:
        monthly_slope = float(theilslopes(np.log(monthly["p90"]), monthly["month_idx"])[0])
        annual_growth = float(np.clip(monthly_slope * 12, 0, np.log(4)))
    else:
        annual_growth = 0.0
    return float(monthly["p90"].iloc[-1]), annual_growth, monthly


def conditional_record(
    model: dict, matched: pd.DataFrame, origin: int, record: float, horizon: int,
    growth_ratio: float = 1.0,
) -> float:
    compute, annual_growth, _ = compute_growth(matched)
    future_compute = compute * math.exp(annual_growth * growth_ratio * horizon / 12.0)
    shift = predict_score(model, future_compute, origin + horizon) - predict_score(model, compute, origin)
    return float(np.clip(record + max(0.0, shift), record, 100.0))


def decompose(model: dict, matched: pd.DataFrame) -> dict:
    data = matched.loc[matched["match_level"] != "unmatched"]
    first = int(data["month_idx"].min())
    last = int(data["month_idx"].max())
    start = data.loc[data["month_idx"] == first, "compute_flop"]
    end = data.loc[data["month_idx"] == last, "compute_flop"]
    if len(start) < 3 or len(end) < 3:
        raise ValueError("At least three compute matches are required at both decomposition endpoints")
    c0, c1 = float(np.percentile(start, 90)), float(np.percentile(end, 90))
    f00 = predict_score(model, c0, first)
    f10 = predict_score(model, c1, first)
    f01 = predict_score(model, c0, last)
    f11 = predict_score(model, c1, last)
    scale = 0.5 * ((f10 - f00) + (f11 - f01))
    time = 0.5 * ((f01 - f00) + (f11 - f10))
    total = f11 - f00
    residual = total - scale - time
    if abs(residual) > 1e-8:
        raise AssertionError("Shapley contributions do not sum to the total")
    return {
        "first_month_idx": first, "last_month_idx": last,
        "first_compute_p90_flop": c0, "last_compute_p90_flop": c1,
        "f00": f00, "f10": f10, "f01": f01, "f11": f11,
        "total_points": total, "compute_points": scale, "time_association_points": time,
        "compute_share_pct": float(scale / total * 100) if abs(total) > 1e-4 else None,
        "time_share_pct": float(time / total * 100) if abs(total) > 1e-4 else None,
        "additivity_residual_points": residual,
    }


def rolling_backtest(monthly: pd.DataFrame, audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = monthly.set_index("month_idx")["cumulative_best"]
    rows = []
    for horizon in (1, 3):
        for target in range(SPLIT_MONTH, 10):
            origin = target - horizon
            if origin < SPLIT_MONTH - 1:
                continue
            history = records.loc[:origin].to_numpy(dtype=float)
            current = float(history[-1])
            mean_gain = float(np.diff(history).mean())
            fitted = fit_capability(audit.loc[audit["month_idx"] <= origin])
            matched = audit.loc[audit["month_idx"] <= origin]
            candidate = conditional_record(fitted, matched, origin, current, horizon)
            actual = float(records.loc[target])
            row = {
                "horizon_months": horizon, "origin_month_idx": origin,
                "target_month_idx": target, "target_month": monthly.loc[target, "month"],
                "actual_cumulative_best": actual,
                "persistence": current,
                "mean_gain": float(min(100, current + horizon * mean_gain)),
                "conditional_model": candidate,
                "training_model_rows": fitted["n_rows"],
                "record_updated": bool(actual > current + 1e-10),
            }
            for method in ("persistence", "mean_gain", "conditional_model"):
                row[f"{method}_abs_error"] = abs(actual - row[method])
            rows.append(row)
    backtest = pd.DataFrame(rows)
    summary = backtest.groupby("horizon_months").agg(
        n_origins=("actual_cumulative_best", "size"),
        n_record_updates=("record_updated", "sum"),
        persistence_mae=("persistence_abs_error", "mean"),
        mean_gain_mae=("mean_gain_abs_error", "mean"),
        conditional_model_mae=("conditional_model_abs_error", "mean"),
    ).reset_index()
    single = summary.loc[summary["horizon_months"] == 1].iloc[0]
    methods = ["persistence", "mean_gain", "conditional_model"]
    selected = min(methods, key=lambda name: (single[f"{name}_mae"], methods.index(name)))
    return backtest, summary, {
        "selected_method": selected,
        "selection_horizon_months": 1,
        "selected_mae_points": float(single[f"{selected}_mae"]),
        "conditional_beats_persistence": bool(single["conditional_model_mae"] < single["persistence_mae"]),
    }


def bootstrap_conditional(
    audit: pd.DataFrame, origin: int, record: float, horizon: int,
    growth_ratio: float, draws: int = 200,
) -> dict:
    matched = audit.loc[audit["match_level"] != "unmatched"].copy()
    cluster_ids = matched["c4_row"].dropna().unique()
    groups = {identity: matched.loc[matched["c4_row"] == identity] for identity in cluster_ids}
    rng = np.random.default_rng(SEED + horizon + round(growth_ratio * 10))
    predictions = []
    for _ in range(draws):
        selection = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        pieces = []
        for draw_id, identity in enumerate(selection):
            piece = groups[identity].copy()
            piece["bootstrap_cluster"] = draw_id
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            model = fit_capability(sample)
            predictions.append(conditional_record(model, audit, origin, record, horizon, growth_ratio))
        except (ValueError, RuntimeError):
            continue
    if len(predictions) < 0.8 * draws:
        raise RuntimeError("Fewer than 80% of cluster bootstrap fits succeeded")
    return {
        "bootstrap_requested": draws,
        "bootstrap_successful": len(predictions),
        "parameter_sensitivity_p10": float(np.percentile(predictions, 10)),
        "parameter_sensitivity_p90": float(np.percentile(predictions, 90)),
        "coverage_calibrated": False,
    }


def future_forecasts(
    monthly: pd.DataFrame, audit: pd.DataFrame, model: dict, selection: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origin = int(monthly["month_idx"].max())
    record = float(monthly["cumulative_best"].iloc[-1])
    mean_gain = float(monthly["record_gain"].iloc[1:].mean())
    current_compute, annual_growth, compute_monthly = compute_growth(audit)
    rows = []
    for horizon in (12, 24):
        persistence = record
        mean_gain_point = float(min(100, record + horizon * mean_gain))
        for key, ratio in (("momentum", 1.0), ("halved", 0.5), ("stagnation", 0.0)):
            condition = conditional_record(model, audit, origin, record, horizon, ratio)
            if selection["selected_method"] == "persistence":
                primary = persistence
            elif selection["selected_method"] == "mean_gain":
                primary = mean_gain_point
            else:
                primary = condition
            uncertainty = bootstrap_conditional(audit, origin, record, horizon, ratio)
            rows.append({
                "horizon_months": horizon, "scenario": key,
                "annual_log_compute_growth": annual_growth * ratio,
                "origin_compute_p90_flop": current_compute,
                "future_compute_p90_flop": current_compute * math.exp(annual_growth * ratio * horizon / 12),
                "current_record_score": record,
                "selected_method": selection["selected_method"],
                "main_point_forecast": primary,
                "conditional_score": condition,
                "persistence_score": persistence,
                "mean_gain_score": mean_gain_point,
                **uncertainty,
                "conditional_status": "unvalidated_sensitivity" if not selection["conditional_beats_persistence"] else "one_month_backtest_selected",
            })
    return pd.DataFrame(rows), compute_monthly


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "experiment_id", EXPERIMENT_ID)
    output.insert(1, "model_version", MODEL_VERSION)
    output.insert(2, "data_version", DATA_VERSION)
    return output


def make_figure(monthly: pd.DataFrame, future: pd.DataFrame, script_hash: str, data_hashes: dict) -> list[Path]:
    plt.rcParams.update({"font.size": 9, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.plot(monthly["month_idx"], monthly["cumulative_best"], color="#1c6b5a", marker="o", label="Observed cumulative record")
    x_future = np.array([9, 21, 33])
    palette = {"momentum": "#b43f38", "halved": "#bd7a24", "stagnation": "#4473a8"}
    for name, group in future.groupby("scenario"):
        ordered = group.sort_values("horizon_months")
        values = np.r_[monthly["cumulative_best"].iloc[-1], ordered["conditional_score"].to_numpy()]
        ax.plot(x_future, values, color=palette[name], linestyle="--", marker="s", label=f"{name} condition")
    main = future.loc[future["scenario"] == "momentum"].sort_values("horizon_months")
    ax.plot(x_future, np.r_[monthly["cumulative_best"].iloc[-1], main["main_point_forecast"].to_numpy()],
            color="#30343b", linewidth=1.7, linestyle=":", label="Selected point forecast")
    ax.axvline(9, color="#777777", linewidth=0.8)
    ax.set(xlabel="Months since 2024-06", ylabel="Leaderboard average score (0-100)", xlim=(0, 33), ylim=(0, 100))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    pdf = FIGURES / "exp407_frontier_v2.pdf"
    png = FIGURES / "exp407_frontier_v2.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    metadata = {
        "figure_id": "EXP407-F01", "experiment_id": EXPERIMENT_ID,
        "model_version": MODEL_VERSION, "data_version": DATA_VERSION,
        "source_data_sha256": data_hashes, "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": script_hash, "formats": ["pdf", "png"], "dpi_png": 300,
        "width_mm": 180, "height_mm": 91, "title_in_figure": False,
        "axes": {"x": "months since 2024-06", "y": "score out of 100"},
        "notes": "Dashed curves are unvalidated conditional sensitivities; the dotted line is the backtest-selected point forecast. No coverage claim.",
    }
    meta_path = FIGURES / "exp407_frontier_v2.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    caption = FIGURES / "exp407_frontier_v2.pdf.caption.md"
    caption.write_text(
        "# EXP-407 cumulative frontier and conditional compute scenarios\n\n"
        "Solid: observed eligible C1 cumulative record. Dashed: M4-v2 conditional compute and time coefficient sensitivities. "
        "Dotted: one-month-backtest-selected point forecast. The sensitivity curves are not calibrated 12/24-month predictive intervals. "
        "Data D4V2-C1/C2/C4; model M4-v2; script src/problem04/exp07_frontier_v2.py.\n",
        encoding="utf-8",
    )
    return [pdf, png, meta_path, caption]


def small_validation(primary: pd.DataFrame, audit: pd.DataFrame, monthly: pd.DataFrame) -> None:
    accepted = audit["match_level"] != "unmatched"
    if len(primary) != len(audit) or (audit.loc[accepted, "c4_pub_date"] > audit.loc[accepted, "date"]).any():
        raise AssertionError("Input/matching invariants failed")
    if not monthly["cumulative_best"].is_monotonic_increasing:
        raise AssertionError("Record is not monotone")
    training = audit.loc[audit["month_idx"] < SPLIT_MONTH]
    model = fit_capability(training)
    current = float(monthly.loc[monthly["month_idx"] == SPLIT_MONTH - 1, "cumulative_best"].iloc[0])
    one_step = conditional_record(model, training, SPLIT_MONTH - 1, current, 1)
    if not current <= one_step <= 100:
        raise AssertionError("Conditional forecast violates record bounds")
    print(json.dumps({
        "validation": "passed", "eligible_unique_models": len(primary),
        "time_consistent_matches": int(accepted.sum()),
        "training_matched_rows": model["n_rows"],
        "one_step_bound_check": one_step,
    }, ensure_ascii=False, indent=2))


def run(validate_only: bool = False) -> None:
    hashes = verify_data()
    primary, filter_counts = load_primary()
    audit = match_compute(primary, load_c4())
    monthly = monthly_record(primary)
    small_validation(primary, audit, monthly)
    if validate_only:
        return
    fitted = fit_capability(audit)
    comparison = fit_capability(audit, use_family=False)
    historical = decompose(fitted, audit)
    backtest, backtest_summary, selection = rolling_backtest(monthly, audit)
    future, compute_monthly = future_forecasts(monthly, audit, fitted, selection)
    if (future["main_point_forecast"] < future["current_record_score"] - 1e-12).any():
        raise AssertionError("A future forecast dropped below the observed record")

    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    output_frames = {
        TABLES / "exp407_match_audit.csv": audit,
        TABLES / "exp407_monthly_record.csv": monthly,
        TABLES / "exp407_compute_monthly.csv": compute_monthly,
        TABLES / "exp407_rolling_backtest.csv": backtest,
        TABLES / "exp407_backtest_summary.csv": backtest_summary,
        TABLES / "exp407_scenario_forecast.csv": future,
    }
    for path, frame in output_frames.items():
        add_metadata(frame).to_csv(path, index=False, encoding="utf-8")
    script_hash = sha256(Path(__file__))
    figure_paths = make_figure(monthly, future, script_hash, hashes)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    summary = {
        "experiment_id": EXPERIMENT_ID, "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION, "script_sha256": script_hash,
        "git_head": git_head, "seed": SEED,
        "filter_counts": filter_counts,
        "match_counts": audit["match_level"].value_counts().to_dict(),
        "rejection_counts": audit.loc[audit["match_level"] == "unmatched", "reject_reason"].value_counts().to_dict(),
        "future_dated_accepted_matches": int(((audit["c4_pub_date"] > audit["date"]) & (audit["match_level"] != "unmatched")).sum()),
        "model": {
            "coefficients": {name: float(value) for name, value in zip(
                ["intercept", "log_compute", "month", "chat"] + fitted["family_names"], fitted["coef"]
            )},
            "weighted_r2_logit": fitted["weighted_r2_logit"],
            "no_family_weighted_r2_logit": comparison["weighted_r2_logit"],
            "n_rows": fitted["n_rows"],
            "n_compute_identities": fitted["n_compute_identities"],
            "solver_status": fitted["solver_status"],
            "solver_message": fitted["solver_message"],
        },
        "historical_decomposition": historical,
        "backtest": backtest_summary.to_dict("records"),
        "selection": selection,
        "forecast_status": "conditional_sensitivity_not_12_24_month_validated",
        "c3_used_in_fit": False,
    }
    summary_path = OUT / "exp407_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs = [*output_frames, *figure_paths, summary_path]
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "data_sha256": hashes,
        "script_sha256": script_hash,
        "git_head": git_head,
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
    }
    (OUT / "exp407_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "matches": summary["match_counts"],
        "backtest": summary["backtest"],
        "selection": selection,
        "forecast_status": summary["forecast_status"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="Check inputs and a small training fit without writing results")
    args = parser.parse_args()
    run(validate_only=args.validate_only)
