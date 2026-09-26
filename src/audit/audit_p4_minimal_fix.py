# -*- coding: utf-8 -*-
"""Minimal Problem-4 repair check.

The old forecast loses to persistence. This script changes only two things:
exact compute matches replace family-level first-row matches, and positive C4
token counts enter a three-term historical decomposition. It writes only to
review/audit_20260924 and does not replace the formal Problem-4 results.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = r"D:\project\MathStruct"
sys.path.insert(0, os.path.join(ROOT, "src"))
from problem04.p4_common import (  # noqa: E402
    load_and_match_datasets,
    DynamicSFAModel,
    compute_shapley_decomposition,
)

OUT = os.path.join(ROOT, "review", "audit_20260924")
SPLIT = "2024-10-01"


def eligible(df):
    lic = df["License_Category"].isin(
        ["Strict_Permissive", "Research_Open", "Research_Open_EpochVerified"])
    compute = df["Training_Compute_FLOP"].notnull() & (df["Training_Compute_FLOP"] >= 1e18)
    regime = df["Regime"].isin(["Pretrained", "Chat_Finetuned"])
    return df[lic & compute & regime].copy()


def monthly_eval(data):
    train = data[data["Sub_Date"] < SPLIT]
    test = data[data["Sub_Date"] >= SPLIT]
    model = DynamicSFAModel(delta=0.95)
    model.fit(train)
    last_month = int(train["month_idx"].max())
    anchor = train[(train["month_idx"] == last_month) & (train["is_chat"] == 1)]
    previous = float(anchor["Average ⬆️"].max())
    flat = previous
    rows = []
    for month, sub in test.groupby("month_idx"):
        chat = sub[sub["is_chat"] == 1]
        if len(sub) < 3 or chat.empty:
            continue
        actual = float(chat["Average ⬆️"].max())
        compute_p90 = float(np.percentile(sub["Training_Compute_FLOP"], 90))
        pred = float(model.predict_frontier(np.array([compute_p90]), int(month), 1)[0])
        rows.append({
            "month_idx": int(month),
            "actual_top_chat": actual,
            "model_prediction": pred,
            "model_abs_error": abs(actual - pred),
            "persistence_abs_error": abs(actual - previous),
            "flat_abs_error": abs(actual - flat),
        })
        previous = actual
    return model, pd.DataFrame(rows)


def main():
    matched, *_ = load_and_match_datasets()
    old = eligible(matched)
    exact = eligible(matched[matched["Match_Level"].isin(["L1", "L2"])])
    _, old_eval = monthly_eval(old)
    exact_model, exact_eval = monthly_eval(exact)

    # Historical decomposition on the exact-match sample. Token count is the
    # required C4 field missing from the old two-term split.
    hist = exact.copy()
    hist["tokens"] = pd.to_numeric(hist["C4_Dataset_Size_Tokens"], errors="coerce")
    hist = hist[hist["tokens"].notna() & (hist["tokens"] > 0)]
    y = np.log(np.clip(hist["Average ⬆️"], 1e-4, 100 - 1e-4) / (100 - np.clip(hist["Average ⬆️"], 1e-4, 100 - 1e-4)))
    x = np.column_stack([
        np.log(hist["Training_Compute_FLOP"]),
        np.log(hist["tokens"]),
        hist["month_idx"],
        hist["is_chat"],
    ])
    fit = LinearRegression().fit(x, y)
    start = hist[hist["month_idx"] == hist["month_idx"].min()]
    end = hist[hist["month_idx"] == hist["month_idx"].max()]
    x0 = np.array([
        np.log(np.percentile(start["Training_Compute_FLOP"], 90)),
        np.log(np.percentile(start["tokens"], 90)),
        start["month_idx"].min(),
        1,
    ])
    x1 = np.array([
        np.log(np.percentile(end["Training_Compute_FLOP"], 90)),
        np.log(np.percentile(end["tokens"], 90)),
        end["month_idx"].max(),
        1,
    ])
    names = ["compute", "tokens", "time", "chat"]
    base = float(fit.predict(x0.reshape(1, -1))[0])
    contributions = {}
    for i, name in enumerate(names):
        changed = x0.copy()
        changed[i] = x1[i]
        contributions[name] = float(fit.predict(changed.reshape(1, -1))[0] - base)
    total = float(fit.predict(x1.reshape(1, -1))[0] - base)

    c0 = float(np.percentile(exact.loc[exact["month_idx"] == exact["month_idx"].min(), "Training_Compute_FLOP"], 90))
    c1 = float(np.percentile(exact.loc[exact["month_idx"] == exact["month_idx"].max(), "Training_Compute_FLOP"], 90))
    full_model = DynamicSFAModel(delta=0.95)
    full_model.fit(exact)
    shapley = compute_shapley_decomposition(
        full_model, c0, c1, int(exact["month_idx"].min()), int(exact["month_idx"].max()), 1)

    summary = {
        "old_fuzzy_match": {
            "rows": int(len(old)),
            "unique_compute_values": int(old["Training_Compute_FLOP"].nunique()),
            "months": int(len(old_eval)),
            "model_mae": float(old_eval["model_abs_error"].mean()),
            "persistence_mae": float(old_eval["persistence_abs_error"].mean()),
            "flat_mae": float(old_eval["flat_abs_error"].mean()),
        },
        "exact_match": {
            "rows": int(len(exact)),
            "unique_compute_values": int(exact["Training_Compute_FLOP"].nunique()),
            "months": int(len(exact_eval)),
            "model_mae": float(exact_eval["model_abs_error"].mean()),
            "persistence_mae": float(exact_eval["persistence_abs_error"].mean()),
            "flat_mae": float(exact_eval["flat_abs_error"].mean()),
            "beats_persistence": bool(exact_eval["model_abs_error"].mean() < exact_eval["persistence_abs_error"].mean()),
        },
        "token_decomposition_exact_match": {
            "rows_with_positive_tokens": int(len(hist)),
            "r2": float(fit.score(x, y)),
            "logit_contributions": contributions,
            "total_logit_change": total,
            "note": "Associational split on exact matches. Not a validated forecast.",
        },
        "old_style_shapley_on_exact_matches": {
            "scale_share_pct": shapley["share_compute_pct"],
            "tech_share_pct": shapley["share_tech_pct"],
            "scale_points": shapley["delta_compute"],
            "tech_points": shapley["delta_tech"],
        },
    }
    exact_eval.to_csv(os.path.join(OUT, "p4_exact_match_monthly.csv"), index=False)
    with open(os.path.join(OUT, "p4_minimal_fix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
