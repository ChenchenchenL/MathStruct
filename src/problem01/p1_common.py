# -*- coding: utf-8 -*-
"""Shared loaders/compression for Problem-1 scripts (A1-A3 JSONL, EQ07-11)."""
import json
import lzma
import numpy as np
import pandas as pd

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
# 22 model features after compression (logit pairs renamed, qurater -> 4 dims)
FEATURES = SCALAR_FIELDS + ["fineweb_edu", "fluency", "adfree",
                            "modernbert_cleanliness", "modernbert_readability",
                            "modernbert_reasoning", "modernbert_professionalism",
                            "qur_ws", "qur_re", "qur_ft", "qur_ev"]
BASE5 = ["fineweb_edu", "fluency", "modernbert_reasoning",
         "modernbert_readability", "modernbert_cleanliness"]


def safe_comp(record):
    """Compress one JSON record into the 22-dim feature vector (M1-EQ07..11)."""
    out = {}
    for f in SCALAR_FIELDS:
        v = record.get(f)
        out[f] = float(v) if isinstance(v, (int, float)) else np.nan
    v = record.get("fineweb_edu")
    out["fineweb_edu"] = float(v[0]) if isinstance(v, list) and len(v) >= 1 and isinstance(v[0], (int, float)) else np.nan
    for f, key in (("fluency_en", "fluency"), ("ad_en", "adfree")):
        v = record.get(f)
        if isinstance(v, list) and len(v) >= 2 and all(isinstance(t, (int, float)) for t in v):
            out[key] = float(v[1]) - float(v[0])  # EQ08/09 position2 - position1
        else:
            out[key] = np.nan
    for f in ("modernbert_cleanliness", "modernbert_readability",
              "modernbert_reasoning", "modernbert_professionalism"):
        v = record.get(f)
        if isinstance(v, list) and len(v) == 6 and all(isinstance(t, (int, float)) for t in v):
            z = np.asarray(v, dtype=float)
            z = z - z.max()
            e = np.exp(z) / np.exp(z).sum()
            out[f] = float((np.arange(6) * e).sum() / 5.0)  # EQ10 expected level in [0,1]
        else:
            out[f] = np.nan
    v = record.get("qurater")
    if isinstance(v, list) and len(v) == 4 and all(isinstance(t, (int, float)) for t in v):
        for j, name in enumerate(["qur_ws", "qur_re", "qur_ft", "qur_ev"]):
            out[name] = float(v[j])
    else:
        for name in ("qur_ws", "qur_re", "qur_ft", "qur_ev"):
            out[name] = np.nan
    return out


def read_split(path, domain, domain_field=None, keep_content=False):
    rows = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = safe_comp(rec)
            row["sid"] = rec.get("id", "")
            row["domain"] = str(rec.get(domain_field, domain)) if domain_field else domain
            if keep_content:
                row["content"] = rec.get("content", "")
            rows.append(row)
    return pd.DataFrame(rows)


def fit_quantile_maps(df, features, grid=1001):
    maps = {}
    for f in features:
        v = df[f].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        maps[f] = np.quantile(v, np.linspace(0, 1, grid))
    return maps


def frozen_rank(df, features, maps, grid=1001):
    out = pd.DataFrame(index=df.index)
    lv = np.linspace(0, 1, grid)
    for f in features:
        v = df[f].to_numpy(dtype=float)
        r = np.interp(v, maps[f], lv)
        out[f] = np.where(np.isnan(v), np.nan, r)
    return out
