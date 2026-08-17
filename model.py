"""
model.py — Heuristic G-League breakout scoring model.

This is intentionally NOT a trained ML classifier. The historical sample is
small (14 players) and every one of them is a "positive" case (they all broke
out), so there's no negative class to fit a real classifier against. Instead,
this is a transparent, weighted composite-z-score heuristic: each prospect's
G-League stat profile is standardized against the pooled distribution of
{historical breakout snapshots + current prospects} and combined into a single
0-100 "Breakout Score" using adjustable weights. Think of it as a systematized
scouting rubric, not a prediction with a calibrated error rate.

Core idea: the 14 historical players give us a rough statistical shape of what
a G-League stint looked like right before a real NBA breakout (elite
efficiency, strong per-36 production, meaningful role/minutes, often young).
Current prospects are scored on how closely their own G-League profile matches
that shape, plus a nearest-neighbor "best comp" from the historical set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Core features used by the model. Every one of these must be computable for
# BOTH the historical and current datasets so scores are apples-to-apples.
FEATURES = ["ts_pct", "pts_per36", "reb_per36", "ast_per36", "def_per36", "age"]

# Which direction is "good" for each feature (used when standardizing so that
# a higher z-score always means "more breakout-like").
FEATURE_SIGN = {
    "ts_pct": 1,
    "pts_per36": 1,
    "reb_per36": 1,
    "ast_per36": 1,
    "def_per36": 1,
    "age": -1,  # younger is better
}

DEFAULT_WEIGHTS = {
    "ts_pct": 0.25,      # efficiency
    "pts_per36": 0.20,   # scoring load
    "ast_per36": 0.15,   # playmaking
    "reb_per36": 0.15,   # rebounding
    "def_per36": 0.15,   # steals + blocks
    "age": 0.10,         # youth premium
}

FEATURE_LABELS = {
    "ts_pct": "Efficiency (TS%)",
    "pts_per36": "Scoring (PTS/36)",
    "ast_per36": "Playmaking (AST/36)",
    "reb_per36": "Rebounding (REB/36)",
    "def_per36": "Defense (STL+BLK/36)",
    "age": "Youth (inverse age)",
}


def load_historical() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "historical_breakouts.csv"))
    df["group"] = "Historical Breakout"
    df["mpg"] = df["mpg_g"]
    df["age"] = df["age_g"]
    df["pts"] = df["pts_g"]
    df["reb"] = df["reb_g"]
    df["ast"] = df["ast_g"]
    df["stl"] = df["stl_g"]
    df["blk"] = df["blk_g"]
    df["ts_pct"] = df["ts_g"]
    return df


def load_current() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "current_prospects.csv"))
    df["group"] = "Current Prospect"
    return df


def _per36(row_value: float, mpg: float) -> float:
    if pd.isna(row_value) or pd.isna(mpg) or mpg == 0:
        return np.nan
    return row_value / mpg * 36.0


def add_per36_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pts_per36"] = df.apply(lambda r: _per36(r["pts"], r["mpg"]), axis=1)
    df["reb_per36"] = df.apply(lambda r: _per36(r["reb"], r["mpg"]), axis=1)
    df["ast_per36"] = df.apply(lambda r: _per36(r["ast"], r["mpg"]), axis=1)
    stl36 = df.apply(lambda r: _per36(r["stl"], r["mpg"]), axis=1)
    blk36 = df.apply(lambda r: _per36(r["blk"], r["mpg"]), axis=1)
    df["stl_per36"] = stl36
    df["blk_per36"] = blk36
    # def_per36 = steals + blocks per 36. Many current prospects are missing
    # steal/block data; median-impute rather than treating missing as zero
    # (zero would unfairly punish players we simply don't have data for).
    combined = stl36.fillna(0) + blk36.fillna(0)
    both_missing = stl36.isna() & blk36.isna()
    df["def_per36"] = combined
    df["def_per36_imputed"] = both_missing
    return df


def build_pool() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (historical_df, current_df, pooled_df) all with per-36 + feature columns."""
    hist = add_per36_columns(load_historical())
    cur = add_per36_columns(load_current())

    pool = pd.concat([hist, cur], ignore_index=True, sort=False)
    med = pool["def_per36"].median()
    pool.loc[pool["def_per36_imputed"], "def_per36"] = med
    hist.loc[hist["def_per36_imputed"], "def_per36"] = med
    cur.loc[cur["def_per36_imputed"], "def_per36"] = med

    # impute any remaining missing feature values with the pooled median
    for col in FEATURES:
        med_val = pool[col].median()
        pool[col] = pool[col].fillna(med_val)
        hist[col] = hist[col].fillna(med_val)
        cur[col] = cur[col].fillna(med_val)

    return hist, cur, pool


def standardize(pool: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Z-score `target`'s FEATURES columns using pool mean/std, sign-adjusted
    so higher-is-always-better."""
    out = target.copy()
    for col in FEATURES:
        mean = pool[col].mean()
        std = pool[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            out[f"z_{col}"] = 0.0
        else:
            out[f"z_{col}"] = ((target[col] - mean) / std) * FEATURE_SIGN[col]
    return out


def composite_score(df: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    total_w = sum(weights.values()) or 1.0
    norm_weights = {k: v / total_w for k, v in weights.items()}

    out = df.copy()
    z_cols = [f"z_{c}" for c in FEATURES]
    out["composite_z"] = sum(out[f"z_{c}"] * norm_weights.get(c, 0) for c in FEATURES)
    return out


def percentile_scores(pool_scored: pd.DataFrame) -> pd.Series:
    """Convert composite_z into a friendly 0-100 percentile-rank score."""
    ranks = pool_scored["composite_z"].rank(pct=True)
    return (ranks * 100).round(1)


def score_all(weights: dict[str, float] | None = None):
    hist, cur, pool = build_pool()
    pool_std = standardize(pool, pool)
    pool_scored = composite_score(pool_std, weights)
    pool_scored["breakout_score"] = percentile_scores(pool_scored)

    hist_scored = pool_scored[pool_scored["group"] == "Historical Breakout"].reset_index(drop=True)
    cur_scored = pool_scored[pool_scored["group"] == "Current Prospect"].reset_index(drop=True)
    return hist_scored, cur_scored, pool_scored


def nearest_comps(prospect_row: pd.Series, hist_scored: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Euclidean distance in standardized feature space -> closest historical comps."""
    z_cols = [f"z_{c}" for c in FEATURES]
    prospect_vec = prospect_row[z_cols].to_numpy(dtype=float)
    hist_vecs = hist_scored[z_cols].to_numpy(dtype=float)
    dists = np.linalg.norm(hist_vecs - prospect_vec, axis=1)
    tmp = hist_scored.copy()
    tmp["distance"] = dists
    # convert distance to a rough 0-100 similarity score (bounded, heuristic)
    max_d = dists.max() if len(dists) and dists.max() > 0 else 1.0
    tmp["similarity_pct"] = (100 * (1 - tmp["distance"] / (max_d + 1e-9))).clip(lower=0).round(1)
    return tmp.sort_values("distance").head(k)[
        ["player", "gleague_team", "gleague_season", "breakout_note", "distance", "similarity_pct"]
    ]


if __name__ == "__main__":
    hist_scored, cur_scored, pool_scored = score_all()
    print("=== Historical Breakouts (sanity check) ===")
    print(hist_scored[["player", "breakout_score"]].sort_values("breakout_score", ascending=False).to_string(index=False))
    print("\n=== Current Prospects Leaderboard ===")
    print(cur_scored[["player", "gleague_team", "breakout_score"]].sort_values("breakout_score", ascending=False).to_string(index=False))
    print("\n=== Example nearest comps for top prospect ===")
    top = cur_scored.sort_values("breakout_score", ascending=False).iloc[0]
    print(f"Top prospect: {top['player']}")
    print(nearest_comps(top, hist_scored).to_string(index=False))
