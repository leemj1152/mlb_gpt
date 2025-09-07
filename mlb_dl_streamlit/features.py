from __future__ import annotations
import pandas as pd
import numpy as np

STAT_COLS = [
    "hit_runs","hit_hits","hit_doubles","hit_homeruns","hit_avg","hit_obp","hit_slg","hit_ops",
    "pit_era","pit_whip","pit_strikeouts","pit_walks","pit_hits","pit_homeruns",
]
RECENT_COLS = ["recent_winrate","recent_run_diff","last_rest_days","b2b","recent_games_3d","recent_games_5d"]

def _as_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=df.index, dtype="float64")

def build_features(df_games: pd.DataFrame, df_team_stats: pd.DataFrame, df_recent: pd.DataFrame | None = None):
    df_games = df_games.copy(); df_team_stats = df_team_stats.copy()
    for c in ["home_id","away_id","season"]:
        df_games[c] = pd.to_numeric(df_games[c], errors="coerce")
    for c in ["team_id","season"]:
        df_team_stats[c] = pd.to_numeric(df_team_stats[c], errors="coerce")
    df_games = df_games.dropna(subset=["home_id","away_id","season"])
    df_team_stats = df_team_stats.dropna(subset=["team_id","season"])
    df_games[["home_id","away_id","season"]] = df_games[["home_id","away_id","season"]].astype("int64")
    df_team_stats[["team_id","season"]] = df_team_stats[["team_id","season"]].astype("int64")

    home = df_team_stats.add_prefix("home_"); away = df_team_stats.add_prefix("away_")
    merged = (
        df_games
        .merge(home, left_on=["home_id","season"], right_on=["home_team_id","home_season"], how="left")
        .merge(away, left_on=["away_id","season"], right_on=["away_team_id","away_season"], how="left")
    )

    if df_recent is not None and not df_recent.empty:
        df_recent = df_recent.copy()
        df_recent["team_id"] = pd.to_numeric(df_recent["team_id"], errors="coerce").astype("int64")
        r_home = df_recent.rename(columns={"team_id":"home_id", **{c: f"home_{c}" for c in RECENT_COLS}})
        r_away = df_recent.rename(columns={"team_id":"away_id", **{c: f"away_{c}" for c in RECENT_COLS}})
        merged = (
            merged
            .merge(r_home[["home_id"] + [f"home_{c}" for c in RECENT_COLS]], on="home_id", how="left")
            .merge(r_away[["away_id"] + [f"away_{c}" for c in RECENT_COLS]], on="away_id", how="left")
        )
    else:
        for c in RECENT_COLS:
            merged[f"home_{c}"] = 0.0; merged[f"away_{c}"] = 0.0

    feat = pd.DataFrame(); feat["gamePk"] = merged["gamePk"].values

    for c in STAT_COLS:
        hc, ac = f"home_{c}", f"away_{c}"
        feat[f"diff_{c}"] = _as_float_series(merged, hc) - _as_float_series(merged, ac)

    for c in RECENT_COLS:
        hc, ac = f"home_{c}", f"away_{c}"
        feat[f"diff_{c}"] = _as_float_series(merged, hc) - _as_float_series(merged, ac)

    for c in ["pp_era","pp_whip","pp_is_rhp"]:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in merged.columns or ac in merged.columns:
            feat[f"diff_{c}"] = _as_float_series(merged, hc) - _as_float_series(merged, ac)

    for c in ["pp_days_rest","pp_recent_ip3","pp_recent_era3","pp_recent_whip3"]:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in merged.columns or ac in merged.columns:
            feat[f"diff_{c}"] = _as_float_series(merged, hc) - _as_float_series(merged, ac)

    home_pp_is_rhp = _as_float_series(merged, "home_pp_is_rhp")
    away_pp_is_rhp = _as_float_series(merged, "away_pp_is_rhp")
    home_hit_ops   = _as_float_series(merged, "home_hit_ops")
    away_hit_ops   = _as_float_series(merged, "away_hit_ops")

    feat["diff_ops_x_rhp"] = home_hit_ops * away_pp_is_rhp - away_hit_ops * home_pp_is_rhp
    feat["diff_ops_x_lhp"] = home_hit_ops * (1.0 - away_pp_is_rhp) - away_hit_ops * (1.0 - home_pp_is_rhp)

    if {"home_score","away_score"}.issubset(merged.columns) and merged["home_score"].notna().any():
        feat["label"] = (pd.to_numeric(merged["home_score"], errors="coerce")
                         > pd.to_numeric(merged["away_score"], errors="coerce")).astype("float32")

    feat = feat.dropna(axis=1, how="all")
    return feat, merged
