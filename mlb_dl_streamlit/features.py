# features.py
from __future__ import annotations
import pandas as pd

# 팀 시즌 스탯 열
STAT_COLS = [
    "hit_runs","hit_hits","hit_doubles","hit_homeruns","hit_avg","hit_obp","hit_slg","hit_ops",
    "pit_era","pit_whip","pit_strikeouts","pit_walks","pit_hits","pit_homeruns",
]

# 최근폼 확장 열
RECENT_COLS = ["recent_winrate","recent_run_diff","last_rest_days","b2b"]

def build_features(
    df_games: pd.DataFrame,
    df_team_stats: pd.DataFrame,
    df_recent: pd.DataFrame | None = None,
):
    """
    - df_games: fetch_schedule → enrich_with_probable_pitchers 로 확장된 스케줄
    - df_team_stats: 팀 시즌 통계
    - df_recent: recent_form 결과(팀 단위 최신 스냅샷)
    반환:
      feat: gamePk, 피처(+학습 때는 label)
      merged: 디버그/표시용 병합본
    """
    df_games = df_games.copy()
    df_team_stats = df_team_stats.copy()

    # 타입 정리
    for c in ["home_id","away_id","season"]:
        df_games[c] = pd.to_numeric(df_games[c], errors="coerce")
    for c in ["team_id","season"]:
        df_team_stats[c] = pd.to_numeric(df_team_stats[c], errors="coerce")

    df_games = df_games.dropna(subset=["home_id","away_id","season"])
    df_team_stats = df_team_stats.dropna(subset=["team_id","season"])

    df_games[["home_id","away_id","season"]] = df_games[["home_id","away_id","season"]].astype("int64")
    df_team_stats[["team_id","season"]] = df_team_stats[["team_id","season"]].astype("int64")

    # 팀 시즌 스탯 병합
    home = df_team_stats.add_prefix("home_")
    away = df_team_stats.add_prefix("away_")
    merged = (
        df_games
        .merge(home, left_on=["home_id","season"], right_on=["home_team_id","home_season"], how="left")
        .merge(away, left_on=["away_id","season"], right_on=["away_team_id","away_season"], how="left")
    )

    # 최근폼 병합
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
            merged[f"home_{c}"] = 0.0
            merged[f"away_{c}"] = 0.0

    feat = pd.DataFrame()
    feat["gamePk"] = merged["gamePk"].values

    # 시즌 스탯 차이(홈-원정)
    for c in STAT_COLS:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc not in merged.columns: merged[hc] = pd.NA
        if ac not in merged.columns: merged[ac] = pd.NA
        feat[f"diff_{c}"] = merged[hc].astype("float64") - merged[ac].astype("float64")

    # 최근폼 차이
    for c in RECENT_COLS:
        hc, ac = f"home_{c}", f"away_{c}"
        feat[f"diff_{c}"] = merged[hc].astype("float64") - merged[ac].astype("float64")

    # 선발투수 스탯 차이(선발 주입 시 생성됨)
    for c in ["pp_era","pp_whip","pp_is_rhp"]:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in merged.columns and ac in merged.columns:
            feat[f"diff_{c}"] = merged[hc].astype("float64") - merged[ac].astype("float64")

    # 라벨(완료 경기일 때만)
    if {"home_score","away_score"}.issubset(merged.columns) and merged["home_score"].notna().any():
        feat["label"] = (merged["home_score"].astype("float64") > merged["away_score"].astype("float64")).astype("float32")

    # 전부 NaN인 열 제거
    feat = feat.dropna(axis=1, how="all")
    return feat, merged
