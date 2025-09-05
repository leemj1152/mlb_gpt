# features.py (전체 교체)
from __future__ import annotations
import pandas as pd

# 간단한 피처: 홈/원정 팀 시즌 스탯 차이 + 최근 승률 차이
STAT_COLS = [
    "hit_runs","hit_hits","hit_doubles","hit_homeruns","hit_avg","hit_obp","hit_slg","hit_ops",
    "pit_era","pit_whip","pit_strikeouts","pit_walks","pit_hits","pit_homeruns"
]

def build_features(df_games: pd.DataFrame, df_team_stats: pd.DataFrame, df_recent: pd.DataFrame | None = None):
    # --- 타입 정리: merge 키(팀ID, season)는 꼭 int로 통일 ---
    df_games = df_games.copy()
    df_team_stats = df_team_stats.copy()

    # 숫자로 강제 변환 (문자 -> 숫자), 변환불가 NaN은 드롭 후 int로
    for c in ["home_id", "away_id", "season"]:
        df_games[c] = pd.to_numeric(df_games[c], errors="coerce")
    for c in ["team_id", "season"]:
        df_team_stats[c] = pd.to_numeric(df_team_stats[c], errors="coerce")

    df_games = df_games.dropna(subset=["home_id", "away_id", "season"])
    df_team_stats = df_team_stats.dropna(subset=["team_id", "season"])

    df_games[["home_id", "away_id", "season"]] = df_games[["home_id", "away_id", "season"]].astype("int64")
    df_team_stats[["team_id", "season"]] = df_team_stats[["team_id", "season"]].astype("int64")

    # 팀 스탯 merge용 프리픽스
    home = df_team_stats.add_prefix("home_")
    away = df_team_stats.add_prefix("away_")

    merged = (
        df_games
        .merge(
            home,
            left_on=["home_id", "season"],
            right_on=["home_team_id", "home_season"],
            how="left"
        )
        .merge(
            away,
            left_on=["away_id", "season"],
            right_on=["away_team_id", "away_season"],
            how="left"
        )
    )

    # 최근 승률 (옵션)
    if df_recent is not None and not df_recent.empty:
        df_recent = df_recent.copy()
        df_recent["team_id"] = pd.to_numeric(df_recent["team_id"], errors="coerce").astype("int64")

        r_home = df_recent.rename(columns={"team_id": "home_id", "recent_winrate": "home_recent"})
        r_away = df_recent.rename(columns={"team_id": "away_id", "recent_winrate": "away_recent"})

        merged = (
            merged
            .merge(r_home[["home_id", "home_recent"]], on="home_id", how="left")
            .merge(r_away[["away_id", "away_recent"]], on="away_id", how="left")
        )
    else:
        merged["home_recent"] = 0.5
        merged["away_recent"] = 0.5

    # 차이 피처 생성 (홈 - 원정)
    feat = pd.DataFrame()
    feat["gamePk"] = merged["gamePk"].values

    # 결측이 있을 수 있으니 안전하게 빼기 (자동으로 NaN 결과 가능)
    for c in STAT_COLS:
        hc = f"home_{c}"
        ac = f"away_{c}"
        if hc not in merged.columns:
            merged[hc] = pd.NA
        if ac not in merged.columns:
            merged[ac] = pd.NA
        feat[f"diff_{c}"] = merged[hc].astype("float64") - merged[ac].astype("float64")

    feat["diff_recent"] = merged["home_recent"].astype("float64") - merged["away_recent"].astype("float64")

    # 라벨 (학습 시점에만 존재): 홈 승리 = 1
    if {"home_score", "away_score"}.issubset(merged.columns) and merged["home_score"].notna().any():
        feat["label"] = (merged["home_score"].astype("float64") > merged["away_score"].astype("float64")).astype("float32")

    return feat, merged
