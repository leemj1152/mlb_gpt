
from __future__ import annotations
import pandas as pd
import numpy as np

def _prep_long(df_games: pd.DataFrame) -> pd.DataFrame:
    df = df_games.copy()
    df["date"] = pd.to_datetime(df["date"])
    home = df[["date","home_id","home_score","away_score"]].rename(
        columns={"home_id":"team_id","home_score":"gf","away_score":"ga"}
    ).assign(is_home=1)
    away = df[["date","away_id","away_score","home_score"]].rename(
        columns={"away_id":"team_id","away_score":"gf","home_score":"ga"}
    ).assign(is_home=0)
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team_id","date"]).reset_index(drop=True)
    long["win"] = (long["gf"] > long["ga"]).astype(int)
    return long

def compute_team_rollups(df_games_done: pd.DataFrame, lookback:int=10) -> pd.DataFrame:
    """
    완료 경기 기준으로 '직전 경기까지' 팀 롤링/누적 스탯을 계산.
    멀티인덱스 문제를 피하기 위해 cumcount/cumsum/transform 패턴을 사용.
    """
    import numpy as np
    long = _prep_long(df_games_done)

    # '직전 경기까지' 정보만 쓰도록 shift
    long["win_shift"] = long.groupby("team_id")["win"].shift(1)
    long["gf_shift"]  = long.groupby("team_id")["gf"].shift(1)
    long["ga_shift"]  = long.groupby("team_id")["ga"].shift(1)

    # 이전까지 치른 경기 수: 현재 행 이전까지의 개수
    long["g_played"] = long.groupby("team_id").cumcount()

    # 누적 승수 (NaN→0 처리 후 누적)
    win_shift_filled = long["win_shift"].fillna(0).astype("float64")
    long["win_cum"]  = win_shift_filled.groupby(long["team_id"]).cumsum()

    # 승률: g_played==0이면 NaN (pd.NA 대신 np.nan 사용)
    den = long["g_played"].astype("float64")
    num = long["win_cum"].astype("float64")
    win_pct = np.divide(num, den, out=np.full_like(num, np.nan), where=(den > 0))
    long["win_pct"] = win_pct  # float64 + np.nan만 포함

    # 롤링 평균들: 인덱스 평탄화
    def _roll_mean(s: pd.Series) -> pd.Series:
        return (
            s.fillna(0)
             .groupby(long["team_id"])
             .rolling(lookback, min_periods=3)
             .mean()
             .reset_index(level=0, drop=True)
        )

    long["recent_winrate"] = _roll_mean(long["win_shift"])
    long["avg_gf_lb"]      = _roll_mean(long["gf_shift"])
    long["avg_ga_lb"]      = _roll_mean(long["ga_shift"])
    long["diff_avg_lb"]    = long["avg_gf_lb"] - long["avg_ga_lb"]

    # 팀별 최종 스냅샷
    last = long.sort_values(["team_id","date"]).groupby("team_id").tail(1)
    out = last[[
        "team_id","date","g_played","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb","diff_avg_lb"
    ]].rename(columns={"date":"last_date"})

    return out

def build_game_features_from_history(df_games_all: pd.DataFrame, asof_date: str, lookback:int=10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features for games on 'asof_date' using only history up to *asof_date - 1 day*.
    df_games_all should include both completed and scheduled games with columns
    [gamePk,date,home_id,away_id,home_score,away_score,status].
    Returns (X_feat, merged_view).
    """
    df = df_games_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    target_date = pd.to_datetime(asof_date)
    # History cutoff: yesterday
    hist_cut = target_date - pd.Timedelta(days=1)

    # Completed games strictly before target_date
    df_done = df.dropna(subset=["home_score","away_score"]).query("date <= @hist_cut").copy()

    # rollups
    team_roll = compute_team_rollups(df_done, lookback=lookback)

    # games to predict on target_date
    df_tgt = df.query("date == @target_date").copy()
    if df_tgt.empty:
        return pd.DataFrame(), df_tgt

    # join rollups for home & away
    home = team_roll.add_prefix("home_")
    away = team_roll.add_prefix("away_")
    merged = (df_tgt
              .merge(home, left_on="home_id", right_on="home_team_id", how="left")
              .merge(away, left_on="away_id", right_on="away_team_id", how="left"))

    # feature diffs
    feat_cols = ["g_played","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb","diff_avg_lb"]
    X = pd.DataFrame({"gamePk": merged["gamePk"].values})
    for c in feat_cols:
        X[f"diff_{c}"] = merged[f"home_{c}"] - merged[f"away_{c}"]

    return X, merged

def build_training_set_rolling(df_games_all: pd.DataFrame, train_end: str, lookback:int=10) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create leak-free training set for games up to train_end (inclusive label uses same day).
    For each historical game date t <= train_end, build features using history up to t-1.
    Returns (X, y, merged_rows)
    """
    df = df_games_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    cutoff = pd.to_datetime(train_end)

    # Only use completed games up to cutoff for labels
    df_done = df.dropna(subset=["home_score","away_score"]).query("date <= @cutoff").copy()
    if df_done.empty:
        return pd.DataFrame(), pd.Series(dtype=float), df_done

    dates = sorted(df_done["date"].unique())
    rows = []
    labels = []
    merged_keep = []

    for current_date in dates:
        # history strictly before current_date
        hist = df_done.query("date < @current_date").copy()
        if hist.empty:
            continue
        team_roll = compute_team_rollups(hist, lookback=lookback)
        home = team_roll.add_prefix("home_")
        away = team_roll.add_prefix("away_")

        day_games = df_done.query("date == @current_date").copy()
        if day_games.empty:
            continue
        merged = (day_games
                  .merge(home, left_on="home_id", right_on="home_team_id", how="left")
                  .merge(away, left_on="away_id", right_on="away_team_id", how="left"))
        # features
        feat_cols = ["g_played","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb","diff_avg_lb"]
        Xrow = pd.DataFrame({"gamePk": merged["gamePk"].values})
        for c in feat_cols:
            Xrow[f"diff_{c}"] = merged[f"home_{c}"] - merged[f"away_{c}"]
        yrow = (merged["home_score"] > merged["away_score"]).astype("float32").values
        rows.append(Xrow)
        labels.append(yrow)
        merged_keep.append(merged)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()
    X = pd.concat(rows, ignore_index=True)
    y = pd.Series([v for arr in labels for v in arr], dtype="float32")
    merged_all = pd.concat(merged_keep, ignore_index=True)
    return X, y, merged_all
