# rolling_features.py
from __future__ import annotations
import pandas as pd
import numpy as np

# =========================
# Helpers
# =========================

def _to_ts(x) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s

def _prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "date" not in df.columns:
        raise ValueError("input dataframe must contain 'date'")

    df["date"] = _to_ts(df["date"])
    for c in ["home_score", "away_score"]:
        if c not in df.columns:
            df[c] = np.nan
    if "status" not in df.columns:
        df["status"] = ""

    sort_cols = []
    if "date" in df.columns:
        sort_cols.append("date")
    if "gamePk" in df.columns:
        sort_cols.append("gamePk")
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df

def _as_long_team_games(df_hist: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame({
        "team_id": df_hist["home_id"].astype("Int64").values,
        "team_name": df_hist.get("home_name", pd.Series([""]*len(df_hist))).values,
        "date": df_hist["date"].values,
        "is_home": 1,
        "gf": df_hist["home_score"].astype("float64").values,
        "ga": df_hist["away_score"].astype("float64").values,
    })
    away = pd.DataFrame({
        "team_id": df_hist["away_id"].astype("Int64").values,
        "team_name": df_hist.get("away_name", pd.Series([""]*len(df_hist))).values,
        "date": df_hist["date"].values,
        "is_home": 0,
        "gf": df_hist["away_score"].astype("float64").values,
        "ga": df_hist["home_score"].astype("float64").values,
    })
    long = pd.concat([home, away], ignore_index=True)

    long["win"] = np.where(
        (long["gf"].notna()) & (long["ga"].notna()),
        (long["gf"] > long["ga"]).astype(int),
        np.nan
    )
    long = long.sort_values(["team_id", "date"], kind="mergesort").reset_index(drop=True)
    return long

def compute_team_rollups(df_done: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    long = _as_long_team_games(df_done)

    played_mask = (long["gf"].notna()) & (long["ga"].notna())
    long["played"] = played_mask.astype(int)

    long["g_played"] = long.groupby("team_id")["played"].transform("cumsum")
    long["win_cum"]  = long.groupby("team_id")["win"].transform(lambda s: s.fillna(0).cumsum())

    with np.errstate(divide="ignore", invalid="ignore"):
        long["win_pct"] = (long["win_cum"] / long["g_played"].replace(0, np.nan)).astype("float64")

    grp = long.groupby("team_id", group_keys=False)

    long["recent_winrate"] = grp["win"].apply(
        lambda s: s.fillna(0).rolling(window=lookback, min_periods=1).mean()
    ).values

    long["avg_gf_lb"] = grp["gf"].apply(
        lambda s: s.ffill().rolling(window=lookback, min_periods=1).mean()
    ).values
    long["avg_ga_lb"] = grp["ga"].apply(
        lambda s: s.ffill().rolling(window=lookback, min_periods=1).mean()
    ).values
    long["diff_avg_lb"] = (long["avg_gf_lb"] - long["avg_ga_lb"]).astype("float64")

    long["gf_ewm"] = grp["gf"].apply(lambda s: s.ffill().ewm(span=lookback, min_periods=1, adjust=False).mean()).values
    long["ga_ewm"] = grp["ga"].apply(lambda s: s.ffill().ewm(span=lookback, min_periods=1, adjust=False).mean()).values
    long["ewm_diff"] = (long["gf_ewm"] - long["ga_ewm"]).astype("float64")

    long["prev_date"] = grp["date"].shift(1)
    long["b2b"] = ((long["date"] - long["prev_date"]).dt.days == 1).astype("int8").fillna(0)

    long["games_3g"] = grp["played"].apply(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    ).astype("int16").values

    snaps = long[[
        "team_id", "date",
        "g_played", "win_pct", "recent_winrate",
        "avg_gf_lb", "avg_ga_lb", "diff_avg_lb",
        "gf_ewm", "ga_ewm", "ewm_diff",
        "b2b", "games_3g"
    ]].copy()

    snaps["team_id"] = snaps["team_id"].astype("Int64")
    snaps["date"] = pd.to_datetime(snaps["date"], errors="coerce")
    snaps = snaps.dropna(subset=["date"]).sort_values(["team_id","date"], kind="mergesort").reset_index(drop=True)
    return snaps

def _monotonic_ok(df: pd.DataFrame) -> bool:
    # 팀별 날짜가 단조 증가인지 확인
    g = df.groupby("team_id")["date"].apply(lambda s: s.is_monotonic_increasing)
    return bool(g.all())

def _merge_side_features(left_df: pd.DataFrame, snaps: pd.DataFrame, side: str) -> pd.DataFrame:
    """팀별로 쪼개서 asof-merge (정렬/타입 문제 확실히 회피)"""
    sid = f"{side}_id"
    if sid not in left_df.columns:
        raise ValueError(f"left_df must contain '{sid}'")

    # 준비: dtype 통일 + 결측 제거
    tmp_left = left_df.rename(columns={sid: "team_id"}).copy()
    tmp_left["team_id"] = pd.to_numeric(tmp_left["team_id"], errors="coerce").astype("Int64")
    tmp_left["date"] = pd.to_datetime(tmp_left["date"], errors="coerce")
    tmp_left = tmp_left.dropna(subset=["team_id", "date"]).copy()
    # asof 요구: int64(non-nullable)로 바꿔 안정성 확보
    tmp_left["team_id"] = tmp_left["team_id"].astype("int64")

    snaps_s = snaps.copy()
    snaps_s["team_id"] = pd.to_numeric(snaps_s["team_id"], errors="coerce").astype("Int64")
    snaps_s["date"] = pd.to_datetime(snaps_s["date"], errors="coerce")
    snaps_s = snaps_s.dropna(subset=["team_id", "date"]).copy()
    snaps_s["team_id"] = snaps_s["team_id"].astype("int64")

    parts = []
    # 팀별 분할 머지 (각 부분에서 date 정렬 보장)
    for tid, lpart in tmp_left.groupby("team_id", sort=False):
        spart = snaps_s[snaps_s["team_id"] == tid]
        if spart.empty:
            # 스냅샷이 없으면 NaN으로 채워진 행 유지
            parts.append(lpart)
            continue
        lpart = lpart.sort_values("date", kind="mergesort")
        spart = spart.sort_values("date", kind="mergesort")

        m = pd.merge_asof(
            lpart, spart.drop(columns=["team_id"]),
            on="date",
            direction="backward",
            allow_exact_matches=False
        )
        parts.append(m)

    merged = pd.concat(parts, ignore_index=True)
    # 원래 컬럼명으로 복구
    merged = merged.rename(columns={"team_id": sid})
    return merged

def _build_diff_features(base: pd.DataFrame) -> pd.DataFrame:
    keep_meta = ["gamePk", "date", "home_id", "away_id", "home_name", "away_name", "status", "home_score", "away_score"]
    meta_cols = [c for c in keep_meta if c in base.columns]
    num_cols = [
        "g_played", "win_pct", "recent_winrate",
        "avg_gf_lb", "avg_ga_lb", "diff_avg_lb",
        "gf_ewm", "ga_ewm", "ewm_diff",
        "b2b", "games_3g"
    ]
    feats = {}
    for col in num_cols:
        h, a = f"home_{col}", f"away_{col}"
        if h in base.columns and a in base.columns:
            feats[f"diff_{col}"] = (base[h] - base[a]).astype("float64")
    X = pd.concat([base[meta_cols], pd.DataFrame(feats, index=base.index)], axis=1)
    return X

# =========================
# Public APIs
# =========================

def build_game_features_from_history(df_hist: pd.DataFrame, target_date: str, lookback: int = 10):
    df_hist = _prepare(df_hist)
    td = pd.to_datetime(target_date)

    df_today = df_hist.loc[df_hist["date"].dt.date == td.date()].copy()
    if df_today.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_done = df_hist.loc[df_hist["date"] < td].copy()
    snaps = compute_team_rollups(df_done, lookback=lookback)

    home = _merge_side_features(df_today[["gamePk","date","home_id"]], snaps, side="home")
    away = _merge_side_features(df_today[["gamePk","date","away_id"]], snaps, side="away")

    merged = df_today.merge(home, on=["gamePk","date"], how="left") \
                     .merge(away, on=["gamePk","date"], how="left")

    # 접두사 정리
    for col in ["g_played","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb","diff_avg_lb","gf_ewm","ga_ewm","ewm_diff","b2b","games_3g"]:
        hx, hy = f"{col}_x", f"{col}_y"
        if hx in merged.columns and f"home_{col}" not in merged.columns:
            merged = merged.rename(columns={hx: f"home_{col}"})
        if hy in merged.columns and f"away_{col}" not in merged.columns:
            merged = merged.rename(columns={hy: f"away_{col}"})

    X = _build_diff_features(merged)
    return X, merged

def build_training_set_rolling(df_hist: pd.DataFrame, train_end: str, lookback: int = 10):
    df_hist = _prepare(df_hist)
    te = pd.to_datetime(train_end)

    base = df_hist.loc[df_hist["date"] <= te].copy()
    if base.empty:
        return pd.DataFrame(), pd.Series(dtype="int8"), pd.DataFrame()

    has_score = base["home_score"].notna() & base["away_score"].notna()
    base_labeled = base.loc[has_score].copy()
    if base_labeled.empty:
        return pd.DataFrame(), pd.Series(dtype="int8"), pd.DataFrame()

    snaps = compute_team_rollups(base, lookback=lookback)

    left_home = base_labeled[["gamePk","date","home_id"]]
    left_away = base_labeled[["gamePk","date","away_id"]]

    home = _merge_side_features(left_home, snaps, side="home")
    away = _merge_side_features(left_away, snaps, side="away")

    merged = base_labeled.merge(home, on=["gamePk","date"], how="left") \
                         .merge(away, on=["gamePk","date"], how="left")

    for col in ["g_played","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb","diff_avg_lb","gf_ewm","ga_ewm","ewm_diff","b2b","games_3g"]:
        hx, hy = f"{col}_x", f"{col}_y"
        if hx in merged.columns and f"home_{col}" not in merged.columns:
            merged = merged.rename(columns={hx: f"home_{col}"})
        if hy in merged.columns and f"away_{col}" not in merged.columns:
            merged = merged.rename(columns={hy: f"away_{col}"})

    Xall = _build_diff_features(merged)
    y = (merged["home_score"] > merged["away_score"]).astype("int8")

    feat_cols = [c for c in Xall.columns if c.startswith("diff_")]
    X = pd.concat([merged[["gamePk","date"]], Xall[feat_cols]], axis=1)

    mask = X[feat_cols].notna().all(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask.values].reset_index(drop=True)
    merged = merged.loc[mask.values].reset_index(drop=True)

    return X, y, merged
