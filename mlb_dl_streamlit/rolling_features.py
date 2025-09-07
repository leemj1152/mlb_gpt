# features.py (교체본)
import pandas as pd
import numpy as np

# 최근폼 df 스키마 가정:
# columns 예시: ['team_id','team_name','date','g_played','win_pct','recent_winrate',
#                'avg_gf_lb','avg_ga_lb','ewm_gf','ewm_ga','b2b','games_3g']
# 팀 시즌스탯 df 스키마 가정:
# ['team_id','team_name','season','W','L','win_pct_season','runs_per_game','runs_allowed_per_game', ...]

SAFE_FILL = {
    "win_pct": 0.5, "recent_winrate": 0.5,
    "avg_gf_lb": 4.5, "avg_ga_lb": 4.5,
    "ewm_gf": 4.5, "ewm_ga": 4.5,
    "b2b": 0.0, "games_3g": 0.0,
}

def _prep_recent(df_recent: pd.DataFrame) -> pd.DataFrame:
    r = df_recent.copy()
    # 결측 안전 채움
    for c, v in SAFE_FILL.items():
        if c in r.columns:
            r[c] = r[c].fillna(v)
    # 필요한 최소 컬럼 보장
    need = ["team_id","team_name","date","g_played","win_pct","recent_winrate",
            "avg_gf_lb","avg_ga_lb","ewm_gf","ewm_ga","b2b","games_3g"]
    for c in need:
        if c not in r.columns:
            r[c] = SAFE_FILL.get(c, 0.0)
    # 파생
    r["ewm_diff"] = r["ewm_gf"] - r["ewm_ga"]
    r["diff_avg_lb"] = r["avg_gf_lb"] - r["avg_ga_lb"]
    return r

def _prep_team(df_team: pd.DataFrame) -> pd.DataFrame:
    t = df_team.copy()
    # 시즌 지표 없으면 생성(안전 기본값)
    if "win_pct_season" not in t.columns:
        if "W" in t.columns and "L" in t.columns:
            t["win_pct_season"] = t["W"] / (t["W"] + t["L"]).replace(0, np.nan)
        else:
            t["win_pct_season"] = 0.5
    t["win_pct_season"] = t["win_pct_season"].fillna(0.5)
    if "runs_per_game" not in t.columns:
        t["runs_per_game"] = t.get("R", pd.Series(0.0, index=t.index)) / \
                             t.get("G", pd.Series(1.0, index=t.index)).replace(0, 1.0)
    if "runs_allowed_per_game" not in t.columns:
        t["runs_allowed_per_game"] = t.get("RA", pd.Series(0.0, index=t.index)) / \
                                     t.get("G", pd.Series(1.0, index=t.index)).replace(0, 1.0)
    t["season_diff_rg"] = t["runs_per_game"] - t["runs_allowed_per_game"]
    return t

def _pick_latest_before(r: pd.DataFrame, date_col="date"):
    # 한 팀당 가장 최근 1행 선택 (이미 recent가 최근치라면 그대로)
    # 스케줄별 조인 시 game date기준으로 당일 이전 값으로 맞추는 게 이상적이나,
    # 현 파이프라인(MLP)은 당일 스냅샷으로 쓰던 구조 → 여기선 "최신 한 줄" 사용
    idx = r.groupby("team_id")[date_col].idxmax()
    return r.loc[idx].reset_index(drop=True)

def build_features(df_games: pd.DataFrame,
                   df_team: pd.DataFrame,
                   df_recent: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    입력:
      - df_games: fetch_schedule(...) 결과, columns 예시:
          ['gamePk','date','status','home_id','home_name','away_id','away_name','home_score','away_score', ...]
      - df_team:  fetch_team_season_stats(...)
      - df_recent: recent_form(fetch_schedule(...), n=10) 결과(팀별 최근폼 스냅샷 누적)

    출력:
      - Xfeat: 학습/추론 피처 (gamePk 포함)
      - merged: 메타(표시용) 포함 원본+피처 일부
    """
    games = df_games.copy()
    # 타입/결측 정리
    games["date"] = pd.to_datetime(games["date"])
    for c in ["home_id","away_id"]:
        if c in games.columns:
            games[c] = pd.to_numeric(games[c], errors="coerce").astype("Int64")

    team = _prep_team(df_team)
    recent = _prep_recent(df_recent)

    # 각 팀 최신 스냅샷 선택
    snap_latest = _pick_latest_before(recent)

    # 홈/원정에 시즌스탯, 최근폼 붙이기
    def _side_merge(side: str):
        id_col = f"{side}_id"
        name_col = f"{side}_name"
        out = games[[ "gamePk","date", id_col, name_col ]].rename(
            columns={id_col:"team_id", name_col:"team_name"}
        )
        # 시즌스탯
        out = out.merge(
            team[["team_id","win_pct_season","runs_per_game","runs_allowed_per_game","season_diff_rg"]],
            on="team_id", how="left", suffixes=("","")
        )
        # 최근폼(최신 스냅샷)
        out = out.merge(
            snap_latest[["team_id","win_pct","recent_winrate","avg_gf_lb","avg_ga_lb",
                         "ewm_gf","ewm_ga","ewm_diff","diff_avg_lb","b2b","games_3g"]],
            on="team_id", how="left", suffixes=("","")
        )
        # 접두사
        out = out.add_prefix(f"{side}_")
        out = out.rename(columns={f"{side}_gamePk":"gamePk", f"{side}_date":"date"})
        return out

    H = _side_merge("home")
    A = _side_merge("away")

    merged = games.merge(H, on=["gamePk","date"], how="left").merge(A, on=["gamePk","date"], how="left")

    # 차이 피처(홈 - 원정)
    diff_map = {
        "win_pct_season": ("home_win_pct_season","away_win_pct_season"),
        "runs_per_game": ("home_runs_per_game","away_runs_per_game"),
        "runs_allowed_per_game": ("home_runs_allowed_per_game","away_runs_allowed_per_game"),
        "season_diff_rg": ("home_season_diff_rg","away_season_diff_rg"),

        "win_pct": ("home_win_pct","away_win_pct"),
        "recent_winrate": ("home_recent_winrate","away_recent_winrate"),
        "avg_gf_lb": ("home_avg_gf_lb","away_avg_gf_lb"),
        "avg_ga_lb": ("home_avg_ga_lb","away_avg_ga_lb"),
        "ewm_gf": ("home_ewm_gf","away_ewm_gf"),
        "ewm_ga": ("home_ewm_ga","away_ewm_ga"),
        "ewm_diff": ("home_ewm_diff","away_ewm_diff"),
        "diff_avg_lb": ("home_diff_avg_lb","away_diff_avg_lb"),
        "b2b": ("home_b2b","away_b2b"),
        "games_3g": ("home_games_3g","away_games_3g"),
    }

    for feat, (hcol, acol) in diff_map.items():
        merged[f"diff_{feat}"] = merged[hcol].fillna(SAFE_FILL.get(feat, 0.0)) - \
                                 merged[acol].fillna(SAFE_FILL.get(feat, 0.0))

    # 최종 피처 선택
    feat_cols = [
        "diff_win_pct_season","diff_runs_per_game","diff_runs_allowed_per_game","diff_season_diff_rg",
        "diff_win_pct","diff_recent_winrate","diff_avg_gf_lb","diff_avg_ga_lb",
        "diff_ewm_gf","diff_ewm_ga","diff_ewm_diff","diff_diff_avg_lb",
        "diff_b2b","diff_games_3g",
    ]

    Xfeat = merged[["gamePk"] + feat_cols].copy()

    # 남겨둘 메타(앱 표시용)
    meta_cols = ["gamePk","date","status","home_name","away_name","home_score","away_score"]
    meta_cols = [c for c in meta_cols if c in merged.columns]
    merged_meta = merged[meta_cols].copy()

    return Xfeat, merged_meta
