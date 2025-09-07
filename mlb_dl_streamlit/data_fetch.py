# data_fetch.py
from __future__ import annotations
import os
from typing import Dict, List
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import get_mlb_client

API_BASE = "https://statsapi.mlb.com/api/v1"

# 세션 재사용(속도↑)
_SESSION = requests.Session()

# 간단 디스크 캐시: 선발 시즌 스탯
CACHE_DIR = ".cache"
def _cache_path(season: int) -> str:
    return os.path.join(CACHE_DIR, f"pitchers_{season}.csv")

def _load_pitcher_cache(season: int) -> pd.DataFrame | None:
    p = _cache_path(season)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None

def _save_pitcher_cache(df: pd.DataFrame, season: int) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.drop_duplicates(subset=["player_id"], inplace=True)
    df.to_csv(_cache_path(season), index=False)

def _to_date_str(d: str | datetime | date) -> str:
    if isinstance(d, (datetime, date)):
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    if isinstance(d, str):
        try:
            return pd.to_datetime(d).strftime("%Y-%m-%d")
        except Exception:
            return d[:10]
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def _get_json(url: str, params: dict | None = None) -> dict:
    r = _SESSION.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    """
    스케줄 + 점수/상태 → probablePitcher id/name 주입
      1) /api/v1/schedule?hydrate=probablePitcher
      2) (부족 시) /api/v1.1/game/{gamePk}/feed/live probablePitchers 폴백
    """
    mlb = get_mlb_client()
    start_s, end_s = _to_date_str(start), _to_date_str(end)

    # 1) 기본 스케줄(여기선 선발 컬럼 만들지 않음)
    schedule = mlb.get_schedule(start_date=start_s, end_date=end_s, sport_id=1)
    if schedule is None or getattr(schedule, "dates", None) is None:
        return pd.DataFrame(columns=[
            "gamePk","date","season","home_id","away_id","home_name","away_name",
            "home_score","away_score","status",
            "home_prob_pitcher_id","home_prob_pitcher_name",
            "away_prob_pitcher_id","away_prob_pitcher_name",
        ])
    rows = []
    for d in schedule.dates:
        for g in d.games:
            if g.status.detailedstate in ("Postponed","Suspended","Cancelled"):
                continue
            rows.append({
                "gamePk": g.gamepk,
                "date": d.date,
                "season": int(g.season),
                "home_id": g.teams.home.team.id,
                "away_id": g.teams.away.team.id,
                "home_name": g.teams.home.team.name,
                "away_name": g.teams.away.team.name,
                "home_score": getattr(g.teams.home, "score", None),
                "away_score": getattr(g.teams.away, "score", None),
                "status": g.status.detailedstate,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["gamePk"])

    # 2) schedule hydrate로 선발 1차 주입
    try:
        js = _get_json(
            f"{API_BASE}/schedule",
            params={"startDate": start_s, "endDate": end_s, "sportId": 1, "hydrate": "probablePitcher"},
        )
        mp = []
        for d in js.get("dates", []):
            for g in d.get("games", []):
                teams = g.get("teams", {})
                home_pp = (teams.get("home", {}) or {}).get("probablePitcher", {}) or {}
                away_pp = (teams.get("away", {}) or {}).get("probablePitcher", {}) or {}
                mp.append({
                    "gamePk": g.get("gamePk"),
                    "home_prob_pitcher_id": home_pp.get("id"),
                    "home_prob_pitcher_name": home_pp.get("fullName"),
                    "away_prob_pitcher_id": away_pp.get("id"),
                    "away_prob_pitcher_name": away_pp.get("fullName"),
                })
        if mp:
            df = df.merge(pd.DataFrame(mp), on="gamePk", how="left", suffixes=("", "_hydr"))
    except Exception:
        pass

    # 2-1) 컬럼 보장(접미사 정리)
    for base in ["home_prob_pitcher_id","home_prob_pitcher_name",
                 "away_prob_pitcher_id","away_prob_pitcher_name"]:
        if base not in df.columns:
            for sfx in ["_hydr","_y","_x"]:
                c = base + sfx
                if c in df.columns:
                    df[base] = df[c]
                    break
            else:
                df[base] = np.nan

    # 3) feed/live 폴백(아직 비어 있는 경기만)
    need_fill = df["home_prob_pitcher_id"].isna() | df["away_prob_pitcher_id"].isna()
    for gpk in df.loc[need_fill, "gamePk"].tolist():
        try:
            live = _get_json(f"https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live")
            pp = (((live or {}).get("gameData", {}) or {}).get("probablePitchers", {}) or {})
            for side in ["home","away"]:
                p = (pp.get(side) or {})
                if p:
                    df.loc[df["gamePk"]==gpk, f"{side}_prob_pitcher_id"] = p.get("id")
                    df.loc[df["gamePk"]==gpk, f"{side}_prob_pitcher_name"] = p.get("fullName")
        except Exception:
            continue

    # 최종 컬럼 세트
    keep = ["gamePk","date","season","home_id","away_id","home_name","away_name",
            "home_score","away_score","status",
            "home_prob_pitcher_id","home_prob_pitcher_name",
            "away_prob_pitcher_id","away_prob_pitcher_name"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]

def fetch_team_season_stats(season: int) -> pd.DataFrame:
    """팀 시즌 타/투 기본 지표"""
    mlb = get_mlb_client()
    raw = mlb.get_teams(sport_id=1)
    teams = raw if isinstance(raw, list) else getattr(raw, "teams", raw)

    rows: List[dict] = []
    for t in teams:
        tid = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        tname = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if tid is None:
            continue

        row = {"team_id": tid, "team_name": tname, "season": int(season)}
        # Hitting
        hit = _get_json(f"{API_BASE}/teams/{tid}/stats",
                        params={"stats": "season", "group": "hitting", "season": int(season)})
        hs = (hit.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))
        # Pitching
        pit = _get_json(f"{API_BASE}/teams/{tid}/stats",
                        params={"stats": "season", "group": "pitching", "season": int(season)})
        ps = (pit.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))

        row.update({
            "hit_runs": hs.get("runs"), "hit_hits": hs.get("hits"), "hit_doubles": hs.get("doubles"),
            "hit_homeruns": hs.get("homeRuns") or hs.get("homeruns"),
            "hit_avg": hs.get("avg"), "hit_obp": hs.get("obp"),
            "hit_slg": hs.get("slg"), "hit_ops": hs.get("ops"),
        })
        row.update({
            "pit_era": ps.get("era"), "pit_whip": ps.get("whip"),
            "pit_strikeouts": ps.get("strikeOuts") or ps.get("strikeouts"),
            "pit_walks": ps.get("baseOnBalls") or ps.get("baseonballs"),
            "pit_hits": ps.get("hits"), "pit_homeruns": ps.get("homeRuns") or ps.get("homeruns"),
        })
        rows.append(row)

    return pd.DataFrame(rows)

def _fetch_pitcher_season_stat(pid: int, season: int) -> dict:
    """단일 투수의 시즌 ERA/WHIP + 투구손"""
    out = {"player_id": pid, "season": int(season), "pp_era": None, "pp_whip": None, "pp_is_rhp": None}
    if pid is None:
        return out
    # 투구손
    try:
        person = _get_json(f"{API_BASE}/people/{pid}")
        hand = (((person or {}).get("people",[{}])[0] or {}).get("pitchHand", {}) or {}).get("code")
        if isinstance(hand, str):
            out["pp_is_rhp"] = 1 if hand.upper().startswith("R") else 0
    except Exception:
        pass
    # 시즌 스탯
    try:
        stats = _get_json(f"{API_BASE}/people/{pid}/stats",
                          params={"stats":"season","group":"pitching","season": int(season)})
        st = (stats.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))
        out["pp_era"], out["pp_whip"] = st.get("era"), st.get("whip")
    except Exception:
        pass
    return out

def enrich_with_probable_pitchers(df_games: pd.DataFrame, season: int) -> pd.DataFrame:
    """스케줄 DF에 선발투수 시즌 지표(ERA/WHIP/투구손) 병합"""
    if df_games is None or df_games.empty:
        return df_games

    df = df_games.copy()
    for side in ["home", "away"]:
        cid = f"{side}_prob_pitcher_id"
        if cid not in df.columns:
            df[cid] = np.nan

    # 유니크 투수 id
    pids = pd.unique(pd.concat(
        [df["home_prob_pitcher_id"].dropna(), df["away_prob_pitcher_id"].dropna()], axis=0
    ))
    pids = [int(x) for x in pids if pd.notna(x)]

    # 캐시 로드
    cache_df = _load_pitcher_cache(season)
    cache = {}
    if cache_df is not None and not cache_df.empty:
        for _, r in cache_df.iterrows():
            cache[int(r["player_id"])] = {
                "player_id": int(r["player_id"]),
                "season": int(r["season"]),
                "pp_era": r.get("pp_era"),
                "pp_whip": r.get("pp_whip"),
                "pp_is_rhp": r.get("pp_is_rhp"),
            }

    missing = [pid for pid in pids if pid not in cache]
    print(f"[Pitcher] season={season} unique={len(pids)} cached={len(cache)} fetch={len(missing)}")

    # 병렬 수집
    fetched_rows = []
    if missing:
        def _fetch(pid: int) -> dict:
            return _fetch_pitcher_season_stat(pid, season)
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(_fetch, pid): pid for pid in missing}
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    fetched_rows.append(fut.result())
                except Exception:
                    pass
                if i % 50 == 0:
                    print(f"[Pitcher] fetched {i}/{len(missing)}")

    # 캐시 + 신규 병합 → 캐시 저장
    all_rows = list(cache.values()) + fetched_rows
    pstat = (pd.DataFrame(all_rows)
             if all_rows else
             pd.DataFrame(columns=["player_id","season","pp_era","pp_whip","pp_is_rhp"]))
    if not pstat.empty:
        _save_pitcher_cache(pstat, season)

    # 홈/원정 조인
    for side in ["home", "away"]:
        df = df.merge(
            pstat.add_prefix(f"{side}_").rename(columns={f"{side}_player_id": f"{side}_prob_pitcher_id"}),
            on=f"{side}_prob_pitcher_id", how="left"
        )

    keep = ["gamePk","date","season","home_id","away_id","home_name","away_name",
            "home_score","away_score","status",
            "home_prob_pitcher_id","home_prob_pitcher_name","home_pp_era","home_pp_whip","home_pp_is_rhp",
            "away_prob_pitcher_id","away_prob_pitcher_name","away_pp_era","away_pp_whip","away_pp_is_rhp"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]

def recent_form(df_schedule: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """팀별 최근 N경기 스냅샷(승률/득실마진/휴식일수/b2b)"""
    if df_schedule.empty:
        return pd.DataFrame(columns=["team_id","recent_winrate","recent_run_diff","last_rest_days","b2b"])

    df = df_schedule.dropna(subset=["home_score","away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    frames = []
    for side in ["home","away"]:
        tmp = df[["date", f"{side}_id", "home_score", "away_score"]].rename(
            columns={f"{side}_id":"team_id"}
        )
        if side == "home":
            tmp["team_runs"] = tmp["home_score"]; tmp["opp_runs"] = tmp["away_score"]
            tmp["win"] = (tmp["home_score"] > tmp["away_score"]).astype(int)
        else:
            tmp["team_runs"] = tmp["away_score"]; tmp["opp_runs"] = tmp["home_score"]
            tmp["win"] = (tmp["away_score"] > tmp["home_score"]).astype(int)

        tmp = tmp.sort_values(["team_id","date"])
        tmp["recent_winrate"]  = tmp.groupby("team_id")["win"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        tmp["recent_run_diff"] = (tmp["team_runs"] - tmp["opp_runs"])
        tmp["recent_run_diff"] = tmp.groupby("team_id")["recent_run_diff"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)

        tmp["prev_date"] = tmp.groupby("team_id")["date"].shift(1)
        tmp["last_rest_days"] = (tmp["date"] - tmp["prev_date"]).dt.days
        tmp["last_rest_days"] = tmp["last_rest_days"].fillna(7)
        tmp["b2b"] = (tmp["last_rest_days"] == 1).astype(int)

        frames.append(tmp[["team_id","date","recent_winrate","recent_run_diff","last_rest_days","b2b"]])

    out = pd.concat(frames, axis=0)
    out = out.sort_values(["team_id","date"]).groupby("team_id").tail(1)
    return out[["team_id","recent_winrate","recent_run_diff","last_rest_days","b2b"]]
