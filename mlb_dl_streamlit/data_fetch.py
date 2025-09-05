# data_fetch.py (교체본)
from __future__ import annotations
import pandas as pd
import requests
from datetime import datetime, date
from typing import Tuple, List, Dict, Any
from utils import get_mlb_client

API_BASE = "https://statsapi.mlb.com/api/v1"

def _to_date_str(d: str | datetime | date) -> str:
    """
    어떤 입력이 들어와도 YYYY-MM-DD 로 강제 정규화.
    (예: '2025-08-06 00:00:00' -> '2025-08-06')
    """
    if isinstance(d, (datetime, date)):
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    if isinstance(d, str):
        try:
            return pd.to_datetime(d).strftime("%Y-%m-%d")
        except Exception:
            # 마지막 안전망: 앞 10자리만
            return d[:10]
    # 기타 타입 방어
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    mlb = get_mlb_client()
    start_s = _to_date_str(start)
    end_s = _to_date_str(end)

    schedule = mlb.get_schedule(start_date=start_s, end_date=end_s, sport_id=1)
    if schedule is None or getattr(schedule, "dates", None) is None:
        # API가 400을 내리면 get_schedule이 None을 반환할 수 있음 → 빈 DF 반환
        return pd.DataFrame(columns=[
            "gamePk","date","season","home_id","away_id",
            "home_name","away_name","home_score","away_score","status"
        ])

    rows = []
    for date_block in schedule.dates:
        date_str = date_block.date
        for game in date_block.games:
            if game.status.detailedstate in ("Postponed", "Suspended", "Cancelled"):
                continue
            home = game.teams.home
            away = game.teams.away
            home_score = getattr(home, "score", None)
            away_score = getattr(away, "score", None)
            season = int(game.season)
            rows.append({
                "gamePk": game.gamepk,
                "date": date_str,
                "season": season,
                "home_id": home.team.id,
                "away_id": away.team.id,
                "home_name": home.team.name,
                "away_name": away.team.name,
                "home_score": home_score,
                "away_score": away_score,
                "status": game.status.detailedstate
            })
    return pd.DataFrame(rows).drop_duplicates(subset=["gamePk"])

def _get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_team_season_stats(season: int) -> pd.DataFrame:
    """
    라이브러리의 dataclass 파싱 이슈를 우회하기 위해
    /api/v1/teams/{id}/stats?stats=season&group=hitting|pitching&season=YYYY 엔드포인트의
    원시 JSON을 직접 읽어 필요한 필드만 수집한다.
    """
    mlb = get_mlb_client()
    # get_teams()는 정상 동작하므로 팀 목록은 라이브러리로 가져옴 (리스트/객체 모두 대응)
    raw_teams = mlb.get_teams(sport_id=1)
    teams = raw_teams if isinstance(raw_teams, list) else getattr(raw_teams, "teams", raw_teams)

    rows: list[dict] = []
    for t in teams:
        team_id = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        team_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if team_id is None:
            continue

        row = {"team_id": team_id, "team_name": team_name, "season": int(season)}

        # Hitting
        hit_json = _get_json(
            f"{API_BASE}/teams/{team_id}/stats",
            params={"stats": "season", "group": "hitting", "season": int(season)},
        )
        try:
            hit_stat = (
                hit_json.get("stats", [{}])[0]
                .get("splits", [{}])[0]
                .get("stat", {})
            )
        except Exception:
            hit_stat = {}

        # Pitching
        pit_json = _get_json(
            f"{API_BASE}/teams/{team_id}/stats",
            params={"stats": "season", "group": "pitching", "season": int(season)},
        )
        try:
            pit_stat = (
                pit_json.get("stats", [{}])[0]
                .get("splits", [{}])[0]
                .get("stat", {})
            )
        except Exception:
            pit_stat = {}

        # 필요한 필드만 안전하게 집계 (API가 항목을 추가/변경해도 여기서만 선택)
        row.update({
            "hit_runs": hit_stat.get("runs"),
            "hit_hits": hit_stat.get("hits"),
            "hit_doubles": hit_stat.get("doubles"),
            "hit_homeruns": hit_stat.get("homeRuns") or hit_stat.get("homeruns"),
            "hit_avg": hit_stat.get("avg"),
            "hit_obp": hit_stat.get("obp"),
            "hit_slg": hit_stat.get("slg"),
            "hit_ops": hit_stat.get("ops"),
        })
        row.update({
            "pit_era": pit_stat.get("era"),
            "pit_whip": pit_stat.get("whip"),
            "pit_strikeouts": pit_stat.get("strikeOuts") or pit_stat.get("strikeouts"),
            "pit_walks": pit_stat.get("baseOnBalls") or pit_stat.get("baseonballs"),
            "pit_hits": pit_stat.get("hits"),
            "pit_homeruns": pit_stat.get("homeRuns") or pit_stat.get("homeruns"),
        })

        rows.append(row)

    return pd.DataFrame(rows)

def recent_form(df_schedule: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    df_done = df_schedule.dropna(subset=["home_score", "away_score"]).copy()
    df_done["date"] = pd.to_datetime(df_done["date"])
    frames = []
    for is_home in [True, False]:
        side = "home" if is_home else "away"
        tmp = df_done[["date", f"{side}_id", "home_score", "away_score"]].copy()
        tmp = tmp.rename(columns={f"{side}_id": "team_id"})
        if is_home:
            tmp["win"] = (tmp["home_score"] > tmp["away_score"]).astype(int)
        else:
            tmp["win"] = (tmp["away_score"] > tmp["home_score"]).astype(int)
        tmp = tmp.sort_values(["team_id", "date"])
        tmp["recent_winrate"] = tmp.groupby("team_id")["win"].rolling(n, min_periods=1).mean().reset_index(level=0, drop=True)
        frames.append(tmp[["date", "team_id", "recent_winrate"]])
    out = pd.concat(frames, axis=0)
    out = out.sort_values(["team_id", "date"]).groupby("team_id").tail(1)
    return out[["team_id", "recent_winrate"]]
