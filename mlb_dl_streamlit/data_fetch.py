from __future__ import annotations
import os
from typing import List, Dict
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

from utils import get_mlb_client

API_BASE = "https://statsapi.mlb.com/api/v1"
_SESSION = requests.Session()
CACHE_DIR = ".cache"

def _ensure_cache_dir(): os.makedirs(CACHE_DIR, exist_ok=True)
def _cache_path_pitchers(season:int)->str: return os.path.join(CACHE_DIR, f"pitchers_{season}.csv")
def _cache_dir_logs(season:int)->str: return os.path.join(CACHE_DIR, f"logs_{season}")

def _get_json(url: str, params: dict | None = None) -> dict:
    r = _SESSION.get(url, params=params, timeout=20); r.raise_for_status(); return r.json()
def _to_date_str(d): return pd.to_datetime(d).strftime("%Y-%m-%d")

def _load_pitcher_cache(season:int)->pd.DataFrame|None:
    p=_cache_path_pitchers(season)
    if os.path.exists(p):
        try: return pd.read_csv(p)
        except Exception: return None
    return None
def _save_pitcher_cache(df:pd.DataFrame, season:int)->None:
    _ensure_cache_dir(); df.drop_duplicates(subset=["player_id"], inplace=True)
    df.to_csv(_cache_path_pitchers(season), index=False)
def _load_logs_cache(pid:int, season:int)->pd.DataFrame|None:
    d=_cache_dir_logs(season); os.makedirs(d, exist_ok=True); p=os.path.join(d, f"{pid}.csv")
    if os.path.exists(p):
        try: return pd.read_csv(p, parse_dates=["log_date"])
        except Exception: return None
    return None
def _save_logs_cache(pid:int, season:int, df:pd.DataFrame)->None:
    d=_cache_dir_logs(season); os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f"{pid}.csv"), index=False)

def _ip_to_float(ip):
    if ip is None or (isinstance(ip, float) and np.isnan(ip)): return None
    s=str(ip)
    if "." not in s: return float(int(s))
    whole, frac = s.split(".")
    base=float(int(whole))
    add = 0.0 if frac=="0" else (1.0/3.0 if frac=="1" else (2.0/3.0 if frac=="2" else float("0."+frac)))
    return base+add

def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    mlb = get_mlb_client()
    start_s, end_s = _to_date_str(start), _to_date_str(end)
    schedule = mlb.get_schedule(start_date=start_s, end_date=end_s, sport_id=1)
    rows=[]
    if schedule and getattr(schedule, "dates", None):
        for d in schedule.dates:
            for g in d.games:
                if g.status.detailedstate in ("Postponed","Suspended","Cancelled"):
                    continue
                rows.append({
                    "gamePk": g.gamepk, "date": d.date, "season": int(g.season),
                    "home_id": g.teams.home.team.id, "away_id": g.teams.away.team.id,
                    "home_name": g.teams.home.team.name, "away_name": g.teams.away.team.name,
                    "home_score": getattr(g.teams.home, "score", None),
                    "away_score": getattr(g.teams.away, "score", None),
                    "status": g.status.detailedstate,
                })
    df = pd.DataFrame(rows).drop_duplicates(subset=["gamePk"])
    # hydrate probable pitchers
    try:
        js = _get_json(f"{API_BASE}/schedule", params={"startDate":start_s,"endDate":end_s,"sportId":1,"hydrate":"probablePitcher"})
        mp=[]
        for d in js.get("dates", []):
            for g in d.get("games", []):
                teams=g.get("teams", {})
                home_pp=(teams.get("home", {}) or {}).get("probablePitcher", {}) or {}
                away_pp=(teams.get("away", {}) or {}).get("probablePitcher", {}) or {}
                mp.append({
                    "gamePk": g.get("gamePk"),
                    "home_prob_pitcher_id": home_pp.get("id"),
                    "home_prob_pitcher_name": home_pp.get("fullName"),
                    "away_prob_pitcher_id": away_pp.get("id"),
                    "away_prob_pitcher_name": away_pp.get("fullName"),
                })
        if mp: df = df.merge(pd.DataFrame(mp), on="gamePk", how="left")
    except Exception:
        pass
    for base in ["home_prob_pitcher_id","home_prob_pitcher_name","away_prob_pitcher_id","away_prob_pitcher_name"]:
        if base not in df.columns: df[base]=np.nan
    # fallback feed/live
    need_fill = df["home_prob_pitcher_id"].isna() | df["away_prob_pitcher_id"].isna()
    for gpk in df.loc[need_fill, "gamePk"].tolist():
        try:
            live = _get_json(f"https://statsapi.mlb.com/api/v1.1/game/{gpk}/feed/live")
            pp = (((live or {}).get("gameData", {}) or {}).get("probablePitchers", {}) or {})
            for side in ["home","away"]:
                p=(pp.get(side) or {})
                if p:
                    df.loc[df["gamePk"]==gpk, f"{side}_prob_pitcher_id"]=p.get("id")
                    df.loc[df["gamePk"]==gpk, f"{side}_prob_pitcher_name"]=p.get("fullName")
        except Exception: continue
    keep=["gamePk","date","season","home_id","away_id","home_name","away_name","home_score","away_score","status",
          "home_prob_pitcher_id","home_prob_pitcher_name","away_prob_pitcher_id","away_prob_pitcher_name"]
    for c in keep:
        if c not in df.columns: df[c]=np.nan
    return df[keep]

def fetch_team_season_stats(season: int) -> pd.DataFrame:
    raw = get_mlb_client().get_teams(sport_id=1)
    teams = raw if isinstance(raw, list) else getattr(raw, "teams", raw)
    rows=[]
    for t in teams:
        tid = t.get("id") if isinstance(t, dict) else getattr(t, "id", None)
        tname = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
        if tid is None: continue
        row={"team_id":tid,"team_name":tname,"season":int(season)}
        hit=_get_json(f"{API_BASE}/teams/{tid}/stats", params={"stats":"season","group":"hitting","season":int(season)})
        pit=_get_json(f"{API_BASE}/teams/{tid}/stats", params={"stats":"season","group":"pitching","season":int(season)})
        hs=(hit.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))
        ps=(pit.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))
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

def _fetch_pitcher_season_stat(pid:int, season:int)->dict:
    out={"player_id":pid,"season":int(season),"pp_era":None,"pp_whip":None,"pp_is_rhp":None}
    if pid is None: return out
    try:
        person=_get_json(f"{API_BASE}/people/{pid}")
        hand=(((person or {}).get("people",[{}])[0] or {}).get("pitchHand", {}) or {}).get("code")
        if isinstance(hand, str): out["pp_is_rhp"]=1 if hand.upper().startswith("R") else 0
    except Exception: pass
    try:
        stats=_get_json(f"{API_BASE}/people/{pid}/stats", params={"stats":"season","group":"pitching","season":int(season)})
        st=(stats.get("stats",[{}])[0].get("splits",[{}])[0].get("stat", {}))
        out["pp_era"], out["pp_whip"]=st.get("era"), st.get("whip")
    except Exception: pass
    return out

def _fetch_pitcher_game_logs(pid:int, season:int)->pd.DataFrame:
    cached=_load_logs_cache(pid, season)
    if cached is not None: return cached
    try:
        js=_get_json(f"{API_BASE}/people/{pid}/stats", params={"stats":"gameLog","group":"pitching","season":int(season)})
        splits=(js.get("stats",[{}])[0].get("splits", [])) or []
    except Exception:
        splits=[]
    rows=[]
    for s in splits:
        st=s.get("stat", {}) or {}
        gs=st.get("gamesStarted", 0)
        if not gs: continue
        d=s.get("date") or s.get("gameDate")
        if not d: continue
        rows.append({
            "player_id": pid, "log_date": pd.to_datetime(d),
            "ip": float(_ip_to_float(st.get("inningsPitched"))) if st.get("inningsPitched") is not None else np.nan,
            "er": float(st.get("earnedRuns")) if st.get("earnedRuns") is not None else np.nan,
            "bb": float(st.get("baseOnBalls") or st.get("baseonballs")) if (st.get("baseOnBalls") or st.get("baseonballs")) is not None else np.nan,
            "h":  float(st.get("hits")) if st.get("hits") is not None else np.nan,
        })
    df=pd.DataFrame(rows)
    if not df.empty: df=df.sort_values("log_date").reset_index(drop=True)
    _save_logs_cache(pid, season, df)
    return df

def enrich_with_probable_pitchers(df_games: pd.DataFrame, season: int) -> pd.DataFrame:
    if df_games is None or df_games.empty: return df_games
    df=df_games.copy()
    for side in ["home","away"]:
        cid=f"{side}_prob_pitcher_id"
        if cid not in df.columns: df[cid]=np.nan

    pids = pd.unique(pd.concat([df["home_prob_pitcher_id"].dropna(), df["away_prob_pitcher_id"].dropna()], axis=0))
    pids = [int(x) for x in pids if pd.notna(x)]
    cache_df=_load_pitcher_cache(season); cache={}
    if cache_df is not None and not cache_df.empty:
        for _,r in cache_df.iterrows():
            cache[int(r["player_id"])]={
                "player_id":int(r["player_id"]), "season":int(r["season"]),
                "pp_era":r.get("pp_era"), "pp_whip":r.get("pp_whip"), "pp_is_rhp":r.get("pp_is_rhp"),
            }
    missing=[pid for pid in pids if pid not in cache]
    print(f"[Pitcher] season={season} unique={len(pids)} cached={len(cache)} fetch={len(missing)}")
    fetched=[]
    if missing:
        def _f(pid:int)->dict: return _fetch_pitcher_season_stat(pid, season)
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs={ex.submit(_f, pid):pid for pid in missing}
            for i,fut in enumerate(as_completed(futs),1):
                try: fetched.append(fut.result())
                except Exception: pass
                if i%50==0: print(f"[Pitcher] fetched {i}/{len(missing)}")
    all_rows=list(cache.values())+fetched
    pstat=pd.DataFrame(all_rows) if all_rows else pd.DataFrame(columns=["player_id","season","pp_era","pp_whip","pp_is_rhp"])
    if not pstat.empty: _save_pitcher_cache(pstat, season)

    logs_map={}
    for pid in pids:
        try: logs_map[pid]=_fetch_pitcher_game_logs(pid, season)
        except Exception: logs_map[pid]=pd.DataFrame(columns=["player_id","log_date","ip","er","bb","h"])

    def recent_for_row(row, side:str):
        pid=row.get(f"{side}_prob_pitcher_id"); gdate=pd.to_datetime(row.get("date"))
        if pd.isna(pid) or pid not in logs_map:
            return pd.Series({f"{side}_pp_days_rest":np.nan, f"{side}_pp_recent_ip3":np.nan,
                              f"{side}_pp_recent_era3":np.nan, f"{side}_pp_recent_whip3":np.nan})
        logs=logs_map[pid]
        if logs is None or logs.empty:
            return pd.Series({f"{side}_pp_days_rest":np.nan, f"{side}_pp_recent_ip3":np.nan,
                              f"{side}_pp_recent_era3":np.nan, f"{side}_pp_recent_whip3":np.nan})
        prev=logs[logs["log_date"]<gdate]
        if prev.empty:
            return pd.Series({f"{side}_pp_days_rest":np.nan, f"{side}_pp_recent_ip3":np.nan,
                              f"{side}_pp_recent_era3":np.nan, f"{side}_pp_recent_whip3":np.nan})
        last_date=prev["log_date"].iloc[-1]; days_rest=(gdate-last_date).days
        last3=prev.tail(3); ip_sum=last3["ip"].replace(0, np.nan).sum(skipna=True)
        if ip_sum and ip_sum>0:
            er_sum=last3["er"].fillna(0).sum(); bb_sum=last3["bb"].fillna(0).sum(); h_sum=last3["h"].fillna(0).sum()
            era3=9.0*er_sum/ip_sum; whip3=(bb_sum+h_sum)/ip_sum; ip3=last3["ip"].mean()
        else:
            era3=np.nan; whip3=np.nan; ip3=last3["ip"].mean()
        return pd.Series({f"{side}_pp_days_rest":float(days_rest),
                          f"{side}_pp_recent_ip3":float(ip3) if pd.notna(ip3) else np.nan,
                          f"{side}_pp_recent_era3":float(era3) if pd.notna(era3) else np.nan,
                          f"{side}_pp_recent_whip3":float(whip3) if pd.notna(whip3) else np.nan})

    for side in ["home","away"]:
        df = df.merge(
            pstat.add_prefix(f"{side}_").rename(columns={f"{side}_player_id": f"{side}_prob_pitcher_id"}),
            on=f"{side}_prob_pitcher_id", how="left"
        )
    for side in ["home","away"]:
        recent_feat = df.apply(lambda r: recent_for_row(r, side), axis=1)
        df = pd.concat([df, recent_feat], axis=1)

    keep = ["gamePk","date","season","home_id","away_id","home_name","away_name",
            "home_score","away_score","status",
            "home_prob_pitcher_id","home_prob_pitcher_name","home_pp_era","home_pp_whip","home_pp_is_rhp",
            "away_prob_pitcher_id","away_prob_pitcher_name","away_pp_era","away_pp_whip","away_pp_is_rhp",
            "home_pp_days_rest","home_pp_recent_ip3","home_pp_recent_era3","home_pp_recent_whip3",
            "away_pp_days_rest","away_pp_recent_ip3","away_pp_recent_era3","away_pp_recent_whip3"]
    for c in keep:
        if c not in df.columns: df[c]=np.nan
    return df[keep]

def recent_form(df_schedule: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    if df_schedule.empty:
        return pd.DataFrame(columns=["team_id","recent_winrate","recent_run_diff","last_rest_days","b2b","recent_games_3d","recent_games_5d"])
    df = df_schedule.dropna(subset=["home_score","away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    frames=[]
    for side in ["home","away"]:
        tmp = df[["date", f"{side}_id", "home_score", "away_score"]].rename(columns={f"{side}_id":"team_id"})
        if side=="home":
            tmp["team_runs"]=tmp["home_score"]; tmp["opp_runs"]=tmp["away_score"]; tmp["win"]=(tmp["home_score"]>tmp["away_score"]).astype(int)
        else:
            tmp["team_runs"]=tmp["away_score"]; tmp["opp_runs"]=tmp["home_score"]; tmp["win"]=(tmp["away_score"]>tmp["home_score"]).astype(int)
        tmp = tmp.sort_values(["team_id","date"])
        tmp["recent_winrate"] = tmp.groupby("team_id")["win"].transform(lambda s: s.rolling(n, min_periods=1).mean())
        tmp["run_diff"] = tmp["team_runs"] - tmp["opp_runs"]
        tmp["recent_run_diff"] = tmp.groupby("team_id")["run_diff"].transform(lambda s: s.rolling(n, min_periods=1).mean())
        tmp["prev_date"] = tmp.groupby("team_id")["date"].shift(1)
        tmp["last_rest_days"] = (tmp["date"] - tmp["prev_date"]).dt.days
        tmp["last_rest_days"] = tmp["last_rest_days"].fillna(7)
        tmp["b2b"] = (tmp["last_rest_days"] == 1).astype(int)
        for k in [1,2,3,4,5]:
            tmp[f"prev_date_{k}"] = tmp.groupby("team_id")["date"].shift(k)
        def count_recent(days:int, row)->int:
            c=0; base=row["date"]
            for k in [1,2,3,4,5]:
                d=row.get(f"prev_date_{k}", pd.NaT)
                if pd.isna(d): continue
                if 0 < (base - d).days <= days: c+=1
            return c
        tmp["recent_games_3d"] = tmp.apply(lambda r: count_recent(3, r), axis=1)
        tmp["recent_games_5d"] = tmp.apply(lambda r: count_recent(5, r), axis=1)
        frames.append(tmp[["team_id","date","recent_winrate","recent_run_diff","last_rest_days","b2b","recent_games_3d","recent_games_5d"]])
    out = pd.concat(frames, axis=0)
    out = out.sort_values(["team_id","date"]).groupby("team_id").tail(1)
    return out[["team_id","recent_winrate","recent_run_diff","last_rest_days","b2b","recent_games_3d","recent_games_5d"]]
