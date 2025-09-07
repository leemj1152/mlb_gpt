# app.py
import os
import json
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from joblib import load

from data_fetch import (
    fetch_schedule,
    fetch_team_season_stats,
    recent_form,
    enrich_with_probable_pitchers,
)
from features import build_features

# 한글 팀명 매핑 (간단 버전)
TEAM_KO = {
    "Arizona Diamondbacks": "애리조나", "Atlanta Braves": "애틀랜타",
    "Baltimore Orioles": "볼티모어", "Boston Red Sox": "보스턴",
    "Chicago Cubs": "시카고C", "Chicago White Sox": "시카고W",
    "Cincinnati Reds": "신시내티 레즈", "Cleveland Guardians": "클리블랜드",
    "Colorado Rockies": "콜로라도", "Detroit Tigers": "디트로이트",
    "Houston Astros": "휴스턴", "Kansas City Royals": "캔자스시티",
    "Los Angeles Angels": "LA에인절스", "Los Angeles Dodgers": "LA다저스",
    "Miami Marlins": "마이애미", "Milwaukee Brewers": "밀워키",
    "Minnesota Twins": "미네소타", "New York Mets": "뉴욕 메츠",
    "New York Yankees": "뉴욕 양키스", "Athletics": "오클랜드",
    "Philadelphia Phillies": "필라델피아", "Pittsburgh Pirates": "피츠버그",
    "San Diego Padres": "샌디에이고", "San Francisco Giants": "샌프란시스코",
    "Seattle Mariners": "시애틀", "St. Louis Cardinals": "세인트루이스",
    "Tampa Bay Rays": "탬파베이", "Texas Rangers": "텍사스",
    "Toronto Blue Jays": "토론토", "Washington Nationals": "워싱턴",
}
def ko(name: str) -> str:
    return TEAM_KO.get(name, name)

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

@st.cache_data(show_spinner=True)
def load_team_stats(season: int):
    return fetch_team_season_stats(season)

@st.cache_resource(show_spinner=True)
def load_model():
    model_path  = os.path.abspath("models/model.pt")
    scaler_path = os.path.abspath("models/scaler.joblib")
    cols_path   = os.path.abspath("models/feature_cols.json")
    for p in (model_path, scaler_path, cols_path):
        if not os.path.exists(p):
            st.error(f"파일이 없습니다: {p}\n→ train.py로 재학습하여 models/ 아래 3개 파일을 생성하세요.")
            st.stop()

    state = torch.load(model_path, map_location="cpu")
    scaler = load(scaler_path)
    with open(cols_path, "r") as f:
        cols = json.load(f)

    ckpt_in = state["net.0.weight"].shape[1]
    if ckpt_in != len(cols):
        st.error(f"모델/피처 불일치: checkpoint in_dim={ckpt_in}, feature_cols={len(cols)}")
        st.stop()

    model = MLP(in_dim=len(cols))
    model.load_state_dict(state)
    model.eval()
    return model, scaler, cols

st.set_page_config(page_title="MLB Predictor", layout="wide")
st.title("⚾ MLB 홈팀 승리 확률 예측 (Demo)")
st.caption("python-mlb-statsapi + PyTorch + Streamlit — 교육용 데모")

# ===== 사이드바 =====
with st.sidebar:
    target_date = st.date_input("Target Date", value=date.today())
    default_season = target_date.year
    season = st.number_input("Season", min_value=2015, max_value=2100, value=default_season, step=1)
    st.write("먼저 `train.py`로 과거 구간 학습을 완료해야 예측이 가능합니다.")

    # 최근폼은 '선택일-1일'까지(누수 방지)
    td = pd.Timestamp(target_date)
    start_recent = (td - pd.Timedelta(days=30)).date()
    end_recent   = (td - pd.Timedelta(days=1)).date()
    df_recent_source = fetch_schedule(str(start_recent), str(end_recent))
    df_recent = recent_form(df_recent_source, n=10)

    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True, index=0)

# ===== 오늘 스케줄 =====
df_games = fetch_schedule(str(target_date), str(target_date))
if df_games.empty:
    st.info("해당 날짜에 예정된 경기가 없습니다.")
    st.stop()

# 선발투수 지표 주입 (이름/ERA/WHIP/투구손)
df_games = enrich_with_probable_pitchers(df_games, season)

# 선발 id 확보 현황(디버그 캡션)
if {"home_prob_pitcher_id","away_prob_pitcher_id"}.issubset(df_games.columns):
    filled = ((~df_games["home_prob_pitcher_id"].isna())
              & (~df_games["away_prob_pitcher_id"].isna())).sum()
    st.caption(f"🧪 선발투수 id 확보: {filled}/{len(df_games)} 경기")

# 시즌 팀 스탯 (캐시)
df_team = load_team_stats(season)

# ===== 피처 생성 & 예측 =====
Xfeat, merged = build_features(df_games, df_team, df_recent)
model, scaler, cols = load_model()

X = Xfeat.set_index("gamePk")[cols].astype(np.float32)
if X.isna().any().any():
    X = X.fillna(X.median(numeric_only=True))
X = X.dropna(axis=0)
if X.empty:
    st.error("예측에 사용할 유효 피처가 없습니다.")
    st.stop()

Xs = scaler.transform(X.values)
with torch.no_grad():
    proba = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()

# ===== 표시 테이블 구성 =====
out = merged.set_index("gamePk").loc[X.index][[
    "date","home_name","away_name","status","home_score","away_score",
    "home_prob_pitcher_name","away_prob_pitcher_name",
    "home_pp_era","away_pp_era","home_pp_whip","away_pp_whip"
]].copy()

out["home_name_ko"] = out["home_name"].map(ko)
out["away_name_ko"] = out["away_name"].map(ko)

out["P(home win)"] = proba
out["pred_home_win"] = (out["P(home win)"] >= 0.5).astype(int)
out["predicted_winner_en"] = np.where(out["pred_home_win"]==1, out["home_name"], out["away_name"])
out["predicted_winner_ko"] = np.where(out["pred_home_win"]==1, out["home_name_ko"], out["away_name_ko"])

# 과거 경기 채점
is_final = out["status"].str.lower().str.contains("final")
out.loc[is_final, "actual_home_win"] = (out.loc[is_final, "home_score"] > out.loc[is_final, "away_score"]).astype(int)
out["correct"] = None
out.loc[is_final, "correct"] = (out.loc[is_final, "pred_home_win"] == out.loc[is_final, "actual_home_win"]).astype(int)

# 정렬 후 마스크 재계산
out = out.sort_values("P(home win)", ascending=False).reset_index(drop=True)
is_final = out["status"].str.lower().str.contains("final")

# 언어별 테이블
if lang == "한국어":
    display_df = out.rename(columns={
        "home_name_ko":"홈팀","away_name_ko":"원정팀","status":"상태",
        "home_prob_pitcher_name":"홈 선발(예정)","away_prob_pitcher_name":"원정 선발(예정)",
        "home_pp_era":"홈 선발 ERA","away_pp_era":"원정 선발 ERA",
        "home_pp_whip":"홈 선발 WHIP","away_pp_whip":"원정 선발 WHIP",
        "P(home win)":"홈 승 확률","predicted_winner_ko":"예상 승자",
        "correct":"정오","home_score":"홈 점수","away_score":"원정 점수",
    })[["홈팀","원정팀","상태","홈 선발(예정)","원정 선발(예정)","홈 선발 ERA","원정 선발 ERA","홈 선발 WHIP","원정 선발 WHIP","홈 승 확률","예상 승자","정오","홈 점수","원정 점수"]]
else:
    display_df = out.rename(columns={
        "home_name":"Home","away_name":"Away","status":"Status",
        "home_prob_pitcher_name":"Home Prob. SP","away_prob_pitcher_name":"Away Prob. SP",
        "home_pp_era":"Home SP ERA","away_pp_era":"Away SP ERA",
        "home_pp_whip":"Home SP WHIP","away_pp_whip":"Away SP WHIP",
        "P(home win)":"P(Home win)","predicted_winner_en":"Predicted Winner",
        "correct":"Correct","home_score":"Home Score","away_score":"Away Score",
    })[["Home","Away","Status","Home Prob. SP","Away Prob. SP","Home SP ERA","Away SP ERA","Home SP WHIP","Away SP WHIP","P(Home win)","Predicted Winner","Correct","Home Score","Away Score"]]

st.subheader(f"예측 결과 — {target_date}")

# Final 경기 정확도 출력
final_rows = out.loc[is_final]
if not final_rows.empty:
    acc = final_rows["correct"].mean()
    st.metric("Final 경기 기준 정확도", f"{acc*100:.1f}% ({int(final_rows['correct'].sum())}/{len(final_rows)})")

# 문자열 "None" → NaN, 숫자 칼럼 강제 수치화
num_candidates = [
    "홈 승 확률", "P(Home win)",
    "홈 선발 ERA","원정 선발 ERA","Home SP ERA","Away SP ERA",
    "홈 선발 WHIP","원정 선발 WHIP","Home SP WHIP","Away SP WHIP",
]
display_df = display_df.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan})
for c in [c for c in num_candidates if c in display_df.columns]:
    display_df[c] = pd.to_numeric(display_df[c], errors="coerce")

# 행 스타일
def row_style(r):
    base = [""] * len(display_df.columns)
    idx = display_df.columns.get_loc
    color_home = "background-color: rgba(59,130,246,0.12);"
    color_away = "background-color: rgba(245,158,11,0.12);"
    color_right = "background-color: rgba(16,185,129,0.18);"
    color_wrong = "background-color: rgba(239,68,68,0.18);"
    prob_col = "홈 승 확률" if lang == "한국어" else "P(Home win)"
    correct_col = "정오" if lang == "한국어" else "Correct"
    pred_home = out.loc[r.name, "pred_home_win"] if r.name < len(out) else 0
    base = [color_home]*len(base) if int(pred_home)==1 else [color_away]*len(base)
    if prob_col in display_df.columns:
        base[idx(prob_col)] = "background-color: rgba(0,0,0,0.06); font-weight:600;"
    if correct_col in display_df.columns and pd.notna(r.get(correct_col, np.nan)):
        base[idx(correct_col)] = color_right if int(r[correct_col])==1 else color_wrong
    return base

# 포맷(존재하는 컬럼에만)
fmt = {}
if "홈 승 확률" in display_df.columns: fmt["홈 승 확률"] = "{:.3f}"
if "P(Home win)" in display_df.columns: fmt["P(Home win)"] = "{:.3f}"
for c in ["홈 선발 ERA","원정 선발 ERA","Home SP ERA","Away SP ERA",
          "홈 선발 WHIP","원정 선발 WHIP","Home SP WHIP","Away SP WHIP"]:
    if c in display_df.columns: fmt[c] = "{:.2f}"

styled = display_df.style.format(fmt).apply(row_style, axis=1)
st.dataframe(styled, width="stretch")
