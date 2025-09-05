import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from joblib import load
import json
from datetime import date
from data_fetch import fetch_schedule, fetch_team_season_stats, recent_form
from features import build_features

TEAM_KO = {
    "Arizona Diamondbacks": "애리조나",
    "Atlanta Braves": "애틀랜타",
    "Baltimore Orioles": "볼티모어",
    "Boston Red Sox": "보스턴",
    "Chicago Cubs": "시카고C",
    "Chicago White Sox": "시카고W",
    "Cincinnati Reds": "신시내티 레즈",
    "Cleveland Guardians": "클리블랜드",
    "Colorado Rockies": "콜로라도",
    "Detroit Tigers": "디트로이트",
    "Houston Astros": "휴스턴",
    "Kansas City Royals": "캔자스시티",
    "Los Angeles Angels": "LA에인절스",
    "Los Angeles Dodgers": "LA다저스",
    "Miami Marlins": "마이애미",
    "Milwaukee Brewers": "밀워키",
    "Minnesota Twins": "미네소타",
    "New York Mets": "뉴욕 메츠",
    "New York Yankees": "뉴욕 양키스",
    "Athletics": "오클랜드",
    "Philadelphia Phillies": "필라델피아",
    "Pittsburgh Pirates": "피츠버그",
    "San Diego Padres": "샌디에이고",
    "San Francisco Giants": "샌프란시스코",
    "Seattle Mariners": "시애틀",
    "St. Louis Cardinals": "세인트루이스",
    "Tampa Bay Rays": "탬파베이",
    "Texas Rangers": "텍사스",
    "Toronto Blue Jays": "토론토",
    "Washington Nationals": "워싱턴",
}

def ko(name: str) -> str:
    return TEAM_KO.get(name, name)  # 매핑 없으면 원문 유지

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

@st.cache_data(show_spinner=True)
def load_team_stats(season: int):
    return fetch_team_season_stats(season)

@st.cache_resource(show_spinner=True)
def load_model():
    state = torch.load("models/model.pt", map_location="cpu")
    scaler = load("models/scaler.joblib")
    with open("models/feature_cols.json","r") as f:
        cols = json.load(f)
    model = MLP(in_dim=len(cols))
    model.load_state_dict(state)
    model.eval()
    return model, scaler, cols

st.set_page_config(page_title="MLB Predictor", layout="wide")
st.title("⚾ MLB 홈팀 승리 확률 예측 (Demo)")
st.caption("python-mlb-statsapi + PyTorch + Streamlit — 교육용 데모")

with st.sidebar:
    season = st.number_input("Season", min_value=2015, max_value=2100, value=date.today().year, step=1)
    target_date = st.date_input("Target Date", value=date.today())
    st.write("먼저 `train.py`로 과거 구간 학습을 완료해야 예측이 가능합니다.")

    td = pd.Timestamp(target_date)
    start_recent = (td - pd.Timedelta(days=30)).date()  # <-- date 객체
    end_recent = td.date()

    df_recent_source = fetch_schedule(str(start_recent), str(end_recent))
    df_recent = recent_form(df_recent_source, n=10)

    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True, index=0)

# 오늘(선택일) 스케줄
df_games = fetch_schedule(str(target_date), str(target_date))
if df_games.empty:
    st.info("해당 날짜에 예정된 경기가 없습니다.")
    st.stop()

# 팀 시즌 스탯 (캐시)
df_team = load_team_stats(season)

# 피처 생성 (라벨 없음)
Xfeat, merged = build_features(df_games, df_team, df_recent)
if Xfeat.drop(columns=["gamePk"], errors="ignore").isna().any().any():
    st.warning("결측치가 있어 일부 경기를 예측에서 제외할 수 있습니다.")

model, scaler, cols = load_model()
# 스케줄 기준으로 정렬
X = Xfeat.set_index("gamePk")[cols].astype(np.float32)
X = X.dropna(axis=0)
if X.empty:
    st.error("예측에 사용할 유효 피처가 없습니다. (팀 스탯/최근폼 결합 실패)")
    st.stop()

Xs = scaler.transform(X.values)
with torch.no_grad():
    proba = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()

# 결과 병합 (점수 포함)
out = merged.set_index("gamePk").loc[X.index][
    ["date","home_name","away_name","status","home_score","away_score"]
].copy()

# 한글 팀명 컬럼 추가
out["home_name_ko"] = out["home_name"].map(ko)
out["away_name_ko"] = out["away_name"].map(ko)

# 예측 확률/클래스/승자(영문+한글)
out["P(home win)"] = proba
out["pred_home_win"] = (out["P(home win)"] >= 0.5).astype(int)
out["predicted_winner_en"] = np.where(out["pred_home_win"] == 1, out["home_name"], out["away_name"])
out["predicted_winner_ko"] = np.where(out["pred_home_win"] == 1, out["home_name_ko"], out["away_name_ko"])

# 과거 경기 채점
is_final = out["status"].str.lower().str.contains("final")
out.loc[is_final, "actual_home_win"] = (out.loc[is_final, "home_score"] > out.loc[is_final, "away_score"]).astype(int)
out["correct"] = None
out.loc[is_final, "correct"] = (out.loc[is_final, "pred_home_win"] == out.loc[is_final, "actual_home_win"]).astype(int)

# 정렬
out = out.sort_values("P(home win)", ascending=False).reset_index(drop=True)

# ---- 언어별 표시 컬럼 구성 (gamePk, date 제거) ----
if lang == "한국어":
    display_df = out.rename(columns={
        "home_name_ko": "홈팀",
        "away_name_ko": "원정팀",
        "status": "상태",
        "P(home win)": "홈 승 확률",
        "predicted_winner_ko": "예상 승자",
        "correct": "정오",
        "home_score": "홈 점수",
        "away_score": "원정 점수",
    })[["홈팀","원정팀","상태","홈 승 확률","예상 승자","정오","홈 점수","원정 점수"]]
else:
    display_df = out.rename(columns={
        "home_name": "Home",
        "away_name": "Away",
        "status": "Status",
        "P(home win)": "P(Home win)",
        "predicted_winner_en": "Predicted Winner",
        "correct": "Correct",
        "home_score": "Home Score",
        "away_score": "Away Score",
    })[["Home","Away","Status","P(Home win)","Predicted Winner","Correct","Home Score","Away Score"]]

st.subheader(f"예측 결과 — {target_date}")

# Final 경기 정확도 메트릭
is_final_mask = out["status"].str.lower().str.contains("final")
final_rows = out.loc[is_final_mask]
if not final_rows.empty:
    acc = final_rows["correct"].mean()
    st.metric("Final 경기 기준 정확도", f"{acc*100:.1f}% ({int(final_rows['correct'].sum())}/{len(final_rows)})")

# ---- 색상 하이라이트 ----
def row_style(r):
    base = [""] * len(display_df.columns)
    idx = display_df.columns.get_loc
    color_home = "background-color: rgba(59,130,246,0.12);"   # 홈 승 예측 = 파랑
    color_away = "background-color: rgba(245,158,11,0.12);"   # 원정 승 예측 = 주황
    color_right = "background-color: rgba(16,185,129,0.18);"  # 정답 = 초록
    color_wrong = "background-color: rgba(239,68,68,0.18);"   # 오답 = 빨강

    # 어떤 컬럼명이 쓰였는지 언어에 따라 선택
    prob_col = "홈 승 확률" if lang == "한국어" else "P(Home win)"
    correct_col = "정오" if lang == "한국어" else "Correct"

    # out의 pred_home_win을 그대로 활용
    pred_home_win_val = out.loc[r.name, "pred_home_win"] if r.name < len(out) else 0
    if int(pred_home_win_val) == 1:
        base = [color_home] * len(base)
    else:
        base = [color_away] * len(base)

    # 확률 칸 강조
    if prob_col in display_df.columns:
        base[idx(prob_col)] = "background-color: rgba(0,0,0,0.06); font-weight:600;"

    # 채점 색
    if correct_col in display_df.columns and pd.notna(r.get(correct_col, np.nan)):
        base[idx(correct_col)] = color_right if int(r[correct_col]) == 1 else color_wrong

    return base

styled = display_df.style.format({
    "홈 승 확률": "{:.3f}",
    "P(Home win)": "{:.3f}",
}).apply(row_style, axis=1)

st.dataframe(styled, use_container_width=True)
