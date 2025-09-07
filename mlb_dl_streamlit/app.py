# Rolling, leak-free inference app with LightGBM calibrated model
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json
from datetime import date
from data_fetch import fetch_schedule
from rolling_features import build_game_features_from_history

# ===================== UI / Names =====================
TEAM_KO = {
    "Arizona Diamondbacks": "애리조나",
    "Atlanta Braves": "애틀랜타",
    "Baltimore Orioles": "볼티모어",
    "Boston Red Sox": "보스턴",
    "Chicago Cubs": "시카고C",
    "Chicago White Sox": "시카고W",
    "Cincinnati Reds": "신시내티",
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
    "Oakland Athletics": "오클랜드",
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
    return TEAM_KO.get(name, name)

# ===================== Model Loader =====================
@st.cache_resource(show_spinner=True)
def load_model():
    """
    LightGBM + Isotonic Calibrated model & scaler & feature cols
    Expect files:
      - models/model_lgbm_calibrated.joblib
      - models/scaler_lgbm.joblib
      - models/feature_cols_lgbm.json
    """
    model = load("models/model_lgbm_calibrated.joblib")  # CalibratedClassifierCV
    scaler = load("models/scaler_lgbm.joblib")
    with open("models/feature_cols_lgbm.json", "r", encoding="utf-8") as f:
        cols = json.load(f)
    return model, scaler, cols

# ===================== Page =====================
st.set_page_config(page_title="MLB Predictor (Rolling, Leak-Free, LightGBM)", layout="wide")
st.title("⚾ MLB 승부 예측 — 롤링 피처 (누수 차단, LightGBM)")
st.caption("학습: 지정한 기준일까지 완료 경기만 / 예측: '전날까지' 히스토리로 피처 생성 / 모델: LightGBM + Isotonic Calibration")

with st.sidebar:
    target_date = st.date_input("예측 날짜", value=date.today())
    lookback = st.slider("최근 경기 반영(N)", 5, 20, 10, 1)
    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True)

# ===================== Data & Features =====================
# Season start to target_date to have history
season_start = f"{target_date.year}-03-20"
df_hist = fetch_schedule(season_start, str(target_date))
if df_hist.empty:
    st.info("해당 날짜의 경기 또는 히스토리가 없습니다.")
    st.stop()

# Features for games ON target_date using history up to target_date-1
Xfeat, merged = build_game_features_from_history(df_hist, str(target_date), lookback=lookback)
if Xfeat.empty:
    st.info("예측할 경기가 없거나 피처를 만들 수 없습니다.")
    st.stop()

model, scaler, cols = load_model()

# X 준비
X = Xfeat.set_index("gamePk")[cols].astype(np.float32).dropna(axis=0)
if X.empty:
    st.error("예측에 사용할 유효 피처가 없습니다.")
    st.stop()

Xs = scaler.transform(X.values)
proba = model.predict_proba(Xs)[:, 1]  # LightGBM calibrated probabilities

# ===================== Output Table =====================
out = merged.set_index("gamePk").loc[X.index][
    ["date", "home_name", "away_name", "status", "home_score", "away_score"]
].copy()
out["home_ko"] = out["home_name"].map(ko)
out["away_ko"] = out["away_name"].map(ko)
out["P(home win)"] = proba
out["pred_home_win"] = (out["P(home win)"] >= 0.5).astype(int)
out["predicted_winner_en"] = np.where(out["pred_home_win"] == 1, out["home_name"], out["away_name"])
out["predicted_winner_ko"] = np.where(out["pred_home_win"] == 1, out["home_ko"], out["away_ko"])

# 정렬 후(인덱스 확정) 정확도 메트릭 계산
out = out.sort_values("P(home win)", ascending=False).reset_index(drop=True)

# === Final 경기 정확도 메트릭 & 정오 컬러 하이라이트를 위한 준비 ===
is_final_mask = out["status"].astype(str).str.lower().str.contains("final")
if is_final_mask.any():
    # 실제 홈 승패 (Final만)
    actual_home_win = (out.loc[is_final_mask, "home_score"] > out.loc[is_final_mask, "away_score"]).astype(int)
    out["correct"] = np.nan
    out.loc[is_final_mask, "actual_home_win"] = actual_home_win
    out.loc[is_final_mask, "correct"] = (
        out.loc[is_final_mask, "pred_home_win"] == out.loc[is_final_mask, "actual_home_win"]
    ).astype(int)

    n_final = int(is_final_mask.sum())
    n_correct = int(np.nansum(out["correct"]))
    acc = n_correct / n_final if n_final > 0 else float("nan")
    st.metric("Final 경기 기준 정확도", f"{acc*100:.1f}% ({n_correct}/{n_final})")

# 언어에 따른 표시 테이블
if lang == "한국어":
    disp = out.rename(columns={
        "home_ko": "홈팀", "away_ko": "원정팀", "status": "상태", "P(home win)": "홈 승 확률",
        "predicted_winner_ko": "예상 승자", "home_score": "홈 점수", "away_score": "원정 점수"
    })[["홈팀", "원정팀", "상태", "홈 승 확률", "예상 승자", "홈 점수", "원정 점수"]]
else:
    disp = out.rename(columns={
        "home_name": "Home", "away_name": "Away", "status": "Status", "P(home win)": "P(Home win)",
        "predicted_winner_en": "Predicted Winner", "home_score": "Home Score", "away_score": "Away Score"
    })[["Home", "Away", "Status", "P(Home win)", "Predicted Winner", "Home Score", "Away Score"]]

# ===================== Styling (정답=파랑, 오답=빨강; 완료 경기만) =====================
def row_style(r: pd.Series):
    i = r.name  # disp와 out은 reset_index(drop=True)로 정렬 동일
    styles = [""] * len(disp.columns)

    # Final 경기만 색칠
    if is_final_mask.iloc[i]:
        corr = out.loc[i, "correct"] if "correct" in out.columns else np.nan
        if pd.notna(corr):
            bg = (
                "background-color: rgba(59,130,246,0.18);"   # 파랑: 정답
                if int(corr) == 1
                else "background-color: rgba(239,68,68,0.18);"  # 빨강: 오답
            )
            styles = [bg] * len(styles)

    # 확률 컬럼은 가독성 강조
    prob_col = "홈 승 확률" if lang == "한국어" else "P(Home win)"
    if prob_col in disp.columns:
        j = disp.columns.get_loc(prob_col)
        styles[j] = styles[j] + " font-weight:600;"

    return styles

# 내부 스크롤 제거: 테이블 높이를 동적으로 크게
row_h = 36      # 대략적 행 높이(px)
header_h = 38   # 헤더 높이(px)
max_h = 3000    # 너무 길어지는 것 방지용 상한(필요시 확장)

tbl_style = (disp.style
             .format({"홈 승 확률": "{:.3f}", "P(Home win)": "{:.3f}"})
             .apply(row_style, axis=1))

st.dataframe(
    tbl_style,
    use_container_width=True,
    height=min(header_h + row_h * len(disp), max_h)
)
