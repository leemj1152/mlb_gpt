import os, json
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

# ---------- 팀명 한글 ----------
TEAM_KO = {
    "Arizona Diamondbacks":"애리조나","Atlanta Braves":"애틀랜타","Baltimore Orioles":"볼티모어","Boston Red Sox":"보스턴",
    "Chicago Cubs":"시카고C","Chicago White Sox":"시카고W","Cincinnati Reds":"신시내티 레즈","Cleveland Guardians":"클리블랜드",
    "Colorado Rockies":"콜로라도","Detroit Tigers":"디트로이트","Houston Astros":"휴스턴","Kansas City Royals":"캔자스시티",
    "Los Angeles Angels":"LA에인절스","Los Angeles Dodgers":"LA다저스","Miami Marlins":"마이애미","Milwaukee Brewers":"밀워키",
    "Minnesota Twins":"미네소타","New York Mets":"뉴욕 메츠","New York Yankees":"뉴욕 양키스","Athletics":"오클랜드",
    "Philadelphia Phillies":"필라델피아","Pittsburgh Pirates":"피츠버그","San Diego Padres":"샌디에이고","San Francisco Giants":"샌프란시스코",
    "Seattle Mariners":"시애틀","St. Louis Cardinals":"세인트루이스","Tampa Bay Rays":"탬파베이","Texas Rangers":"텍사스",
    "Toronto Blue Jays":"토론토","Washington Nationals":"워싱턴",
}
def ko(name: str) -> str: return TEAM_KO.get(name, name)

# ---------- 모델 ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

@st.cache_data(show_spinner=True)
def load_team_stats(season: int):
    return fetch_team_season_stats(season)

@st.cache_resource(show_spinner=True)
def load_model_bundle():
    model_path  = os.path.abspath("models/model.pt")
    scaler_path = os.path.abspath("models/scaler.joblib")
    cols_path   = os.path.abspath("models/feature_cols.json")
    thr_path    = os.path.abspath("models/threshold.json")
    calib_path  = os.path.abspath("models/calibrator.joblib")
    meta_path   = os.path.abspath("models/meta.json")

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

    thr = 0.5
    if os.path.exists(thr_path):
        try:
            with open(thr_path, "r") as f:
                thr = float(json.load(f).get("threshold", 0.5))
        except Exception:
            pass

    calibrator = None
    if os.path.exists(calib_path):
        try:
            calibrator = load(calib_path)  # IsotonicRegression 또는 LogisticRegression
        except Exception:
            calibrator = None

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return model, scaler, cols, calibrator, thr, meta

# ---------- 보정 안전장치 ----------
def _apply_calibrator(raw: np.ndarray, calibrator):
    """
    보정 결과가 (1) 표준편차가 너무 작거나 (2) 유니크 값이 너무 적거나 (3) 양끝단 클리핑이 심하면
    보정을 '붕괴'로 간주하고 원시 확률(raw)을 그대로 사용한다.
    """
    raw = np.clip(raw, 0.0, 1.0)
    if calibrator is None or raw.size == 0:
        return raw, False, "no-calibrator"

    try:
        if hasattr(calibrator, "transform"):
            cal = calibrator.transform(raw)
        elif hasattr(calibrator, "predict_proba"):
            cal = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            return raw, False, "unknown-calibrator"
    except Exception:
        return raw, False, "transform-error"

    cal = np.clip(cal, 0.0, 1.0)

    # ---- 붕괴 기준(강화) ----
    std_raw = float(np.std(raw))
    uniq_raw = int(np.unique(np.round(raw, 3)).size)

    std_out = float(np.std(cal))
    uniq_out = int(np.unique(np.round(cal, 3)).size)
    minv, maxv = cal.min(), cal.max()
    p_min = float(np.mean(np.isclose(cal, minv)))
    p_max = float(np.mean(np.isclose(cal, maxv)))
    clip = max(p_min, p_max)

    # ① 절대 기준 ② raw대비 과도 압축
    collapsed = (
        std_out < 0.01 or uniq_out < 10 or clip > 0.6 or
        (std_raw > 0 and std_out < std_raw * 0.25)
    )
    if collapsed:
        return raw, False, f"collapsed(std={std_out:.5f}, uniq={uniq_out}, clip={clip:.2f})"

    return cal, True, f"ok(std={std_out:.3f}, uniq={uniq_out})"

# ---------- 피처→입력 ----------
def make_inputs(df_games, df_team, df_recent, cols, scaler):
    Xfeat, merged = build_features(df_games, df_team, df_recent)
    X = Xfeat.set_index("gamePk")[cols].astype(np.float32)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
    X = X.dropna(axis=0)
    if X.empty:
        return merged.iloc[0:0].copy(), None
    Xs = scaler.transform(X.values)
    merged = merged.set_index("gamePk").loc[X.index]
    return merged, Xs

# ---------- 페이지 ----------
st.set_page_config(page_title="MLB Predictor", layout="wide")
st.title("⚾ MLB 홈팀 승리 확률 예측 (Demo)")
st.caption("python-mlb-statsapi + PyTorch + Streamlit")

# ===== 사이드바 =====
with st.sidebar:
    target_date = st.date_input("Target Date", value=date.today())
    default_season = target_date.year
    season = st.number_input("Season", min_value=2015, max_value=2100, value=default_season, step=1)
    st.write("먼저 `train.py`로 해당 시즌 과거 구간 학습을 완료해야 예측이 가능합니다.")

    td = pd.Timestamp(target_date)
    start_recent = (td - pd.Timedelta(days=30)).date()
    end_recent   = (td - pd.Timedelta(days=1)).date()
    df_recent_source = fetch_schedule(str(start_recent), str(end_recent))
    df_recent = recent_form(df_recent_source, n=10)

    lang = st.radio("표시 언어", ["한국어", "English"], horizontal=True, index=0)

# ===== 모델 로드 & 메타 표시 =====
model, scaler, cols, calibrator, thr, meta = load_model_bundle()
st.caption(
    f"📦 model meta | season={meta.get('season')} | train={meta.get('train_start')}~{meta.get('train_end')} "
    f"| n_feat={meta.get('n_features')} | calibrator={meta.get('calibrator')} | thr={meta.get('threshold')}"
)

# ===== 오늘 스케줄 예측 =====
df_games = fetch_schedule(str(target_date), str(target_date))
if df_games.empty:
    st.info("해당 날짜에 예정된 경기가 없습니다.")
else:
    df_games = enrich_with_probable_pitchers(df_games, season)

    if {"home_prob_pitcher_id","away_prob_pitcher_id"}.issubset(df_games.columns):
        filled = ((~df_games["home_prob_pitcher_id"].isna())
                  & (~df_games["away_prob_pitcher_id"].isna())).sum()
        st.caption(f"🧪 선발투수 id 확보: {filled}/{len(df_games)} 경기")

    df_team = load_team_stats(season)

    merged, Xs = make_inputs(df_games, df_team, df_recent, cols, scaler)
    if Xs is None:
        st.error("예측에 사용할 유효 피처가 없습니다."); st.stop()

    with torch.no_grad():
        raw_proba = model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()

    proba, used_calib, reason = _apply_calibrator(raw_proba, calibrator)
    thr_eff = thr  # ✅ 항상 저장된 threshold 사용

    # 디버그
    st.caption(
        f"🔍 확률분포 | raw std={np.std(raw_proba):.3f}, uniq≈{np.unique(np.round(raw_proba,3)).size} "
        f"→ out std={np.std(proba):.3f}, uniq≈{np.unique(np.round(proba,3)).size} | "
        f"calib={'ON' if used_calib else 'OFF'} ({reason}) | thr={thr_eff:.3f}"
    )
    if not used_calib and calibrator is not None:
        st.caption("⚠️ 보정이 단차/클리핑으로 비활성화되었습니다. 원시 확률과 저장된 임계값으로 판정합니다.")

    out = merged.copy()
    out["home_name_ko"] = out["home_name"].map(ko)
    out["away_name_ko"] = out["away_name"].map(ko)
    out["P(home win)"] = np.clip(proba, 0.0, 1.0)
    out["pred_home_win"] = (out["P(home win)"] >= thr_eff).astype(int)
    out["predicted_winner_en"] = np.where(out["pred_home_win"]==1, out["home_name"], out["away_name"])
    out["predicted_winner_ko"] = np.where(out["pred_home_win"]==1, out["home_name_ko"], out["away_name_ko"])

    out["status"] = out["status"].fillna("")
    is_final = out["status"].astype(str).str.lower().str.contains("final")
    out.loc[is_final, "actual_home_win"] = (out.loc[is_final, "home_score"] > out.loc[is_final, "away_score"]).astype(int)
    out["correct"] = None
    out.loc[is_final, "correct"] = (out.loc[is_final, "pred_home_win"] == out.loc[is_final, "actual_home_win"]).astype(int)

    out = out.sort_values("P(home win)", ascending=False).reset_index(drop=True)
    is_final = out["status"].astype(str).str.lower().str.contains("final")

    # 표시 컬럼(요청 순서)
    if lang == "한국어":
        display_df = out.rename(columns={
            "home_name_ko":"홈팀","away_name_ko":"원정팀",
            "home_prob_pitcher_name":"홈팀 선발","away_prob_pitcher_name":"원정팀 선발",
            "P(home win)":"홈승 확률","predicted_winner_ko":"예상 승자",
            "home_score":"홈 점수","away_score":"원정 점수","status":"상태",
        })[["홈팀","원정팀","홈팀 선발","원정팀 선발","홈승 확률","예상 승자","홈 점수","원정 점수","상태"]]
    else:
        display_df = out.rename(columns={
            "home_name":"Home","away_name":"Away",
            "home_prob_pitcher_name":"Home SP","away_prob_pitcher_name":"Away SP",
            "P(home win)":"P(Home win)","predicted_winner_en":"Predicted Winner",
            "home_score":"Home Score","away_score":"Away Score","status":"Status",
        })[["Home","Away","Home SP","Away SP","P(Home win)","Predicted Winner","Home Score","Away Score","Status"]]

    st.subheader(f"예측 결과 — {target_date}")

    done = out.loc[is_final]
    if not done.empty and "correct" in done.columns:
        correct_cnt = int(done["correct"].sum())
        total_done = int(len(done))
        acc = (correct_cnt / total_done) * 100 if total_done > 0 else 0.0
        c1, c2, c3 = st.columns([1, 1, 1.2])
        c1.metric("완료 경기 수", f"{total_done}")
        c2.metric("적중 경기 수", f"{correct_cnt}")
        c3.metric("적중률", f"{acc:.1f}%")
    else:
        st.caption("오늘은 아직 완료된 경기가 없어요.")

    num_cols = [c for c in ["홈승 확률","P(Home win)","홈 점수","원정 점수","Home Score","Away Score"] if c in display_df.columns]
    display_df = display_df.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan})
    for c in num_cols: display_df[c] = pd.to_numeric(display_df[c], errors="coerce")

    def row_style(r):
        base = [""] * len(display_df.columns)
        if r.name >= len(out): return base
        if is_final.iloc[r.name]:
            corr = out.loc[r.name, "correct"]
            if pd.notna(corr) and int(corr) == 1:
                base = ["background-color: rgba(59,130,246,0.18);"] * len(base)
            else:
                base = ["background-color: rgba(239,68,68,0.18);"] * len(base)
        return base

    fmt = {}
    if "홈승 확률" in display_df.columns: fmt["홈승 확률"] = "{:.3f}"
    if "P(Home win)" in display_df.columns: fmt["P(Home win)"] = "{:.3f}"

    styled = display_df.style.format(fmt).apply(row_style, axis=1)
    st.dataframe(styled, width="stretch")

# ===== 9월 성능 검사 (학습은 8월까지) =====
st.markdown("---")
st.subheader("🧪 9월 성능 검사 (학습 기간: 8월까지)")

colA, colB = st.columns([1, 2])
with colA:
    run_eval = st.button("9월 성능 검사 실행", type="primary")
with colB:
    st.caption("모델은 **해당 시즌 8월 말까지** 학습했다고 가정합니다. 9월 1일 ~ 오늘/9월 말의 **완료 경기**만 집계합니다.")

if run_eval:
    today = date.today()
    season_eval = meta.get("season", today.year)
    start_eval = date(season_eval, 9, 1)
    end_eval = min(today, date(season_eval, 9, 30))

    df_eval = fetch_schedule(str(start_eval), str(end_eval))
    if df_eval.empty:
        st.info("해당 구간의 경기 데이터가 없습니다."); st.stop()
    df_eval = enrich_with_probable_pitchers(df_eval, season_eval)
    df_team_eval = load_team_stats(season_eval)

    start_recent2 = (pd.Timestamp(end_eval) - pd.Timedelta(days=30)).date()
    end_recent2   = (pd.Timestamp(end_eval) - pd.Timedelta(days=1)).date()
    df_recent_src2 = fetch_schedule(str(start_recent2), str(end_recent2))
    df_recent2 = recent_form(df_recent_src2, n=10)

    mergedE, XsE = make_inputs(df_eval, df_team_eval, df_recent2, cols, scaler)
    if XsE is None:
        st.error("평가 구간에서 유효 피처가 없어 성능을 계산할 수 없습니다."); st.stop()

    with torch.no_grad():
        rawE = model(torch.tensor(XsE, dtype=torch.float32)).numpy().ravel()

    probaE, used_calibE, reasonE = _apply_calibrator(rawE, calibrator)
    thr_effE = thr  # ✅ 항상 저장된 threshold 사용

    st.caption(
        f"🔍 (평가) 확률분포 | raw std={np.std(rawE):.3f}, uniq≈{np.unique(np.round(rawE,3)).size} "
        f"→ out std={np.std(probaE):.3f}, uniq≈{np.unique(np.round(probaE,3)).size} | "
        f"calib={'ON' if used_calibE else 'OFF'} ({reasonE}) | thr={thr_effE:.3f}"
    )

    outE = mergedE.copy()
    outE["P(home win)"] = np.clip(probaE, 0.0, 1.0)
    outE["pred_home_win"] = (outE["P(home win)"] >= thr_effE).astype(int)
    outE["status"] = outE["status"].fillna("")
    is_final_E = outE["status"].astype(str).str.lower().str.contains("final")
    doneE = outE.loc[is_final_E].copy()
    if doneE.empty:
        st.info("9월 구간에서 아직 완료된 경기가 없습니다."); st.stop()
    doneE["actual_home_win"] = (doneE["home_score"] > doneE["away_score"]).astype(int)
    doneE["correct"] = (doneE["pred_home_win"] == doneE["actual_home_win"]).astype(int)

    total_done = int(len(doneE))
    correct_cnt = int(doneE["correct"].sum())
    acc = (correct_cnt / total_done) * 100 if total_done > 0 else 0.0

    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.metric("9월 완료 경기 수", f"{total_done}")
    c2.metric("적중 경기 수", f"{correct_cnt}")
    c3.metric("9월 적중률", f"{acc:.1f}%")

    with st.expander("9월 평가 표 (일부 열) 보기"):
        tmp = doneE[["date","home_name","away_name","home_score","away_score","status","P(home win)","pred_home_win","actual_home_win","correct"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
        tmp = tmp.sort_values("date")
        tmp["정오"] = tmp["correct"].map({1:"✅",0:"❌"})
        tmp = tmp.drop(columns=["correct"])
        st.dataframe(tmp, width="stretch")
