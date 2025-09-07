# compare_models.py
# 두 모델(이전 MLP + rolling_features_1 / 현재 LGBM + rolling_features)을
# 동일 기간에 대해 평가하고 지표 및 그래프로 비교

from __future__ import annotations
import argparse, os, json
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, accuracy_score
)
from joblib import load

# 프로젝트 내부 모듈
from data_fetch import fetch_schedule
import rolling_features as rf_new          # 현재 파이프라인
import rolling_features_1 as rf_old        # 이전(MLP) 파이프라인으로 네이밍했다고 가정

# -----------------------------
# 설정 & 유틸
# -----------------------------

@dataclass
class ModelArtifacts:
    name: str
    model: object
    scaler: object
    feat_cols: List[str]
    threshold: float | None  # 분류 임계값(없으면 0.5)

def _bool_final(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.contains("final")

def _safe_logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    # logloss는 0/1 확률 방지를 위해 clip
    p_clip = np.clip(p, 1e-6, 1-1e-6)
    return log_loss(y_true, p_clip, labels=[0,1])

def _cal_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    ll = _safe_logloss(y_true, p)
    auc = roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan
    br = brier_score_loss(y_true, p)
    pred = (p >= thr).astype(int)
    acc = accuracy_score(y_true, pred)
    return {"logloss": ll, "auc": auc, "brier": br, "accuracy": acc}

def _ensure_float_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

# -----------------------------
# 아티팩트 로더
# -----------------------------

def load_artifacts_mlp(models_dir: str = "models") -> ModelArtifacts:
    """
    이전 모델(Multi-layer Perceptron) 아티팩트 로드
    - model.pt / scaler.joblib / feature_cols.json
    - 분류 임계값은 0.5 고정
    """
    import torch
    from torch import nn

    # MLP 구조는 app_mlp.py에서 사용하던 것과 동일해야 함
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

    state = torch.load(os.path.join(models_dir, "model.pt"), map_location="cpu")
    scaler = load(os.path.join(models_dir, "scaler.joblib"))
    with open(os.path.join(models_dir, "feature_cols.json"), "r", encoding="utf-8") as f:
        cols = json.load(f)

    model = MLP(in_dim=len(cols))
    model.load_state_dict(state)
    model.eval()
    return ModelArtifacts(name="MLP(old)", model=model, scaler=scaler, feat_cols=cols, threshold=0.5)

def load_artifacts_lgbm(models_dir: str = "models") -> ModelArtifacts:
    """
    현재 모델(LightGBM) 아티팩트 로드
    - model_lgbm_calibrated.joblib / scaler_lgbm.joblib / feature_cols_lgbm.json / threshold_lgbm.json
    """
    model = load(os.path.join(models_dir, "model_lgbm_calibrated.joblib"))
    scaler = load(os.path.join(models_dir, "scaler_lgbm.joblib"))
    with open(os.path.join(models_dir, "feature_cols_lgbm.json"), "r", encoding="utf-8") as f:
        cols = json.load(f)
    thr_path = os.path.join(models_dir, "threshold_lgbm.json")
    if os.path.exists(thr_path):
        with open(thr_path, "r", encoding="utf-8") as f:
            thr = float(json.load(f).get("threshold", 0.5))
    else:
        thr = 0.5
    return ModelArtifacts(name="LightGBM(new)", model=model, scaler=scaler, feat_cols=cols, threshold=thr)

# -----------------------------
# 피처 생성 & 예측 파이프라인
# -----------------------------

def build_features_for_range(
    feature_builder_module,
    df_hist: pd.DataFrame,
    start_date: str,
    end_date: str,
    lookback: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    지정한 기간 [start_date, end_date]의 게임에 대해
    feature_builder_module.build_game_features_from_history()로
    피처(Xfeat)와 메타(merged) 생성

    feature_builder_module는 rolling_features 또는 rolling_features_1 (이전) 모듈
    """
    # 빌더는 target_date의 경기만 생성하므로, 날짜를 하루씩 순회해서 쌓는다.
    dates = pd.date_range(start_date, end_date, freq="D")
    X_list, M_list = [], []
    for d in dates:
        Xd, Md = feature_builder_module.build_game_features_from_history(df_hist, str(d.date()), lookback=lookback)
        if not Xd.empty:
            X_list.append(Xd)
            M_list.append(Md)
    if len(X_list) == 0:
        return pd.DataFrame(), pd.DataFrame()
    Xfeat = pd.concat(X_list, ignore_index=True)
    merged = pd.concat(M_list, ignore_index=True)
    return Xfeat, merged

def predict_with_mlp(art: ModelArtifacts, X: pd.DataFrame) -> np.ndarray:
    import torch
    Xs = art.scaler.transform(X.values.astype(np.float32))
    with torch.no_grad():
        p = art.model(torch.tensor(Xs, dtype=torch.float32)).numpy().ravel()
    return p

def predict_with_sklearn(art: ModelArtifacts, X: pd.DataFrame) -> np.ndarray:
    Xs = art.scaler.transform(X.values.astype(np.float32))
    p = art.model.predict_proba(Xs)[:, 1]
    return p

# -----------------------------
# 메인 비교 루틴
# -----------------------------

def compare_models(
    season_start: str,
    eval_start: str,
    eval_end: str,
    lookback: int,
    models_dir: str = "models",
    out_dir: str = "compare_out"
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 히스토리 로드
    print(f"[Data] fetch schedule {season_start} ~ {eval_end}")
    df_hist = fetch_schedule(season_start, eval_end)
    if df_hist.empty:
        raise SystemExit("No games in the given range.")

    # 2) 이전 MLP용 피처
    print("[Feature][OLD] build features with rolling_features_1")
    X_old, M_old = build_features_for_range(rf_old, df_hist, eval_start, eval_end, lookback)
    # 3) 현재 LGBM용 피처
    print("[Feature][NEW] build features with rolling_features")
    X_new, M_new = build_features_for_range(rf_new, df_hist, eval_start, eval_end, lookback)

    # 공통 필터: NaN 제거 및 feature alignment
    results = []

    # ---- OLD (MLP) ----
    if not X_old.empty:
        art_old = load_artifacts_mlp(models_dir)
        # 정렬/정렬: feature column 맞추기
        cols_old = [c for c in art_old.feat_cols if c in X_old.columns]
        Xo = X_old.set_index("gamePk")[cols_old].copy()
        Xo = _ensure_float_df(Xo, cols_old).dropna(axis=0)
        if not Xo.empty:
            p_old = predict_with_mlp(art_old, Xo)
            # 라벨 및 메타
            Mo = M_old.set_index("gamePk").loc[Xo.index]
            is_final = _bool_final(Mo["status"])
            y_old = (Mo.loc[is_final, "home_score"] > Mo.loc[is_final, "away_score"]).astype(int)
            p_old_final = p_old[is_final.values]

            if len(y_old) > 0:
                met_old = _cal_metrics(y_old.values, p_old_final, thr=art_old.threshold or 0.5)
                results.append(("MLP(old)", met_old, len(y_old)))
                # 저장
                pd.DataFrame({
                    "gamePk": Xo.index,
                    "date": Mo["date"].values,
                    "home": Mo["home_name"].values,
                    "away": Mo["away_name"].values,
                    "status": Mo["status"].values,
                    "p_homewin": p_old,
                }).to_csv(os.path.join(out_dir, "pred_old_detail.csv"), index=False)
            else:
                print("[OLD] No final games for metrics.")
        else:
            print("[OLD] No valid rows after NA drop.")
    else:
        print("[OLD] Empty X_old features.")

    # ---- NEW (LGBM) ----
    if not X_new.empty:
        art_new = load_artifacts_lgbm(models_dir)
        cols_new = [c for c in art_new.feat_cols if c in X_new.columns]
        Xn = X_new.set_index("gamePk")[cols_new].copy()
        Xn = _ensure_float_df(Xn, cols_new).dropna(axis=0)
        if not Xn.empty:
            p_new = predict_with_sklearn(art_new, Xn)
            Mn = M_new.set_index("gamePk").loc[Xn.index]
            is_final = _bool_final(Mn["status"])
            y_new = (Mn.loc[is_final, "home_score"] > Mn.loc[is_final, "away_score"]).astype(int)
            p_new_final = p_new[is_final.values]

            if len(y_new) > 0:
                met_new = _cal_metrics(y_new.values, p_new_final, thr=art_new.threshold or 0.5)
                results.append(("LightGBM(new)", met_new, len(y_new)))
                pd.DataFrame({
                    "gamePk": Xn.index,
                    "date": Mn["date"].values,
                    "home": Mn["home_name"].values,
                    "away": Mn["away_name"].values,
                    "status": Mn["status"].values,
                    "p_homewin": p_new,
                }).to_csv(os.path.join(out_dir, "pred_new_detail.csv"), index=False)
            else:
                print("[NEW] No final games for metrics.")
        else:
            print("[NEW] No valid rows after NA drop.")
    else:
        print("[NEW] Empty X_new features.")

    if not results:
        raise SystemExit("No comparable results.")

    # 4) 결과 표 & 저장
    rows = []
    for name, met, n in results:
        rows.append({
            "model": name,
            "final_games": n,
            "logloss": met["logloss"],
            "auc": met["auc"],
            "brier": met["brier"],
            "accuracy": met["accuracy"],
        })
    df_res = pd.DataFrame(rows)
    print("\n=== Metrics (Final games only) ===")
    print(df_res)
    df_res.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)

    # 5) 시각화
    # Bar chart: accuracy / auc (↑ good), logloss / brier(↓ good)
    def _bar(ax, metric_key, title, invert=False):
        xs = np.arange(len(df_res))
        vals = df_res[metric_key].values.astype(float)
        ax.bar(xs, vals)
        ax.set_xticks(xs)
        ax.set_xticklabels(df_res["model"].values, rotation=0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        if invert:
            # 낮을수록 좋은 지표는 막대 위에 숫자만
            pass
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    _bar(axes[0,0], "accuracy", "Accuracy (↑ good)")
    _bar(axes[0,1], "auc", "ROC-AUC (↑ good)")
    _bar(axes[1,0], "logloss", "LogLoss (↓ good)")
    _bar(axes[1,1], "brier", "Brier Score (↓ good)")
    fig.suptitle(f"Model Comparison on Final Games\nRange: {eval_start} ~ {eval_end} (lookback={lookback})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(out_dir, "compare_metrics.png")
    plt.savefig(fig_path, dpi=160)
    print(f"[Save] {fig_path}")

    # 6) Calibration curve (선택) — NEW/OLD 모두 확률분포 보기
    #   히스토그램 형태만 간단히 저장
    fig2, ax2 = plt.subplots(figsize=(8,4))
    for name, file in [("old", "pred_old_detail.csv"), ("new", "pred_new_detail.csv")]:
        path = os.path.join(out_dir, file)
        if os.path.exists(path):
            arr = pd.read_csv(path)["p_homewin"].values
            ax2.hist(arr, bins=20, histtype="step", label=name, density=True)
    ax2.set_title("Predicted Probability Distribution")
    ax2.set_xlabel("P(Home win)")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2_path = os.path.join(out_dir, "pred_prob_hist.png")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=160)
    print(f"[Save] {fig2_path}")

# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season_start", type=str, default="2024-03-20",
                    help="히스토리 시작일 (피처 생성을 위해 과거부터 로드)")
    ap.add_argument("--eval_start", type=str, required=True,
                    help="평가 시작일(이 날~종료일까지의 경기만 스코어 비교)")
    ap.add_argument("--eval_end", type=str, required=True,
                    help="평가 종료일")
    ap.add_argument("--lookback", type=int, default=10)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--out_dir", type=str, default="compare_out")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    compare_models(
        season_start=args.season_start,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        lookback=args.lookback,
        models_dir=args.models_dir,
        out_dir=args.out_dir,
    )
