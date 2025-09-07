# train.py
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import dump

from data_fetch import (
    fetch_schedule,
    fetch_team_season_stats,
    recent_form,
    enrich_with_probable_pitchers,
)
from features import build_features

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--no_pitchers", action="store_true", help="Skip probable-pitcher enrichment")
    args = ap.parse_args()

    print(f"[Data] Fetching schedule {args.start} ~ {args.end}")
    df_games = fetch_schedule(args.start, args.end)
    print(f"[Data] games: {len(df_games)}")

    if args.no_pitchers:
        print("[Data] Skipping probable pitchers enrichment (--no_pitchers)")
    else:
        print("[Data] Enrich games with probable pitchers (season stats/handedness)")
        df_games = enrich_with_probable_pitchers(df_games, args.season)

    print(f"[Data] Fetching team season stats for {args.season}")
    df_team = fetch_team_season_stats(args.season)

    print("[Data] Computing recent form")
    df_recent = recent_form(df_games, n=10)

    print("[Feature] Building features")
    Xy, merged = build_features(df_games, df_team, df_recent)

    # 결측/무한대 정리 + 거의 비어있는 컬럼 제거 + 중앙값 임퓨트
    Xy = Xy.replace([np.inf, -np.inf], np.nan)
    if "label" not in Xy.columns:
        raise SystemExit("No completed games in given range to train on (label missing). Choose a past range.")
    y = Xy["label"].values.astype(np.float32)

    feature_cols = [c for c in Xy.columns if c not in ("gamePk","label")]
    X = Xy[feature_cols].copy()

    na_ratio = X.isna().mean()
    drop_cols = na_ratio[na_ratio > 0.98].index.tolist()
    if drop_cols:
        print(f"[Clean] Dropping near-empty columns: {drop_cols}")
        X = X.drop(columns=drop_cols)
        feature_cols = [c for c in feature_cols if c not in drop_cols]

    X = X.fillna(X.median(numeric_only=True))
    if len(X) == 0:
        raise SystemExit("All rows became invalid after cleaning. Check your date range or feature merges.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values.astype(np.float32))

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

    # 간단 학습(전체 데이터, 10 epoch)
    model = MLP(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    print("[Train] Start training (10 epochs)")
    model.train()
    for epoch in range(10):
        opt.zero_grad()
        pred = model(Xt)
        loss = bce(pred, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            auc = roc_auc_score(yt.numpy(), pred.numpy())
        print(f"epoch {epoch+1:02d}: loss={loss.item():.4f}, auc={auc:.4f}")

    # 저장
    save_dir = os.path.abspath(args.models_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_path  = os.path.join(save_dir, "model.pt")
    scaler_path = os.path.join(save_dir, "scaler.joblib")
    cols_path   = os.path.join(save_dir, "feature_cols.json")

    torch.save(model.state_dict(), model_path)
    dump(scaler, scaler_path)
    with open(cols_path, "w") as f:
        import json; json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print("[Save]")
    print("  model :", model_path)
    print("  scaler:", scaler_path)
    print("  cols  :", cols_path)

if __name__ == "__main__":
    main()
