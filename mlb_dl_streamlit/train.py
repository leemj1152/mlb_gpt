from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from joblib import dump
from data_fetch import fetch_schedule, fetch_team_season_stats, recent_form
from features import build_features

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--models_dir", type=str, default="models")
    args = ap.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print(f"[Data] Fetching schedule {args.start} ~ {args.end}")
    df_games = fetch_schedule(args.start, args.end)
    print(f"[Data] games: {len(df_games)}")

    print(f"[Data] Fetching team season stats for {args.season}")
    df_team = fetch_team_season_stats(args.season)

    print("[Data] Computing recent form")
    df_recent = recent_form(df_games, n=10)

    print("[Feature] Building features")
    Xy, merged = build_features(df_games, df_team, df_recent)
    Xy = Xy.dropna(axis=0)  # 결측치 제거
    if "label" not in Xy.columns:
        raise SystemExit("No completed games in given range to train on (label missing). Choose a past range.")

    y = Xy["label"].values.astype(np.float32)
    X = Xy.drop(columns=["gamePk","label"]).values.astype(np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # torch tensor
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32)

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

    # save
    torch.save(model.state_dict(), os.path.join(args.models_dir, "model.pt"))
    dump(scaler, os.path.join(args.models_dir, "scaler.joblib"))
    cols = Xy.drop(columns=["gamePk","label"]).columns.tolist()
    with open(os.path.join(args.models_dir, "feature_cols.json"), "w") as f:
        import json; json.dump(cols, f, ensure_ascii=False, indent=2)
    print("[Save] models/model.pt, models/scaler.joblib, models/feature_cols.json")

if __name__ == "__main__":
    main()
