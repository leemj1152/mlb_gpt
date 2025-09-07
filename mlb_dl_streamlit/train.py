
# Rolling-learning train script (replaces previous train.py)
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from joblib import dump
from data_fetch import fetch_schedule
from rolling_features import build_training_set_rolling

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
    ap.add_argument("--season_start", type=str, default="2024-03-20")
    ap.add_argument("--train_end", type=str, required=True, help="YYYY-MM-DD inclusive (use games <= this date)")
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--lookback", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    print(f"[Data] Fetch season schedule {args.season_start} ~ {args.train_end}")
    df_games = fetch_schedule(args.season_start, args.train_end)
    print(f"[Data] games fetched: {len(df_games)}")

    print(f"[Feature] Build leak-free rolling training set up to {args.train_end}")
    X, y, merged = build_training_set_rolling(df_games, args.train_end, lookback=args.lookback)
    if X.empty:
        raise SystemExit("No training rows produced. Check date range.")
    # drop NA rows
    mask = X.drop(columns=["gamePk"]).notna().all(axis=1)
    X = X.loc[mask]
    y = y.loc[mask.values]

    Xmat = X.drop(columns=["gamePk"]).values.astype(np.float32)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xmat)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y.values.reshape(-1,1), dtype=torch.float32)

    model = MLP(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    print("[Train] 12 epochs")
    model.train()
    for epoch in range(12):
        opt.zero_grad()
        pred = model(Xt)
        loss = bce(pred, yt)
        loss.backward()
        opt.step()
        with torch.no_grad():
            auc = roc_auc_score(yt.numpy(), pred.numpy())
            ll = log_loss(yt.numpy(), pred.numpy(), labels=[0,1])
            bs = brier_score_loss(yt.numpy().ravel(), pred.numpy().ravel())
        print(f"epoch {epoch+1:02d}: loss={loss.item():.4f} auc={auc:.4f} logloss={ll:.4f} brier={bs:.4f}")

    # save
    torch.save(model.state_dict(), os.path.join(args.models_dir, "model_roll.pt"))
    dump(scaler, os.path.join(args.models_dir, "scaler_roll.joblib"))
    feat_cols = [c for c in X.columns if c != "gamePk"]
    with open(os.path.join(args.models_dir, "feature_cols_roll.json"), "w", encoding="utf-8") as f:
        import json; json.dump(feat_cols, f, ensure_ascii=False, indent=2)
    print("[Save] models/model_roll.pt, models/scaler_roll.joblib, models/feature_cols_roll.json")

if __name__ == "__main__":
    main()
