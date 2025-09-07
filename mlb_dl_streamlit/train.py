from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from joblib import dump

from data_fetch import fetch_schedule, fetch_team_season_stats, recent_form, enrich_with_probable_pitchers
from features import build_features

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x)

def _collapse_flags(arr: np.ndarray):
    arr = np.clip(arr, 0.0, 1.0)
    std = float(np.std(arr))
    uniq = int(np.unique(np.round(arr, 3)).size)
    minv, maxv = arr.min(), arr.max()
    p_min = float(np.mean(np.isclose(arr, minv)))
    p_max = float(np.mean(np.isclose(arr, maxv)))
    clip = max(p_min, p_max)
    collapsed = (std < 1e-4) or (uniq < 4) or (clip > 0.80)
    return collapsed, std, uniq, clip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--no_pitchers", action="store_true")
    ap.add_argument("--pp_basic_only", action="store_true", help="선발 피처 중 ERA/WHIP만 사용")
    ap.add_argument("--recent_n", type=int, default=10, help="recent_form 윈도우 크기")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--calibration", type=str, default="auto",
                    choices=["auto","isotonic","platt","none"])
    ap.add_argument("--feature_set", type=str, default="full",
                    choices=["full","baseline"])
    ap.add_argument("--force_threshold", type=float, default=None)
    args = ap.parse_args()

    print(f"[Data] Fetching schedule {args.start} ~ {args.end}")
    df_games = fetch_schedule(args.start, args.end)
    print(f"[Data] games: {len(df_games)}")

    if not args.no_pitchers:
        print("[Data] Enrich games with probable pitchers (season stats + recent form + rest)")
        df_games = enrich_with_probable_pitchers(df_games, args.season)

    print(f"[Data] Fetching team season stats for {args.season}")
    df_team = fetch_team_season_stats(args.season)

    print("[Data] Computing recent form")
    df_recent = recent_form(df_games, n=args.recent_n)

    print("[Feature] Building features")
    Xy, merged = build_features(df_games, df_team, df_recent)

    Xy = Xy.replace([np.inf, -np.inf], np.nan)
    if "label" not in Xy.columns or Xy["label"].isna().all():
        raise SystemExit("No completed games in given range to train on (label missing). Choose a past range.")
    dates = merged[["gamePk","date"]].drop_duplicates()
    Xy = Xy.merge(dates, on="gamePk", how="left")
    Xy["date"] = pd.to_datetime(Xy["date"])

    all_feature_cols = [c for c in Xy.columns if c not in ("gamePk","label","date")]
    X_all = Xy[all_feature_cols].copy()
    y_all = Xy["label"].astype(np.float32).values

    Xy_sorted = Xy.sort_values("date").reset_index(drop=True)
    n = len(Xy_sorted)
    val_len = int(max(1, round(n * args.val_ratio))) if args.val_ratio > 0 else 0
    cut = n - val_len if val_len > 0 else n

    X_all = X_all.loc[Xy_sorted.index].reset_index(drop=True)
    y_all = Xy_sorted["label"].astype(np.float32).values

    X_tr_raw = X_all.iloc[:cut].copy(); y_tr = y_all[:cut]
    X_val_raw = X_all.iloc[cut:].copy() if val_len > 0 else None
    y_val = y_all[cut:] if val_len > 0 else None

    na_ratio_tr = X_tr_raw.isna().mean()
    drop_cols = na_ratio_tr[na_ratio_tr > 0.98].index.tolist()
    if drop_cols:
        print(f"[Clean] Dropping near-empty columns (train-based): {drop_cols}")
        X_tr_raw = X_tr_raw.drop(columns=drop_cols)
        if X_val_raw is not None: X_val_raw = X_val_raw.drop(columns=drop_cols, errors="ignore")
    feature_cols = [c for c in X_tr_raw.columns]

    # baseline 최소 셋만
    if args.feature_set == "baseline":
        allow = {
            "diff_hit_ops","diff_pit_era","diff_pit_whip",
            "diff_recent_winrate","diff_recent_run_diff",
            "diff_last_rest_days","diff_b2b"
        }
        feature_cols = [c for c in feature_cols if c in allow]
        X_tr_raw = X_tr_raw[feature_cols]
        if X_val_raw is not None: X_val_raw = X_val_raw.reindex(columns=feature_cols)
        print(f"[Feature] baseline cols used ({len(feature_cols)}): {feature_cols}")

    # 선발 ERA/WHIP만 강제할 경우
    if args.pp_basic_only:
        keep = {"diff_pp_era","diff_pp_whip"}
        pp_cols = [c for c in feature_cols if c.startswith("diff_pp_")]
        drop_pp = [c for c in pp_cols if c not in keep]
        if drop_pp:
            print(f"[Feature] drop pitcher extras: {drop_pp}")
            X_tr_raw = X_tr_raw.drop(columns=drop_pp, errors="ignore")
            if X_val_raw is not None: X_val_raw = X_val_raw.drop(columns=drop_pp, errors="ignore")
            feature_cols = [c for c in feature_cols if c not in drop_pp]

    med = X_tr_raw.median(numeric_only=True)
    X_tr_raw = X_tr_raw.fillna(med)
    if X_val_raw is not None: X_val_raw = X_val_raw.fillna(med)
    if len(X_tr_raw) == 0: raise SystemExit("All rows became invalid after cleaning.")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw.values.astype(np.float32))
    X_va = scaler.transform(X_val_raw.values.astype(np.float32)) if X_val_raw is not None else None

    Xt_tr = torch.tensor(X_tr, dtype=torch.float32)
    yt_tr = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)
    Xt_va = torch.tensor(X_va, dtype=torch.float32) if X_va is not None else None
    yt_va = torch.tensor(y_val.reshape(-1,1), dtype=torch.float32) if X_va is not None else None

    model = MLP(in_dim=Xt_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    bce = nn.BCELoss()

    best_state = None; best_auc = -1.0; patience_left = args.patience
    print(f"[Train] epochs={args.epochs}  val_ratio={args.val_ratio:.2f}  patience={args.patience}  wd={args.weight_decay}")

    for epoch in range(1, args.epochs + 1):
        model.train(); opt.zero_grad()
        pred_tr = model(Xt_tr)
        loss_tr = bce(pred_tr, yt_tr)
        loss_tr.backward(); opt.step()

        with torch.no_grad():
            auc_tr = roc_auc_score(yt_tr.numpy(), pred_tr.numpy())
            log = f"epoch {epoch:02d}: loss_tr={loss_tr.item():.4f}, auc_tr={auc_tr:.4f}"
            if Xt_va is not None:
                model.eval()
                pred_va = model(Xt_va)
                auc_va = roc_auc_score(yt_va.numpy(), pred_va.numpy())
                log += f", auc_va={auc_va:.4f}"
                if auc_va > best_auc:
                    best_auc = auc_va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    patience_left = args.patience
                else:
                    patience_left -= 1
            print(log)
        if Xt_va is not None and patience_left <= 0:
            print(f"[EarlyStop] no val AUC improvement for {args.patience} epochs. stop at {epoch}.")
            break

    if best_state is not None: model.load_state_dict(best_state)

     # ---- 보정 + 임계값 ----
    calibrator = None
    calib_type = "none"
    best_threshold = 0.5

    if Xt_va is not None and len(Xt_va) > 0:
        model.eval()
        with torch.no_grad():
            val_raw = model(Xt_va).numpy().ravel()
        yv = yt_va.numpy().ravel()

        def fit_iso():
            iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(val_raw, yv)
            out = np.clip(iso.transform(val_raw), 0, 1)
            return iso, out, "isotonic"

        def fit_platt():
            lr = LogisticRegression(max_iter=1000); lr.fit(val_raw.reshape(-1,1), yv)
            out = lr.predict_proba(val_raw.reshape(-1,1))[:,1]
            return lr, out, "platt"

        def collapsed(raw, out):
            raw = np.clip(raw, 0, 1); out = np.clip(out, 0, 1)
            std_raw = float(np.std(raw)); std_out = float(np.std(out))
            uniq_out = int(np.unique(np.round(out, 3)).size)
            minv, maxv = out.min(), out.max()
            p_min = float(np.mean(np.isclose(out, minv)))
            p_max = float(np.mean(np.isclose(out, maxv)))
            return (std_out < 0.01) or (uniq_out < 10) or (max(p_min, p_max) > 0.6) or (std_raw > 0 and std_out < std_raw * 0.25)

        def brier(p): 
            p = np.clip(p, 0, 1)
            return float(np.mean((p - yv)**2))

        mode = args.calibration
        cand = []

        if mode in ("auto", "isotonic"):
            try:
                iso, out_iso, _ = fit_iso()
                cand.append(("isotonic", iso, out_iso))
            except Exception:
                pass
        if mode in ("auto", "platt"):
            try:
                lr, out_lr, _ = fit_platt()
                cand.append(("platt", lr, out_lr))
            except Exception:
                pass
        if mode == "none" or not cand:
            cand = [("none", None, val_raw)]

        # 후보 중 '붕괴 아님' & BrierScore가 가장 낮은 것 선택. 모두 붕괴면 raw 사용.
        best = ("none", None, val_raw, brier(val_raw))
        for name, cali, out in cand:
            if collapsed(val_raw, out):
                print(f"[Calib] drop {name}: collapsed")
                continue
            br = brier(out)
            if br < best[3] - 1e-5:   # 유의미하게 더 낮을 때만 채택
                best = (name, cali, out, br)

        calib_type, calibrator, proba_cal, _ = best
        print(f"[Calib] chosen={calib_type}")

        # ---- 임계값 ----
        if args.force_threshold is not None:
            best_threshold = float(args.force_threshold)
            print(f"[Tune] threshold forced to {best_threshold:.3f}")
        else:
            grid = np.linspace(0.30, 0.70, 81)
            accs = [accuracy_score(yv.astype(int), (proba_cal >= t).astype(int)) for t in grid]
            best_threshold = float(grid[int(np.argmax(accs))])
            print(f"[Tune] acc@best={max(accs):.4f}  thr={best_threshold:.3f}  (calibrator={calib_type})")

        raw_acc = ((val_raw >= 0.5).astype(int) == yv.astype(int)).mean()
        print(f"[Diag] Val raw@0.5 acc={raw_acc:.3f}")

    base_rate_tr = float(yt_tr.mean())
    print(f"[Diag] Train base-rate(home win)={base_rate_tr:.3f}")

    os.makedirs(args.models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.models_dir, "model.pt"))
    dump(scaler, os.path.join(args.models_dir, "scaler.joblib"))
    with open(os.path.join(args.models_dir, "feature_cols.json"), "w") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    if calibrator is not None:
        dump(calibrator, os.path.join(args.models_dir, "calibrator.joblib"))
    with open(os.path.join(args.models_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": best_threshold}, f)
    with open(os.path.join(args.models_dir, "meta.json"), "w") as f:
        json.dump({
            "season": int(args.season),
            "train_start": args.start,
            "train_end": args.end,
            "n_features": len(feature_cols),
            "calibrator": calib_type,
            "threshold": best_threshold,
            "val_auc": float(best_auc) if best_auc >= 0 else None,
            "recent_n": int(args.recent_n),
            "pp_basic_only": bool(args.pp_basic_only),
            "no_pitchers": bool(args.no_pitchers),
        }, f, ensure_ascii=False, indent=2)

    print("[Save] model.pt / scaler.joblib / feature_cols.json / threshold.json / meta.json"
          + ("/ calibrator.joblib" if calibrator is not None else ""))

if __name__ == "__main__":
    main()
