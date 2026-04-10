"""
Model Comparison: Static Fitness Prediction on GB1
====================================================
Compares predictive quality (R² and Spearman) across model families on a fixed
budget of n=200 random training points, repeated 20 times.

Each method runs as a separate subprocess — avoids memory conflicts between
torch, sklearn, and xgboost on macOS.

Models evaluated
────────────────
  RF-ESM2        Random Forest on ESM2-650M 4-site embeddings  (our original)
  RF-ESMc        Random Forest on ESMc-600M 4-site embeddings
  XGB-ESMc       XGBoost ensemble on ESMc-600M                 (ALDE-style boosting)
  DNN-ESMc-S     DNN Ensemble [4608→256→128→1]     LeakyReLU   (small config)
  DNN-ESMc-L     DNN Ensemble [4608→500→150→50→1]  LeakyReLU   (ALDE ESM2 config)
  ALDE-XGB       XGBoost (tweedie, early_stop=10) on 4-site onehot (80-dim)
                 Li 2024 BOOSTING_ENSEMBLE, ~18% hit rate no-ZS
  ALDE-DNN       DNN Ensemble [80→30→30→1] LeakyReLU on 4-site onehot
                 Li 2024 DNN_ENSEMBLE, ~62% hit rate no-ZS (best no-ZS in paper)

DNN training details (matching jsunn-y/ALDE exactly)
─────────────────────────────────────────────────────
  Optimizer : Adam, lr=1e-3
  Loss      : MSE
  Max iters : 300
  Early stop: stop if min(loss[-30:]) >= min(loss[:-30])
  Ensemble  : 5 independent models, predictions averaged

Usage
─────
  cd /Users/aleksanderheino/protein/v1

  # Run one method at a time (each saves to results/compare_<method>.json):
  python3.11 experiments/compare_models.py --method RF-ESMc
  python3.11 experiments/compare_models.py --method XGB-ESMc
  python3.11 experiments/compare_models.py --method DNN-ESMc-S
  python3.11 experiments/compare_models.py --method DNN-ESMc-L

  # Print combined results table from saved JSON files:
  python3.11 experiments/compare_models.py --summarize
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT, "results", "compare")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
N_TRAIN      = 200   # training points per replicate
N_REPS       = 20    # number of random replicates
N_ENS        = 5     # DNN ensemble size
DNN_ITERS    = 300   # max training iterations per DNN
DNN_LR       = 1e-3  # Adam learning rate
EARLY_STOP_W = 30    # early stopping window size

METHOD_NAMES = ["RF-ESM2", "RF-ESMc", "XGB-ESMc", "DNN-ESMc-S", "DNN-ESMc-L", "ALDE-XGB", "ALDE-DNN", "DNN-ESMc-mean"]


ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs, same order as SSMuLA
GB1_SITES = [38, 39, 40, 53]           # 0-indexed positions that vary in GB1


def sequences_to_onehot(seqs):
    """4-site onehot: shape (N, 4*20) = (N, 80). Exact SSMuLA encoding."""
    aa_idx = {aa: i for i, aa in enumerate(ALL_AAS)}
    out = np.zeros((len(seqs), len(GB1_SITES) * len(ALL_AAS)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, site in enumerate(GB1_SITES):
            aa = seq[site]
            if aa in aa_idx:
                out[i, j * len(ALL_AAS) + aa_idx[aa]] = 1.0
    return out


def load_data(method):
    df = pd.read_csv(os.path.join(ROOT, "data/gb1/gb1_fitness.csv"))
    fitness = df["label"].values.astype(np.float32)
    fitness = fitness / fitness.max()

    if method == "RF-ESM2":
        emb = np.load(os.path.join(ROOT, "data/gb1/embeddings_esm2_650m_4site.npz"))["embeddings"]
    elif method == "DNN-ESMc-mean":
        emb = np.load(os.path.join(ROOT, "data/gb1/embeddings_esmc600m_4site_mean.npy"))
    elif method in ("ALDE-XGB", "ALDE-DNN"):
        seqs = df["protein"].values
        emb = sequences_to_onehot(seqs)
    else:
        emb = np.load(os.path.join(ROOT, "data/gb1/embeddings_esmc600m_4site.npy"))

    assert len(emb) == len(fitness)
    return emb, fitness


# ── ALDE baselines (Li 2024) ─────────────────────────────────────────────────
def run_alde_xgb(emb, fitness):
    """
    Li 2024 BOOSTING_ENSEMBLE: XGBoost, objective=reg:tweedie, early_stopping_rounds=10
    on 4-site onehot (80-dim). Their lower-end no-ZS baseline (~18% hit rate GB1).
    """
    import xgboost as xgb
    r2_list, sp_list = [], []
    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        tr = rng.choice(len(fitness), size=N_TRAIN, replace=False)
        te = np.setdiff1d(np.arange(len(fitness)), tr)
        X_tr, y_tr = emb[tr], fitness[tr]
        X_te, y_te = emb[te], fitness[te]
        m = xgb.XGBRegressor(
            objective="reg:tweedie",
            early_stopping_rounds=10,
            nthread=-1,
            verbosity=0,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        pred = m.predict(X_te)
        r2_list.append(float(r2_score(y_te, pred)))
        sp_list.append(float(spearmanr(y_te, pred).statistic))
        print(f"  rep {rep+1}/{N_REPS}  R²={r2_list[-1]:.4f}  Sp={sp_list[-1]:.4f}", flush=True)
    return {"r2": r2_list, "sp": sp_list, "convergence": None}


def run_alde_dnn(emb, fitness):
    """
    Li 2024 DNN_ENSEMBLE on 4-site onehot:
      architecture: [80, 30, 30, 1]
      activation: LeakyReLU
      lr: 1e-3, max_iter: 300, early_stop window: 30
      ensemble: 5 independent models
    Their best no-ZS model (~62% hit rate GB1).
    """
    import torch
    import torch.nn as nn

    architecture = [emb.shape[1], 30, 30, 1]  # [80, 30, 30, 1] for GB1 onehot

    class DNN_FF(nn.Module):
        def __init__(self, arch):
            super().__init__()
            layers = []
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i+1]).double())
                if i < len(arch) - 2:
                    layers.append(nn.LeakyReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_one(X_tr, y_tr):
        model = DNN_FF(architecture)
        X_t = torch.tensor(X_tr).double()
        y_t = torch.tensor(y_tr).double()
        opt = torch.optim.Adam(model.parameters(), lr=DNN_LR)
        mse = nn.MSELoss()
        losses = []
        model.train()
        for i in range(DNN_ITERS):
            opt.zero_grad()
            loss = mse(model(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if i > EARLY_STOP_W:
                if min(losses[i-EARLY_STOP_W+1:]) >= min(losses[:i-EARLY_STOP_W+1]):
                    break
        model.eval()
        return model, losses

    r2_list, sp_list = [], []
    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        tr = rng.choice(len(fitness), size=N_TRAIN, replace=False)
        te = np.setdiff1d(np.arange(len(fitness)), tr)
        X_tr, y_tr = emb[tr], fitness[tr]
        X_te, y_te = emb[te], fitness[te]

        models = [train_one(X_tr, y_tr)[0] for _ in range(N_ENS)]

        X_t = torch.tensor(X_te).double()
        with torch.no_grad():
            preds = np.stack([m(X_t).numpy() for m in models])
        pred = preds.mean(axis=0)

        r2_list.append(float(r2_score(y_te, pred)))
        sp_list.append(float(spearmanr(y_te, pred).statistic))
        print(f"  rep {rep+1}/{N_REPS}  R²={r2_list[-1]:.4f}  Sp={sp_list[-1]:.4f}", flush=True)
    return {"r2": r2_list, "sp": sp_list, "convergence": None}


# ── RF ──────────────────────────────────────────────────────────────────────
def run_rf(emb, fitness):
    from sklearn.ensemble import RandomForestRegressor
    r2_list, sp_list = [], []
    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        tr = rng.choice(len(fitness), size=N_TRAIN, replace=False)
        te = np.setdiff1d(np.arange(len(fitness)), tr)
        m = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=rep)
        m.fit(emb[tr], fitness[tr])
        pred = m.predict(emb[te])
        r2_list.append(float(r2_score(fitness[te], pred)))
        sp_list.append(float(spearmanr(fitness[te], pred).statistic))
        print(f"  rep {rep+1}/{N_REPS}  R²={r2_list[-1]:.4f}  Sp={sp_list[-1]:.4f}", flush=True)
    return {"r2": r2_list, "sp": sp_list, "convergence": None}


# ── XGBoost ─────────────────────────────────────────────────────────────────
def run_xgb(emb, fitness):
    import xgboost as xgb
    r2_list, sp_list = [], []
    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        tr = rng.choice(len(fitness), size=N_TRAIN, replace=False)
        te = np.setdiff1d(np.arange(len(fitness)), tr)
        m = xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, verbosity=0, seed=rep
        )
        m.fit(emb[tr], fitness[tr])
        pred = m.predict(emb[te])
        r2_list.append(float(r2_score(fitness[te], pred)))
        sp_list.append(float(spearmanr(fitness[te], pred).statistic))
        print(f"  rep {rep+1}/{N_REPS}  R²={r2_list[-1]:.4f}  Sp={sp_list[-1]:.4f}", flush=True)
    return {"r2": r2_list, "sp": sp_list, "convergence": None}


# ── DNN ──────────────────────────────────────────────────────────────────────
def run_dnn(emb, fitness, architecture):
    import torch
    import torch.nn as nn

    class DNN_FF(nn.Module):
        def __init__(self, arch):
            super().__init__()
            layers = []
            for i in range(len(arch) - 1):
                layers.append(nn.Linear(arch[i], arch[i+1]).double())
                if i < len(arch) - 2:
                    layers.append(nn.LeakyReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def train_one(X_tr, y_tr):
        model = DNN_FF(architecture)
        X_t = torch.tensor(X_tr).double()
        y_t = torch.tensor(y_tr).double()
        opt = torch.optim.Adam(model.parameters(), lr=DNN_LR)
        mse = nn.MSELoss()
        losses = []
        model.train()
        for i in range(DNN_ITERS):
            opt.zero_grad()
            loss = mse(model(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if i > EARLY_STOP_W:
                if min(losses[i-EARLY_STOP_W+1:]) >= min(losses[:i-EARLY_STOP_W+1]):
                    break
        model.eval()
        return model, losses

    r2_list, sp_list = [], []
    conv_curves = None  # save loss curves from rep 0

    for rep in range(N_REPS):
        rng = np.random.default_rng(rep)
        tr = rng.choice(len(fitness), size=N_TRAIN, replace=False)
        te = np.setdiff1d(np.arange(len(fitness)), tr)
        X_tr, y_tr = emb[tr], fitness[tr]
        X_te, y_te = emb[te], fitness[te]

        # Train ensemble
        models, all_losses = [], []
        for _ in range(N_ENS):
            m, lc = train_one(X_tr, y_tr)
            models.append(m)
            all_losses.append(lc)

        if rep == 0:
            conv_curves = all_losses

        # Predict: average ensemble
        X_t = torch.tensor(X_te).double()
        with torch.no_grad():
            preds = np.stack([m(X_t).numpy() for m in models])
        pred = preds.mean(axis=0)

        r2_list.append(float(r2_score(y_te, pred)))
        sp_list.append(float(spearmanr(y_te, pred).statistic))
        stopped = len(all_losses[0])
        print(f"  rep {rep+1}/{N_REPS}  R²={r2_list[-1]:.4f}  Sp={sp_list[-1]:.4f}  "
              f"stopped@{stopped}", flush=True)

    # Convergence summary for rep 0, model 0
    CKPTS = [0, 50, 100, 150, 200, 250]
    c = conv_curves[0]
    ckpt_str = [float(c[min(i, len(c)-1)]) for i in CKPTS]
    print(f"\n  Convergence (rep0, model0): {dict(zip(CKPTS, [f'{v:.5f}' for v in ckpt_str]))}", flush=True)

    return {"r2": r2_list, "sp": sp_list, "convergence": conv_curves}


# ── Summarize ────────────────────────────────────────────────────────────────
def summarize():
    print(f"\n{'Method':<16} {'R²':>8} {'±R²':>6}  {'Spearman':>9} {'±Sp':>6}")
    print("─" * 52)
    for name in METHOD_NAMES:
        path = os.path.join(OUT_DIR, f"{name}.json")
        if not os.path.exists(path):
            print(f"{name:<16}  (not run yet)")
            continue
        with open(path) as f:
            d = json.load(f)
        r2 = d["r2"]
        sp = d["sp"]
        print(f"{name:<16} {np.mean(r2):>8.4f} {np.std(r2):>6.4f}  "
              f"{np.mean(sp):>9.4f} {np.std(sp):>6.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_NAMES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        sys.exit(0)

    if not args.method:
        parser.error("Provide --method or --summarize")

    name = args.method
    out_path = os.path.join(OUT_DIR, f"{name}.json")
    if os.path.exists(out_path):
        print(f"{name} already done ({out_path}). Delete to rerun.")
        summarize()
        sys.exit(0)

    print(f"Loading data...", flush=True)
    emb, fitness = load_data(name)
    print(f"  {len(fitness)} variants  emb={emb.shape}  fitness max={fitness.max():.4f}", flush=True)
    print(f"\nRunning {name} ({N_REPS} reps × {N_TRAIN} train points)...", flush=True)

    if name in ("RF-ESM2", "RF-ESMc"):
        result = run_rf(emb, fitness)
    elif name == "ALDE-XGB":
        result = run_alde_xgb(emb, fitness)
    elif name == "ALDE-DNN":
        result = run_alde_dnn(emb, fitness)
    elif name == "DNN-ESMc-mean":
        result = run_dnn(emb, fitness, [emb.shape[1], 500, 150, 50, 1])
    elif name == "XGB-ESMc":
        result = run_xgb(emb, fitness)
    elif name == "DNN-ESMc-S":
        result = run_dnn(emb, fitness, [emb.shape[1], 256, 128, 1])
    elif name == "DNN-ESMc-L":
        result = run_dnn(emb, fitness, [emb.shape[1], 500, 150, 50, 1])

    # Save (exclude non-serializable convergence curves for now)
    save = {"method": name, "r2": result["r2"], "sp": result["sp"]}
    with open(out_path, "w") as f:
        json.dump(save, f)
    print(f"\nSaved: {out_path}", flush=True)

    r2, sp = result["r2"], result["sp"]
    print(f"\n{name}: R²={np.mean(r2):.4f}±{np.std(r2):.4f}  "
          f"Spearman={np.mean(sp):.4f}±{np.std(sp):.4f}", flush=True)
    summarize()
