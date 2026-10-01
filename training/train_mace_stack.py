"""MACE tier: LightGBM + MLP members combined by an L1 (median) stack; replaces E_mace in the registry.

For the MACE reference, an L1 stack of a GBDT and an MLP gave a lower held-out error than
the GBDT alone on all ten partitions (2.02 % vs 2.06 % MAPE).

Member k (partition seed 42+k): LightGBM (compact params) and an MLP, both trained on the
train part with early stopping on val; stack weights w_gbdt, w_mlp, b from a median
regression on the val predictions of the majority-protocol rows. Test residuals give the
reported accuracy and the interval quantiles, as in train_final.py.

The MLP is exported as plain arrays (npz) and evaluated in numpy at prediction time.
Run after train_final.py.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.linear_model import QuantileRegressor

import common as C
from evaluate import PARAMS
from train_final import MODELS

TIER, N_EL = "E_mace", 103
YS = 0.023  # WHY: std of y = ln(Vexp/V0); rescale the MLP target to O(1) for Adam


def frac_matrix(formulas):
    import sys

    sys.path.insert(0, str(C.HERE.parent / "src"))
    from volcorr.features import frac_features, parse_formula

    cache, rows = {}, []
    for f in formulas:
        if f not in cache:
            cache[f] = list(frac_features(parse_formula(f)).values())
        rows.append(cache[f])
    return np.array(rows, dtype=np.float32)


class Net(nn.Module):
    def __init__(self, p, hidden=256):
        super().__init__()
        self.sg = nn.Embedding(231, 16)
        self.head = nn.Sequential(nn.Linear(p + 16, hidden), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.1),
                                  nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, x, sg):
        return self.head(torch.cat([x, self.sg(sg)], -1)).squeeze(-1)


def fit_mlp(X, sg, y, tr, va, seed, max_epochs=200, patience=25, bs=256):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    mu = np.nanmean(X[tr], 0)
    sd = np.nanstd(X[tr], 0) + 1e-6
    Z = np.nan_to_num(np.clip((X - mu) / sd, -6, 6)).astype(np.float32)  # NaN -> training mean
    Zt, St, yt = torch.from_numpy(Z), torch.from_numpy(sg), torch.from_numpy(y / YS)
    net = Net(X.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    rng = np.random.default_rng(seed)
    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        net.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = (net(Zt[idx], St[idx]) - yt[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            mae = (net(Zt[va], St[va]) - yt[va]).abs().mean().item()
        sched.step(mae)
        if mae < best - 1e-6:
            best, bad, state = mae, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    sdict = {k: v.numpy() for k, v in state.items()}
    arrays = {"mu": mu, "sd": sd, "sg_emb": sdict["sg.weight"], "scale": np.array(YS)}
    for j, li in enumerate((0, 3, 6, 8)):  # Linear layers inside head (after SiLU/Dropout)
        arrays[f"W{j}"], arrays[f"b{j}"] = sdict[f"head.{li}.weight"], sdict[f"head.{li}.bias"]
    return arrays


def mlp_numpy(a, X, sg):
    """Must match volcorr.model.mlp_predict."""
    h = np.nan_to_num(np.clip((X - a["mu"]) / a["sd"], -6, 6))
    h = np.concatenate([h, a["sg_emb"][sg]], 1)
    for j in range(4):
        h = h @ a[f"W{j}"].T + a[f"b{j}"]
        if j < 3:
            h = h / (1.0 + np.exp(-h))  # SiLU
    return h[:, 0] * float(a["scale"])


def member(seed, X, sg, y, maj):
    tr, va, te = C.split(seed, len(y))
    m = lgb.train(dict(PARAMS["compact"], n_jobs=4), lgb.Dataset(X[tr], y[tr]), 4000,
                  valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
    arrays = fit_mlp(X, sg, y, tr, va, seed)
    P = lambda idx: np.stack([m.predict(X[idx], num_iteration=m.best_iteration),
                              mlp_numpy(arrays, X[idx], sg[idx])], 1)
    Pv, Pt = P(va), P(te)
    fm = maj[va]
    # WHY: L1 (median) stacking matches the MAPE-like metric; L2 stacks were outlier-dominated.
    q = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(Pv[fm], y[va][fm])
    w, b = q.coef_.astype(float), float(q.intercept_)
    return m.model_to_string(num_iteration=m.best_iteration), arrays, w, b, te, y[te] - (Pt @ w + b), Pt


def main():
    D = C.load()
    Xb, _, names, v0 = C.tier_matrices(D, TIER, "none")
    F = frac_matrix(D["formulas"])
    X = np.hstack([Xb, F]).astype(np.float32)
    names = list(names) + [f"frac_Z{z}" for z in range(1, N_EL + 1)]
    sg = np.nan_to_num(Xb[:, names.index("sym_spacegroup")]).astype(np.int64)
    y = np.log(D["v_exp"] / v0).astype(np.float32)
    maj = ~D["energy_criterion"]
    reg = json.loads((MODELS / "registry.json").read_text())
    k_members = len(reg["tiers"][TIER]["files"])
    out = Parallel(n_jobs=k_members, verbose=5)(delayed(member)(s, X, sg, y, maj)
                                                 for s in C.SEEDS[:k_members])

    files, residuals, te_all, gb_only = [], [], [], []
    for k, (text, arrays, w, b, te, res, Pt) in enumerate(out):
        blob = gzip.compress(text.encode(), compresslevel=9, mtime=0)
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        entry = {"weights": list(w), "intercept": b}
        for kind, name, data in (("gbdt", f"{TIER}_{k}.txt.gz", blob), ("mlp", f"{TIER}_{k}_mlp.npz", buf.getvalue())):
            (MODELS / name).write_bytes(data)
            entry[kind] = {"name": name, "sha256": hashlib.sha256(data).hexdigest(), "mb": len(data) / 1e6}
        files.append(entry)
        residuals.append(res)
        te_all.append(te)
        gb_only.append(y[te] - Pt[:, 0])
        print(f"member {k}: weights gbdt {w[0]:.3f} mlp {w[1]:.3f} b {b:+.5f}")
    res, te = np.concatenate(residuals), np.concatenate(te_all)
    yhat = np.log(D["v_exp"][te] / v0[te]) - res
    m_all = C.mape(yhat, v0[te], D["v_exp"][te])
    mm = maj[te]
    m_maj = C.mape(yhat[mm], v0[te][mm], D["v_exp"][te][mm])
    yg = np.log(D["v_exp"][te] / v0[te]) - np.concatenate(gb_only)
    print(f"stack held-out MAPE {m_all:.3f}% (majority {m_maj:.3f}%); gbdt member alone "
          f"{C.mape(yg, v0[te], D['v_exp'][te]):.3f}%")
    reg["tiers"][TIER] = {
        "kind": "stack", "features": names, "files": files,
        "residual_quantiles": {"q68": float(np.quantile(np.abs(res), 0.68)),
                               "q90": float(np.quantile(np.abs(res), 0.90))},
        "heldout_mape_percent": m_all, "heldout_mape_majority_percent": m_maj,
        "heldout_uncorrected_mape_percent": C.mape(np.zeros(len(te)), v0[te], D["v_exp"][te]),
        "n_heldout": int(len(res))}
    (MODELS / "registry.json").write_text(json.dumps(reg, indent=1))


if __name__ == "__main__":
    main()
