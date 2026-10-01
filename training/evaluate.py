"""LightGBM variants on the paper's ten partitions, for every input tier.

Writes training/results/lgbm_eval.csv (one row per tier x settings x params x seed), with
deployment-view MAPE on all test rows and on the majority protocol group, plus the
training-view MAPE (recorded settings, the paper's own evaluation).
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import common as C

COMPACT = {"objective": "regression_l1", "metric": "mae", "learning_rate": 0.03, "num_leaves": 63,
           "min_child_samples": 20, "feature_fraction": 0.8, "bagging_fraction": 0.9,
           "bagging_freq": 1, "reg_lambda": 1e-3, "seed": 929032, "verbose": -1}
PARAMS = {"paper": C.PAPER_PARAMS, "compact": COMPACT}


def run(tier, settings, pname, seed, D):
    Xtr, Xdep, names, v0 = C.tier_matrices(D, tier, settings)
    y = np.log(D["v_exp"] / v0).astype(np.float32)
    tr, va, te = C.split(seed, len(y))
    t0 = time.perf_counter()
    m = lgb.train(dict(PARAMS[pname], n_jobs=4), lgb.Dataset(Xtr[tr], y[tr]), 4000,
                  valid_sets=[lgb.Dataset(Xtr[va], y[va])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    fit_s = time.perf_counter() - t0
    maj = ~D["energy_criterion"][te]
    p_tr = m.predict(Xtr[te], num_iteration=m.best_iteration)
    p_dep = m.predict(Xdep[te], num_iteration=m.best_iteration)
    s = m.model_to_string(num_iteration=m.best_iteration)
    return {"tier": tier, "settings": settings, "params": pname, "seed": seed,
            "mape_recorded_view": C.mape(p_tr, v0[te], D["v_exp"][te]),
            "mape_all": C.mape(p_dep, v0[te], D["v_exp"][te]),
            "mape_majority": C.mape(p_dep[maj], v0[te][maj], D["v_exp"][te][maj]),
            "mape_uncorrected": C.mape(np.zeros(len(te)), v0[te], D["v_exp"][te]),
            "best_iter": m.best_iteration, "fit_seconds": fit_s, "model_mb": len(s) / 1e6}


def main():
    D = C.load()
    jobs = [(t, s, p, seed) for t in C.TIERS for s in (["recorded", "rule"] if t != "E_mace" else ["none"])
            for p in PARAMS for seed in C.SEEDS]
    rows = Parallel(n_jobs=7, verbose=5)(delayed(run)(*j, D) for j in jobs)
    df = pd.DataFrame(rows)
    out = C.HERE / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "lgbm_eval.csv", index=False)
    g = df.groupby(["tier", "settings", "params"])
    summ = g[["mape_uncorrected", "mape_recorded_view", "mape_all", "mape_majority", "best_iter",
              "fit_seconds", "model_mb"]].mean()
    summ["std_all"] = g["mape_all"].std()
    pd.set_option("display.width", 200)
    print(summ.round(3))


if __name__ == "__main__":
    main()
