"""Train the shipped ensembles and write src/volcorr/models/{registry,domain}.json.

Each tier ships N members, member k trained on the train part of the paper's partition
seed 42+k (early stopping on its val part). Its test part was never seen by that member,
so the pooled deployment-view test residuals give (i) the reported accuracy and (ii) the
residual quantiles used for the 68% / 90% intervals. The deployed prediction is the
member mean, which is at least as accurate as a single member, so the intervals are
slightly conservative.

usage: python train_final.py --params compact --settings recorded [--members 10]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import lightgbm as lgb
import numpy as np
from joblib import Parallel, delayed

import common as C
from evaluate import PARAMS

MODELS = C.HERE.parent / "src" / "volcorr" / "models"


def member(tier, settings, pname, seed, D):
    Xtr, Xdep, names, v0 = C.tier_matrices(D, tier, settings)
    y = np.log(D["v_exp"] / v0).astype(np.float32)
    tr, va, te = C.split(seed, len(y))
    m = lgb.train(dict(PARAMS[pname], n_jobs=4), lgb.Dataset(Xtr[tr], y[tr]), 4000,
                  valid_sets=[lgb.Dataset(Xtr[va], y[va])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    text = m.model_to_string(num_iteration=m.best_iteration)
    res = y[te] - m.predict(Xdep[te], num_iteration=m.best_iteration)
    return text, names, te, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="compact", choices=list(PARAMS))
    ap.add_argument("--settings", default="recorded", choices=["recorded", "rule"])
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--base-url", default="https://huggingface.co/utksi/volcorr-models/resolve/main")
    a = ap.parse_args()

    D = C.load()
    MODELS.mkdir(parents=True, exist_ok=True)
    seeds = C.SEEDS[: a.members]
    jobs = [(t, a.settings if t != "E_mace" else "none", a.params, s) for t in C.TIERS for s in seeds]
    out = Parallel(n_jobs=10, verbose=5)(delayed(member)(*j, D) for j in jobs)

    reg = {"version": 1, "base_url": a.base_url, "params": a.params, "settings": a.settings,
           "target": "y = ln(V_exp / V_ref)", "tiers": {}}
    for tier in C.TIERS:
        rows = [o for j, o in zip(jobs, out) if j[0] == tier]
        files = []
        for k, (text, names, _, _) in enumerate(rows):
            blob = gzip.compress(text.encode(), compresslevel=9, mtime=0)
            name = f"{tier}_{k}.txt.gz"
            (MODELS / name).write_bytes(blob)
            files.append({"name": name, "sha256": hashlib.sha256(blob).hexdigest(), "mb": len(blob) / 1e6})
        res = np.concatenate([r[3] for r in rows])
        v0 = D["v_mace"] if tier == "E_mace" else D["v_dft"]
        te_all = np.concatenate([r[2] for r in rows])
        y_hat = np.log(D["v_exp"][te_all] / v0[te_all]) - res
        mape = C.mape(y_hat, v0[te_all], D["v_exp"][te_all])
        reg["tiers"][tier] = {
            "features": rows[0][1], "files": files,
            # WHY: quantiles of |y - y_hat| in log-volume; symmetric intervals in y map to
            # slightly asymmetric intervals in delta.
            "residual_quantiles": {"q68": float(np.quantile(np.abs(res), 0.68)),
                                   "q90": float(np.quantile(np.abs(res), 0.90))},
            "heldout_mape_percent": mape,
            "heldout_uncorrected_mape_percent": C.mape(np.zeros(len(te_all)), v0[te_all], D["v_exp"][te_all]),
            "n_heldout": int(len(res))}
        print(f"{tier:13s} held-out MAPE {mape:.3f}%  q68 {reg['tiers'][tier]['residual_quantiles']['q68']:.4f}"
              f"  size {sum(f['mb'] for f in files):.1f} MB")
    (MODELS / "registry.json").write_text(json.dumps(reg, indent=1))

    # Aggregate applicability statistics (counts and ranges only; no entries).
    from volcorr import features as F

    el = Counter()
    for f in D["formulas"]:
        el.update(F.parse_formula(f).keys())
    cols = D["columns"]
    sg = Counter(int(v) for v in D["X"][:, cols.index("sym_spacegroup")] if np.isfinite(v))
    rng = {}
    for k in ("shell_n_elements", "shell_z_max", "comp_mean_GSvolume_pa", "shell_en_contrast"):
        v = D["X"][:, cols.index(k)]
        rng[k] = [float(np.nanmin(v)), float(np.nanmax(v))]
    dom = {"n_training_compounds": int(len(D["formulas"])), "element_counts": dict(sorted(el.items())),
           "spacegroup_counts": {str(k): v for k, v in sorted(sg.items())}, "feature_ranges": rng,
           "rare_element_threshold": 30, "rare_spacegroup_threshold": 10}
    (MODELS / "domain.json").write_text(json.dumps(dom, indent=0))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(C.HERE.parent / "src"))
    main()
