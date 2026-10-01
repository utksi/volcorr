"""Shared by train/evaluate scripts: data loading, the paper's partitions, tier definitions.

Evaluation protocol (identical to the paper):
  ten partitions, seeds 42..51, split(seed, n) -> train | val | test (80/10/10) over the
  29,372 benchmark rows in split-file order; metric = MAPE of the corrected volume,
  V_corr = V0 * exp(y_hat), y = ln(V_exp / V0).

"Deployment" scoring: test rows are featurized as a new user query would be, i.e. with
the MP-rule settings (rule_set_*) and energy_criterion = 0, and are reported both on all
test rows and on the majority protocol group (recorded force-based EDIFFG).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "_private" / "dataset.npz"
SEEDS = list(range(42, 52))

PAPER_PARAMS = {
    "objective": "regression_l1", "metric": "mae",
    "learning_rate": 0.012999059682890706, "num_leaves": 383,
    "min_child_samples": 40, "feature_fraction": 0.9603139046451887,
    "bagging_fraction": 0.9880615845171192, "bagging_freq": 1,
    "reg_alpha": 0.006036247839988367, "reg_lambda": 0.0023759783819383077,
    "max_depth": 12, "min_gain_to_split": 0.008402957521794496,
    "seed": 929032, "verbose": -1,
}

# Tier -> (column prefixes, reference volume). Settings columns are handled separately.
TIERS = {
    "A_formula":   (("comp_", "shell_"), "dft"),
    "B_spacegroup": (("comp_", "shell_", "sym_"), "dft"),
    "C_cif":       (("comp_", "shell_", "sym_", "cell_"), "dft"),
    "D_vdft":      (("comp_", "shell_", "sym_", "dft_"), "dft"),
    "E_mace":      (("comp_", "shell_", "sym_", "mace_"), "mace"),
}


def split(seed: int, n: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = max(1, int(round(0.1 * n)))
    test, pool = perm[:n_test], perm[n_test:]
    n_val = max(1, int(round(0.1 * len(pool))))
    return pool[n_val:], pool[:n_val], test


def load():
    """Benchmark rows only. Returns dict with X (n, p), columns, v_exp, v_dft, v_mace, ..."""
    d = np.load(DATA, allow_pickle=True)
    k = d["keep"]
    out = {c: d[c][k] for c in ("X", "mp_ids", "formulas", "v_exp", "v_dft", "v_mace",
                                "pbeu_recorded", "energy_criterion")}
    out["columns"] = [str(c) for c in d["columns"]]
    return out


def tier_matrices(D: dict, tier: str, settings: str = "recorded"):
    """(X_train_view, X_deploy_view, names, v0).

    settings: 'recorded' -> train on recorded set_* (incl. the EDIFFG-group nuisance flag),
              deploy view swaps in rule_set_* with energy_criterion = 0;
              'rule'     -> rule_set_* without the nuisance flag in both views;
              'none'     -> no settings columns.
    The MACE tier never uses DFT settings.
    """
    prefixes, ref = TIERS[tier]
    cols = D["columns"]
    base = [i for i, c in enumerate(cols) if c.startswith(prefixes)]
    names = [cols[i] for i in base]
    Xb = D["X"][:, base]
    if ref == "mace" or settings == "none":
        Xtr = Xdep = Xb
    elif settings == "recorded":
        s_rec = [cols.index(c) for c in cols if c.startswith("set_")]
        s_rule = [cols.index("rule_" + cols[i]) for i in s_rec]
        Xs_dep = D["X"][:, s_rule].copy()
        Xs_dep[:, [cols[i] for i in s_rec].index("set_energy_criterion")] = 0.0
        Xtr = np.hstack([Xb, D["X"][:, s_rec]])
        Xdep = np.hstack([Xb, Xs_dep])
        names += [cols[i] for i in s_rec]
    elif settings == "rule":
        s_rule = [i for i, c in enumerate(cols) if c.startswith("rule_set_") and c != "rule_set_energy_criterion"]
        Xtr = Xdep = np.hstack([Xb, D["X"][:, s_rule]])
        names += [cols[i].removeprefix("rule_") for i in s_rule]
    else:
        raise ValueError(settings)
    v0 = D["v_dft"] if ref == "dft" else D["v_mace"]
    return Xtr.astype(np.float32), Xdep.astype(np.float32), names, v0


def mape(y_hat, v0, v_exp) -> float:
    return float(np.mean(np.abs(v0 * np.exp(y_hat) - v_exp) / v_exp) * 100.0)
