"""Load the trained ensembles and turn descriptors into a volume-error prediction.

Convention: y = ln(V_exp / V_ref); the ensemble predicts y_hat, and
  predicted reference error   delta = 100 (exp(-y_hat) - 1)   [% of V_exp]
  corrected volume            V_corr = V_ref exp(y_hat)       (when V_ref is known)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from functools import cache, lru_cache
from importlib import resources
from pathlib import Path

import numpy as np

TIER_INFO = {
    "A_formula": "formula only",
    "B_spacegroup": "formula + space group",
    "C_cif": "experimental structure",
    "D_vdft": "formula + space group + your PBE(+U) volume",
    "E_mace": "MACE-MPA-0 relaxation of a structure",
}


def _cache_dir() -> Path:
    return Path(os.environ.get("VOLCORR_CACHE", Path.home() / ".cache" / "volcorr"))


@lru_cache(maxsize=1)
def registry() -> dict:
    return json.loads((resources.files("volcorr") / "models" / "registry.json").read_text())


def _fetch(name: str, sha256: str) -> Path:
    """Model file from the package, the local cache, or the release URL (checked by sha256)."""
    pkg = resources.files("volcorr") / "models" / name
    if pkg.is_file():
        return Path(str(pkg))
    path = _cache_dir() / name
    if not path.exists():
        base = os.environ.get("VOLCORR_MODEL_URL", registry()["base_url"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{base.rstrip('/')}/{name}", tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        path.unlink()
        raise RuntimeError(f"checksum mismatch for {name}; deleted, retry to re-download")
    return path


@dataclass
class Prediction:
    tier: str
    y_hat: float                      # ln(V_exp / V_ref)
    y_members: list[float]
    reference: str                    # "PBE(+U), MP settings" or "MACE-MPA-0"
    interval68: tuple[float, float]   # delta bounds, % (empirical, from held-out residuals)
    interval90: tuple[float, float]
    v_ref: float | None = None        # A^3/atom
    v_exp_input: float | None = None  # A^3/atom (tier C: volume of the given cell)
    warnings: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    @property
    def delta(self) -> float:
        """Predicted error of the reference volume relative to experiment, %."""
        return 100.0 * (np.exp(-self.y_hat) - 1.0)

    @property
    def spread(self) -> float:
        """Ensemble standard deviation of delta, % (model disagreement, not total error)."""
        return float(np.std([100.0 * (np.exp(-y) - 1.0) for y in self.y_members]))

    @property
    def v_corrected(self) -> float | None:
        return None if self.v_ref is None else float(self.v_ref * np.exp(self.y_hat))

    @property
    def v_ref_predicted(self) -> float | None:
        """Tier C: the reference-method volume implied for the given experimental cell."""
        return None if self.v_exp_input is None else float(self.v_exp_input * np.exp(-self.y_hat))

    def as_dict(self) -> dict:
        return {"tier": self.tier, "input": TIER_INFO[self.tier], "reference": self.reference,
                "predicted_error_percent": round(self.delta, 3),
                "ensemble_spread_percent": round(self.spread, 3),
                "interval68_percent": [round(v, 2) for v in self.interval68],
                "interval90_percent": [round(v, 2) for v in self.interval90],
                "v_reference_A3_per_atom": self.v_ref, "v_corrected_A3_per_atom": self.v_corrected,
                "v_input_A3_per_atom": self.v_exp_input,
                "v_reference_predicted_A3_per_atom": self.v_ref_predicted,
                "settings_assumed": self.settings, "warnings": self.warnings}


class TierModel:
    def __init__(self, tier: str):
        import lightgbm as lgb

        spec = registry()["tiers"][tier]
        self.tier, self.spec = tier, spec
        self.features: list[str] = spec["features"]
        self.stack = spec.get("kind") == "stack"
        self.members = []
        for f in spec["files"]:
            gb = f["gbdt"] if self.stack else f
            text = gzip.decompress(_fetch(gb["name"], gb["sha256"]).read_bytes()).decode()
            booster = lgb.Booster(model_str=text)
            if self.stack:
                with np.load(_fetch(f["mlp"]["name"], f["mlp"]["sha256"])) as z:
                    mlp = {k: z[k] for k in z.files}
                self.members.append((booster, mlp, np.array(f["weights"]), f["intercept"]))
            else:
                self.members.append((booster, None, None, 0.0))

    def predict(self, x: dict[str, float]) -> tuple[float, list[float]]:
        row = np.array([[x.get(k, np.nan) for k in self.features]], dtype=np.float64)
        ys = []
        for booster, mlp, w, b in self.members:
            g = float(booster.predict(row)[0])
            if mlp is None:
                ys.append(g)
            else:
                sg = int(x.get("sym_spacegroup", 0) or 0)
                ys.append(float(w[0] * g + w[1] * mlp_predict(mlp, row, sg) + b))
        return float(np.mean(ys)), ys

    def intervals(self, y_hat: float):
        """Delta intervals from held-out absolute residual quantiles in y (split-conformal style)."""
        q = self.spec["residual_quantiles"]
        to_d = lambda y: 100.0 * (np.exp(-y) - 1.0)
        return ((to_d(y_hat + q["q68"]), to_d(y_hat - q["q68"])),
                (to_d(y_hat + q["q90"]), to_d(y_hat - q["q90"])))


def mlp_predict(a: dict, row: np.ndarray, sg: int) -> float:
    """Numpy forward pass of the exported MLP (standardise, clip, space-group embedding, SiLU)."""
    h = np.nan_to_num(np.clip((row - a["mu"]) / a["sd"], -6, 6))
    h = np.concatenate([h, a["sg_emb"][[sg]]], 1)
    for j in range(4):
        h = h @ a[f"W{j}"].T + a[f"b{j}"]
        if j < 3:
            h = h / (1.0 + np.exp(-h))
    return float(h[0, 0] * a["scale"])


@cache
def load(tier: str) -> TierModel:
    return TierModel(tier)
