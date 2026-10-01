# volcorr

volcorr predicts the volume error of a PBE/PBE+U relaxation, done with Materials Project
settings, relative to the experimental (ICSD) volume. It can also correct a PBE(+U) or
MACE-MPA-0 volume.

Web version with this option: https://huggingface.co/spaces/utksi/volcorr-dft

This branch (`finished-calc`) adds an optional mode that uses the results of your own finished
PBE(+U) relaxation (relaxed structure, band gap, magnetization; optionally formation energy and
energy above the hull). The version without it is the `main` branch.

## Installation

```
pip install git+https://github.com/utksi/volcorr@finished-calc
pip install "volcorr[mace] @ git+https://github.com/utksi/volcorr"   # for the MACE mode
```

The trained models (about 140 MB) are downloaded from
https://huggingface.co/utksi/volcorr-models on first use and checked against the SHA-256
hashes in `src/volcorr/models/registry.json`.

## Usage

```
volcorr Fe2O3 --sg 167                  # formula and space group
volcorr NaCl                            # formula only
volcorr --cif structure.cif             # experimental structure (CIF, POSCAR, extxyz)
volcorr NaCl --sg 225 --vdft 23.0       # correct your own PBE(+U) volume, A^3/atom
volcorr --cif structure.cif --mace      # relax with MACE-MPA-0, then correct
volcorr --relaxed CONTCAR --gap 1.2 --mag 0      # your finished PBE(+U) relaxation (best)
```

A positive error means the calculation overestimates the volume. `--json` prints all
fields. From Python:

```python
from volcorr.predict import predict
predict("Fe2O3", 167).as_dict()
```

## Accuracy

Mean absolute percentage error of the corrected volume for held-out compounds, averaged
over ten random 80/10/10 partitions of the data (standard deviation about 0.02). The
test compounds are those relaxed with the standard force-based Materials Project protocol.

| Input | Corrected | Uncorrected |
|---|---|---|
| formula | 1.33 % | 2.46 % |
| formula and space group | 1.33 % | 2.46 % |
| experimental structure | 1.24 % | 2.46 % |
| formula, space group and PBE(+U) volume | 1.32 % | 2.46 % |
| structure, MACE-MPA-0 relaxation | 1.79 % | 3.67 % |
| your finished PBE(+U) relaxation: relaxed structure, band gap, magnetization | 1.30 % | 2.46 % |
| the same plus formation energy and energy above hull | 1.29 % | 2.46 % |

The 68 % and 90 % intervals are quantiles of the held-out residuals. They have the same
width for every compound.

## Calculation settings

The training volumes come from Materials Project relaxations: PBE, PAW, 520 eV,
spin-polarized, force-based ionic relaxation, and PBE+U for oxides and fluorides of
Co (3.32 eV), Cr (3.7), Fe (5.3), Mn (3.9), Mo (4.38), Ni (6.2), V (3.25) and W (6.2).
Use `--plus-u` or `--no-u` to override the choice of PBE+U. Other functionals, such as
r2SCAN, are not covered.

The MACE mode relaxes the structure with MACE-MPA-0 (medium), keeping the space group
fixed, with FIRE to 0.01 eV/A.

## Limitations

- Training used only compounds whose PBE(+U) volume is within 5 % of experiment.
- The experimental volumes are at room temperature, so thermal expansion is part of the
  predicted error.
- A formula can have several polymorphs; give the space group or the structure.
- Warnings are printed for elements and space groups that are rare in the training data.

## Training

The scripts in `training/` rebuild the models (`build_dataset.py`, `evaluate.py`,
`train_final.py`, `train_mace_stack.py`, `make_reference.py`). They need the
ICSD-derived training set, which is not distributed.

## Licence

Code: MIT. Models: see `MODEL_LICENSE.md`.
