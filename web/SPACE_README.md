---
title: volcorr
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Predict the DFT volume error of a crystal vs experiment
---

# volcorr

How far will a PBE/PBE+U relaxation with Materials Project settings miss the experimental
volume of a crystal? Enter a formula, optionally a space group, or upload/paste a structure
(CIF, POSCAR, extxyz, ...). Runs entirely in the browser (Pyodide + LightGBM + moyo); models
are loaded from [utksi/volcorr-models](https://huggingface.co/utksi/volcorr-models).
