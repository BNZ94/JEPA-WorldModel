"""Data assembly for the Tahoe perturbation probe: build (z_control, z_treated, drug, cell_line)
pairs from the centroid npz, fit a per-dim standardizer on TRAIN only, and produce the two splits.

PAIR construction:
  For each treated population (cell_line, drug) with a centroid, the baseline state is the vehicle
  (DMSO_TF) centroid of the SAME cell line. So z_control = control_centroid[cell_line],
  z_treated = treated_centroid[(cell_line, drug)]. The true drug effect is the SHIFT
  Δ = z_treated - z_control (the discriminating signal; absolute z is dominated by cell-line identity).

SPLITS:
  * pairs   (PRIMARY): random split of (cell_line, drug) pairs. Both the cell line and the drug are
    seen in train (in other pairs); the specific combination is held out -> tests whether the model
    composes cell-line context (via z_control) with the drug.
  * celllines (SECONDARY): hold out whole cell lines; every test pair's cell line is unseen.

Standardization is fit on TRAIN pairs only (control+treated stacked) to avoid leakage. Everything
downstream (predictor, baselines, metrics) lives in this standardized space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class Split:
    name: str
    z_ctrl_tr: np.ndarray
    z_treat_tr: np.ndarray
    drug_tr: np.ndarray
    cl_tr: np.ndarray
    z_ctrl_te: np.ndarray
    z_treat_te: np.ndarray
    drug_te: np.ndarray
    cl_te: np.ndarray
    mean: np.ndarray  # per-dim, fit on train
    std: np.ndarray
    n_drugs: int
    n_cell_lines: int

    def std_(self, x):
        return (x - self.mean) / self.std


def _load_pairs(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    # control centroid per cell line (aggregate if >1 control pop per cell line — should be 1)
    ctrl_cl = d["control_cell_line"]
    ctrl_c = d["control_centroid"]
    ctrl_n = d["control_n"]
    ctrl_by_cl = {}
    ctrl_n_by_cl = {}
    for c, cl, n in zip(ctrl_c, ctrl_cl, ctrl_n):
        if cl in ctrl_by_cl:  # weighted combine (shouldn't generally happen)
            tot = ctrl_n_by_cl[cl] + n
            ctrl_by_cl[cl] = (ctrl_by_cl[cl] * ctrl_n_by_cl[cl] + c * n) / tot
            ctrl_n_by_cl[cl] = tot
        else:
            ctrl_by_cl[cl] = c.astype(np.float64)
            ctrl_n_by_cl[cl] = n
    # treated pairs that have a matching control cell line
    z_ctrl, z_treat, drug, cl = [], [], [], []
    n_missing_ctrl = 0
    for c, cli, dri in zip(d["treated_centroid"], d["treated_cell_line"], d["treated_drug"]):
        if cli not in ctrl_by_cl:
            n_missing_ctrl += 1
            continue
        z_ctrl.append(ctrl_by_cl[cli]); z_treat.append(c.astype(np.float64))
        drug.append(int(dri)); cl.append(int(cli))
    return (np.asarray(z_ctrl), np.asarray(z_treat), np.asarray(drug), np.asarray(cl),
            d["drug_names"], d["cell_line_names"], meta, n_missing_ctrl)


def make_split(npz_path, kind="pairs", test_frac=0.2, seed=0):
    z_ctrl, z_treat, drug, cl, drug_names, cl_names, meta, n_missing = _load_pairs(npz_path)
    n = len(z_ctrl)
    rng = np.random.default_rng(seed)
    if kind == "pairs":
        idx = rng.permutation(n)
        n_te = int(round(test_frac * n))
        te, tr = idx[:n_te], idx[n_te:]
    elif kind == "celllines":
        cls = np.unique(cl)
        rng.shuffle(cls)
        n_te_cl = max(1, int(round(test_frac * len(cls))))
        te_cls = set(cls[:n_te_cl].tolist())
        te = np.array([i for i in range(n) if cl[i] in te_cls])
        tr = np.array([i for i in range(n) if cl[i] not in te_cls])
    else:
        raise ValueError(kind)

    mean = z_ctrl[tr].mean(0) * 0.0  # placeholder, recomputed below over stacked train
    stacked = np.concatenate([z_ctrl[tr], z_treat[tr]], axis=0)
    mean = stacked.mean(0)
    std = stacked.std(0) + 1e-6
    return Split(
        name=kind,
        z_ctrl_tr=z_ctrl[tr], z_treat_tr=z_treat[tr], drug_tr=drug[tr], cl_tr=cl[tr],
        z_ctrl_te=z_ctrl[te], z_treat_te=z_treat[te], drug_te=drug[te], cl_te=cl[te],
        mean=mean, std=std, n_drugs=len(drug_names), n_cell_lines=len(cl_names),
    ), {"n_pairs": n, "n_missing_ctrl": n_missing, "n_train": len(tr), "n_test": len(te),
        "kind": kind, "test_frac": test_frac, "seed": seed, **meta}
