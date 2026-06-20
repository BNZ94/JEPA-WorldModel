"""Synthetic centroids.npz for PIPELINE DEBUGGING ONLY (clearly not real data).

Structure mirrors the real difficulty: absolute z is dominated by a large cell-line baseline (so
no-op scores high on absolute metrics, low on shift), the drug effect is a per-drug shift PLUS a
cell-line-specific modulation that is a linear readout of z_control (so it is recoverable by a model
that sees z_control, i.e. the model should beat the per-drug mean-shift baseline).
"""
import json
import numpy as np

D, n_cl, n_drug = 2560, 40, 60
rng = np.random.default_rng(0)
cl_base = rng.normal(0, 5.0, (n_cl, D))          # big -> dominates absolute z
drug_shift = rng.normal(0, 1.0, (n_drug, D))     # shared per-drug effect
W = rng.normal(0, 1.0 / np.sqrt(D), (n_drug, D, 8))  # per-drug readout of an 8-dim cl summary
S = rng.normal(0, 1.0 / np.sqrt(D), (8, D))      # cl summary projection

control_c, control_cl, control_n = [], [], []
treated_c, treated_cl, treated_dr, treated_n = [], [], [], []
for cl in range(n_cl):
    zc = cl_base[cl] + rng.normal(0, 0.1, D)
    control_c.append(zc); control_cl.append(cl); control_n.append(200)
    cl_sum = S @ cl_base[cl]                       # 8-dim summary of the cell line
    for dr in range(n_drug):
        modulation = W[dr] @ cl_sum                # cell-line-specific, recoverable from z_control
        shift = drug_shift[dr] + 0.8 * modulation
        zt = zc + shift + rng.normal(0, 0.05, D)
        treated_c.append(zt); treated_cl.append(cl); treated_dr.append(dr); treated_n.append(150)

meta = {"synthetic": True, "n_cell_lines": n_cl, "n_drugs_total": n_drug,
        "n_treated_pairs": len(treated_c), "min_cells": 50}
np.savez_compressed(
    "examples/tahoe_probe/data/centroids_synth.npz",
    treated_centroid=np.asarray(treated_c, np.float32), treated_cell_line=np.asarray(treated_cl),
    treated_drug=np.asarray(treated_dr), treated_n=np.asarray(treated_n),
    control_centroid=np.asarray(control_c, np.float32), control_cell_line=np.asarray(control_cl),
    control_n=np.asarray(control_n),
    cell_line_names=np.asarray([f"CL{i}" for i in range(n_cl)]),
    drug_names=np.asarray([f"D{i}" for i in range(n_drug)]),
    control_labels=np.asarray(["DMSO_TF"]), meta=json.dumps(meta))
print("wrote synth npz", meta)
