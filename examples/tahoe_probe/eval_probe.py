"""Evaluation for the Tahoe perturbation probe.

The drug effect is SMALL relative to cell-line identity, so absolute z_treated is approximately
z_control and EVERYTHING (even no-op) scores high on R²/cosine to z_treated. The discriminating
signal is the SHIFT Δ = z_treated - z_control. We therefore report metrics primarily on the shift
(refinement #1) and against three baselines, the PRIMARY one being the PER-DRUG mean shift
(refinement #2): beating it requires capturing cell-line-SPECIFIC modulation of the drug effect.

All inputs are in the standardized latent space (fit on train).
"""

from __future__ import annotations

import numpy as np
import torch


# ----------------------------- metrics -----------------------------
def _r2(pred, true):
    """Fraction-of-variance-explained over all pairs and dims; baseline = mean of `true`."""
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def _cosine(pred, true):
    pn = np.linalg.norm(pred, axis=1)
    tn = np.linalg.norm(true, axis=1)
    ok = (pn > 1e-8) & (tn > 1e-8)
    if ok.sum() == 0:
        return float("nan")
    c = (pred[ok] * true[ok]).sum(1) / (pn[ok] * tn[ok])
    return float(c.mean())


def eval_predictions(pred_treat, ctrl, true_treat):
    """Return shift-space and absolute-space metrics for a set of predicted treated centroids."""
    true_shift = true_treat - ctrl
    pred_shift = pred_treat - ctrl
    return {
        "R2_shift": _r2(pred_shift, true_shift),
        "cos_shift": _cosine(pred_shift, true_shift),
        "R2_abs": _r2(pred_treat, true_treat),
        "cos_abs": _cosine(pred_treat, true_treat),
    }


# ----------------------------- baselines -----------------------------
def baseline_predictions(split, which, drug_te=None):
    """Predicted treated centroid (standardized) for a baseline on the TEST set.

    no_op             : z_treated = z_control                      (shift 0)
    global_mean_shift : z_treated = z_control + mean_train_shift   (ignores drug identity)
    per_drug_mean_shift: z_treated = z_control + mean_train_shift_for_that_drug  (PRIMARY baseline;
                         the per-drug average effect across TRAIN cell lines — beating it proves
                         cell-line-specific modulation)
    """
    ctrl_tr = split.std_(split.z_ctrl_tr); treat_tr = split.std_(split.z_treat_tr)
    ctrl_te = split.std_(split.z_ctrl_te)
    shift_tr = treat_tr - ctrl_tr
    if which == "no_op":
        return ctrl_te.copy()
    if which == "global_mean_shift":
        return ctrl_te + shift_tr.mean(0, keepdims=True)
    if which == "per_drug_mean_shift":
        gmean = shift_tr.mean(0)
        per_drug = {}
        for dr in np.unique(split.drug_tr):
            per_drug[int(dr)] = shift_tr[split.drug_tr == dr].mean(0)
        pred = ctrl_te.copy()
        for i, dr in enumerate(split.drug_te):
            pred[i] = ctrl_te[i] + per_drug.get(int(dr), gmean)
        return pred
    raise ValueError(which)


# ----------------------------- action-decodability probe -----------------------------
def decodability(train_shift, train_drug, test_shift, test_drug, n_drugs, seed=0,
                 epochs=300, lr=0.05, device="cpu"):
    """Train a linear (multinomial-logistic) probe to predict drug from the latent SHIFT.

    Returns top-1 and top-5 accuracy on the test shifts plus the chance rate. Used both as a
    property of the frozen embedding space (on TRUE shifts) and to compare idm_on vs idm_off models
    (on PREDICTED shifts). This is the 'action information recoverable from the latent transition'
    analog of the gLV M4 decodability — but here it is a diagnostic, not a collapse rescue.
    """
    torch.manual_seed(seed)
    Xtr = torch.tensor(train_shift, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_drug, dtype=torch.long, device=device)
    Xte = torch.tensor(test_shift, dtype=torch.float32, device=device)
    yte = torch.tensor(test_drug, dtype=torch.long, device=device)
    clf = torch.nn.Linear(Xtr.shape[1], n_drugs).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = clf(Xte)
        top1 = (logits.argmax(1) == yte).float().mean().item()
        k = min(5, n_drugs)
        top5 = (logits.topk(k, 1).indices == yte[:, None]).any(1).float().mean().item()
    # chance = picking the most frequent test class
    vals, cnts = np.unique(test_drug, return_counts=True)
    chance = float(cnts.max() / cnts.sum())
    return {"top1": top1, "top5": top5, "chance_top1": chance, "n_test": len(test_drug)}
