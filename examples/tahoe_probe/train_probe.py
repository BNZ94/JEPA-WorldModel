"""Train the action-conditioned one-step predictor (+ optional IDM auxiliary) on Tahoe centroids,
and run the IDM ablation. Faithful light reimplementation of the gLV recipe (see model.py header).

OBJECTIVE (mirrors examples/microbiome_jepa/train_worldmodel.run):
    pred_loss = MSE(z_pred, z_treated_true)                       # latent prediction (standardized)
    idm_loss  = CE(idm(z_control, z_pred), drug)                  # categorical action recovery
    total     = pred_loss + idm_coeff * idm_loss                  # ablation: idm_coeff in {1.0, 0.0}

IMPORTANT (refinement #3): the encoder is FROZEN (real mosaicfm-3b embeddings), so there is no
representation collapse to fight; the gLV M4 IDM benefit was collapse-specific. To make idm_coeff a
real knob on a frozen-input model, the IDM here reads the PREDICTOR'S OUTPUT z_pred (so its gradient
flows into the predictor + action encoder) — i.e. it is an auxiliary that regularizes the predictor
toward an action-aware transition. We therefore do NOT claim M4 'replicates'; we test a different
mechanism and report it honestly either way.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire
import numpy as np
import torch

from examples.tahoe_probe.data import make_split
from examples.tahoe_probe.eval_probe import (
    baseline_predictions, decodability, eval_predictions,
)
from examples.tahoe_probe.model import TahoeWorldModel

BASELINES = ["no_op", "global_mean_shift", "per_drug_mean_shift"]


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


@torch.no_grad()
def _val_shift_r2(model, Xc, Xt, dr):
    """Shift-space R² on a validation tensor set (same metric the test eval reports)."""
    model.eval()
    zp, _ = model(Xc, dr)
    true_shift = Xt - Xc
    pred_shift = zp - Xc
    ss_res = ((pred_shift - true_shift) ** 2).sum()
    ss_tot = ((true_shift - true_shift.mean(0, keepdim=True)) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def train_arm(split, idm_coeff, seed=0, action_dim=128, idm_hidden=256,
              epochs=400, lr=1e-3, batch_size=256, weight_decay=1e-5, device="cpu",
              val_frac=0.15, eval_every=5, patience=8, verbose=False):
    """Train with VALIDATION-BASED EARLY STOPPING (a 15% slice of TRAIN held out by seed). We keep the
    weights with the best val shift-R² and stop after `patience` non-improving evals. This (a) selects
    each run's best generalization point automatically — important because the GRU overshoots the shift
    magnitude if trained too long — and (b) avoids any tuning on the TEST set."""
    _set_seed(seed)
    dev = torch.device(device)
    Xc_all = torch.tensor(split.std_(split.z_ctrl_tr), dtype=torch.float32, device=dev)
    Xt_all = torch.tensor(split.std_(split.z_treat_tr), dtype=torch.float32, device=dev)
    dr_all = torch.tensor(split.drug_tr, dtype=torch.long, device=dev)
    n_all = Xc_all.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 777)
    vperm = torch.randperm(n_all, generator=g).to(dev)
    n_val = max(1, int(round(val_frac * n_all)))
    vi, ti = vperm[:n_val], vperm[n_val:]
    Xc, Xt, dr = Xc_all[ti], Xt_all[ti], dr_all[ti]
    Xcv, Xtv, drv = Xc_all[vi], Xt_all[vi], dr_all[vi]
    n = Xc.shape[0]

    model = TahoeWorldModel(state_dim=Xc.shape[1], action_dim=action_dim,
                            n_drugs=split.n_drugs, idm_hidden=idm_hidden).to(dev)
    # Exclude biases (incl. the update-gate bias that encodes the no-op prior) from weight decay, so
    # decay shrinks the delta-producing WEIGHTS (curbs held-out overshoot) without eroding that prior.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "bias_" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    mse = torch.nn.MSELoss(); ce = torch.nn.CrossEntropyLoss()

    best_val, best_state, best_ep, bad = -1e9, None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot = pl = il = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            zc, zt, d = Xc[idx], Xt[idx], dr[idx]
            opt.zero_grad()
            z_pred, _ = model(zc, d)
            pred_loss = mse(z_pred, zt)
            idm_loss = ce(model.idm(zc, z_pred), d) if idm_coeff > 0 else torch.zeros((), device=dev)
            loss = pred_loss + idm_coeff * idm_loss
            loss.backward(); opt.step()
            tot += loss.item(); pl += pred_loss.item(); il += float(idm_loss.detach())
        if ep % eval_every == 0 or ep == epochs - 1:
            vr2 = _val_shift_r2(model, Xcv, Xtv, drv)
            if vr2 > best_val + 1e-4:
                best_val, best_ep, bad = vr2, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if verbose and ep % 50 == 0:
                print(f"  ep{ep} pred={pl:.3f} idm={il:.3f} val_R2_shift={vr2:+.3f} best={best_val:+.3f}@{best_ep}", flush=True)
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_val_r2 = best_val; model._best_ep = best_ep
    return model


@torch.no_grad()
def predict(model, split, device="cpu"):
    dev = torch.device(device)
    Xc_te = torch.tensor(split.std_(split.z_ctrl_te), dtype=torch.float32, device=dev)
    d_te = torch.tensor(split.drug_te, dtype=torch.long, device=dev)
    Xc_tr = torch.tensor(split.std_(split.z_ctrl_tr), dtype=torch.float32, device=dev)
    d_tr = torch.tensor(split.drug_tr, dtype=torch.long, device=dev)
    model.eval()
    zt_te = model(Xc_te, d_te)[0].cpu().numpy()
    zt_tr = model(Xc_tr, d_tr)[0].cpu().numpy()
    return zt_tr, zt_te


def eval_arm(model, split, seed=0, device="cpu"):
    ctrl_te = split.std_(split.z_ctrl_te); treat_te = split.std_(split.z_treat_te)
    ctrl_tr = split.std_(split.z_ctrl_tr); treat_tr = split.std_(split.z_treat_tr)
    zt_tr, zt_te = predict(model, split, device=device)

    out = {"model": eval_predictions(zt_te, ctrl_te, treat_te)}
    for b in BASELINES:
        pred = baseline_predictions(split, b)
        out[b] = eval_predictions(pred, ctrl_te, treat_te)

    # action-decodability: true shift (data property) and model's predicted shift
    true_shift_tr, true_shift_te = treat_tr - ctrl_tr, treat_te - ctrl_te
    pred_shift_tr, pred_shift_te = zt_tr - ctrl_tr, zt_te - ctrl_te
    out["decode_true_shift"] = decodability(true_shift_tr, split.drug_tr, true_shift_te,
                                            split.drug_te, split.n_drugs, seed=seed, device=device)
    out["decode_pred_shift"] = decodability(pred_shift_tr, split.drug_tr, pred_shift_te,
                                            split.drug_te, split.n_drugs, seed=seed, device=device)
    return out


def run_single(npz="examples/tahoe_probe/data/centroids.npz", kind="pairs", idm_coeff=1.0,
               seed=0, epochs=400, test_frac=0.2, device="cpu", verbose=True):
    """Train + eval a single arm; prints the metric table. Used for smoke tests."""
    split, info = make_split(npz, kind=kind, test_frac=test_frac, seed=seed)
    print(f"split={kind} info={json.dumps(info)}", flush=True)
    model = train_arm(split, idm_coeff=idm_coeff, seed=seed, epochs=epochs, device=device, verbose=verbose)
    res = eval_arm(model, split, seed=seed, device=device)
    print(json.dumps(res, indent=2), flush=True)
    return res


if __name__ == "__main__":
    fire.Fire({"single": run_single})
