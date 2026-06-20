"""Light, faithful reimplementation of our gLV action-conditioned predictor + IDM,
adapted for ONE-STEP Tahoe drug-perturbation transfer (population/centroid level).

WHY A REIMPLEMENTATION (documented deviation, see CLAUDE.md integrity rules):
  The gLV Layer-B pipeline (examples/microbiome_jepa/train_worldmodel.py) is entangled with the
  5-D temporal JEPA contract: SetTransformerEncoder -> [B,D,T,1,1], jepa.unroll over T steps,
  sequence regularizers (VC_IDM_Sim_Regularizer). Tahoe is ONE-STEP, uses FROZEN 2560-d
  mosaicfm-3b embeddings (no encoder to train), and the prediction target is a REAL embedding
  centroid (supervised), so there is no representation-collapse to fight and the sequence reg is
  not needed. We therefore reuse the *component designs*, not the temporal scaffolding:

  - GRUPredictor mirrors eb_jepa.architectures.RNNPredictor EXACTLY in wiring: the action is the
    GRU *input*, the state is the GRU *hidden*; one step gives z_treated = f(z_control, action).
    (eb_jepa/architectures.py:412-454.) We drop RNNPredictor's `final_ln`: that LayerNorm only made
    sense because the gLV encoder shared its final LN so predictions and targets lived in the same
    LN'd latent. Here targets are frozen embeddings (standardized per-dim instead), so an output
    LayerNorm would discard per-sample magnitude that the target carries. Documented deviation.

  - InverseDynamicsModel mirrors eb_jepa.architectures.InverseDynamicsModel: an MLP on
    concat(z_control, z_treated). The gLV action is a continuous panel vector (IDM regresses it);
    here the action is a categorical DRUG (~hundreds), so our IDM is a CLASSIFIER (cross-entropy).
    This is the only structural change, and it is forced by the modality. (Documented deviation.)

  - Objective mirrors train_worldmodel.run: prediction loss + idm_coeff * IDM loss, with the
    idm_coeff in {1.0 (on), 0.0 (off)} ablation exactly as run_ablation.py. The VICReg anti-collapse
    term is omitted (supervised target, frozen latent -> no collapse). Documented deviation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DrugActionEncoder(nn.Module):
    """Learned per-drug embedding -> action vector u_t (the GRU input).

    NOTE (stated in the brief): a learned per-drug table CANNOT generalize to unseen drugs, so
    held-out-DRUG eval is not available with this encoder. Held-out (cell_line, drug) PAIRS and
    held-out CELL LINES *are* available, because the drug index is seen in training there.
    A SMILES-based encoder would be the drop-in replacement for unseen-drug generalization.
    """

    def __init__(self, n_drugs: int, action_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_drugs, action_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, drug_idx: torch.Tensor) -> torch.Tensor:  # [B] long -> [B, action_dim]
        return self.emb(drug_idx)


class GRUPredictor(nn.Module):
    """One-step latent predictor; action = GRU input, z_control = GRU hidden (RNNPredictor wiring).

    Output = GRU output directly (= predicted z_treated), exactly like RNNPredictor. A GRU's output is
    (1-z)*h0 + z*cand, i.e. it ALREADY carries most of the hidden state z_control, so it can represent
    "z_control + small delta" natively; an explicit residual skip (z_control + out) double-counts the
    baseline and overshoots (verified: R²_abs collapses, shift explodes). The action=input /
    state=hidden wiring is unchanged from eb_jepa.architectures.RNNPredictor.
    """

    def __init__(self, state_dim: int, action_dim: int, num_layers: int = 1, residual: bool = False,
                 update_gate_bias: float = 3.0):
        super().__init__()
        self.rnn = nn.GRU(input_size=action_dim, hidden_size=state_dim, num_layers=num_layers)
        self.num_layers = num_layers
        self.residual = residual
        # Init the UPDATE-gate bias high so z≈1 and h'≈h0 at start: the predictor begins at the
        # "no change" prior (z_pred≈z_control) and only has to learn the small drug delta. PyTorch GRU
        # gate order in bias_*_l0 is [reset, update, new]; the middle block is the update gate z, and
        # h' = (1-z)*n + z*h0. Without this, z≈0.5 halves the 2560-d baseline and the model underfits
        # badly (verified: pred-loss stuck ~0.64 >> no-op ~0.04 on synthetic data).
        if update_gate_bias is not None:
            h = state_dim
            for layer in range(num_layers):
                with torch.no_grad():
                    getattr(self.rnn, f"bias_ih_l{layer}")[h:2 * h].fill_(update_gate_bias)
                    getattr(self.rnn, f"bias_hh_l{layer}")[h:2 * h].fill_(update_gate_bias)

    def forward(self, z_control: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # z_control [B, D], action [B, A]
        h0 = z_control.unsqueeze(0).repeat(self.num_layers, 1, 1).contiguous()  # [L, B, D]
        inp = action.unsqueeze(0).contiguous()                                  # [1, B, A]
        out, _ = self.rnn(inp, h0)
        delta = out[0]  # [B, D]
        return z_control + delta if self.residual else delta


class IDMClassifier(nn.Module):
    """Inverse-dynamics head: predict the drug (categorical) from concat(z_control, z_treated).

    Same MLP shape as eb_jepa.architectures.InverseDynamicsModel; final layer outputs n_drugs
    logits instead of an action vector (cross-entropy instead of MSE).
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_drugs: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_drugs),
        )

    def forward(self, z_control: torch.Tensor, z_treated: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([z_control, z_treated], dim=1))  # [B, n_drugs] logits


class TahoeWorldModel(nn.Module):
    """Bundles the action encoder + GRU predictor + IDM, mirroring the gLV JEPA bundle."""

    def __init__(self, state_dim: int, action_dim: int, n_drugs: int,
                 idm_hidden: int = 256, num_layers: int = 1):
        super().__init__()
        self.action_encoder = DrugActionEncoder(n_drugs, action_dim)
        self.predictor = GRUPredictor(state_dim, action_dim, num_layers=num_layers)
        self.idm = IDMClassifier(state_dim, idm_hidden, n_drugs)

    def forward(self, z_control: torch.Tensor, drug_idx: torch.Tensor):
        action = self.action_encoder(drug_idx)
        z_pred = self.predictor(z_control, action)
        return z_pred, action
