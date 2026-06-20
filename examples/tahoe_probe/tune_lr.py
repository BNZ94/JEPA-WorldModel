"""Quick LR/epoch grid on REAL data (held-out pairs, seed 0, idm_on) to pick a stable setting before
the multi-seed ablation. Prints model shift-R²/cosine vs the per-drug baseline."""
import fire
from examples.tahoe_probe.data import make_split
from examples.tahoe_probe.train_probe import train_arm, eval_arm


def _lst(x, cast):
    return [cast(v) for v in x] if isinstance(x, (list, tuple)) else [cast(v) for v in str(x).split(",")]


def run(npz="examples/tahoe_probe/data/centroids.npz", device="cuda",
        lrs="3e-4,1e-4", epochs="40,100,200", wds="1e-4,1e-3,1e-2", dropout="0.0", action_dim=64):
    lrs = _lst(lrs, float); eps = _lst(epochs, int); wds = _lst(wds, float); drops = _lst(dropout, float)
    split, info = make_split(npz, kind="pairs", test_frac=0.2, seed=0)
    print("info", info["n_train"], info["n_test"], "n_drugs", split.n_drugs, flush=True)
    per_drug = None
    best = None
    for ep in eps:
        for lr in lrs:
            for wd in wds:
                m = train_arm(split, idm_coeff=1.0, seed=0, epochs=ep, lr=lr, device=device,
                              weight_decay=wd, action_dim=action_dim)
                r = eval_arm(m, split, seed=0, device=device)
                per_drug = r['per_drug_mean_shift']['R2_shift']
                ms = r['model']['R2_shift']
                if best is None or ms > best[0]:
                    best = (ms, ep, lr, wd)
                print(f"ep={ep} lr={lr} wd={wd}: model.R2_shift={ms:+.3f} cos={r['model']['cos_shift']:+.3f} "
                      f"| per_drug={per_drug:+.3f} global={r['global_mean_shift']['R2_shift']:+.3f} "
                      f"no_op={r['no_op']['R2_shift']:+.3f} | decode_pred_top1={r['decode_pred_shift']['top1']:.3f} "
                      f"true_top1={r['decode_true_shift']['top1']:.3f}", flush=True)
    print(f"BEST model.R2_shift={best[0]:+.3f} at epochs={best[1]} lr={best[2]} wd={best[3]} "
          f"(per_drug baseline={per_drug:+.3f})", flush=True)


if __name__ == "__main__":
    fire.Fire(run)
