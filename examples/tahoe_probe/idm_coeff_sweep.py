"""Sensitivity of the predictor to the IDM auxiliary weight (held-out pairs). The main ablation is
on/off (coeff 1.0 vs 0.0, matching gLV run_ablation); this checks intermediate weights so the
"IDM hurts on real data" finding is not an artifact of one (large) coefficient. 3 seeds, mean±se."""
import fire
import numpy as np
from examples.tahoe_probe.data import make_split
from examples.tahoe_probe.train_probe import train_arm, eval_arm


def run(npz="examples/tahoe_probe/data/centroids.npz", device="cuda",
        coeffs="0.0,0.01,0.1,0.3,1.0", seeds="0,1,2", epochs=300, lr=3e-4, wd=1e-3, action_dim=64):
    coeffs = [float(x) for x in coeffs] if isinstance(coeffs, (list, tuple)) else [float(x) for x in str(coeffs).split(",")]
    seeds = [int(x) for x in seeds] if isinstance(seeds, (list, tuple)) else [int(x) for x in str(seeds).split(",")]
    for c in coeffs:
        r2s, decs, pdrug = [], [], []
        for s in seeds:
            split, _ = make_split(npz, kind="pairs", test_frac=0.2, seed=s)
            m = train_arm(split, idm_coeff=c, seed=s, epochs=epochs, lr=lr, weight_decay=wd,
                          action_dim=action_dim, device=device)
            res = eval_arm(m, split, seed=s, device=device)
            r2s.append(res["model"]["R2_shift"]); decs.append(res["decode_pred_shift"]["top1"])
            pdrug.append(res["per_drug_mean_shift"]["R2_shift"])
        a = np.array(r2s); d = np.array(decs)
        se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
        print(f"idm_coeff={c}: model.R2_shift={a.mean():+.3f}±{se:.3f} decode_pred_top1={d.mean():.3f} "
              f"(per_drug={np.mean(pdrug):+.3f})", flush=True)


if __name__ == "__main__":
    fire.Fire(run)
