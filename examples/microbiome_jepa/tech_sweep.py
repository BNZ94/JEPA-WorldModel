"""M2 tech-invariance sweep orchestrator (DANN invariance_coeff sweep, multi-seed).

Trains a grid of Layer-A set-JEPA encoders on the REAL MicrobeAtlas corpus with the
DANN technology adversary at several invariance_coeff values (incl. 0 = matched
baseline) and several seeds, for one or both SSL losses (VICReg / SIGReg-BCS). Then
freezes each encoder and runs the post-hoc dual-axis probe (tech bal-acc DOWN,
biome bal-acc MAINTAINED) via tech_invariance.run — a FRESH probe, never the
training adversary. Collects every number into one sweep JSON + the tradeoff curve.

INTEGRITY: every number is measured from a real run; the per-run checkpoint dir and
config are recorded. coeff=0 is the matched re-baseline (same labelled+balanced corpus
and recipe, adversary simply not built). The committed `realenc` (unlabelled first-N
corpus) is a separate historical reference, not this sweep's baseline.

Run (cluster GPU node, 4 GPUs):
  sbatch examples/microbiome_jepa/run_tech_sweep.sh
or directly:
  python -m examples.microbiome_jepa.tech_sweep --losses vicreg,bcs \
      --coeffs 0,0.3,1.0,3.0 --seeds 0,1,2 --n_gpus 4 --epochs 30 --ns 16000
"""

import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import fire

REPO = os.environ.get("EBJEPA_REPO", ".")
PY = os.path.join(os.environ.get("UV_PROJECT_ENVIRONMENT", ""), "bin", "python") \
    if os.environ.get("UV_PROJECT_ENVIRONMENT") else "python"
CFG = "examples/microbiome_jepa/cfgs/layerA_real.yaml"


def _csv(x):
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    return [s.strip() for s in str(x).split(",") if str(s).strip() != ""]


def _tag(loss, coeff, seed):
    c = str(coeff).replace(".", "p")
    return f"techinv_{loss}_c{c}_s{seed}"


def train_one(loss, coeff, seed, gpu, data_dir, ckpt_root, epochs, ns, d_model, log_dir):
    """Spawn one training subprocess pinned to `gpu`. Returns (Popen, tag, ckpt_dir, logf)."""
    tag = _tag(loss, coeff, seed)
    ckpt_dir = os.path.join(ckpt_root, tag)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [
        PY, "-m", "examples.microbiome_jepa.main",
        "--fname", CFG, "--folder", ckpt_dir,
        "--meta.seed", str(seed),
        "--data.data_dir", data_dir,
        "--data.with_tech_labels", "true",
        "--data.tech_balanced", "true",
        "--data.synth_n_samples", str(ns),
        "--model.d_model", str(d_model),
        "--loss.type", loss,
        "--loss.invariance_coeff", str(coeff),
        "--optim.epochs", str(epochs),
        "--logging.tqdm_silent", "true",
    ]
    logf = open(os.path.join(log_dir, f"train_{tag}.log"), "w")
    logf.write("CMD: " + " ".join(cmd) + f"\nGPU={gpu}\n")
    logf.flush()
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p, tag, ckpt_dir, logf


def eval_one(tag, ckpt_dir, loss, coeff, seed, data_dir, d_model, per_class_cap, out_root, log_dir):
    """Frozen-encoder dual-axis probe (CPU). Returns the parsed result dict (or None)."""
    out_dir = os.path.join(out_root, tag)
    cmd = [
        PY, "-m", "examples.microbiome_jepa.tech_invariance",
        "--checkpoint", os.path.join(ckpt_dir, "latest.pth.tar"),
        "--fname", CFG, "--data_dir", data_dir,
        "--d_model", str(d_model), "--n_max", "256",
        "--per_class_cap", str(per_class_cap), "--device", "cpu",
        "--mlp_probe", "true", "--out", out_dir,
    ]
    logf = open(os.path.join(log_dir, f"eval_{tag}.log"), "w")
    logf.write("CMD: " + " ".join(cmd) + "\n")
    logf.flush()
    r = subprocess.run(cmd, cwd=REPO, env=os.environ, stdout=logf, stderr=subprocess.STDOUT)
    logf.close()
    jpath = os.path.join(out_dir, "tech_invariance.json")
    if r.returncode != 0 or not os.path.exists(jpath):
        return None
    with open(jpath) as fh:
        res = json.load(fh)
    res["_meta"] = {"loss": loss, "coeff": float(coeff), "seed": int(seed),
                    "tag": tag, "checkpoint": ckpt_dir}
    return res


def run(
    losses="vicreg,bcs",
    coeffs="0,0.3,1.0,3.0",
    seeds="0,1,2",
    n_gpus: int = 4,
    epochs: int = 30,
    ns: int = 16000,
    d_model: int = 128,
    per_class_cap: int = 2500,
    data_dir: str = None,
    ckpt_root: str = None,
    out_root: str = None,
    skip_train: bool = False,   # only re-run eval on existing checkpoints
):
    work = os.environ.get("WORK", ".")
    data_dir = data_dir or os.path.join(os.environ.get("EBJEPA_DSETS", "."), "susagi", "data")
    ckpt_root = ckpt_root or os.path.join(work, "checkpoints/microbiome_jepa/tech_sweep")
    out_root = out_root or os.path.join(work, "checkpoints/microbiome_jepa/tech_sweep_eval")
    log_dir = os.path.join(ckpt_root, "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    losses, coeffs, seeds = _csv(losses), _csv(coeffs), _csv(seeds)
    grid = list(itertools.product(losses, coeffs, seeds))
    print(f"[sweep] {len(grid)} runs: losses={losses} coeffs={coeffs} seeds={seeds} "
          f"n_gpus={n_gpus} epochs={epochs} ns={ns} d_model={d_model}", flush=True)

    # ---- Phase 1: train across the GPU pool (n_gpus concurrent) ----
    if not skip_train:
        pending = list(grid)
        running = {}   # gpu -> (Popen, tag, ckpt_dir, logf, job)
        free_gpus = list(range(n_gpus))
        t0 = time.time()
        while pending or running:
            while pending and free_gpus:
                loss, coeff, seed = pending.pop(0)
                gpu = free_gpus.pop(0)
                p, tag, ckpt_dir, logf = train_one(
                    loss, coeff, seed, gpu, data_dir, ckpt_root, epochs, ns, d_model, log_dir)
                running[gpu] = (p, tag, ckpt_dir, logf, (loss, coeff, seed))
                print(f"[train] start {tag} on gpu{gpu}", flush=True)
            done = []
            for gpu, (p, tag, ckpt_dir, logf, job) in running.items():
                if p.poll() is not None:
                    logf.close()
                    ok = p.returncode == 0 and os.path.exists(os.path.join(ckpt_dir, "latest.pth.tar"))
                    print(f"[train] done  {tag} rc={p.returncode} ok={ok} "
                          f"({(time.time()-t0)/60:.1f}m elapsed)", flush=True)
                    done.append(gpu)
            for gpu in done:
                running.pop(gpu)
                free_gpus.append(gpu)
            time.sleep(5)

    # ---- Phase 2: frozen-probe eval (CPU) for every run ----
    results = []
    for loss, coeff, seed in grid:
        tag = _tag(loss, coeff, seed)
        ckpt_dir = os.path.join(ckpt_root, tag)
        if not os.path.exists(os.path.join(ckpt_dir, "latest.pth.tar")):
            print(f"[eval] SKIP {tag} (no checkpoint)", flush=True)
            continue
        res = eval_one(tag, ckpt_dir, loss, coeff, seed, data_dir, d_model, per_class_cap, out_root, log_dir)
        if res is None:
            print(f"[eval] FAIL {tag}", flush=True)
            continue
        tp = res["tech_probe"]["jepa_pretrained"]["balanced_acc_mean"]
        bp = res.get("biome_probe", {}).get("jepa_pretrained", {}).get("balanced_acc_mean", float("nan"))
        print(f"[eval] {tag}: TECH bal {tp:.3f} | BIOME bal {bp:.3f}", flush=True)
        results.append(res)

    Path(out_root).mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    with open(os.path.join(out_root, "sweep_results.json"), "w") as fh:
        json.dump({"grid": grid, "config": {"epochs": epochs, "ns": ns, "d_model": d_model,
                   "per_class_cap": per_class_cap}, "runs": results, "summary": summary}, fh, indent=2)
    print(f"\n[sweep] saved -> {out_root}/sweep_results.json", flush=True)
    _print_tradeoff(summary)
    return summary


def summarize(results):
    """Aggregate over seeds -> mean/se of (tech bal-acc, biome bal-acc) per (loss, coeff)."""
    import math
    from collections import defaultdict
    buckets = defaultdict(lambda: {"tech": [], "biome": [], "tech_mlp": [], "biome_mlp": []})
    for r in results:
        m = r["_meta"]
        key = (m["loss"], m["coeff"])
        buckets[key]["tech"].append(r["tech_probe"]["jepa_pretrained"]["balanced_acc_mean"])
        if "jepa_pretrained" in r.get("biome_probe", {}):
            buckets[key]["biome"].append(r["biome_probe"]["jepa_pretrained"]["balanced_acc_mean"])
        if "jepa_pretrained" in r.get("tech_probe_mlp", {}):
            buckets[key]["tech_mlp"].append(r["tech_probe_mlp"]["jepa_pretrained"]["balanced_acc_mean"])
        if "jepa_pretrained" in r.get("biome_probe_mlp", {}):
            buckets[key]["biome_mlp"].append(r["biome_probe_mlp"]["jepa_pretrained"]["balanced_acc_mean"])

    def ms(v):
        if not v:
            return {"mean": None, "se": None, "n": 0}
        mean = sum(v) / len(v)
        se = (sum((x - mean) ** 2 for x in v) / (len(v) - 1)) ** 0.5 / math.sqrt(len(v)) if len(v) > 1 else 0.0
        return {"mean": mean, "se": se, "n": len(v)}

    out = []
    for (loss, coeff), d in sorted(buckets.items()):
        out.append({"loss": loss, "coeff": coeff,
                    "tech_balacc": ms(d["tech"]), "biome_balacc": ms(d["biome"]),
                    "tech_balacc_mlp": ms(d["tech_mlp"]), "biome_balacc_mlp": ms(d["biome_mlp"])})
    return out


def _print_tradeoff(summary):
    print("\n================ TRADEOFF CURVE (linear probe, mean±se over seeds) ================")
    print(f"{'loss':8s} {'coeff':>6s} | {'TECH bal (down=good)':>22s} | {'BIOME bal (keep)':>18s}")
    for s in summary:
        t, b = s["tech_balacc"], s["biome_balacc"]
        ts = f"{t['mean']:.3f}±{t['se']:.3f} (n{t['n']})" if t["mean"] is not None else "—"
        bs = f"{b['mean']:.3f}±{b['se']:.3f}" if b["mean"] is not None else "—"
        print(f"{s['loss']:8s} {str(s['coeff']):>6s} | {ts:>22s} | {bs:>18s}")


if __name__ == "__main__":
    fire.Fire(run)
