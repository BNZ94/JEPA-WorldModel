"""Render the ablation figure locally from a saved ablation_results.json (so it can be produced even
if matplotlib was absent on the compute node)."""
import json
import sys

from examples.tahoe_probe.run_ablation_tahoe import _make_figure


def main(json_path, out_png):
    d = json.load(open(json_path))
    _make_figure(d["summary"], d["splits"], d["seeds"], d["epochs"], out_png)
    print("wrote", out_png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
