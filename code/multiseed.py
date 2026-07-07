#!/usr/bin/env python3
"""Multi-seed sweep: regenerate all three dataset realizations at 5 seeds,
rerun every offline pipeline, and aggregate mean/std per numeric leaf.

Realizations (matching the paper's three scopes):
  main    = T1 only,              --per-cell 25  -> provenance_eval.py
  twotype = T1 + T5,              --per-cell 15  -> provenance_eval_multitype.py, gnn_baseline.py
  fivetype= all five techniques,  --per-cell 6   -> fusion_eval.py, taxonomy_matrix.py, routing_sep.py

LLM-as-detector is NOT rerun (API cost; stays single-run, noted in the paper).
Model-internal seeds (Louvain/KMeans/GNN) stay fixed at 42; variance reported is
over dataset seeds.

Outputs:
  multiseed/seed{S}_{name}.json     raw per-seed results
  multiseed/mean_{name}.json        mean-valued results in the ORIGINAL schema
                                    (drop-in for plot_*.py)
  results_multiseed.json            {name: {path: {mean, std, n}}} flat summary
Usage: python3 multiseed.py [--seeds 42,101,202,303,404]
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "multiseed" / "data"
OUT = HERE / "multiseed"

T1 = "account_matrix"
T5 = "fake_review_cluster"

REALIZATIONS = {
    "main": {"types": T1, "per_cell": 25},
    "twotype": {"types": f"{T1},{T5}", "per_cell": 15},
    "fivetype": {"types": None, "per_cell": 6},
}

PIPELINES = [
    # (name, script, realization)
    ("results", "provenance_eval.py", "main"),
    ("multitype", "provenance_eval_multitype.py", "twotype"),
    ("gnn", "gnn_baseline.py", "twotype"),
    ("fusion", "fusion_eval.py", "fivetype"),
    ("matrix", "taxonomy_matrix.py", "fivetype"),
    ("routing", "routing_sep.py", "fivetype"),
]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def gen(seed):
    paths = {}
    for real, cfg in REALIZATIONS.items():
        p = DATA / f"seed{seed}_{real}.json"
        cmd = [sys.executable, "generate_synthetic.py", "--out", str(p),
               "--per-cell", str(cfg["per_cell"]), "--seed", str(seed)]
        if cfg["types"]:
            cmd += ["--types", cfg["types"]]
        run(cmd)
        paths[real] = p
    return paths


def walk(node, prefix, sink):
    """Collect numeric leaves as path -> value."""
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{prefix}.{k}" if prefix else k, sink)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{prefix}[{i}]", sink)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        sink[prefix] = float(node)


def mean_shaped(nodes):
    """Rebuild the original schema with numeric leaves replaced by the mean."""
    first = nodes[0]
    if isinstance(first, dict):
        return {k: mean_shaped([n[k] for n in nodes if isinstance(n, dict) and k in n])
                for k in first}
    if isinstance(first, list):
        if all(isinstance(n, list) and len(n) == len(first) for n in nodes):
            return [mean_shaped([n[i] for n in nodes]) for i in range(len(first))]
        return first
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        vals = [n for n in nodes if isinstance(n, (int, float)) and not isinstance(n, bool)]
        return round(statistics.fmean(vals), 4)
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,101,202,303,404")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    DATA.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        paths = gen(seed)
        for name, script, real in PIPELINES:
            out = OUT / f"seed{seed}_{name}.json"
            if out.exists():
                print(f"  skip {out.name} (exists)", flush=True)
                continue
            run([sys.executable, script, "--data", str(paths[real]), "--out", str(out)])

    summary = {}
    for name, _, _ in PIPELINES:
        per_seed = [json.loads((OUT / f"seed{s}_{name}.json").read_text()) for s in seeds]
        json.dump(mean_shaped(per_seed), open(OUT / f"mean_{name}.json", "w"), indent=1)
        leaves = []
        for ps in per_seed:
            sink = {}
            walk(ps, "", sink)
            leaves.append(sink)
        agg = {}
        for path in leaves[0]:
            vals = [lv[path] for lv in leaves if path in lv]
            if len(vals) >= 2:
                agg[path] = {"mean": round(statistics.fmean(vals), 4),
                             "std": round(statistics.stdev(vals), 4), "n": len(vals)}
        summary[name] = agg
    json.dump({"seeds": seeds, "summary": summary},
              open(HERE / "results_multiseed.json", "w"), indent=1)
    print(f"-> results_multiseed.json ({len(seeds)} seeds)")


if __name__ == "__main__":
    main()
