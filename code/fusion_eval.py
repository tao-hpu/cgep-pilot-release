#!/usr/bin/env python3
"""Confidence-gated multi-substrate fusion detector on the 5-type dataset.

Finding 6 showed naive fusion loses on pooled ARI because the weaker substrate's
IMPURE clusters add false-positive pairs. Fix: a confidence gate — only trust a
substrate's assignment when it places the item in a COORDINATED community (>=3
members). Route in priority order interaction -> citation -> content-sim, else
singleton. Expected: one detector recovers the whole taxonomy on the pooled stream,
beating every single substrate.

Usage: python3 fusion_eval.py --data ../02-data/pilot/dataset_5type.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score

CLUSTER_TYPES = ["account_matrix", "citation_stuffing", "parasite_content", "fake_review_cluster"]


def louvain_sizes(G):
    lab, size = {}, {}
    for ci, nodes in enumerate(nx.community.louvain_communities(G, seed=42)):
        for n in nodes:
            lab[n] = ci
        size[ci] = len(nodes)
    return lab, size


def interaction_sub(ds):
    G = nx.Graph(); G.add_nodes_from(a["account_id"] for a in ds["accounts"])
    for u, v, w in ds["edges"]:
        G.add_edge(u, v, weight=w)
    acc, size = louvain_sizes(G)
    lab = {it["content_id"]: acc[it["account_id"]] for it in ds["content"]}
    csize = {it["content_id"]: size[acc[it["account_id"]]] for it in ds["content"]}
    return lab, csize


def citation_sub(ds):
    G = nx.Graph(); G.add_nodes_from(c["content_id"] for c in ds["content"])
    for u, v in ds["citations"]:
        G.add_edge(u, v)
    lab, size = louvain_sizes(G)
    return ({c["content_id"]: lab.get(c["content_id"], -c["content_id"] - 1) for c in ds["content"]},
            {c["content_id"]: size.get(lab.get(c["content_id"], -1), 1) for c in ds["content"]})


def content_sub(ds, k=8, thresh=0.35):
    items = ds["content"]
    sig = np.array([it["framework_sig"] for it in items])
    A = kneighbors_graph(sig, n_neighbors=min(k, len(items) - 1),
                         mode="distance", metric="euclidean", include_self=False).tocoo()
    G = nx.Graph(); G.add_nodes_from(range(len(items)))
    for i, j, d in zip(A.row, A.col, A.data):
        if d < thresh:
            G.add_edge(int(i), int(j))
    lab, size = louvain_sizes(G)
    return ({items[i]["content_id"]: lab.get(i, -i - 1) for i in range(len(items))},
            {items[i]["content_id"]: size.get(lab.get(i, -1), 1) for i in range(len(items))})


def run(ds):
    inter, isz = interaction_sub(ds)
    cite, csz = citation_sub(ds)
    cont, ctsz = content_sub(ds)
    cid_of = {it["content_id"]: it["cluster_id"] for it in ds["content"]}

    def fuse(cid, gate=3):
        if isz[cid] >= gate:
            return f"I{inter[cid]}"
        if csz[cid] >= gate:
            return f"C{cite[cid]}"
        if ctsz[cid] >= gate:
            return f"T{cont[cid]}"
        return f"S{cid}"

    subs = {"interaction": inter, "citation": cite, "content-sim": cont,
            "fusion": {c["content_id"]: fuse(c["content_id"]) for c in ds["content"]}}

    res = {"by_type": {}, "pooled": {}}
    for mtype in CLUSTER_TYPES:
        idx = [it["content_id"] for it in ds["content"] if it["mtype"] == mtype]
        true = [cid_of[c] for c in idx]
        res["by_type"][mtype] = {name: round(adjusted_rand_score(true, [lab[c] for c in idx]), 3)
                                 for name, lab in subs.items()}
    pidx = [it["content_id"] for it in ds["content"] if it["mtype"] in CLUSTER_TYPES]
    ptrue = [cid_of[c] for c in pidx]
    res["pooled"] = {name: round(adjusted_rand_score(ptrue, [lab[c] for c in pidx]), 3)
                     for name, lab in subs.items()}
    return res


def show(res):
    subs = ["interaction", "citation", "content-sim", "fusion"]
    print("== Provenance ARI: substrate x type (+ confidence-gated fusion) ==\n")
    print(f"  {'type':20s}" + "".join(f"{s:>13s}" for s in subs))
    for mtype, cell in res["by_type"].items():
        print(f"  {mtype:20s}" + "".join(f"{cell[s]:>13.3f}" for s in subs))
    print(f"\n  {'POOLED (all types)':20s}" + "".join(f"{res['pooled'][s]:>13.3f}" for s in subs))
    p = res["pooled"]
    best_single = max(("interaction", "citation", "content-sim"), key=lambda s: p[s])
    print(f"\n  fusion pooled ARI {p['fusion']:.3f}  vs  best single ({best_single}) {p[best_single]:.3f}"
          f"  ->  {'FUSION WINS' if p['fusion'] > p[best_single] else 'fusion does not win'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_fusion.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
