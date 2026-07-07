#!/usr/bin/env python3
"""Full-taxonomy signal crossover: detector (substrate) x manipulation type.

Substrates / detectors:
  interaction  : Louvain on account interaction graph          (native to ①)
  citation     : Louvain on citation graph                     (native to ②)
  content-sim  : Louvain on framework_sig kNN graph            (native to ③ ⑤)
  authority-gap: per-item detection via host_authority-page_q  (native to ③, detection)
  dom-flag     : per-item detection via dom_carrier            (native to ④, detection)

Provenance ARI (T2) reported for the clustering substrates on ①②③⑤; detection recall
for the per-item detectors on ③④. Expected: block-diagonal — each type recovered only
by its native substrate.

Usage: python3 taxonomy_matrix.py --data ../02-data/pilot/dataset.json --out results_matrix.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score

CLUSTER_TYPES = ["account_matrix", "citation_stuffing", "parasite_content", "fake_review_cluster"]
SYMBOL = {"account_matrix": "(1)", "citation_stuffing": "(2)", "parasite_content": "(3)",
          "hidden_injection": "(4)", "fake_review_cluster": "(5)"}


def louvain(G):
    lab = {}
    for ci, nodes in enumerate(nx.community.louvain_communities(G, seed=42)):
        for n in nodes:
            lab[n] = ci
    return lab


def graph_from_accounts(ds):
    G = nx.Graph(); G.add_nodes_from(a["account_id"] for a in ds["accounts"])
    for u, v, w in ds["edges"]:
        G.add_edge(u, v, weight=w)
    acc = louvain(G)
    return {it["content_id"]: acc[it["account_id"]] for it in ds["content"]}


def graph_from_citations(ds):
    G = nx.Graph(); G.add_nodes_from(c["content_id"] for c in ds["content"])
    for u, v in ds["citations"]:
        G.add_edge(u, v)
    lab = louvain(G)
    return {c["content_id"]: lab.get(c["content_id"], -c["content_id"] - 1) for c in ds["content"]}


def graph_from_content(ds, k=8, thresh=0.35):
    items = ds["content"]
    sig = np.array([it["framework_sig"] for it in items])
    A = kneighbors_graph(sig, n_neighbors=min(k, len(items) - 1),
                         mode="distance", metric="euclidean", include_self=False).tocoo()
    G = nx.Graph(); G.add_nodes_from(range(len(items)))
    for i, j, d in zip(A.row, A.col, A.data):
        if d < thresh:
            G.add_edge(int(i), int(j))
    lab = louvain(G)
    return {items[i]["content_id"]: lab.get(i, -i - 1) for i in range(len(items))}


def run(ds):
    items = ds["content"]
    cid_of = {it["content_id"]: it["cluster_id"] for it in items}
    detectors = {"interaction": graph_from_accounts(ds),
                 "citation": graph_from_citations(ds),
                 "content-sim": graph_from_content(ds)}
    res = {"provenance": {}, "detection": {}}
    for det, lab in detectors.items():
        res["provenance"][det] = {}
        for mtype in CLUSTER_TYPES:
            idx = [it["content_id"] for it in items if it["mtype"] == mtype]
            true = [cid_of[c] for c in idx]
            res["provenance"][det][mtype] = round(adjusted_rand_score(true, [lab[c] for c in idx]), 3)

    # per-item detection anchors
    def recall(flag_fn, mtype):
        idx = [it for it in items if it["mtype"] == mtype]
        return round(sum(flag_fn(it) for it in idx) / len(idx), 3) if idx else 0.0
    res["detection"]["dom-flag@(4)"] = recall(lambda it: it["dom_carrier"], "hidden_injection")
    res["detection"]["authority-gap@(3)"] = recall(
        lambda it: it["host_authority"] - it["page_quality"] > 0.35, "parasite_content")
    # off-target: dom-flag should NOT fire on organic
    org = [it for it in items if not it["is_manipulated"]]
    res["detection"]["dom-flag@organic(FP)"] = round(sum(it["dom_carrier"] for it in org) / len(org), 3)
    return res


def show(res):
    dets = ["interaction", "citation", "content-sim"]
    print("== Provenance ARI: detector (rows) x manipulation type (cols) ==\n")
    print(f"  {'detector':14s}" + "".join(f"{SYMBOL[t]:>9s}" for t in CLUSTER_TYPES))
    for det in dets:
        row = res["provenance"][det]
        cells = []
        for t in CLUSTER_TYPES:
            v = row[t]
            best = v == max(res["provenance"][d][t] for d in dets)
            cells.append(f"{v:>8.3f}{'*' if best else ' '}")
        print(f"  {det:14s}" + "".join(cells))
    print("\n  (* = best detector for that type; expect block-diagonal)")
    print("\n== Per-item detection anchors ==")
    for k, v in res["detection"].items():
        print(f"  {k:24s} {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_matrix.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
