#!/usr/bin/env python3
"""轴1 signal-crossover experiment: interaction graph vs fingerprint graph, per type.

Thesis (see RESULTS.md Finding 5): coordination is always a graph, but ① account_matrix
lives in the ACCOUNT INTERACTION graph while ⑤ fake_review_cluster lives in the CONTENT
FINGERPRINT similarity graph. A single-signal detector covers only half the taxonomy.

Detectors (both = Louvain community detection, differing only in which graph):
  interaction : Louvain on the crawled account interaction graph
  fingerprint : Louvain on a kNN similarity graph over framework signatures

Reports T2 cluster recovery (NMI) for each detector on each manipulation type.
Expected crossover: interaction wins on ①, fingerprint wins on ⑤.

Usage: python3 provenance_eval_multitype.py --data ../02-data/pilot/dataset.json --out results_multitype.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def bcubed_f(true, pred):
    n = len(true)
    if n == 0:
        return 0.0
    bp, bt = defaultdict(list), defaultdict(list)
    for i, (t, p) in enumerate(zip(true, pred)):
        bp[p].append(i); bt[t].append(i)
    pr = rc = 0.0
    for i in range(n):
        sp, st = bp[pred[i]], bt[true[i]]
        both = sum(1 for j in sp if true[j] == true[i])
        pr += both / len(sp); rc += both / len(st)
    pr /= n; rc /= n
    return 2 * pr * rc / (pr + rc) if pr + rc else 0.0


def louvain_labels(G):
    lab = {}
    for ci, nodes in enumerate(nx.community.louvain_communities(G, seed=42)):
        for n in nodes:
            lab[n] = ci
    return lab


def interaction_graph_labels(ds):
    G = nx.Graph()
    G.add_nodes_from(a["account_id"] for a in ds["accounts"])
    for u, v, w in ds["edges"]:
        G.add_edge(u, v, weight=w)
    acc_lab = louvain_labels(G)
    size = defaultdict(set)
    for aid, cm in acc_lab.items():
        size[cm].add(aid)
    csize = {cm: len(a) for cm, a in size.items()}
    lab = {it["content_id"]: acc_lab[it["account_id"]] for it in ds["content"]}
    return lab, csize


def fingerprint_graph_labels(ds, k=8, dist_thresh=0.35):
    items = ds["content"]
    sig = np.array([it["framework_sig"] for it in items], dtype=float)
    # Euclidean, NOT cosine: signatures live in the positive orthant [0,1]^6 where
    # random vectors share high cosine by chance, swamping real clusters. A small
    # Euclidean distance is what actually marks a shared review framework.
    A = kneighbors_graph(sig, n_neighbors=min(k, len(items) - 1),
                         mode="distance", metric="euclidean", include_self=False)
    G = nx.Graph()
    G.add_nodes_from(range(len(items)))
    Acoo = A.tocoo()
    for i, j, d in zip(Acoo.row, Acoo.col, Acoo.data):
        if d < dist_thresh:                            # only tight-fingerprint neighbours
            G.add_edge(int(i), int(j), weight=float(1.0 / (1.0 + d)))
    node_lab = louvain_labels(G)
    csize = defaultdict(int)
    for n in range(len(items)):
        csize[node_lab.get(n, -n - 1)] += 1
    lab = {items[i]["content_id"]: node_lab.get(i, -i - 1) for i in range(len(items))}
    return lab, dict(csize)


def scores(true, pred):
    return {"nmi": round(normalized_mutual_info_score(true, pred), 3),
            "ari": round(adjusted_rand_score(true, pred), 3),
            "b3f": round(bcubed_f(list(true), list(pred)), 3)}


def run(ds):
    items = ds["content"]
    inter, inter_sz = interaction_graph_labels(ds)
    finger, finger_sz = fingerprint_graph_labels(ds)
    cid_by_content = {it["content_id"]: it["cluster_id"] for it in items}

    # type-aware fusion: per item, trust whichever graph puts it in a coordinated
    # community (>=3 members); interaction first, else fingerprint, else singleton.
    # namespace labels so I-communities and F-communities never collide.
    fusion = {}
    for it in items:
        c = it["content_id"]
        ic, fc = inter[c], finger[c]
        if inter_sz.get(ic, 0) >= 3:
            fusion[c] = f"I{ic}"
        elif finger_sz.get(fc, 0) >= 3:
            fusion[c] = f"F{fc}"
        else:
            fusion[c] = f"S{c}"

    res = {"by_type": {}, "pooled": {}}
    for mtype in ("account_matrix", "fake_review_cluster"):
        idx = [it["content_id"] for it in items if it["mtype"] == mtype]
        true = [cid_by_content[c] for c in idx]
        res["by_type"][mtype] = {
            "n": len(idx),
            "interaction": scores(true, [inter[c] for c in idx]),
            "fingerprint": scores(true, [finger[c] for c in idx]),
            "fusion": scores(true, [fusion[c] for c in idx]),
        }
    # pooled over BOTH types — where single-signal detectors are forced to face a
    # mixed stream and fusion should dominate.
    pidx = [it["content_id"] for it in items if it["mtype"]]
    ptrue = [cid_by_content[c] for c in pidx]
    res["pooled"] = {
        "n": len(pidx),
        "interaction": scores(ptrue, [inter[c] for c in pidx]),
        "fingerprint": scores(ptrue, [finger[c] for c in pidx]),
        "fusion": scores(ptrue, [fusion[c] for c in pidx]),
    }
    return res


def show(res):
    print("== Signal crossover: T2 cluster recovery (NMI / ARI / B3) ==\n")
    print("  (winner judged by ARI — chance-adjusted; NMI over-rewards singleton fragmentation)\n")
    print(f"  {'manipulation type':22s} {'detector':14s}   NMI    ARI    B3")
    for mtype, cell in res["by_type"].items():
        best = max(("interaction", "fingerprint", "fusion"), key=lambda d: cell[d]["ari"])
        for det in ("interaction", "fingerprint", "fusion"):
            s = cell[det]
            star = "  <-- best" if det == best else ""
            print(f"  {mtype:22s} {det:14s}  {s['nmi']:.3f}  {s['ari']:.3f}  {s['b3f']:.3f}{star}")
        print()
    p = res["pooled"]
    pbest = max(("interaction", "fingerprint", "fusion"), key=lambda d: p[d]["ari"])
    print(f"  {'POOLED (both types)':22s}")
    for det in ("interaction", "fingerprint", "fusion"):
        s = p[det]
        print(f"  {'':22s} {det:14s}  {s['nmi']:.3f}  {s['ari']:.3f}  {s['b3f']:.3f}"
              f"{'  <-- best' if det == pbest else ''}")
    print()
    am, fr = res["by_type"]["account_matrix"], res["by_type"]["fake_review_cluster"]
    crossover = (am["interaction"]["ari"] > am["fingerprint"]["ari"] and
                 fr["fingerprint"]["ari"] > fr["interaction"]["ari"])
    print("CROSSOVER CONFIRMED (by ARI):" if crossover else "No crossover (by ARI):")
    print(f"  interaction ARI: ①={am['interaction']['ari']:.3f}  ⑤={fr['interaction']['ari']:.3f}")
    print(f"  fingerprint ARI: ①={am['fingerprint']['ari']:.3f}  ⑤={fr['fingerprint']['ari']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_multitype.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
