#!/usr/bin/env python3
"""CIB (coordinated-inauthentic-behavior) multi-view baseline for seller attribution.

Imports the classic CIB detection lens (Mannocci et al. survey, arXiv:2408.01257):
build one coordination network per behavioral view, blend them, run community
detection. Views available in the synthetic pilot:
  - interaction  : the account co-activity graph (ds['edges']) — the canonical CIB
                   artifact (co-retweet / co-URL). Included so this is a faithful
                   CIB detector, not a strawman that omits its primary view.
  - co-timing    : accounts posting in the same hour bucket
  - content-sim  : small Euclidean distance in framework_sig (shared template)
  - co-target    : same topic_id (same attacked entity)
The blended graph is a weighted union; communities via Louvain; clusters scored
per manipulated item with the SAME Hungarian attribution as provenance_eval.

Point of comparison: a general CIB detector that fuses all views is NOT
substrate-matched. On T1 (which coordinates in the interaction graph) it should
trail the native interaction-graph Louvain, and on a mixed T1/T5 stream it cannot
match a substrate-routed detector — the paper's thesis, tested against the most
relevant prior lens rather than a strawman.

Usage: python3 cib_baseline.py --data ../data/seed42_main.json --out results_cib.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from scipy.optimize import linear_sum_assignment

from provenance_eval import clustering_scores


def louvain_labels(G, n_items):
    lab = {}
    for ci, nodes in enumerate(nx.community.louvain_communities(G, seed=42)):
        for n in nodes:
            lab[n] = ci
    # isolated items get singleton labels
    return [lab.get(i, -i - 1) for i in range(n_items)]


def cib_graph(items, ds, w_inter=2.0, w_time=1.0, w_content=1.0, w_target=0.5,
              k=8, dist_thresh=0.35):
    """Blended multi-view coordination graph over content items (nodes = items)."""
    n = len(items)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # view 0: interaction / co-activity (the canonical CIB network)
    item_of_acc = {it["account_id"]: idx for idx, it in enumerate(items)}
    for u, v, w in ds.get("edges", []):
        if u in item_of_acc and v in item_of_acc:
            i, j = item_of_acc[u], item_of_acc[v]
            e = G.get_edge_data(i, j, {})
            G.add_edge(i, j, weight=e.get("weight", 0.0) + w_inter * float(w))

    # view 1: content-sim (framework_sig kNN, tight Euclidean neighbours)
    sig = np.array([it["framework_sig"] for it in items], dtype=float)
    A = kneighbors_graph(sig, n_neighbors=min(k, n - 1), mode="distance",
                         metric="euclidean", include_self=False).tocoo()
    for i, j, d in zip(A.row, A.col, A.data):
        if d < dist_thresh:
            G.add_edge(int(i), int(j),
                       weight=G.get_edge_data(int(i), int(j), {}).get("weight", 0.0)
                       + w_content / (1.0 + d))

    # view 2: co-timing (same hour bucket) — restricted within shared topic to
    # avoid an all-pairs blowup and match how CIB co-activity is scoped in practice
    by_key = defaultdict(list)
    for idx, it in enumerate(items):
        by_key[(it["features"]["hour"], it["topic_id"])].append(idx)
    for group in by_key.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                e = G.get_edge_data(i, j, {})
                G.add_edge(i, j, weight=e.get("weight", 0.0) + w_time)

    # view 3: co-target (same topic) — weak prior, links same-entity items
    by_topic = defaultdict(list)
    for idx, it in enumerate(items):
        by_topic[it["topic_id"]].append(idx)
    for group in by_topic.values():
        if len(group) > 40:            # skip pathological hubs
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                e = G.get_edge_data(i, j, {})
                G.add_edge(i, j, weight=e.get("weight", 0.0) + w_target)
    return G


def hungarian_attr(pred, true_sellers):
    ps = sorted(set(pred)); ss = sorted(set(true_sellers))
    pi = {p: i for i, p in enumerate(ps)}; si = {s: i for i, s in enumerate(ss)}
    cont = np.zeros((len(ps), len(ss)), dtype=int)
    for p, s in zip(pred, true_sellers):
        cont[pi[p], si[s]] += 1
    r, c = linear_sum_assignment(-cont)
    return round(cont[r, c].sum() / len(pred), 3)


def single_view_graphs(items, ds, k=8, dist_thresh=0.35):
    """Each CIB view as its own graph, for the view-selection config."""
    n = len(items)
    views = {}

    Gi = nx.Graph(); Gi.add_nodes_from(range(n))
    io = {it["account_id"]: idx for idx, it in enumerate(items)}
    for u, v, w in ds.get("edges", []):
        if u in io and v in io:
            Gi.add_edge(io[u], io[v], weight=float(w))
    views["interaction"] = Gi

    Gc = nx.Graph(); Gc.add_nodes_from(range(n))
    sig = np.array([it["framework_sig"] for it in items], dtype=float)
    A = kneighbors_graph(sig, n_neighbors=min(k, n - 1), mode="distance",
                         metric="euclidean", include_self=False).tocoo()
    for i, j, d in zip(A.row, A.col, A.data):
        if d < dist_thresh:
            Gc.add_edge(int(i), int(j), weight=1.0 / (1.0 + d))
    views["content"] = Gc
    return views


def run(ds):
    items = ds["content"]
    idx_of = {it["content_id"]: i for i, it in enumerate(items)}
    manip = [it for it in items if it["is_manipulated"]]
    m_idx = [idx_of[it["content_id"]] for it in manip]
    m_true = [it["cluster_id"] for it in manip]
    m_seller = [it["seller_id"] for it in manip]

    def score(lab_all):
        pred = [lab_all[i] for i in m_idx]
        sc = clustering_scores(m_true, pred)
        sc["attrib_hungarian"] = hungarian_attr(pred, m_seller)
        return sc

    # config A: generic equal-weight multi-view blend (naive CIB import)
    G = cib_graph(items, ds)
    blend = score(louvain_labels(G, len(items)))

    # config B: view selection — pick the single view with highest modularity,
    # i.e. CIB done substrate-aware. Reduces to the native substrate per technique.
    views = single_view_graphs(items, ds)
    best_name, best_mod, best_lab = None, -1.0, None
    for name, Gv in views.items():
        comms = nx.community.louvain_communities(Gv, seed=42)
        mod = nx.community.modularity(Gv, comms) if Gv.number_of_edges() else -1.0
        if mod > best_mod:
            lab = {n_: ci for ci, nodes in enumerate(comms) for n_ in nodes}
            best_name, best_mod = name, mod
            best_lab = [lab.get(i, -i - 1) for i in range(len(items))]
    selected = score(best_lab)
    selected["view"] = best_name

    return {"n_items": len(items), "n_manip": len(manip),
            "cib_multiview_blend": blend,
            "cib_view_selected": selected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_cib.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    print(json.dumps(res, indent=1))
    json.dump(res, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
