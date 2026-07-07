#!/usr/bin/env python3
"""轴1 provenance evaluation — full baselines (sklearn + networkx).

Upgrades the stdlib pilot script with credible, standard baselines and metrics:
  content-only : LogisticRegression (T1, 5-fold CV) / KMeans (T2)
  graph-based  : connected components / Louvain / label propagation (networkx)
  metrics      : P/R/F1 (T1); NMI / ARI / B3-F1 (T2); item-level acc (T3)

Reports overall + severity breakdowns (by cluster size, by homogeneity) and
dumps results.json for the figure. LLM-as-detector is intentionally NOT run here
(see RESULTS.md 'walls'): the synthetic set carries no persuasive text — LLM
detection belongs to the ITW/real-text track (轴2) and needs an API key.

Usage: python3 provenance_eval.py --data ../02-data/pilot/dataset.json --out results.json
"""
import argparse
import json
from collections import defaultdict, Counter

import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_fscore_support,
                             normalized_mutual_info_score, adjusted_rand_score)


def bcubed_f(true, pred):
    n = len(true)
    if n == 0:
        return 0.0
    by_pred, by_true = defaultdict(list), defaultdict(list)
    for i, (t, p) in enumerate(zip(true, pred)):
        by_pred[p].append(i); by_true[t].append(i)
    prec = rec = 0.0
    for i in range(n):
        sp, st = by_pred[pred[i]], by_true[true[i]]
        both = sum(1 for j in sp if true[j] == true[i])
        prec += both / len(sp); rec += both / len(st)
    prec /= n; rec /= n
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def feature_matrix(items):
    return np.array([[it["features"]["hour"] / 23.0, it["features"]["stance"],
                      it["features"]["length"], it["features"]["prof_mismatch"]]
                     for it in items], dtype=float)


def build_graph(ds):
    G = nx.Graph()
    G.add_nodes_from(a["account_id"] for a in ds["accounts"])
    for u, v, w in ds["edges"]:
        G.add_edge(u, v, weight=w)
    return G


def account_labels(G, method):
    if method == "components":
        comms = nx.connected_components(G)
    elif method == "louvain":
        comms = nx.community.louvain_communities(G, seed=42)
    elif method == "labelprop":
        comms = nx.community.asyn_lpa_communities(G, weight="weight", seed=42)
    else:
        raise ValueError(method)
    lab = {}
    for cidx, nodes in enumerate(comms):
        for n in nodes:
            lab[n] = cidx
    return lab


def clustering_scores(true, pred):
    return {"nmi": round(normalized_mutual_info_score(true, pred), 3),
            "ari": round(adjusted_rand_score(true, pred), 3),
            "b3f": round(bcubed_f(list(true), list(pred)), 3)}


def run(ds):
    items = ds["content"]
    camp = {c["cluster_id"]: c for c in ds["campaigns"]}
    X = feature_matrix(items)
    Xs = StandardScaler().fit_transform(X)
    y = np.array([1 if it["is_manipulated"] else 0 for it in items])
    G = build_graph(ds)
    comp_size = Counter()
    graph_lab = {m: account_labels(G, m) for m in ("components", "louvain", "labelprop")}
    for n, l in graph_lab["components"].items():
        comp_size[l] += 1

    results = {"n_items": len(items), "n_manip": int(y.sum()), "t1": {}, "t2": {},
               "t3": {}, "by_size": {}, "by_homog": {}}

    # ---- T1 detection ----
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    yc = cross_val_predict(clf, Xs, y, cv=5)
    p, r, f, _ = precision_recall_fscore_support(y, yc, average="binary", zero_division=0)
    results["t1"]["content_logreg"] = {"P": round(p, 3), "R": round(r, 3), "F1": round(f, 3)}
    comp = graph_lab["components"]
    yg = np.array([1 if comp_size[comp[it["account_id"]]] >= 3 else 0 for it in items])
    p, r, f, _ = precision_recall_fscore_support(y, yg, average="binary", zero_division=0)
    results["t1"]["graph_components"] = {"P": round(p, 3), "R": round(r, 3), "F1": round(f, 3)}

    # ---- T2/T3 on manipulated items ----
    m_idx = [i for i, it in enumerate(items) if it["is_manipulated"]]
    m_true = [items[i]["cluster_id"] for i in m_idx]
    k = len(camp)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
    m_content = [int(km[i]) for i in m_idx]
    results["t2"]["content_kmeans"] = clustering_scores(m_true, m_content)
    graph_preds = {}
    for m in ("components", "louvain", "labelprop"):
        pred = [graph_lab[m][items[i]["account_id"]] for i in m_idx]
        graph_preds[m] = pred
        results["t2"][f"graph_{m}"] = clustering_scores(m_true, pred)

    # ---- hybrid: Louvain communities, re-merge graph-fragmented items by content ----
    # Finding-3 prediction: graph loses small clusters to partial observation; using
    # content-centroid proximity to fold fragments back into coordinated communities
    # should recover them without hurting large clusters.
    louv = graph_lab["louvain"]
    comm_acc = defaultdict(list)
    for acc, cm in louv.items():
        comm_acc[cm].append(acc)
    coord = {cm for cm, accs in comm_acc.items() if len(accs) >= 3}
    comm_items = defaultdict(list)
    for i, it in enumerate(items):
        comm_items[louv[it["account_id"]]].append(i)
    centroids = {cm: X[comm_items[cm]].mean(axis=0) for cm in coord if comm_items[cm]}

    def hybrid_label(i):
        cm = louv[items[i]["account_id"]]
        if cm in coord or not centroids:
            return cm
        xi = X[i]
        best, bd = None, 1e9
        for c2, ce in centroids.items():
            d = float(np.linalg.norm(xi - ce))
            if d < bd:
                bd, best = d, c2
        return best if best is not None and bd < 0.20 else cm

    m_hybrid = [hybrid_label(i) for i in m_idx]
    graph_preds["hybrid"] = m_hybrid
    results["t2"]["hybrid"] = clustering_scores(m_true, m_hybrid)

    # ---- T3 seller attribution ----
    seller_of = {it["cluster_id"]: it["seller_id"] for it in items if it["is_manipulated"]}

    def attribution(pred):
        # majority mapping: each predicted cluster takes its majority seller label.
        # Rewards over-segmentation (a singleton is always "correct") — kept for
        # reference only; the paper reports the Hungarian variant below.
        groups = defaultdict(list)
        for pr, tr in zip(pred, m_true):
            groups[pr].append(seller_of[tr])
        maj = {pr: Counter(s).most_common(1)[0][0] for pr, s in groups.items()}
        return round(sum(1 for pr, tr in zip(pred, m_true)
                         if maj[pr] == seller_of[tr]) / len(pred), 3)

    def attribution_hungarian(pred):
        # one-to-one mapping: Hungarian assignment on the pred-cluster x seller
        # contingency matrix; unmatched predicted clusters score zero, so
        # over-segmentation is penalized instead of rewarded.
        from scipy.optimize import linear_sum_assignment
        true_sellers = [seller_of[tr] for tr in m_true]
        ps = sorted(set(pred)); ss = sorted(set(true_sellers))
        pi = {p: i for i, p in enumerate(ps)}; si = {s: i for i, s in enumerate(ss)}
        cont = np.zeros((len(ps), len(ss)), dtype=int)
        for p, s in zip(pred, true_sellers):
            cont[pi[p], si[s]] += 1
        r, c = linear_sum_assignment(-cont)
        return round(cont[r, c].sum() / len(pred), 3)

    for key, pred in (("content_kmeans", m_content),
                      ("graph_louvain", graph_preds["louvain"]),
                      ("hybrid", m_hybrid)):
        results["t3"][key] = attribution(pred)
        results["t3_hungarian"] = results.get("t3_hungarian", {})
        results["t3_hungarian"][key] = attribution_hungarian(pred)

    # ---- severity: T2 by cluster size (NMI + B3 for the three graph + content) ----
    for s in sorted({c["size"] for c in camp.values()}):
        idx = [j for j, i in enumerate(m_idx) if camp[m_true[j]]["size"] == s]
        t = [m_true[j] for j in idx]
        cell = {"content_kmeans": clustering_scores(t, [m_content[j] for j in idx])}
        for m in ("components", "louvain", "labelprop", "hybrid"):
            cell[f"graph_{m}" if m != "hybrid" else "hybrid"] = \
                clustering_scores(t, [graph_preds[m][j] for j in idx])
        results["by_size"][str(s)] = {"n": len(idx), **cell}

    # ---- severity: content T1 recall by homogeneity ----
    for h in ("high", "med", "low"):
        hit = tot = 0
        for i in m_idx:
            if camp[items[i]["cluster_id"]]["homogeneity"] == h:
                tot += 1; hit += int(yc[i] == 1)
        results["by_homog"][h] = {"content_recall": round(hit / tot, 3), "n": tot}

    return results


def show(res):
    print(f"items={res['n_items']}  manipulated={res['n_manip']}\n")
    print("== T1 detection ==")
    for k, v in res["t1"].items():
        print(f"  {k:18s}  P={v['P']:.3f} R={v['R']:.3f} F1={v['F1']:.3f}")
    print("\n== T2 cluster recovery (NMI / ARI / B3-F1) ==")
    for k, v in res["t2"].items():
        print(f"  {k:18s}  NMI={v['nmi']:.3f} ARI={v['ari']:.3f} B3={v['b3f']:.3f}")
    print("\n== T3 seller attribution (item acc) ==")
    for k, v in res["t3"].items():
        print(f"  {k:18s}  acc={v:.3f}")
    print("\n== severity: T2 by cluster size (NMI) ==")
    hdr = ["content_kmeans", "graph_components", "graph_louvain", "graph_labelprop"]
    print("  size  " + "  ".join(f"{h.split('_')[-1]:>10s}" for h in hdr))
    for s, cell in res["by_size"].items():
        print(f"  {s:>4s}  " + "  ".join(f"{cell[h]['nmi']:>10.3f}" for h in hdr) + f"   (n={cell['n']})")
    print("\n== severity: content T1 recall by homogeneity ==")
    for h, v in res["by_homog"].items():
        print(f"  {h:4s}  recall={v['content_recall']:.3f}  (n={v['n']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
