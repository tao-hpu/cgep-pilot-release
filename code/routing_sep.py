#!/usr/bin/env python3
"""③/⑤ routing separation: split the content stream by authority gap BEFORE clustering.

Both ③ parasite_content and ⑤ fake_review_cluster live on the CONTENT substrate, so
content-sim Louvain must separate them without a graph to lean on (RESULTS.md Finding 7:
content-sim ARI ③ 0.55 / ⑤ 0.36). sep_test.py tried a *soft* fix — append a scaled
authority-gap dimension to the signature — and got only +0.04 ARI, weight-sensitive,
turning negative at weight >=0.6 because the extra dim fragments ③'s own tight clusters.

This script tests the *routing* alternative flagged in RESULTS.md / PROJECT-TRACKER:
don't concatenate the gap into the feature space, ROUTE on it. Split items into a
high-gap branch (expected ③) and a low-gap branch (expected ⑤), run content-sim Louvain
INSIDE each branch on its own kNN graph, then merge the (namespaced) labels and score
the pooled ③+⑤ stream. Mechanism it could exploit: a global kNN over the mixed stream
spends an item's k neighbour slots partly on the other type; routing first gives each
item a denser same-type neighbourhood.

Three settings compared, same NMI/ARI/B3 as the rest of the pipeline:
  (a) baseline    : content-sim Louvain over the pooled ③+⑤ stream (no separation)
  (b) soft-append : baseline + scaled authority-gap dim (sep_test.py's approach)
  (c) routing     : split on gap, cluster each branch, merge

Threshold is chosen with principled 1-D methods (median / Otsu / GMM-2), NOT tuned to
the metric, and a sweep reports sensitivity.

Usage: python3 routing_sep.py --data ../02-data/pilot/dataset_5type.json --out results_routing.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import networkx as nx
from sklearn.neighbors import kneighbors_graph
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

TYPES = ["parasite_content", "fake_review_cluster"]     # the two content-substrate types
K = 8
DIST_THRESH = 0.35


# ------------------------------------------------------------------ metrics
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


def scores(true, pred):
    return {"nmi": round(normalized_mutual_info_score(true, pred), 3),
            "ari": round(adjusted_rand_score(true, pred), 3),
            "b3f": round(bcubed_f(list(true), list(pred)), 3)}


# ------------------------------------------------------------------ clustering primitive
def louvain(G):
    lab = {}
    for ci, nodes in enumerate(nx.community.louvain_communities(G, seed=42)):
        for n in nodes:
            lab[n] = ci
    return lab


def content_cluster(vecs, k=K, thresh=DIST_THRESH):
    """Louvain over a tight-neighbour kNN graph on `vecs`; returns list of int labels.

    Identical construction to fusion_eval.content_sub (euclidean kNN, distance-gated),
    so the baseline here matches the pipeline's content substrate exactly.
    """
    n = len(vecs)
    if n == 0:
        return []
    if n == 1:
        return [0]
    A = kneighbors_graph(vecs, n_neighbors=min(k, n - 1),
                         mode="distance", metric="euclidean", include_self=False).tocoo()
    G = nx.Graph(); G.add_nodes_from(range(n))
    for i, j, d in zip(A.row, A.col, A.data):
        if d < thresh:
            G.add_edge(int(i), int(j))
    lab = louvain(G)
    return [lab.get(i, -i - 1) for i in range(n)]


# ------------------------------------------------------------------ threshold pickers (no p-hacking)
def otsu_threshold(x, bins=256):
    """Classic Otsu: maximise between-class variance of a 1-D histogram."""
    x = np.asarray(x, dtype=float)
    if x.max() <= x.min():
        return float(x.min())
    hist, edges = np.histogram(x, bins=bins, range=(x.min(), x.max()))
    hist = hist.astype(float)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    mu0 = np.cumsum(p * centers) / np.clip(w0, 1e-12, None)
    mu_t = (p * centers).sum()
    mu1 = (mu_t - np.cumsum(p * centers)) / np.clip(w1, 1e-12, None)
    sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[np.nanargmax(sigma_b)])


def gmm_threshold(x):
    """Decision boundary of a 2-component 1-D Gaussian mixture (posterior crossover)."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    if x.max() <= x.min():
        return float(x.min())
    gm = GaussianMixture(n_components=2, random_state=42, n_init=3).fit(x)
    lo, hi = sorted(gm.means_.ravel())
    grid = np.linspace(lo, hi, 1001).reshape(-1, 1)
    pred = gm.predict(grid).ravel()
    # first index where the assigned component flips = boundary between the two means
    flip = np.where(pred[1:] != pred[:-1])[0]
    return float(grid[flip[0] + 1, 0]) if len(flip) else float((lo + hi) / 2)


# ------------------------------------------------------------------ label builders
def baseline_labels(items):
    sig = np.array([it["framework_sig"] for it in items], dtype=float)
    lab = content_cluster(sig)
    return {items[i]["content_id"]: lab[i] for i in range(len(items))}


def soft_append_labels(items, gap_weight):
    sig = np.array([it["framework_sig"] for it in items], dtype=float)
    gap = np.array([[max(0.0, it["host_authority"] - it["page_quality"]) * gap_weight]
                    for it in items])
    lab = content_cluster(np.hstack([sig, gap]))
    return {items[i]["content_id"]: lab[i] for i in range(len(items))}


def routing_labels(items, gap, tau):
    """Split on gap>=tau, cluster each branch on its own kNN graph, namespace + merge."""
    hi_idx = [i for i in range(len(items)) if gap[i] >= tau]
    lo_idx = [i for i in range(len(items)) if gap[i] < tau]
    out = {}
    for prefix, idx in (("H", hi_idx), ("L", lo_idx)):
        if not idx:
            continue
        sub = np.array([items[i]["framework_sig"] for i in idx], dtype=float)
        sub_lab = content_cluster(sub)
        for local, i in enumerate(idx):
            out[items[i]["content_id"]] = f"{prefix}{sub_lab[local]}"
    return out, len(hi_idx), len(lo_idx)


# ------------------------------------------------------------------ evaluation
def eval_all(items, labmap, cid_of):
    idx_all = [it["content_id"] for it in items]
    res = {"pooled": scores([cid_of[c] for c in idx_all], [labmap[c] for c in idx_all]),
           "by_type": {}}
    for t in TYPES:
        idx = [it["content_id"] for it in items if it["mtype"] == t]
        res["by_type"][t] = scores([cid_of[c] for c in idx], [labmap[c] for c in idx])
    return res


def routing_purity(items, gap, tau):
    """Fraction of each branch that is a single type — diagnostic of the router itself."""
    hi = [items[i]["mtype"] for i in range(len(items)) if gap[i] >= tau]
    lo = [items[i]["mtype"] for i in range(len(items)) if gap[i] < tau]
    def pure(branch, want):
        return round(sum(1 for m in branch if m == want) / len(branch), 3) if branch else None
    return {"tau": round(float(tau), 4),
            "hi_n": len(hi), "hi_frac_parasite": pure(hi, "parasite_content"),
            "lo_n": len(lo), "lo_frac_review": pure(lo, "fake_review_cluster")}


def run(ds):
    items = [it for it in ds["content"] if it["mtype"] in TYPES]
    cid_of = {it["content_id"]: it["cluster_id"] for it in items}
    gap = [max(0.0, it["host_authority"] - it["page_quality"]) for it in items]
    gap = np.array(gap)

    # (a) baseline
    base = eval_all(items, baseline_labels(items), cid_of)

    # (b) soft-append at two weights (sep_test used 0.3 best / 0.6 harmful)
    soft = {f"w{w}": eval_all(items, soft_append_labels(items, w), cid_of)
            for w in (0.3, 0.6)}

    # principled thresholds
    picks = {"median": float(np.median(gap)),
             "otsu": otsu_threshold(gap),
             "gmm": gmm_threshold(gap)}
    # a robust default: midpoint of the empty band between the two gap masses
    picks["midband"] = float((gap[gap > 0].min() + gap[gap == 0].max()) / 2) \
        if np.any(gap > 0) and np.any(gap == 0) else picks["otsu"]

    # (c) routing at each principled threshold
    routing = {}
    for name, tau in picks.items():
        lab, hn, ln = routing_labels(items, gap, tau)
        routing[name] = {"tau": round(float(tau), 4), "hi_n": hn, "lo_n": ln,
                         **eval_all(items, lab, cid_of)}

    # sensitivity sweep — stability of routing across the threshold, not a search for a peak
    sweep = []
    for tau in np.linspace(0.0, 0.7, 15):
        lab, hn, ln = routing_labels(items, gap, tau)
        r = eval_all(items, lab, cid_of)
        sweep.append({"tau": round(float(tau), 3), "hi_n": hn, "lo_n": ln,
                      "pooled_ari": r["pooled"]["ari"], "pooled_nmi": r["pooled"]["nmi"],
                      "pooled_b3f": r["pooled"]["b3f"]})

    purity = [routing_purity(items, gap, t) for t in
              (0.05, picks["otsu"], picks["gmm"], 0.2, 0.3)]

    return {"n_items": len(items),
            "n_by_type": {t: sum(1 for it in items if it["mtype"] == t) for t in TYPES},
            "baseline": base, "soft_append": soft, "thresholds": picks,
            "routing": routing, "sweep": sweep, "router_purity": purity}


# ------------------------------------------------------------------ reporting
def _row(name, cell):
    p = cell["pooled"]; a = cell["by_type"]["parasite_content"]; f = cell["by_type"]["fake_review_cluster"]
    return (f"  {name:24s}  {p['nmi']:.3f} {p['ari']:.3f} {p['b3f']:.3f}   "
            f"{a['ari']:.3f}   {f['ari']:.3f}")


def show(res):
    print(f"③/⑤ routing separation — pooled n={res['n_items']} "
          f"(③ {res['n_by_type']['parasite_content']} / ⑤ {res['n_by_type']['fake_review_cluster']})\n")
    print("  authority-gap thresholds picked (no metric tuning):")
    for k, v in res["thresholds"].items():
        print(f"    {k:8s} tau={v:.4f}")
    print()
    print(f"  {'setting':24s}  {'pooled NMI/ARI/B3':17s}   ③ARI   ⑤ARI")
    print(_row("(a) baseline (pooled)", res["baseline"]))
    print(_row("(b) soft-append w=0.3", res["soft_append"]["w0.3"]))
    print(_row("(b) soft-append w=0.6", res["soft_append"]["w0.6"]))
    for name, cell in res["routing"].items():
        print(_row(f"(c) routing/{name} t={cell['tau']:.2f}", cell))
    print("\n  routing sensitivity sweep (pooled):")
    print(f"    {'tau':>6s} {'hi_n':>6s} {'lo_n':>6s}   {'ARI':>6s} {'NMI':>6s} {'B3':>6s}")
    for s in res["sweep"]:
        print(f"    {s['tau']:6.3f} {s['hi_n']:6d} {s['lo_n']:6d}   "
              f"{s['pooled_ari']:6.3f} {s['pooled_nmi']:6.3f} {s['pooled_b3f']:6.3f}")
    print("\n  router purity (branch composition):")
    for r in res["router_purity"]:
        print(f"    tau={r['tau']:.3f}  hi_n={r['hi_n']:4d} frac③={r['hi_frac_parasite']}   "
              f"lo_n={r['lo_n']:4d} frac⑤={r['lo_frac_review']}")

    base_ari = res["baseline"]["pooled"]["ari"]
    best_route = max(res["routing"].values(), key=lambda c: c["pooled"]["ari"])
    print(f"\n  pooled ARI: baseline {base_ari:.3f}  ->  best routing {best_route['pooled']['ari']:.3f}  "
          f"(delta {best_route['pooled']['ari'] - base_ari:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_routing.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
