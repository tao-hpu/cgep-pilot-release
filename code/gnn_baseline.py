#!/usr/bin/env python3
"""轴1 provenance — GNN-on-graph baseline (PyG, pure-graph, no API key).

Adds a learned graph-representation baseline alongside the existing unsupervised
provenance baselines (KMeans / connected-components / Louvain / label-prop). Runs on
the SAME dataset and the SAME evaluation protocol (T2 cluster recovery NMI/ARI/B3-F1
on manipulated items, T3 seller attribution item-accuracy) — metric functions are
imported from provenance_eval so scores are bit-identical.

Models
  feat_kmeans      : KMeans on the 12-d node features, NO graph (ablation isolating
                     "does the graph help beyond richer features").
  gae_kmeans       : unsupervised Graph Auto-Encoder (2-layer GCN encoder, inner-product
                     decoder, reconstruct the account interaction graph) -> embedding
                     -> KMeans. Directly comparable to Louvain (both unsupervised, both
                     read only the interaction graph + no labels).
  dgi_kmeans       : unsupervised Deep Graph Infomax (contrastive) -> embedding -> KMeans.
  gcn_supervised   : GCN node classifier over campaign labels, 50/50 split, test-item
                     accuracy. USES GROUND-TRUTH LABELS -> a supervised upper bound, NOT
                     comparable to the unsupervised baselines. Reported separately.

Node = account (1 content item per account). Features = content 4-d
(hour/stance/length/prof_mismatch) + framework_sig 6-d + host_authority + page_quality
(12-d), standardized. Edges = ds['edges'] (account interaction graph, undirected).

Usage: python3 gnn_baseline.py --data ../02-data/pilot/dataset.json --out results_gnn.json
"""
import argparse
import json
import random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE, DeepGraphInfomax

# reuse the *exact* baseline metric + graph code so the comparison is apples-to-apples
from provenance_eval import clustering_scores, build_graph, account_labels

SEED = 42


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def node_features(items, acc_of_node):
    """12-d standardized feature per node (account). acc_of_node: node_idx -> content item."""
    rows = []
    for it in acc_of_node:
        f = it["features"]
        sig = it.get("framework_sig", [0.0] * 6)
        rows.append([f["hour"] / 23.0, f["stance"], f["length"], f["prof_mismatch"],
                     *sig, it.get("host_authority", 0.0), it.get("page_quality", 0.0)])
    return StandardScaler().fit_transform(np.asarray(rows, dtype=float))


def build_pyg(ds):
    """Return (Data, node_items, id2idx). node i <-> one account/content item."""
    # one content item per account (generator posts exactly 1 item per account)
    item_of_acc = {c["account_id"]: c for c in ds["content"]}
    accounts = [a["account_id"] for a in ds["accounts"]]
    id2idx = {aid: i for i, aid in enumerate(accounts)}
    node_items = [item_of_acc[aid] for aid in accounts]
    X = node_features(items=ds["content"], acc_of_node=node_items)
    src, dst = [], []
    for u, v, w in ds["edges"]:
        if u in id2idx and v in id2idx:
            a, b = id2idx[u], id2idx[v]
            src += [a, b]; dst += [b, a]                      # undirected
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=torch.tensor(X, dtype=torch.float), edge_index=edge_index, num_nodes=len(accounts))
    return data, node_items, id2idx


# ---------------- encoders ----------------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, out)

    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei))
        return self.c2(x, ei)


class DGIEncoder(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.conv = GCNConv(in_dim, hid)
        self.act = nn.PReLU(hid)

    def forward(self, x, ei):
        return self.act(self.conv(x, ei))


def corruption(x, ei):
    return x[torch.randperm(x.size(0))], ei


# ---------------- embedders ----------------
def train_gae(data, hid=32, out=16, epochs=200, lr=0.01):
    set_seed()
    model = GAE(GCNEncoder(data.x.size(1), hid, out))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model.encode(data.x, data.edge_index).cpu().numpy(), float(loss)


def train_dgi(data, hid=32, epochs=200, lr=0.01):
    set_seed()
    model = DeepGraphInfomax(
        hidden_channels=hid, encoder=DGIEncoder(data.x.size(1), hid),
        summary=lambda z, *a, **k: torch.sigmoid(z.mean(dim=0)), corruption=corruption)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        pos, neg, summ = model(data.x, data.edge_index)
        loss = model.loss(pos, neg, summ)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z, _, _ = model(data.x, data.edge_index)
        return z.cpu().numpy(), float(loss)


def kmeans_labels(Z, k):
    return KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(Z)


# ---------------- supervised GCN upper bound ----------------
def train_supervised(data, node_items, camp_ids, epochs=200, lr=0.01):
    """Node classification over campaign labels. Manip nodes only get real labels;
    organic -> a shared 'organic' class. 50/50 split on manip nodes; report test acc.
    Uses labels => upper bound, not comparable to unsupervised methods."""
    set_seed()
    cid2y = {cid: i for i, cid in enumerate(camp_ids)}
    organic_y = len(cid2y)
    y = torch.tensor([cid2y[it["cluster_id"]] if it["is_manipulated"] else organic_y
                      for it in node_items], dtype=torch.long)
    n_classes = organic_y + 1
    manip_nodes = np.array([i for i, it in enumerate(node_items) if it["is_manipulated"]])
    rng = np.random.RandomState(SEED); rng.shuffle(manip_nodes)
    half = len(manip_nodes) // 2
    train_idx = torch.tensor(manip_nodes[:half], dtype=torch.long)
    test_idx = torch.tensor(manip_nodes[half:], dtype=torch.long)

    model = GCNEncoder(data.x.size(1), 64, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        acc = float((pred[test_idx] == y[test_idx]).float().mean())
    return round(acc, 3), len(train_idx), len(test_idx)


def run(ds):
    set_seed()
    camp = {c["cluster_id"]: c for c in ds["campaigns"]}
    camp_ids = [c["cluster_id"] for c in ds["campaigns"]]
    k = len(camp)
    data, node_items, id2idx = build_pyg(ds)

    # evaluate PER CONTENT ITEM (matches provenance_eval): a content item inherits the
    # cluster label of its account node. Some accounts post 2 items, so #manip items
    # (4241) > #manip nodes; both items of a shared account get that node's label.
    manip_items = [it for it in ds["content"] if it["is_manipulated"]]
    m_node = [id2idx[it["account_id"]] for it in manip_items]      # node idx per manip item
    m_true = [it["cluster_id"] for it in manip_items]
    seller_of = {it["cluster_id"]: it["seller_id"] for it in ds["content"] if it["is_manipulated"]}

    def t3(labels_all):
        # majority mapping — rewards over-segmentation; reference only.
        pred = [labels_all[n] for n in m_node]
        groups = defaultdict(list)
        for p, t in zip(pred, m_true):
            groups[p].append(seller_of[t])
        maj = {p: Counter(s).most_common(1)[0][0] for p, s in groups.items()}
        return round(sum(1 for p, t in zip(pred, m_true) if maj[p] == seller_of[t]) / len(pred), 3)

    def t3_hungarian(labels_all):
        # one-to-one Hungarian mapping on the pred-cluster x seller contingency;
        # unmatched predicted clusters score zero. Paper headline metric.
        import numpy as _np
        from scipy.optimize import linear_sum_assignment
        pred = [labels_all[n] for n in m_node]
        true_sellers = [seller_of[t] for t in m_true]
        ps = sorted(set(pred)); ss = sorted(set(true_sellers))
        pi = {p: i for i, p in enumerate(ps)}; si = {s: i for i, s in enumerate(ss)}
        cont = _np.zeros((len(ps), len(ss)), dtype=int)
        for p, s in zip(pred, true_sellers):
            cont[pi[p], si[s]] += 1
        r, c = linear_sum_assignment(-cont)
        return round(cont[r, c].sum() / len(pred), 3)

    def eval_clustering(labels_all):
        pred = [int(labels_all[n]) for n in m_node]
        sc = clustering_scores(m_true, pred)
        sc_by_size = {}
        for s in sorted({camp[c]["size"] for c in camp}):
            idx = [j for j, t in enumerate(m_true) if camp[t]["size"] == s]
            sc_by_size[str(s)] = {"n": len(idx),
                                  **clustering_scores([m_true[j] for j in idx], [pred[j] for j in idx])}
        return sc, sc_by_size

    results = {"n_nodes": data.num_nodes, "n_edges": int(data.edge_index.size(1) // 2),
               "n_manip": len(manip_items), "k": k, "features": "12-d (content4 + sig6 + host_auth + page_q)",
               "seed": SEED, "t2": {}, "t3": {}, "t3_hungarian": {}, "by_size": {}, "loss": {}}

    # ---- ablation: features only, no graph ----
    feat_lab = kmeans_labels(data.x.numpy(), k)
    sc, bs = eval_clustering(feat_lab)
    results["t2"]["feat_kmeans"] = sc; results["by_size"]["feat_kmeans"] = bs
    results["t3"]["feat_kmeans"] = t3(feat_lab); results["t3_hungarian"]["feat_kmeans"] = t3_hungarian(feat_lab)

    # ---- GAE (unsupervised, graph) ----
    z_gae, l_gae = train_gae(data)
    gae_lab = kmeans_labels(z_gae, k)
    sc, bs = eval_clustering(gae_lab)
    results["t2"]["gae_kmeans"] = sc; results["by_size"]["gae_kmeans"] = bs
    results["t3"]["gae_kmeans"] = t3(gae_lab); results["t3_hungarian"]["gae_kmeans"] = t3_hungarian(gae_lab); results["loss"]["gae"] = round(l_gae, 4)

    # ---- DGI (unsupervised, graph) ----
    z_dgi, l_dgi = train_dgi(data)
    dgi_lab = kmeans_labels(z_dgi, k)
    sc, bs = eval_clustering(dgi_lab)
    results["t2"]["dgi_kmeans"] = sc; results["by_size"]["dgi_kmeans"] = bs
    results["t3"]["dgi_kmeans"] = t3(dgi_lab); results["t3_hungarian"]["dgi_kmeans"] = t3_hungarian(dgi_lab); results["loss"]["dgi"] = round(l_dgi, 4)

    # ---- reference: Louvain on the same interaction graph (from provenance_eval) ----
    G = build_graph(ds)
    louv = account_labels(G, "louvain")
    louv_all = [louv[node_items[i]["account_id"]] for i in range(len(node_items))]
    sc, bs = eval_clustering(louv_all)
    results["t2"]["louvain_ref"] = sc; results["by_size"]["louvain_ref"] = bs
    results["t3"]["louvain_ref"] = t3(louv_all); results["t3_hungarian"]["louvain_ref"] = t3_hungarian(louv_all)

    # ---- supervised upper bound (uses labels) ----
    acc, ntr, nte = train_supervised(data, node_items, camp_ids)
    results["gcn_supervised"] = {"test_item_acc": acc, "n_train": ntr, "n_test": nte,
                                 "note": "uses ground-truth labels; upper bound, not comparable to unsupervised rows"}
    return results


def show(res):
    print(f"nodes={res['n_nodes']} edges={res['n_edges']} manip={res['n_manip']} k={res['k']}")
    print(f"features: {res['features']}  seed={res['seed']}\n")
    print("== T2 cluster recovery (NMI / ARI / B3-F1) — manipulated items ==")
    order = ["feat_kmeans", "gae_kmeans", "dgi_kmeans", "louvain_ref"]
    for name in order:
        v = res["t2"][name]
        print(f"  {name:14s}  NMI={v['nmi']:.3f} ARI={v['ari']:.3f} B3={v['b3f']:.3f}")
    print("\n== T3 seller attribution (item acc) ==")
    for name in order:
        print(f"  {name:14s}  acc={res['t3'][name]:.3f}")
    print("\n== T2 NMI by cluster size ==")
    print("  size  " + "  ".join(f"{n.split('_')[0]:>10s}" for n in order))
    for s in res["by_size"][order[0]]:
        if s == "n":
            continue
        print(f"  {s:>4s}  " + "  ".join(f"{res['by_size'][n][s]['nmi']:>10.3f}" for n in order))
    sup = res["gcn_supervised"]
    print(f"\n== supervised GCN upper bound (uses labels) ==")
    print(f"  test-item acc={sup['test_item_acc']:.3f}  (train={sup['n_train']} / test={sup['n_test']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_gnn.json")
    args = ap.parse_args()
    ds = json.loads(open(args.data, encoding="utf-8").read())
    res = run(ds)
    show(res)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
