#!/usr/bin/env python3
"""Construct synthetic GEO manipulation campaigns — full 5-type taxonomy.

Each manipulation type encodes its coordination in a DIFFERENT substrate, so that a
detector on the wrong substrate fails (the benchmark's core thesis):
  ① account_matrix      -> account INTERACTION graph (ds['edges'])
  ② citation_stuffing   -> CITATION graph, self-citation ring (ds['citations'])
  ③ parasite_content    -> CONTENT duplication (tight framework_sig) + authority gap
  ④ hidden_injection    -> per-item DOM-carrier flag (trivial detection, easy anchor)
  ⑤ fake_review_cluster -> CONTENT fingerprint (moderate framework_sig similarity)

STRUCTURE + feature vectors only, no persuasive text (双重用途红线). Ground-truth
(cluster_id/seller_id, ring membership, 同源组, carrier type) is free by construction.

Usage: python3 generate_synthetic.py --out ../../02-data/pilot/dataset.json [--per-cell 6]
"""
import argparse
import json
import random
from pathlib import Path

CLUSTER_SIZES = [3, 10, 30]
HOMOGENEITY = {"high": 0.04, "med": 0.15, "low": 0.34}
PLATFORMS = ["zhihu", "xiaohongshu", "baijiahao", "csdn", "toutiao"]
TYPES = ["account_matrix", "citation_stuffing", "parasite_content",
         "hidden_injection", "fake_review_cluster"]
SIG_DIM = 6


def clamp(x):
    return max(0.0, min(1.0, x))


def rand_sig(rng):
    return [round(rng.uniform(0, 1), 3) for _ in range(SIG_DIM)]


def base_item(aid, cid, seller, mtype, topic, rng, sig, hour=None, stance=0.5,
              length=0.5, prof=0.3, host_auth=0.3, page_q=0.5, dom=False):
    return {"account_id": aid, "cluster_id": cid, "seller_id": seller,
            "is_manipulated": mtype is not None, "mtype": mtype, "topic_id": topic,
            "platform": rng.choice(PLATFORMS),
            "features": {"hour": rng.randint(0, 23) if hour is None else hour,
                         "stance": clamp(stance), "length": clamp(length),
                         "prof_mismatch": clamp(prof)},
            "framework_sig": sig, "host_authority": round(host_auth, 3),
            "page_quality": round(page_q, 3), "dom_carrier": dom}


def gen(mtype, cid, seller, topic, size, homog, rng, acc_ctr):
    """Return (accounts, content, interaction_edges, citation_edges)."""
    sd = HOMOGENEITY[homog]
    accounts, content, edges, cites = [], [], [], []
    aids = []
    for _ in range(size):
        aid = acc_ctr[0]; acc_ctr[0] += 1
        accounts.append({"account_id": aid, "cluster_id": cid, "seller_id": seller})
        aids.append(aid)

    if mtype == "account_matrix":
        base_hour = rng.randint(8, 22)
        t_stance, t_prof = rng.uniform(0.72, 0.95), rng.uniform(0.6, 0.9)
        for aid in aids:
            content.append(base_item(aid, cid, seller, mtype, topic, rng, rand_sig(rng),
                                      hour=max(0, min(23, int(rng.gauss(base_hour, 1)))),
                                      stance=rng.gauss(t_stance, sd), prof=rng.gauss(t_prof, sd)))
        p = 0.6 if size <= 10 else 0.28
        for i in range(len(aids)):
            for j in range(i + 1, len(aids)):
                if rng.random() < p and rng.random() < 0.55:
                    edges.append([aids[i], aids[j], round(rng.uniform(0.5, 1.0), 2)])

    elif mtype == "citation_stuffing":
        cids_local = []
        for aid in aids:
            it = base_item(aid, cid, seller, mtype, topic, rng, rand_sig(rng),
                           length=rng.gauss(0.8, sd))
            content.append(it); cids_local.append(it)         # content_id filled later
        # self-citation ring: dense mutual citations among the fake docs (recorded by
        # account_id here; remapped to content_id after global id assignment)
        for i in range(size):
            cites.append(("ring", aids[i], aids[(i + 1) % size]))   # cycle
            for j in range(size):
                if i != j and rng.random() < (0.5 if size <= 10 else 0.2):
                    cites.append(("ring", aids[i], aids[j]))

    elif mtype == "parasite_content":
        tmpl = [rng.uniform(0, 1) for _ in range(SIG_DIM)]     # duplicated content
        for aid in aids:
            content.append(base_item(
                aid, cid, seller, mtype, topic, rng,
                [round(clamp(rng.gauss(tmpl[d], sd * 0.5)), 3) for d in range(SIG_DIM)],
                host_auth=rng.uniform(0.7, 0.95),              # borrowed high authority
                page_q=rng.uniform(0.1, 0.35)))                # low actual quality

    elif mtype == "hidden_injection":
        for aid in aids:
            content.append(base_item(aid, cid, seller, mtype, topic, rng, rand_sig(rng),
                                     dom=True))                # DOM carrier present

    elif mtype == "fake_review_cluster":
        tmpl = [rng.uniform(0, 1) for _ in range(SIG_DIM)]
        t_stance = rng.uniform(0.72, 0.95)
        for aid in aids:
            content.append(base_item(
                aid, cid, seller, mtype, topic, rng,
                [round(clamp(rng.gauss(tmpl[d], sd)), 3) for d in range(SIG_DIM)],
                stance=rng.gauss(t_stance, sd), length=rng.gauss(0.6, sd), prof=rng.gauss(0.4, 0.2)))

    return accounts, content, edges, cites


def gen_organic(topic, size, rng, acc_ctr):
    accounts, content, cites = [], [], []
    aids = []
    for _ in range(size):
        aid = acc_ctr[0]; acc_ctr[0] += 1
        accounts.append({"account_id": aid, "cluster_id": None, "seller_id": None})
        aids.append(aid)
        content.append(base_item(aid, None, None, None, topic, rng, rand_sig(rng),
                                  stance=rng.gauss(0.5, 0.3), length=rng.uniform(0.1, 0.9),
                                  prof=rng.uniform(0, 0.5),
                                  host_auth=rng.uniform(0.2, 0.6), page_q=rng.uniform(0.4, 0.8)))
    for i in range(size):                                      # sparse incidental citations
        if rng.random() < 0.3:
            cites.append(("ext", aids[i], aids[rng.randrange(size)]))
    return accounts, content, cites


def build(per_cell, seed, types=None):
    types = types or TYPES
    rng = random.Random(seed)
    acc_ctr = [0]
    accounts, content, edges, cites, campaigns = [], [], [], [], []
    cid = seller = topic = 0
    acc_content = {}       # account_id -> content_id (each account posts 1 item here)
    for size in CLUSTER_SIZES:
        for homog in HOMOGENEITY:
            for _ in range(per_cell):
                for mtype in types:
                    ma, mc, me, mci = gen(mtype, cid, seller, topic, size, homog, rng, acc_ctr)
                    oa, oc, oci = gen_organic(topic, size, rng, acc_ctr)
                    accounts += ma + oa; content += mc + oc; edges += me
                    cites += mci + oci
                    campaigns.append({"cluster_id": cid, "seller_id": seller, "topic_id": topic,
                                      "size": size, "homogeneity": homog, "mtype": mtype})
                    cid += 1; seller += 1; topic += 1
    for i, c in enumerate(content):
        c["content_id"] = i
        acc_content[c["account_id"]] = i
    # remap citation edges from account_id to content_id
    citations = [[acc_content[u], acc_content[v]] for _, u, v in cites
                 if u in acc_content and v in acc_content]
    return {"campaigns": campaigns, "accounts": accounts, "content": content,
            "edges": edges, "citations": citations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--types", default=None,
                    help="comma-separated subset of manipulation types (default: all 5)")
    args = ap.parse_args()
    types = args.types.split(",") if args.types else None
    ds = build(args.per_cell, args.seed, types)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ds, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    by = Counter(c["mtype"] for c in ds["content"] if c["mtype"])
    print(f"campaigns={len(ds['campaigns'])} accounts={len(ds['accounts'])} "
          f"content={len(ds['content'])} edges={len(ds['edges'])} citations={len(ds['citations'])}")
    print("  manipulated by type:", dict(by))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
