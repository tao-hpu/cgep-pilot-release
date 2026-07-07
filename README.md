# CGEP Synthetic Provenance Pilot

A fully synthetic, structure-only dataset and reproducible evaluation harness for the
account-level provenance experiments in *CGEP: Toward Detecting and Attributing GEO
Poisoning in Chinese AI Search* (Tao An, FIM Labs). This release covers the synthetic
pilot only. The full CGEP benchmark (controlled injection over real seed content) and
the in-the-wild evaluation are separate future releases.

## What this is

GEO poisoning is coordinated, inauthentic effort to bias the descriptions a generative
search engine produces about a target entity. The paper reframes defense against it as
three sub-tasks: **detection** (was this coordinated manipulation), **cluster recovery**
(which coordinated cluster does an item belong to), and **seller attribution** (trace an
item back to the actor cluster behind it). This pilot tests one prediction of that
framing: detecting manipulation and attributing it to an actor are separate problems
that need different signals.

Each of five manipulation techniques encodes its coordination in a different substrate,
so a detector reading the wrong substrate fails:

| Technique | Substrate that carries the coordination signal |
|-----------|------------------------------------------------|
| T1 account matrix | account interaction graph (`edges`) |
| T2 citation stuffing | citation graph, self-citation ring (`citations`) |
| T3 parasite content | content duplication (tight `framework_sig`) + host-authority gap |
| T4 hidden injection | per-item DOM carrier flag (`dom_carrier`) |
| T5 fake review cluster | content fingerprint (moderate `framework_sig` similarity) |

### Red lines this data respects

- **No text.** Items are structure and numeric feature vectors only. The dataset carries
  no persuasive copy, no attack prompts, and no manipulation-generation pipeline. This is
  a deliberate dual-use red line: the artifact supports building detectors, not attacks.
- **No real accounts or platform data.** Account graphs are fully fictitious. Their
  *structure* is parameter-calibrated to publicly reported coordination patterns, but no
  real identity, account, or scraped content is present. Ground-truth labels
  (cluster, seller, ring membership, carrier type) are free by construction, not human
  annotation.
- **No in-the-wild content.** Nothing here was collected from a live engine or website.

## Contents

```
release/
  README.md
  DATASHEET.md              datasheet-for-datasets description
  LICENSE                   CC-BY-4.0 (data) + MIT (code); scope noted below
  data/
    seed42_main.json        T1 only, per-cell 25  -> 6450 items (3225 manip / 3225 organic)
    seed42_twotype.json     T1 + T5, per-cell 15  -> 7740 items (3870 manip)
    seed42_fivetype.json    all five, per-cell 6  -> 7740 items (3870 manip)
  code/
    generate_synthetic.py         the generator (all three realizations)
    provenance_eval.py            detection / cluster recovery / attribution (main set)
    provenance_eval_multitype.py  substrate crossover (two-technique set)
    gnn_baseline.py               learned GNN vs. classical community detection
    fusion_eval.py                confidence-gated substrate fusion (five-technique set)
    taxonomy_matrix.py            substrate x technique detectability matrix
    routing_sep.py                authority-gap routing for content-substrate types
    multiseed.py                  5-seed sweep driver, aggregates mean/std
  results/
    results_multiseed.json   5-seed aggregate; the numeric anchor for all paper claims
```

`data/` holds the seed-42 realizations. `results/results_multiseed.json` holds the
mean and standard deviation over five dataset-generation seeds (42, 101, 202, 303, 404);
this is the source of truth for the numbers below.

### Data schema

Each `.json` file is a single object with five arrays:

- `campaigns` — one record per coordinated campaign: `cluster_id`, `seller_id`,
  `topic_id`, `size` (3, 10, or 30), `homogeneity` (high / med / low), `mtype`.
- `accounts` — `account_id` mapped to its `cluster_id` / `seller_id` (both `null` for
  organic accounts).
- `content` — one item per account: `is_manipulated`, `mtype`, `platform`,
  `features` (`hour`, `stance`, `length`, `prof_mismatch`), a 6-dim `framework_sig`
  vector, `host_authority`, `page_quality`, `dom_carrier`, plus `content_id`.
- `edges` — account interaction edges `[account_id, account_id, weight]` (the T1 substrate).
- `citations` — citation edges `[content_id, content_id]` (the T2 substrate).

Each campaign is paired with a same-topic organic control of equal size, so the detection
label is balanced by construction.

## Reproducing the numbers

Requires Python 3.9+ with `numpy`, `scikit-learn`, `networkx`, and `scipy`. The GNN
baseline (`gnn_baseline.py`) additionally needs `torch` and `torch-geometric`; the other
six scripts do not. Run all commands **from `code/`** (scripts resolve `../data/`
relatively, and `gnn_baseline.py` imports `provenance_eval` as a sibling module).

Evaluate the provided seed-42 data:

```bash
cd code
python3 provenance_eval.py            --data ../data/seed42_main.json    --out results.json
python3 provenance_eval_multitype.py  --data ../data/seed42_twotype.json --out results_multitype.json
python3 gnn_baseline.py               --data ../data/seed42_twotype.json --out results_gnn.json
python3 fusion_eval.py                --data ../data/seed42_fivetype.json --out results_fusion.json
python3 taxonomy_matrix.py            --data ../data/seed42_fivetype.json --out results_matrix.json
python3 routing_sep.py                --data ../data/seed42_fivetype.json --out results_routing.json
```

Regenerate the data from scratch (each realization is a different `--types` / `--per-cell`
setting of the same generator):

```bash
python3 generate_synthetic.py --out ../data/seed42_main.json     --per-cell 25 --seed 42 --types account_matrix
python3 generate_synthetic.py --out ../data/seed42_twotype.json  --per-cell 15 --seed 42 --types account_matrix,fake_review_cluster
python3 generate_synthetic.py --out ../data/seed42_fivetype.json --per-cell  6 --seed 42
```

Run the full five-seed sweep that produced `results/results_multiseed.json` (writes its
own `multiseed/` working directory under `code/`):

```bash
python3 multiseed.py --seeds 42,101,202,303,404
```

Dataset-generation seeds vary across the sweep; model-internal seeds (KMeans, Louvain,
GNN) are held fixed at 42, so the reported variance is over data realizations.

## Headline results

All figures are means over five seeds from `results/results_multiseed.json`. Standard
deviation is at most 0.012 on every number quoted below.

**Main set (T1, N=6450), the paper's empirical core.** Content features detect that
manipulation occurred but cannot attribute it to a seller cluster; the account interaction
graph recovers the provenance.

| Sub-task | Content-only | Account graph |
|----------|-------------|---------------|
| Detection (F1) | 0.934 | 0.965 |
| Cluster recovery (NMI) | 0.500 | 0.991 |
| Seller attribution (item acc.) | 0.19 | 0.96 |

Attribution accuracy uses a **one-to-one Hungarian mapping** between predicted clusters
and ground-truth sellers, with unmatched predicted clusters scoring zero. A majority-label
mapping is not used for headline numbers: like cluster purity, it rewards over-segmentation
(a singleton predicted cluster is always "correct"), so it inflates the score of a detector
that fragments. For the same reason ARI and B³-F1 are the primary cluster-recovery metrics
and purity is dropped.

The remaining scripts extend this to the taxonomy: each technique is detectable only on
its own substrate (`taxonomy_matrix.py`), a learned GNN does not beat classical Louvain on
the interaction graph (`gnn_baseline.py`), and a confidence-gated fusion of substrates with
authority-gap routing covers the full five-technique family with one detector
(`fusion_eval.py`, `routing_sep.py`).

### Scope and honesty about these numbers

The absolute scores are optimistic **by construction**: features are generated to encode
the manipulation signal, so these numbers are not a claim about real black-industry
separability. What the pilot establishes is *structural* — which signal recovers which
sub-task — not ecological validity. Establishing this on real data is the goal of the
in-the-wild axis, which this release does not cover.

## Ethics and dual use

Consistent with the paper's Ethics section:

- The dataset contains **no attack-generation pipeline and no persuasive text** — only
  structure and feature vectors. It supports building detectors, not attacks.
- All account graphs are **synthetic and fictitious**; no real account is labeled a
  "manipulator," avoiding the defamation and privacy hazards of labeling real identities.
- Maliciousness is defined strictly on **coordination and authenticity evidence**
  (provenance, clustering, timing, templating), never on sentiment. Genuine organic
  criticism is benign in this schema regardless of how negative it is. This is both a
  validity condition (otherwise the task collapses into sentiment classification) and an
  ethical red line (otherwise a detector could be used to suppress legitimate criticism).

## Citation

```bibtex
@misc{an2026cgep,
  title        = {{CGEP}: Toward Detecting and Attributing {GEO} Poisoning in Chinese AI Search},
  author       = {An, Tao},
  year         = {2026},
  howpublished = {arXiv preprint},
  note         = {FIM Labs, Singapore}
}
```

## License

Dual-licensed. The **data** in `data/` and `results/` is released under
**CC-BY-4.0**; the **code** in `code/` is released under the **MIT License**. See
`LICENSE` for both texts and the scope split.
