# Datasheet: CGEP Synthetic Provenance Pilot

Following the *Datasheets for Datasets* framework (Gebru et al., 2021). This datasheet
describes the synthetic pilot release only, not the full CGEP benchmark.

## Motivation

**For what purpose was the dataset created?**
To test one prediction of the CGEP task formulation: that detecting GEO manipulation and
attributing it to an actor cluster are separate problems requiring separate signals, and
that each manipulation technique hides its coordination in a different substrate (account
interaction graph, citation graph, or content similarity). The dataset provides
structure-and-feature realizations with ground-truth cluster, seller, and technique labels
so that detection, cluster recovery, and seller attribution can be measured without human
annotation or real-world data collection.

**Who created the dataset and for whom?**
Tao An, FIM Labs (Singapore), as the reproducibility artifact for the CGEP preprint.

**Who funded the creation of the dataset?**
FIM Labs.

## Composition

**What do the instances represent?**
Three realizations of a single synthetic generator. Each realization is one JSON object
containing four coupled entity types: coordinated **campaigns**, **accounts**, per-account
**content** items (structure plus a numeric feature vector, no text), and two relation
graphs — account **interaction edges** and content **citation edges**. Every manipulated
campaign is paired with an equal-size organic control on the same topic.

**How many instances are there?**

| File | Techniques | Campaigns | Content items | Manipulated | Organic |
|------|-----------|-----------|---------------|-------------|---------|
| `seed42_main.json` | T1 | 225 | 6450 | 3225 | 3225 |
| `seed42_twotype.json` | T1, T5 | 270 | 7740 | 3870 | 3870 |
| `seed42_fivetype.json` | T1–T5 | 270 | 7740 | 3870 | 3870 |

Each realization crosses cluster size {3, 10, 30} with template homogeneity {high, med,
low} (9 cells). `results/results_multiseed.json` aggregates all three realizations over
five dataset-generation seeds (42, 101, 202, 303, 404).

**What data does each instance consist of?**
Content items carry: `is_manipulated`, `mtype` (technique or null), `platform` (one of five
Chinese platform names used as a categorical only), `features` (`hour`, `stance`, `length`,
`prof_mismatch`), a 6-dimensional `framework_sig` vector, `host_authority`, `page_quality`,
`dom_carrier`, `content_id`, and identity fields (`account_id`, `cluster_id`, `seller_id`,
`topic_id`). No natural-language text of any kind is included.

**Is there a label or target associated with each instance?**
Yes, and all labels are free by construction: per-item `is_manipulated`, technique `mtype`,
and the `cluster_id` / `seller_id` mapping used for cluster recovery and attribution. The
citation ring membership and DOM-carrier flag are recorded directly.

**Is any information missing?**
By design there is no text, no images, no timestamps beyond a synthetic `hour` feature, and
no real-world identifiers. Organic accounts have `null` cluster and seller ids.

**Are there recommended data splits?**
The pilot uses cross-validated detection and whole-set clustering rather than a fixed
train/test split. Cluster recovery and attribution are scored per manipulated item. When
splitting, keep paired campaign/control twins on the same side.

**Does the dataset contain data that might be considered confidential, offensive, or
sensitive?**
No. There is no personal data, no real account, no scraped content, and no persuasive or
offensive text. Platform names are used only as categorical values.

**Does the dataset relate to people?**
No. Account nodes are fully fictitious. Their graph *structure* is calibrated to publicly
reported coordination patterns, but no node corresponds to a real person or account.

## Collection process

**How was the data collected?**
It was not collected. It is **generated** by `code/generate_synthetic.py`, a
seeded procedural generator. No internet data and no human annotation are involved. Each
manipulation technique is generated so that its coordination signal lives in a specific
substrate (interaction graph, citation ring, content-fingerprint similarity, host-authority
gap, or DOM carrier), with a paired organic control for every campaign.

**Over what timeframe was the data collected?**
Not applicable; the release realizations were generated in 2026 for the preprint.

**Were any ethical review processes conducted?**
The design follows the paper's Ethics and Dual-Use section: no attack-generation pipeline
is released, no real system is attacked, account graphs are fictitious, and labels are
grounded in coordination evidence rather than sentiment.

## Preprocessing / cleaning / labeling

**Was any preprocessing done?**
None beyond generation. Content ids are assigned after generation and citation edges are
remapped from account ids to content ids. There is no cleaning step because there is no
collected raw data.

**Is the raw generator available?**
Yes; `code/generate_synthetic.py` reproduces every realization deterministically from a
seed, and `code/multiseed.py` reproduces the full five-seed sweep.

## Uses

**What has the dataset been used for?**
The synthetic provenance pilot in the CGEP preprint: detection, cluster recovery, seller
attribution, a substrate-detectability matrix, a learned-GNN comparison, a multi-view CIB
baseline, a zero-shot LLM-as-detector baseline, confidence-gated substrate fusion, and
authority-gap routing.

**What (else) could the dataset be used for?**
Prototyping and unit-testing coordinated-inauthentic-behavior detectors, community-detection
and graph-clustering baselines, and multi-substrate fusion methods, in a setting where
ground truth is exact.

**Is there anything about the dataset that could cause harm or unfair treatment?**
The absolute detection and attribution scores are optimistic by construction and must not
be cited as real-world detectability. Because features are generated to encode the
manipulation signal, the dataset shows the *structure* of the problem, not ecological
validity. Do not use it to claim that real GEO manipulation is this separable.

**Are there tasks for which the dataset should not be used?**
It should not be used to train or benchmark an attack/manipulation generator (it contains
no attack content by design), nor as evidence about real accounts, platforms, or sellers.

## Distribution

**How is the dataset distributed?**
As this local release package (JSON data, Python code, aggregate results). It is a public,
desensitized split; the paper notes that full data is access-controlled.

**Under what license?**
Data (`data/`, `results/`) under CC-BY-4.0; code (`code/`) under MIT. See `LICENSE`.

## Maintenance

**Who maintains the dataset?**
Tao An, FIM Labs (`tony@fim.ai`).

**Will the dataset be updated?**
This pilot is a fixed artifact tied to the preprint. The full CGEP benchmark (controlled
injection over real seed content) and the in-the-wild evaluation are planned as separate
future releases and are not part of this package.

**How can others contribute or extend it?**
The generator is parameterized (`--types`, `--per-cell`, `--seed`); new realizations and
severity gradients can be produced deterministically and evaluated with the provided scripts.
