"""W1: LLM-as-detector baseline on serialized features (zero-shot, 6-way).

The synthetic set carries no persuasive text (dual-use red line), so the LLM
sees a serialized feature record per item — the same 12-d + platform/dom_carrier
view the classical baselines get, never the label fields (is_manipulated, mtype,
cluster_id, seller_id).

Reproduce:
    python3 llm_detector.py --env /path/to/.env --models gpt-4o,claude-sonnet-5
Raw responses checkpoint to results_llm_raw_<model>.jsonl (resume-safe);
aggregated metrics go to results_llm.json.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "seed42_fivetype.json"

LABELS = [
    "organic",
    "account_matrix",
    "citation_stuffing",
    "parasite_content",
    "hidden_injection",
    "fake_review_cluster",
]

SYSTEM_PROMPT = """You are auditing content items from a Chinese AI-search (GEO) ecosystem for coordinated manipulation. You see only a structured feature record per item (no text is available).

Feature glossary:
- platform: publishing platform (toutiao / zhihu / weibo / xhs / website).
- hour: posting hour of day (0-23).
- stance: promotional-stance score in [0,1]; higher = more one-sidedly promotional.
- length: normalized content length in [0,1].
- prof_mismatch: mismatch between the account's stated profile/expertise and the topic, in [0,1].
- framework_sig: 6-d rhetorical-framework signature; campaigns reusing one template have similar signatures.
- host_authority: authority of the hosting domain in [0,1].
- page_quality: intrinsic quality of the page in [0,1].
- dom_carrier: true if the page DOM carries content invisible to human readers but visible to crawlers.

Manipulation taxonomy (5 types, plus organic):
1. account_matrix — a matrix of sock-puppet accounts posts templated promotional content in coordinated bursts.
2. citation_stuffing — low-quality pages mass-manufacture citations to inflate a target's apparent source support.
3. parasite_content — low-quality promotional pages parasitize high-authority hosts to borrow their credibility (high host authority, low page quality).
4. hidden_injection — instructions/content hidden in the page DOM for AI crawlers, invisible to human readers.
5. fake_review_cluster — clusters of fake reviews/answers stamped from one template with a shared rhetorical fingerprint.

Classify the item. Respond with ONLY a JSON object:
{"label": "<one of: organic, account_matrix, citation_stuffing, parasite_content, hidden_injection, fake_review_cluster>", "manipulated": <true|false>}"""

LABEL_FIELDS = {"is_manipulated", "mtype", "cluster_id", "seller_id", "account_id", "topic_id", "content_id"}


def serialize(item):
    f = item["features"]
    return json.dumps(
        {
            "platform": item["platform"],
            "hour": f["hour"],
            "stance": round(f["stance"], 3),
            "length": round(f["length"], 3),
            "prof_mismatch": round(f["prof_mismatch"], 3),
            "framework_sig": [round(v, 3) for v in item["framework_sig"]],
            "host_authority": round(item["host_authority"], 3),
            "page_quality": round(item["page_quality"], 3),
            "dom_carrier": item["dom_carrier"],
        },
        ensure_ascii=False,
    )


def gold_label(item):
    return item["mtype"] if item["is_manipulated"] else "organic"


def stratified_sample(content, per_class, seed):
    rng = random.Random(seed)
    by_label = {}
    for it in content:
        by_label.setdefault(gold_label(it), []).append(it)
    sample = []
    for lab in LABELS:
        pool = sorted(by_label[lab], key=lambda x: x["content_id"])
        sample.extend(rng.sample(pool, per_class))
    rng.shuffle(sample)
    return sample


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def call_llm(base, key, model, record, retries=5):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Item: " + record},
        ],
        "max_tokens": 60,
    }
    # some newer models (claude-*) reject the deprecated `temperature` param
    if "claude" not in model.lower():
        body["temperature"] = 0
    for attempt in range(retries):
        try:
            r = requests.post(
                base.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=120,
            )
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError("HTTP %d" % r.status_code)
            r.raise_for_status()
            out = r.json()
            return out["choices"][0]["message"]["content"], out.get("usage", {})
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt + random.random())


def parse_response(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    lab = str(obj.get("label", "")).strip()
    return lab if lab in LABELS else None


def run_model(model, sample, base, key, workers):
    raw_path = HERE / ("results_llm_raw_%s.jsonl" % model.replace("/", "_"))
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["content_id"]] = rec
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}

    def work(item):
        text, usage = call_llm(base, key, model, serialize(item))
        return item, text, usage

    # up to 3 passes: items whose calls fail (even after per-call retries) are
    # simply absent from the checkpoint and get retried on the next pass
    for sweep in range(3):
        todo = [it for it in sample if it["content_id"] not in done]
        if not todo:
            break
        print("[%s] pass %d: %d cached, %d to run" % (model, sweep + 1, len(done), len(todo)))
        with open(raw_path, "a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, it) for it in todo]
            for i, fut in enumerate(as_completed(futs)):
                try:
                    item, text, usage = fut.result()
                except Exception as e:
                    print("[%s] item failed, will retry next pass: %s" % (model, e))
                    continue
                rec = {
                    "content_id": item["content_id"],
                    "gold": gold_label(item),
                    "pred": parse_response(text),
                    "raw": text,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["content_id"]] = rec
                for k in usage_tot:
                    usage_tot[k] += usage.get(k, 0)
                if (i + 1) % 50 == 0:
                    print("[%s] %d/%d" % (model, i + 1, len(todo)))
    missing = [it["content_id"] for it in sample if it["content_id"] not in done]
    if missing:
        raise RuntimeError("%s: %d items still missing after 3 passes" % (model, len(missing)))
    return [done[it["content_id"]] for it in sample], usage_tot


def metrics(records):
    n = len(records)
    parsed = [r for r in records if r["pred"] is not None]
    acc6 = sum(r["pred"] == r["gold"] for r in parsed) / n
    # binary detection: manipulated = any non-organic label; unparseable counts as wrong
    tp = sum(r["pred"] != "organic" and r["gold"] != "organic" for r in parsed)
    fp = sum(r["pred"] != "organic" and r["gold"] == "organic" for r in parsed)
    fn = sum(r["gold"] != "organic" for r in records) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    conf = {g: {p: 0 for p in LABELS + ["unparsed"]} for g in LABELS}
    for r in records:
        conf[r["gold"]][r["pred"] or "unparsed"] += 1
    per_class_f1 = {}
    for lab in LABELS:
        ctp = conf[lab][lab]
        cfp = sum(conf[g][lab] for g in LABELS if g != lab)
        cfn = sum(v for p, v in conf[lab].items() if p != lab)
        d = 2 * ctp + cfp + cfn
        per_class_f1[lab] = 2 * ctp / d if d else 0.0
    return {
        "n": n,
        "n_unparsed": n - len(parsed),
        "six_way_acc": round(acc6, 3),
        "six_way_macro_f1": round(sum(per_class_f1.values()) / len(per_class_f1), 3),
        "binary_precision": round(prec, 3),
        "binary_recall": round(rec, 3),
        "binary_f1": round(f1, 3),
        "per_class_f1": {k: round(v, 3) for k, v in per_class_f1.items()},
        "confusion": conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=".env", help="dotenv file with LLM_API_BASE and LLM_API_KEY")
    ap.add_argument("--models", default="gpt-4o,claude-sonnet-5")
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results_llm.json"))
    args = ap.parse_args()

    env = load_env(args.env)
    base, key = env["LLM_API_BASE"], env["LLM_API_KEY"]
    content = json.load(open(DATA))["content"]
    sample = stratified_sample(content, args.per_class, args.seed)
    print("sample: %d items (%d per class), seed=%d" % (len(sample), args.per_class, args.seed))

    results = {"sample_size": len(sample), "per_class": args.per_class, "seed": args.seed, "models": {}}
    for model in args.models.split(","):
        model = model.strip()
        records, usage = run_model(model, sample, base, key, args.workers)
        m = metrics(records)
        m["usage"] = usage
        results["models"][model] = m
        print("[%s] 6-way acc %.3f | binary F1 %.3f | unparsed %d" % (model, m["six_way_acc"], m["binary_f1"], m["n_unparsed"]))
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
