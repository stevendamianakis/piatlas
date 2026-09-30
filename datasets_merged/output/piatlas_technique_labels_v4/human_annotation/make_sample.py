#!/usr/bin/env python3
"""
Verification sample for the first author's annotation (seed 2027).

300 records, 3.5% of the 8,663 labeled records, in the design of the paper's earlier verification
subset: 150 curated templates stratified by source (at least 2 per source, the rest proportional
to source size), 100 LLMail-Inject templates drawn uniformly, and 50 records oversampled from those
whose production labels carry a rare class (F3, F5, or F6), so that every class has support.

Writes sample.json (with the production labels, for scoring only), a blank
sheet_author_A.csv, and a blank 20-record sheet_practice_A.csv drawn from
outside the sample. A sheet that already holds annotations is never
overwritten.
"""
import csv, json, os, random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
SEED, N_CUR, N_LL, N_RARE, MIN_PER_SOURCE = 2027, 150, 100, 50, 2
NORM = {"Agentdojo": "AgentDojo", "BrowseSafeBench": "BrowseSafe", "PIArena-refined": "PIArena", "SEP_train": "SEP",
        "SEP_validation": "SEP", "AgentSafetyBench": "Agent-SafetyBench", "OpenPromptInjection": "Open-Prompt-Injection"}
CLASSES = ["F1", "F2", "F3", "F4", "F5", "F6"]
TECH_COL = "techniques (optional, e.g. F1.T2;F2.T4)"
PRACTICE_SEED, N_PRACTICE = 2028, 20


def load_records():
    cur = [r for r in json.load(open(os.path.join(V4, "piatlas_v4_curated_4663.json"))) if r["piatlas_status"] == "ok"]
    ll = [r for r in json.load(open(os.path.join(V4, "piatlas_v4_llmail_4000.json"))) if r["piatlas_status"] == "ok"]
    recs = {f"c{r['noll_pos']}": {"id": f"c{r['noll_pos']}", "partition": "curated",
                                  "source": NORM.get(r["sources"][0], r["sources"][0]), "template": r["template"],
                                  "production": r["piatlas"]} for r in cur}
    recs.update({f"l{r['llmail_pos']}": {"id": f"l{r['llmail_pos']}", "partition": "LLMail-Inject",
                                         "source": "LLMail-Inject", "template": r["template"], "production": r["piatlas"]}
                 for r in ll})
    return recs


def draw(recs, rng):
    """The sampling design: returns {record id: stratum} and the curated allocation per source."""
    # 1. curated, stratified by source
    by_src = defaultdict(list)
    for k, r in sorted(recs.items()):
        if r["partition"] == "curated":
            by_src[r["source"]].append(k)
    alloc = {s: min(MIN_PER_SOURCE, len(v)) for s, v in by_src.items()}
    rest, total = N_CUR - sum(alloc.values()), sum(len(v) for v in by_src.values())
    for s, v in by_src.items():
        alloc[s] = min(len(v), alloc[s] + int(round(rest * len(v) / total)))
    while sum(alloc.values()) > N_CUR:
        alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < N_CUR:
        alloc[max(alloc, key=alloc.get)] += 1
    stratum = {}
    for s in sorted(by_src):
        for k in rng.sample(by_src[s], alloc[s]):
            stratum[k] = "curated, stratified by source"
    # 2. LLMail-Inject, uniform
    for k in rng.sample(sorted(k for k, r in recs.items() if r["partition"] == "LLMail-Inject"), N_LL):
        stratum[k] = "LLMail-Inject, uniform"
    # 3. rare-class oversample
    for cls, n in (("F3", 17), ("F5", 17), ("F6", 16)):
        pool = sorted(k for k, r in recs.items() if k not in stratum and any(c.startswith(cls) for c in r["production"]))
        for k in rng.sample(pool, n):
            stratum[k] = f"oversampled ({cls} in production labels)"
    return stratum, alloc


def write_sheets(rows, names):
    """Blank annotator sheets; a sheet that already holds any annotation is never overwritten."""
    head = ["id", "template"] + CLASSES + [TECH_COL, "notes"]
    for name in names:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as fh:
                if any(any((r.get(c) or "").strip() for c in CLASSES + [TECH_COL, "notes"]) for r in csv.DictReader(fh)):
                    print(f"kept {name}: it already holds annotations")
                    continue
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(head)
            for r in rows:
                w.writerow([r["id"], r["template"]] + [""] * len(CLASSES) + ["", ""])


def main():
    rng = random.Random(SEED)
    recs = load_records()
    stratum, alloc = draw(recs, rng)
    order = sorted(stratum)
    rng.shuffle(order)
    sample = [dict(recs[k], stratum=stratum[k]) for k in order]
    json.dump(sample, open(os.path.join(HERE, "sample.json"), "w"), ensure_ascii=False, indent=1)

    # annotator sheets: no labels, no source, no strata (annotators stay blind to the LLM, the dataset, and oversampling)
    write_sheets(sample, ("sheet_author_A.csv",))
    # practice round: 20 records outside the sample (10 curated, 10 LLMail-Inject), same blind format
    prng = random.Random(PRACTICE_SEED)
    practice = []
    for part in ("curated", "LLMail-Inject"):
        pool = sorted(k for k, r in recs.items() if r["partition"] == part and k not in stratum)
        practice += [{"id": k, "template": recs[k]["template"]} for k in prng.sample(pool, N_PRACTICE // 2)]
    prng.shuffle(practice)
    json.dump([r["id"] for r in practice], open(os.path.join(HERE, "practice_ids.json"), "w"))
    write_sheets(practice, ("sheet_practice_A.csv",))
    from collections import Counter
    print(f"sample: {len(sample)} records | strata {dict(Counter(r['stratum'] for r in sample))}")
    print(f"curated allocation: {dict(sorted(alloc.items()))}")
    print("production class support:", {c: sum(any(x.startswith(c) for x in r['production']) for r in sample) for c in CLASSES})


if __name__ == "__main__":
    main()
