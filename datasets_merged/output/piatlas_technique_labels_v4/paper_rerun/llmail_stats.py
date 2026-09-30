#!/usr/bin/env python3
"""
Finding 4 and the LLMail-Inject row of the benchmark matrix with the new labels.

Inputs: ../piatlas_v4_llmail_4000.json (new labels of the paper's 4,000-template LLMail-Inject
sample), ../piatlas_v4_curated_4663.json (new labels of the curated templates), and the paper's
old labels (pi_labels_llmail.json, keyed by position in the merged template set; pi_labels.json).
Writes llmail_stats.json and prints a markdown summary.
"""
import json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["F1", "F2", "F3", "F4", "F5", "F6"]
OLD_LLMAIL = os.path.join(HERE, "pi_labels_llmail.json")
OLD_CURATED = os.path.join(HERE, "pi_labels.json")


def cls(codes):
    return {c.upper().split(".")[0] for c in codes}


def tech(codes):
    return {".".join(c.upper().split(".")[:2]) for c in codes}


def prevalence(sets):
    return {c: round(100 * float(np.mean([c in s for s in sets])), 1) for c in CLASSES}


def main():
    from sklearn.metrics import cohen_kappa_score
    ll = [r for r in json.load(open(os.path.join(HERE, "..", "piatlas_v4_llmail_4000.json"))) if r["piatlas_status"] == "ok"]
    cur = [r for r in json.load(open(os.path.join(HERE, "..", "piatlas_v4_curated_4663.json"))) if r["piatlas_status"] == "ok"]
    old_ll = json.load(open(OLD_LLMAIL))
    old_cur = json.load(open(OLD_CURATED))
    new_ll = [cls(r["piatlas"]) for r in ll]
    old_ll_sets = [cls(old_ll.get(str(r["merged_index"]), [])) for r in ll]
    new_cur = [cls(r["piatlas"]) for r in cur]
    old_cur_sets = [cls(old_cur.get(str(r["noll_pos"]), [])) for r in cur]
    p = {"llmail_new": prevalence(new_ll), "llmail_old": prevalence(old_ll_sets),
         "curated_new": prevalence(new_cur), "curated_old": prevalence(old_cur_sets)}
    cov = {k: sum(v[c] >= 5 for c in CLASSES) for k, v in p.items()}
    kappa = {c: round(float(cohen_kappa_score([c in s for s in old_ll_sets], [c in s for s in new_ll])), 3) for c in CLASSES}
    ntech = [len(tech(r["piatlas"])) for r in ll]
    techs = Counter(t for r in ll for t in tech(r["piatlas"]))
    out = {"n_llmail": len(ll), "n_curated": len(cur), "prevalence": p, "coverage": cov, "llmail_old_new_kappa": kappa,
           "llmail_median_techniques": float(np.median(ntech)), "llmail_mean_techniques": round(float(np.mean(ntech)), 2),
           "llmail_technique_pct": {t: round(100 * n / len(ll), 1) for t, n in techs.most_common()}}
    json.dump(out, open(os.path.join(HERE, "llmail_stats.json"), "w"), indent=1)
    print(f"LLMail-Inject: {len(ll)} labeled; curated: {len(cur)}")
    print("| set | " + " | ".join(CLASSES) + " | Cov. |\n|---|" + "--:|" * 7)
    for k in ("llmail_old", "llmail_new", "curated_old", "curated_new"):
        print(f"| {k} | " + " | ".join(f"{p[k][c]}" for c in CLASSES) + f" | {cov[k]}/6 |")
    print("LLMail old-vs-new kappa per class:", kappa)
    print("LLMail techniques per template: median", out["llmail_median_techniques"], "mean", out["llmail_mean_techniques"])
    print("LLMail top techniques:", list(out["llmail_technique_pct"].items())[:12])


if __name__ == "__main__":
    main()
