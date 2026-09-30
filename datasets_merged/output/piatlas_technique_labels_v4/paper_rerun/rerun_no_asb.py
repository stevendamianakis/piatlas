#!/usr/bin/env python3
"""
Robustness check for the section-7 rerun: the same cached clusterings (fit on all 4,663, as in the
paper), scored on the records outside ASB only, with the old and the new class+technique sets.
ASB's 2,000 templates are five wrappers x 400 tool names, and its old labels vary in blocks of 50
(the July batch size), so this shows how much of the old-vs-new difference ASB carries.
Writes rerun_no_asb.json and appends a section to RERUN.md.
"""
import json, os
from collections import defaultdict
import numpy as np
import rerun_paper_experiments as R

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    old, ds, new_by_noll, idx = R.load_inputs()
    keep = np.array([i for i in idx if not any(s == "ASB" for s in ds[i])])
    y_old = R.catify([R.tech(old[i]) for i in keep])
    y_new = R.catify([R.tech(new_by_noll[i]) for i in keep])
    out = {"n": int(len(keep)), "rows": {}, "five_seed": {}}
    for name, f in R.ENCODERS:
        for algo in ("k-means", "Ward", "GMM", "HDBSCAN"):
            lab = np.load(os.path.join(HERE, "cluster_cache", f"{f}_{algo}" + (R.KTAG if algo != "HDBSCAN" else "") + ".npy"))[keep]
            out["rows"][f"{name}|{algo}"] = {"old": R.scores(y_old, lab), "new": R.scores(y_new, lab)}
        runs = defaultdict(list)
        for s in R.SEEDS5:
            lab = np.load(os.path.join(HERE, "cluster_cache", f"{f}_kmeans_ninit10_seed{s}{R.KTAG}.npy"))[keep]
            for v, y in (("old", y_old), ("new", y_new)):
                for m, x in R.scores(y, lab).items():
                    runs[f"{v}|{m}"].append(x)
        out["five_seed"][name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in runs.items()}
    json.dump(out, open(os.path.join(HERE, f"rerun_no_asb{R.KTAG}.json"), "w"), indent=1)
    L = ["", "## Robustness: section 7 without ASB", "",
         f"The same clusterings, scored on the {out['n']:,} labeled records outside ASB.", "",
         "| encoder | clusterer | NMI old | NMI new | ARI old | ARI new | AMI old | AMI new |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for key, r in out["rows"].items():
        e, a = key.split("|")
        L.append(f"| {e} | {a} | {r['old']['NMI']:.3f} | {r['new']['NMI']:.3f} | {r['old']['ARI']:.3f} | {r['new']['ARI']:.3f} "
                 f"| {r['old']['AMI']:.3f} | {r['new']['AMI']:.3f} |")
    L += ["", "Five k-means seeds without ASB, mean (sd):", "",
          "| encoder | NMI old | NMI new | ARI old | ARI new | AMI old | AMI new |", "|---|--:|--:|--:|--:|--:|--:|"]
    for name, f in out["five_seed"].items():
        g = lambda k: f"{f[k][0]:.3f} ({f[k][1]:.3f})"
        L.append(f"| {name} | {g('old|NMI')} | {g('new|NMI')} | {g('old|ARI')} | {g('new|ARI')} | {g('old|AMI')} | {g('new|AMI')} |")
    with open(os.path.join(HERE, f"RERUN{R.KTAG}.md"), "a") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
