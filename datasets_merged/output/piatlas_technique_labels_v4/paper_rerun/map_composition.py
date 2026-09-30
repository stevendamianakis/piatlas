#!/usr/bin/env python3
"""
The steered map of Figure fig:a3-umap and Table cluster-composition, with the new labels.

map/map_steer_pca50umap_kmeans.npy holds the ten k-means clusters of the steered PCA-50 + UMAP map,
regenerated in the July environment on 6308 (umap 0.5.12, sklearn 1.8.0); its cluster sizes equal
the paper's table, so the figure is unchanged and only the class shares move. map/dr_sweep_labels.npz
(if present) holds compute_full.py's 2-D sweep (five reducers x three encoders), scored here by the
2-D class+technique NMI that App. C uses to choose the reducer. Percentages round half up, as in the paper.
Writes map_composition.json and prints markdown.
"""
import json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = ["F1", "F2", "F3", "F4", "F5", "F6"]
NORM = {"Agentdojo": "AgentDojo", "BrowseSafeBench": "BrowseSafe", "PIArena-refined": "PIArena", "SEP_train": "SEP",
        "SEP_validation": "SEP", "AgentSafetyBench": "Agent-SafetyBench", "OpenPromptInjection": "Open-Prompt-Injection"}


def half_up(x):
    return int(np.floor(x + 0.5))


def main():
    from sklearn.metrics import normalized_mutual_info_score as NMI
    lab = np.load(os.path.join(HERE, "map", "map_steer_pca50umap_kmeans.npy"))
    old = json.load(open(os.path.join(HERE, "pi_labels.json")))
    new = {int(k): v["codes"] for k, v in json.load(open(os.path.join(HERE, "new_labels_noll.json"))).items() if v["status"] == "ok"}
    ds = json.load(open(os.path.join(HERE, "datasets_noll.json")))
    rows = []
    for c in sorted(set(lab.tolist()), key=lambda c: -int(np.sum(lab == c))):
        m = np.where(lab == c)[0]
        src = Counter(s for i in m for s in {NORM.get(x, x) for x in ds[i]})
        share = lambda L, idx: {k: half_up(100 * np.mean([k in {x.upper().split(".")[0] for x in L(i)} for i in idx])) for k in C}
        mn = [i for i in m if i in new]
        rows.append({"cluster": int(c), "n": int(len(m)), "old": share(lambda i: old[str(i)], m), "new": share(lambda i: new[i], mn),
                     "sources": [(s, half_up(100 * n / len(m))) for s, n in src.most_common(2)]})
    out = {"rows": rows}
    print("| # | n | " + " | ".join(C) + " | dominant sources |\n|--:|--:|" + "--:|" * 6 + "---|")
    for r in rows:
        print(f"| {r['cluster']} | {r['n']} | " + " | ".join(f"{r['old'][k]} → {r['new'][k]}" for k in C) + " | "
              + ", ".join(f"{s} ({p}%)" for s, p in r["sources"]) + " |")
    sweep = os.path.join(HERE, "map", "dr_sweep_labels.npz")
    if os.path.exists(sweep):
        z = np.load(sweep)
        keys = sorted(new)
        cat = lambda seq: np.unique([str(s) for s in seq], return_inverse=True)[1]
        y_old = cat([tuple(sorted({".".join(x.upper().split(".")[:2]) for x in old[str(i)]})) for i in range(4663)])
        y_new = cat([tuple(sorted({".".join(x.upper().split(".")[:2]) for x in new[i]})) for i in keys])
        out["dr_nmi"] = {k: {"old": float(NMI(y_old, z[k])), "new": float(NMI(y_new, z[k][keys])),
                             "k": int(len(set(z[k].tolist())))} for k in z.files}
        print("\n2-D class+technique NMI by (encoder, reducer), old → new:")
        for k, v in sorted(out["dr_nmi"].items(), key=lambda kv: -kv[1]["new"]):
            print(f"  {k:18s} k={v['k']:2d}  {v['old']:.3f} → {v['new']:.3f}")
    json.dump(out, open(os.path.join(HERE, "map_composition.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
