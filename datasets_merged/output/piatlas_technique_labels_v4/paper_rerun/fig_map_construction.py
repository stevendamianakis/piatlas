#!/usr/bin/env python3
"""
Construction-first version of Figure fig:a3-umap (figures/fig_encoder_nmi_umap.pdf) and the rows of
Table cluster-composition.

Same map as before: the steered F2LLM embeddings of the 4,663 curated templates, PCA-50 then UMAP
(July environment on 6308), and the ten k-means clusters, found without labels. Everything drawn is
computed here from the production labels (piatlas_v4_curated_4663.json):

  signature  the techniques that more than half of a cluster's members carry (bold label);
  color      the class, other than instruction manipulation (F1), that more than half of the
             members carry (the most frequent one if several); blue when no class besides F1 does;
             gray for a mixed cluster, in which no technique reaches MIXED_BELOW percent of the
             members (only cluster #1: its most common technique covers 52%, against at least 89%
             in every other cluster, so any threshold between 55 and 85 gives the same result);
  source     the cluster's main sources, those holding at least 15% of its members (small label, with
             the cluster number of the table); a template counts for every benchmark it appears in.

Colors are the first four slots of the validated categorical palette (dataviz skill; all checks pass
in light mode, and aqua and yellow need the direct labels every cluster has) plus gray. Drawn at
print size for one IEEE column (3.5 in wide, the published aspect ratio).

Usage: python3 fig_map_construction.py [output.pdf]   (default: the paper's figure path)
Also writes map/cluster_signatures.json, which the table rows are generated from.
"""
import json, os, sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..", "..", "paper_draft", "PIAtlas_final",
                                                          "figures", "fig_encoder_nmi_umap.pdf")
NORM = {"Agentdojo": "AgentDojo", "BrowseSafeBench": "BrowseSafe", "PIArena-refined": "PIArena", "SEP_train": "SEP",
        "SEP_validation": "SEP", "AgentSafetyBench": "Agent-SafetyBench", "OpenPromptInjection": "Open-Prompt-Injection"}
C = ["F1", "F2", "F3", "F4", "F5", "F6"]
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"
# categorical slots in fixed order, assigned to classes by how many templates their clusters hold
FAMILY = {"F1": (BLUE, "F1 only"), "F4": (ORANGE, "F1 + F4"), "F2": (AQUA, "F1 + F2"), "F5": (YELLOW, "F1 + F5"),
          "mixed": (GRAY, "mixed")}
MIXED_BELOW = 60
# label layout: text position (x, y) and leader start -> cluster target(s), tuned to avoid collisions
LAYOUT = {
    3: ((-5.1, -5.0), [((-5.1, -3.3), (-4.95, 0.23))]),
    0: ((-5.2, 15.0), [((-5.2, 13.7), (-5.2, 10.08))]),
    6: ((2.2, 13.4), [((1.2, 12.1), (-0.9, 8.93))]),
    4: ((19.4, 11.6), [((16.3, 12.4), (13.11, 13.69))]),
    8: ((17.2, 20.6), [((14.0, 21.3), (8.69, 17.39)), ((14.8, 22.0), (12.65, 25.18))]),
    2: ((2.4, -8.0), [((7.0, -7.0), (10.85, -6.38))]),
    5: ((5.4, 5.6), [((5.4, 7.2), (5.38, 9.86))]),
    1: ((6.0, 1.4), [((9.9, 3.0), (13.13, 6.45))]),
    9: ((18.8, -2.4), [((16.6, -0.9), (15.35, 3.38))]),
    7: ((20.4, 7.4), [((22.0, 6.5), (22.56, 2.86))]),
}


def clusters():
    lab = np.load(os.path.join(HERE, "map", "map_steer_pca50umap_kmeans.npy"))
    cur = sorted(json.load(open(os.path.join(V4, "piatlas_v4_curated_4663.json"))), key=lambda r: r["noll_pos"])
    assert [r["noll_pos"] for r in cur] == list(range(len(lab)))
    T = [frozenset(".".join(c.upper().split(".")[:2]) for c in r["piatlas"]) for r in cur]
    M = [frozenset(NORM.get(x, x) for x in r["sources"]) for r in cur]  # every benchmark a template appears in
    out = {}
    for c in sorted(set(lab.tolist())):
        m = np.where(lab == c)[0]
        tp = Counter(t for i in m for t in T[i])
        sig = sorted(t for t, k in tp.items() if k > len(m) / 2)
        cls = {k: 100 * float(np.mean([any(t.startswith(k + ".") for t in T[i]) for i in m])) for k in C}
        major = [k for k in C[1:] if cls[k] > 50]
        top = 100 * max(tp.values()) / len(m)
        fam = "mixed" if top < MIXED_BELOW else (max(major, key=lambda k: cls[k]) if major else "F1")
        src = Counter(s for i in m for s in M[i])
        out[int(c)] = {"n": int(len(m)), "signature": sig,
                       "signature_share": {t: 100 * tp[t] / len(m) for t in sig},
                       "carry_all": 100 * float(np.mean([set(sig) <= T[i] for i in m])),
                       "class_share": cls, "family": fam, "top_technique_share": top,
                       "sources": [(s, 100 * k / len(m)) for s, k in src.most_common(3)]}
    return lab, out


def source_text(info):
    main = [s for s, p in info["sources"] if p >= 15][:2]
    return ", ".join(main) if main else info["sources"][0][0]


def table_rows(info):
    """LaTeX rows of Table cluster-composition: color swatch and number, dominant techniques, n, class
    shares (rounded half up, as before), and the sources holding at least 15% of the members."""
    swatch = {"F1": "mapblue", "F4": "maporange", "F2": "mapaqua", "F5": "mapyellow", "mixed": "mapgray"}
    order = ["F1", "F4", "F2", "F5", "mixed"]
    rows = []
    for fam in order:
        for c, d in sorted(((c, d) for c, d in info.items() if d["family"] == fam), key=lambda kv: (-kv[1]["n"], len(kv[1]["signature"]))):
            sig = "mixed" if fam == "mixed" else "+".join(d["signature"])
            cls = " & ".join(str(int(np.floor(d["class_share"][k] + 0.5))) for k in C)
            rows.append(f"\\mapkey{{{swatch[fam]}}}{c} & {sig} & {d['n']} & {cls} & {source_text(d)} \\\\")
    return "\n".join(rows)


def main():
    xy = np.load(os.path.join(HERE, "map", "map_steer_pca50umap_xy.npy"))
    lab, info = clusters()
    json.dump(info, open(os.path.join(HERE, "map", "cluster_signatures.json"), "w"), indent=1)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6.5, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 3.5 * 328.2 / 371.4))
    for c, ((tx, ty), leaders) in LAYOUT.items():
        d = info[c]
        col = FAMILY[d["family"]][0]
        m = lab == c
        ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=col, alpha=0.8, linewidths=0, rasterized=True, zorder=2)
        for (lx, ly), (cx, cy) in leaders:
            ax.annotate("", xy=(cx, cy), xytext=(lx, ly), zorder=1,
                        arrowprops=dict(arrowstyle="-", color="#b8b7b2", lw=0.5, shrinkA=0, shrinkB=2.5))
        head = "mixed" if d["family"] == "mixed" else " + ".join(d["signature"])
        if d["family"] != "mixed" and len(d["signature"]) > 3:
            half = (len(d["signature"]) + 1) // 2
            head = " + ".join(d["signature"][:half]) + " +\n" + " + ".join(d["signature"][half:])
        ax.text(tx, ty + 0.45, head, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK,
                zorder=3, linespacing=1.05)
        ax.text(tx, ty + 0.35, f"#{c} {source_text(d)}", ha="center", va="top", fontsize=5.6, color=INK2, zorder=3)
    order = ["F1", "F4", "F2", "F5", "mixed"]
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=FAMILY[k][0], mec="none", label=FAMILY[k][1])
               for k in order if any(d["family"] == k for d in info.values())]
    ax.legend(handles=handles, title="Classes most members carry", loc="upper left", frameon=False, fontsize=6.0,
              title_fontsize=6.0, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3, alignment="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#d4d3cd"); s_.set_linewidth(0.6)
    ax.set_xlim(-10.5, 25.3); ax.set_ylim(-10.3, 26.6)
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT, dpi=600)
    fig.savefig(os.path.join(HERE, "map", "fig_map_construction_preview.png"), dpi=300)
    print("wrote", os.path.normpath(OUT))
    print("\nTable cluster-composition rows (grouped by color, then size):")
    print(table_rows(info))
    for c, d in sorted(info.items(), key=lambda kv: -kv[1]["n"]):
        print(f"#{c} n={d['n']} family={d['family']} sig={'+'.join(d['signature'])} carry_all={d['carry_all']:.0f}% "
              f"sources={[(s, round(p)) for s, p in d['sources']]}")


if __name__ == "__main__":
    main()
