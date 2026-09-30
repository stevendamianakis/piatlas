#!/usr/bin/env python3
"""
Annotated version of Figure fig:a3-umap (figures/fig_encoder_nmi_umap.pdf).

Same data as the published map: the steered F2LLM embeddings of the 4,663 curated templates,
PCA-50 then UMAP (regenerated in the July environment on 6308; cluster sizes identical to the
paper's table), and the ten k-means clusters, which are found without any labels. Color shows the
cluster's dominant source (three validated hues, gray for the smaller sources). Each cluster is
labeled with its number from Table cluster-composition, its source, and the constructions that set
it apart under the new PIAtlas labels. Drawn at print size for one IEEE column (3.5 in wide, the
published aspect ratio), so fonts appear at their nominal size.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "..", "paper_draft", "PIAtlas_final", "figures", "fig_encoder_nmi_umap.pdf")
xy = np.load(os.path.join(HERE, "map", "map_steer_pca50umap_xy.npy"))
lab = np.load(os.path.join(HERE, "map", "map_steer_pca50umap_kmeans.npy"))

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"
# cluster: color, bold head, construction line, text position (x, y), leader start(s) -> target(s)
CLUSTERS = {
    3: (BLUE, "#3 ASB", "command only", (-5.1, -5.0), [((-5.1, -3.3), (-4.95, 0.23))]),
    0: (BLUE, "#0 ASB", "+ override", (-5.2, 15.0), [((-5.2, 13.7), (-5.2, 10.08))]),
    6: (BLUE, "#6 ASB", "+ fake completion", (2.2, 13.4), [((1.2, 12.1), (-0.9, 8.93))]),
    4: (ORANGE, "#4 BrowseSafe", "authority, persona, pretext", (19.4, 11.6), [((16.3, 12.4), (13.11, 13.69))]),
    8: (ORANGE, "#8 BrowseSafe, PIArena", "fiction, pretext", (17.2, 20.6), [((14.0, 21.3), (8.69, 17.39)),
                                                                            ((14.8, 22.0), (12.65, 25.18))]),
    2: (GRAY, "#2 TaskTracker, SEP", "emphasized command", (2.4, -8.0), [((7.0, -7.0), (10.85, -6.38))]),
    5: (GRAY, "#5 InjecAgent", "plain command", (5.4, 5.6), [((5.4, 7.2), (5.38, 9.86))]),
    1: (AQUA, "#1 PIArena, CyberSecEval", "forced format, execution rules", (6.0, 1.4), [((9.9, 3.0), (13.13, 6.45))]),
    9: (AQUA, "#9 PIArena, AgentDojo", "authority, task embedding", (18.8, -2.4), [((16.6, -0.9), (15.35, 3.38))]),
    7: (AQUA, "#7 PIArena", "authority, rules, forced format", (19.1, 6.2), [((21.2, 4.9), (22.56, 2.86))]),
}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 6.5, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 3.5 * 328.2 / 371.4))
for c, (col, head, body, (tx, ty), leaders) in CLUSTERS.items():
    m = lab == c
    ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=col, alpha=0.75, linewidths=0, rasterized=True, zorder=2)
    for (lx, ly), (cx, cy) in leaders:
        ax.annotate("", xy=(cx, cy), xytext=(lx, ly), zorder=1,
                    arrowprops=dict(arrowstyle="-", color="#b8b7b2", lw=0.5, shrinkA=0, shrinkB=2.5))
    ax.text(tx, ty + 0.45, head, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK, zorder=3)
    ax.text(tx, ty + 0.35, body, ha="center", va="top", fontsize=6.0, color=INK2, zorder=3)
handles = [plt.Line2D([], [], marker="o", ls="", ms=4, mfc=col, mec="none", label=name)
           for col, name in ((BLUE, "ASB"), (ORANGE, "BrowseSafe"), (AQUA, "PIArena"), (GRAY, "other sources"))]
ax.legend(handles=handles, title="Dominant source", loc="upper left", frameon=False, fontsize=6.0,
          title_fontsize=6.0, handletextpad=0.2, borderaxespad=0.3, labelspacing=0.3, alignment="left")
ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values():
    s_.set_color("#d4d3cd"); s_.set_linewidth(0.6)
ax.set_xlim(-10.5, 25.3); ax.set_ylim(-10.3, 26.6)
fig.tight_layout(pad=0.15)
fig.savefig(OUT, dpi=600)
fig.savefig(os.path.join(HERE, "map", "fig_map_annotated_preview.png"), dpi=300)
print("wrote", os.path.normpath(OUT))
