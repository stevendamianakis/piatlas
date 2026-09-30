#!/usr/bin/env python3
"""
Full 3-model cluster-recovery on the 4,663 non-LLMail template-separable records.
(A) N-dim space: each model x {k-means, HDBSCAN, Ward, GMM} -> internal metrics
    (NWCD, silhouette) + external agreement with PIAtlas class+technique labels
    (NMI/ARI/AMI) + dataset-NMI context control.
(B) 2-D: for each model x DR method, cluster the 2-D coords and score NMI vs labels,
    to pick the DR that best suits NMI; plot only that best (model, DR).
"""
import os, json, numpy as np
from sklearn.cluster import MiniBatchKMeans, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (normalized_mutual_info_score as NMI, adjusted_rand_score as ARI,
                             adjusted_mutual_info_score as AMI, silhouette_score)
import umap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RS = 42; N = 4663
def l2n(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)

pil = json.load(open("pi_labels.json")); codes = [list(pil.get(str(i), [])) for i in range(N)]
tech = lambda c: tuple(sorted(set('.'.join(x.split('.')[:2]) for x in c)))
cls  = lambda c: tuple(sorted(set(x.split('.')[0] for x in c)))
sub  = lambda c: tuple(sorted(set(c)))
def catify(seq):
    u = {}; return np.array([u.setdefault(s, len(u)) for s in seq])
y_tech, y_cls, y_sub = catify([tech(c) for c in codes]), catify([cls(c) for c in codes]), catify([sub(c) for c in codes])
y_ds = catify([tuple(sorted(d)) for d in json.load(open("datasets_noll.json"))])

def nwcd_cos(X, lab):
    m = lab != -1; Xn = l2n(X[m]); y = lab[m]
    if len(set(y.tolist())) < 2: return float('nan')
    g = Xn.mean(0); g /= max(np.linalg.norm(g), 1e-12); G = np.mean(1 - Xn @ g)
    nrm = lambda c: c / max(np.linalg.norm(c), 1e-12)
    ws = [np.mean(1 - Xn[y == k] @ nrm(Xn[y == k].mean(0))) for k in np.unique(y)]
    return float(np.mean(ws) / G)
def sil(X, lab, metric='cosine'):
    m = lab != -1
    if len(set(lab[m].tolist())) < 2: return float('nan')
    return float(silhouette_score(X[m], lab[m], metric=metric, sample_size=min(4000, int(m.sum())), random_state=RS))
def kmeans_auto(X, lo=8, hi=40):
    best = None
    for k in range(lo, hi + 1):
        lab = MiniBatchKMeans(n_clusters=k, random_state=RS, n_init=3, batch_size=1024).fit_predict(X)
        s = silhouette_score(X, lab, metric='cosine', sample_size=min(4000, len(X)), random_state=RS)
        if best is None or s > best[0]: best = (s, k, lab)
    return best[2]
def ward_auto(X, lo=8, hi=40):
    Z = linkage(X, method='ward'); best = None
    for k in range(lo, hi + 1):
        lab = fcluster(Z, k, 'maxclust') - 1
        s = silhouette_score(X, lab, metric='euclidean', sample_size=min(4000, len(X)), random_state=RS)
        if best is None or s > best[0]: best = (s, k, lab)
    return best[2]

MODELS = [("F2LLM-14B plain", "out_f2llm_plain_noll"),
          ("F2LLM-14B steered", "out_f2llm_steer_noll"),
          ("Qwen3-8B", "out_qwen_noll")]
ndim_rows = []; dr_scores = {}; dr_coords = {}; dr_clus = {}
for name, d in MODELS:
    emb = l2n(np.load(os.path.join(d, "emb_full.npy")).astype(np.float32))
    km = kmeans_auto(emb); kk = len(set(km.tolist()))
    clus = {"k-means": km,
            "HDBSCAN": HDBSCAN(min_cluster_size=15, metric='euclidean').fit_predict(emb),
            "Ward": ward_auto(emb),
            "GMM": GaussianMixture(n_components=kk, covariance_type='diag', random_state=RS, max_iter=100).fit_predict(emb)}
    for algo, lab in clus.items():
        lab = np.asarray(lab)
        nc = len(set(lab.tolist()) - {-1}); noise = 100.0 * float(np.mean(lab == -1))
        ndim_rows.append(dict(model=name, algo=algo, k=nc, noise=round(noise, 1),
                              nwcd=nwcd_cos(emb, lab), sil=sil(emb, lab),
                              nmi=float(NMI(y_tech, lab)), ari=float(ARI(y_tech, lab)),
                              ami=float(AMI(y_tech, lab)), ds=float(NMI(y_ds, lab))))
        print(f"{name:18s} {algo:8s} k={nc:2d} noise={noise:4.0f}% NWCD={ndim_rows[-1]['nwcd']:.3f} "
              f"sil={ndim_rows[-1]['sil']:.3f} NMI={ndim_rows[-1]['nmi']:.3f} ARI={ndim_rows[-1]['ari']:.3f} "
              f"AMI={ndim_rows[-1]['ami']:.3f} dsNMI={ndim_rows[-1]['ds']:.3f}", flush=True)
    # ---- 2D DR sweep, scored by NMI of a k-means clustering of the 2D coords ----
    p50 = PCA(n_components=50, random_state=RS).fit_transform(emb)
    drs = {"PCA": PCA(2, random_state=RS).fit_transform(emb),
           "UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=RS).fit_transform(emb),
           "PCA50-UMAP": umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='euclidean', random_state=RS).fit_transform(p50),
           "PCA50-tSNE": TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=RS).fit_transform(p50),
           "tSNE": TSNE(n_components=2, perplexity=30, init='random', learning_rate='auto', metric='cosine', random_state=RS).fit_transform(emb)}
    for drn, xy in drs.items():
        lab2 = kmeans_auto(xy)
        dr_scores[(name, drn)] = float(NMI(y_tech, lab2))
        dr_coords[(name, drn)] = xy; dr_clus[(name, drn)] = lab2
        print(f"    2D {drn:11s} NMI={dr_scores[(name,drn)]:.3f}", flush=True)

# best (model, DR) by 2D NMI
best_md = max(dr_scores, key=dr_scores.get)
print(f"\nBEST 2D by NMI: {best_md}  NMI={dr_scores[best_md]:.3f}", flush=True)

# ---- N-dim table (markdown) ----
L = ["| Encoder | Clusterer | k | NWCD↓ | sil↑ | **NMI** | ARI | AMI | dataset-NMI |",
     "|---|---|---|---|---|---|---|---|---|"]
for r in ndim_rows:
    nz = f"{r['k']} ({r['noise']:.0f}% n)" if r['noise'] > 0 else str(r['k'])
    L.append(f"| {r['model']} | {r['algo']} | {nz} | {r['nwcd']:.3f} | {r['sil']:.3f} | "
             f"{r['nmi']:.3f} | {r['ari']:.3f} | {r['ami']:.3f} | {r['ds']:.3f} |")
open("ndim_table.md", "w").write("\n".join(L) + "\n")
L2 = ["| Encoder | " + " | ".join(sorted(set(d for _, d in dr_scores))) + " |",
      "|" + "---|" * (1 + len(set(d for _, d in dr_scores)))]
drnames = ["PCA", "UMAP", "PCA50-UMAP", "PCA50-tSNE", "tSNE"]
L2 = ["| Encoder | " + " | ".join(drnames) + " |", "|" + "---|" * (1 + len(drnames))]
for name, _ in MODELS:
    L2.append("| " + name + " | " + " | ".join(f"{dr_scores[(name,drn)]:.3f}" for drn in drnames) + " |")
open("dr_nmi_table.md", "w").write("2D NMI (class+tech) by DR method:\n\n" + "\n".join(L2) + "\n")
json.dump({"ndim": ndim_rows, "dr_nmi": {f"{m}|{d}": v for (m, d), v in dr_scores.items()},
           "best_2d": {"model": best_md[0], "dr": best_md[1], "nmi": dr_scores[best_md]}},
          open("full_results.json", "w"), indent=2)

# ---- best-NMI 2D figure (clean, for paper) ----
xy = dr_coords[best_md]; lab = dr_clus[best_md]; k = len(set(lab.tolist()))
cols = plt.cm.tab20(np.linspace(0, 1, 20))
fig, ax = plt.subplots(figsize=(5.0, 4.4))
for c in range(k):
    m = lab == c
    ax.scatter(xy[m, 0], xy[m, 1], s=5, c=[cols[c % 20]], alpha=0.8, linewidths=0)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.15); fig.savefig("fig_best_nmi_2d.pdf", bbox_inches='tight')
# titled png for review
ax.set_title(f"{best_md[0]} + {best_md[1]} (k={k}) — class+tech NMI={dr_scores[best_md]:.3f}", fontsize=9)
fig.savefig("fig_best_nmi_2d.png", dpi=160, bbox_inches='tight')
print("wrote ndim_table.md, dr_nmi_table.md, full_results.json, fig_best_nmi_2d.{pdf,png}", flush=True)
