# compute_full.py's 2-D sweep for the steered encoder (July environment): five reducers, each clustered
# by k-means with k chosen by silhouette over [8, 40]; saves the cluster labels of each reducer.
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
RS = 42
def l2n(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
def kmeans_auto(X, lo=8, hi=40):
    best = None
    for k in range(lo, hi + 1):
        lab = MiniBatchKMeans(n_clusters=k, random_state=RS, n_init=3, batch_size=1024).fit_predict(X)
        s = silhouette_score(X, lab, metric='cosine', sample_size=min(4000, len(X)), random_state=RS)
        if best is None or s > best[0]: best = (s, k, lab)
    return best[2]
out = {}
for name, d in [("plain", "out_f2llm_plain_noll"), ("steer", "out_f2llm_steer_noll"), ("qwen", "out_qwen_noll")]:
    emb = l2n(np.load(d + "/emb_full.npy").astype(np.float32))
    p50 = PCA(n_components=50, random_state=RS).fit_transform(emb)
    drs = {"PCA": lambda: PCA(2, random_state=RS).fit_transform(emb),
           "UMAP": lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=RS).fit_transform(emb),
           "PCA50-UMAP": lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='euclidean', random_state=RS).fit_transform(p50),
           "PCA50-tSNE": lambda: TSNE(n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=RS).fit_transform(p50),
           "tSNE": lambda: TSNE(n_components=2, perplexity=30, init='random', learning_rate='auto', metric='cosine', random_state=RS).fit_transform(emb)}
    for dr, fn in drs.items():
        out[f"{name}|{dr}"] = kmeans_auto(fn())
        print(name, dr, "k", len(set(out[f'{name}|{dr}'].tolist())), flush=True)
np.savez("dr_sweep_labels.npz", **out)
print("saved dr_sweep_labels.npz", flush=True)
