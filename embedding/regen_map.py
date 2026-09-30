# Regenerate the paper's steered PCA-50 + UMAP map exactly as compute_full.py did (July environment),
# then k-means with k chosen by silhouette over [8, 40]; save coordinates and cluster labels.
import json, numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap, sklearn
RS = 42
def l2n(X): return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
emb = l2n(np.load("out_f2llm_steer_noll/emb_full.npy").astype(np.float32))
p50 = PCA(n_components=50, random_state=RS).fit_transform(emb)
xy = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric='euclidean', random_state=RS).fit_transform(p50)
best = None
for k in range(8, 41):
    lab = MiniBatchKMeans(n_clusters=k, random_state=RS, n_init=3, batch_size=1024).fit_predict(xy)
    s = silhouette_score(xy, lab, metric='cosine', sample_size=min(4000, len(xy)), random_state=RS)
    if best is None or s > best[0]: best = (s, k, lab)
np.save("map_steer_pca50umap_xy.npy", xy); np.save("map_steer_pca50umap_kmeans.npy", best[2])
print("umap", umap.__version__, "sklearn", sklearn.__version__, "k", best[1], "sizes", sorted(Counter(best[2].tolist()).values(), reverse=True), flush=True)
