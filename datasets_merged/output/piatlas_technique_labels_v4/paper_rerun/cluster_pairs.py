# Table tab:cluster-pairs and the pair rates of Finding G3: how often two templates share a
# cluster, split by same/different technique set and same/different benchmark. Writes
# map/cluster_pairs.json.  Usage: python3 cluster_pairs.py
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(HERE))  # paths below are relative to piatlas_technique_labels_v4/
from collections import Counter, defaultdict
from sklearn.metrics import normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
SP = os.path.join(HERE, 'cluster_cache') + '/'
cur = sorted(json.load(open('piatlas_v4_curated_4663.json')), key=lambda r: r['noll_pos'])
NORM = {"Agentdojo": "AgentDojo", "BrowseSafeBench": "BrowseSafe", "PIArena-refined": "PIArena", "SEP_train": "SEP",
        "SEP_validation": "SEP", "AgentSafetyBench": "Agent-SafetyBench", "OpenPromptInjection": "Open-Prompt-Injection"}
T = [frozenset(".".join(c.upper().split(".")[:2]) for c in r['piatlas']) for r in cur]
M = [frozenset(NORM.get(x, x) for x in r['sources']) for r in cur]
ref = np.unique([str(sorted(t | {x.split('.')[0] for x in t})) for t in T], return_inverse=True)[1]
for e in ("f2llm_steer", "f2llm_plain", "qwen"):
    l = np.load(SP + f"{e}_noll_k-means_k33.npy")
    print(f"{e}: NMI {NMI(ref, l):.3f} ARI {ARI(ref, l):.3f}  (Table VII: steered 0.609/0.387, plain 0.490/0.137, Qwen 0.410/0.055)")
def rates(lab):
    keys = defaultdict(Counter)
    for t, m, c in zip(T, M, lab.tolist()): keys[(t, m)][c] += 1
    items = list(keys.items()); acc = defaultdict(lambda: [0, 0])
    for a in range(len(items)):
        (ta, ma), ca = items[a]; na = sum(ca.values())
        acc[(True, True)][0] += na*(na-1)//2; acc[(True, True)][1] += sum(v*(v-1)//2 for v in ca.values())
        for b in range(a+1, len(items)):
            (tb, mb), cb = items[b]; nb = sum(cb.values())
            k = (ta == tb, bool(ma & mb)); acc[k][0] += na*nb; acc[k][1] += sum(ca[c]*cb.get(c, 0) for c in ca)
    tot = [sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())]
    f = lambda k: 100*acc[k][1]/acc[k][0]
    return [f((True, True)), f((True, False)), f((False, True)), f((False, False)), 100*tot[1]/tot[0]], {str(k): v[0] for k, v in acc.items()}
res = {}
for name, path in (("map", 'paper_rerun/map/map_steer_pca50umap_kmeans.npy'), ("steered", SP+"f2llm_steer_noll_k-means_k33.npy"),
                   ("plain", SP+"f2llm_plain_noll_k-means_k33.npy"), ("qwen", SP+"qwen_noll_k-means_k33.npy")):
    r, pairs = rates(np.load(path)); res[name] = r
    print(f"{name:8s} same set/same bench {r[0]:.1f} | same set/other bench {r[1]:.1f} | other set/same bench {r[2]:.1f} | other/other {r[3]:.1f} | random {r[4]:.1f}")
print("pair counts:", pairs)
json.dump({"definition": "percent of template pairs in the same cluster; same benchmark = the two templates share at least one benchmark",
           "columns": ["same set, same benchmark", "same set, other benchmark", "other set, same benchmark", "other set, other benchmark", "random pair"],
           "rows": res, "pairs": pairs}, open('paper_rerun/map/cluster_pairs.json', 'w'), indent=1)
