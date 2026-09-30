#!/usr/bin/env python3
"""
Rerun the label-dependent experiments of the paper with the new PIAtlas technique labels.

The paper (sections 6-7, App. C) labels the 4,663 curated templates with the July labels
(pi_labels.json, 31-technique labeling vocabulary). The new labels (current listing, 34 techniques)
cover the same templates (new_labels_noll.json, keyed by position in the 4,663). Every comparison
below is made on the same records: those whose new label has status ok.

Section 7 / App. C (cluster recovery). The twelve clusterings are recomputed exactly as in the
paper, on all 4,663 embeddings: k-means (MiniBatchKMeans, k=31, n_init=3, seed 42, batch 1024),
Ward (scipy, 31 clusters), a diagonal Gaussian mixture (31 components, seed 42), and HDBSCAN
(min_cluster_size 15); noise counts as its own cluster, as in the paper. Step 1 checks that the
old labels on all 4,663 reproduce the published table. Step 2 scores the same clusterings on the
labeled records with the old and the new class+technique sets (each distinct set is one
category, as in the paper), and with class sets. Step 3 repeats the five-seed k-means
(n_init=10, seeds 0, 1, 2, 3, 42). NWCD and silhouette do not depend on labels.

Section 6 (mapping and findings), on the same records with old and new labels: the
benchmark-by-class matrix (percent of a source's templates carrying each class; Cov. = classes in
at least 5%), the composition statistics of Finding 1, the class prevalence of Finding 2, and the
source coverage of Finding 3. Sources come from datasets_noll.json; a template listed under
several sources counts in each, as in the paper.

Inputs (same folder): pi_labels.json, datasets_noll.json, new_labels_noll.json ({position: {"codes":
[...], "status": ...}}; without it, new_labels.json keyed by whole_only_false position plus
eval_targets.json), and the three embedding files in EMB_DIR.
Writes rerun.json and RERUN.md.
"""
import json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.environ.get("EMB_DIR", os.path.join(HERE, "paper_text_emb"))
RS, N = 42, 4663
K = int(os.environ.get("K", 31))  # the paper fixes k to the number of techniques that occur in the labels
KTAG = "" if K == 31 else f"_k{K}"
ENCODERS = [("F2LLM-14B plain", "f2llm_plain_noll"), ("F2LLM-14B steered", "f2llm_steer_noll"), ("Qwen3-8B", "qwen_noll")]
PAPER = {  # App. C table (k, NWCD, sil, NMI, ARI, AMI) and the five-seed means of Finding G1
    ("F2LLM-14B plain", "k-means"): (31, .577, .083, .500, .121, .382), ("F2LLM-14B plain", "HDBSCAN"): (4, .642, .319, .183, .019, .136),
    ("F2LLM-14B plain", "Ward"): (31, .575, .088, .461, .083, .333), ("F2LLM-14B plain", "GMM"): (31, .561, .084, .501, .123, .381),
    ("F2LLM-14B steered", "k-means"): (31, .589, .130, .598, .255, .505), ("F2LLM-14B steered", "HDBSCAN"): (2, .784, .333, .100, .008, .080),
    ("F2LLM-14B steered", "Ward"): (31, .561, .089, .521, .137, .408), ("F2LLM-14B steered", "GMM"): (31, .499, .110, .594, .248, .497),
    ("Qwen3-8B", "k-means"): (31, .606, .092, .427, .072, .299), ("Qwen3-8B", "HDBSCAN"): (16, .365, .403, .115, -.008, .032),
    ("Qwen3-8B", "Ward"): (31, .599, .086, .442, .077, .310), ("Qwen3-8B", "GMM"): (31, .574, .088, .438, .076, .304)}
PAPER_FIVE_SEED = {"NMI": (0.583, 0.004), "ARI": (0.233, 0.015)}
SEEDS5 = [0, 1, 2, 3, 42]
CLASSES = ["F1", "F2", "F3", "F4", "F5", "F6"]
SOURCE_ORDER = ["ASB", "BrowseSafe", "TaskTracker", "SEP", "PIArena", "CyberSecEval", "InjecAgent", "AgentDojo",
                "Agent-SafetyBench", "Greshake", "Open-Prompt-Injection", "AgentDyn", "PromptInject"]


def l2n(X):
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def catify(seq):
    u = {}
    return np.array([u.setdefault(s, len(u)) for s in seq])


def tech(codes):
    return tuple(sorted({".".join(c.upper().split(".")[:2]) for c in codes}))


def cls(codes):
    return tuple(sorted({c.upper().split(".")[0] for c in codes}))


def source_name(s):
    s = s.replace("_train", "").replace("_validation", "").replace("_test", "")
    return {"Agentdojo": "AgentDojo", "AgentSafetyBench": "Agent-SafetyBench", "BrowseSafeBench": "BrowseSafe", "OpenPromptInjection": "Open-Prompt-Injection",
            "PIArena-refined": "PIArena", "CyberSecEval2": "CyberSecEval", "CyberSecEval3": "CyberSecEval"}.get(s, s)


def load_inputs():
    old = json.load(open(os.path.join(HERE, "pi_labels.json")))
    old = [list(old.get(str(i), [])) for i in range(N)]
    ds = json.load(open(os.path.join(HERE, "datasets_noll.json")))
    new_by_noll = {}
    if os.path.exists(os.path.join(HERE, "new_labels_noll.json")):  # all curated templates, keyed by position
        for k, x in json.load(open(os.path.join(HERE, "new_labels_noll.json"))).items():
            if x["status"] == "ok":
                new_by_noll[int(k)] = x["codes"]
    else:  # whole_only_false labels only, mapped through eval_targets.json
        tg = json.load(open(os.path.join(HERE, "eval_targets.json")))
        new = json.load(open(os.path.join(HERE, "new_labels.json")))
        for r in tg:
            x = new.get(str(r["pos"]))
            if x and x["status"] == "ok":
                new_by_noll[r["noll_pos"]] = x["codes"]
    idx = np.array(sorted(new_by_noll))
    return old, ds, new_by_noll, idx


# ---------------------------------------------------------------- section 7 / App. C
def nwcd_cos(X, lab):
    m = lab != -1
    Xn, y = l2n(X[m]), lab[m]
    if len(set(y.tolist())) < 2:
        return float("nan")
    g = Xn.mean(0); g /= max(np.linalg.norm(g), 1e-12)
    G = np.mean(1 - Xn @ g)
    nrm = lambda c: c / max(np.linalg.norm(c), 1e-12)
    return float(np.mean([np.mean(1 - Xn[y == k] @ nrm(Xn[y == k].mean(0))) for k in np.unique(y)]) / G)


def sil(X, lab):
    from sklearn.metrics import silhouette_score
    m = lab != -1
    if len(set(lab[m].tolist())) < 2:
        return float("nan")
    return float(silhouette_score(X[m], lab[m], metric="cosine", sample_size=min(4000, int(m.sum())), random_state=RS))


def scores(y, lab):
    from sklearn.metrics import (normalized_mutual_info_score as NMI, adjusted_rand_score as ARI,
                                 adjusted_mutual_info_score as AMI)
    return {"NMI": float(NMI(y, lab)), "ARI": float(ARI(y, lab)), "AMI": float(AMI(y, lab))}


def cached(name, fn):
    """Cluster labels do not depend on the PIAtlas labels, so each clustering is computed once."""
    path = os.path.join(HERE, "cluster_cache", name + ".npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lab = np.asarray(fn())
    np.save(path, lab)
    return lab


def clusterings(emb, enc):
    from sklearn.cluster import MiniBatchKMeans, HDBSCAN
    from sklearn.mixture import GaussianMixture
    from scipy.cluster.hierarchy import linkage, fcluster
    fns = {"k-means": lambda: MiniBatchKMeans(n_clusters=K, random_state=RS, n_init=3, batch_size=1024).fit_predict(emb),
           "HDBSCAN": lambda: HDBSCAN(min_cluster_size=15, metric="euclidean").fit_predict(emb),
           "Ward": lambda: fcluster(linkage(emb, method="ward"), K, "maxclust") - 1,
           "GMM": lambda: GaussianMixture(n_components=K, covariance_type="diag", random_state=RS, max_iter=100).fit_predict(emb)}
    return {k: cached(f"{enc}_{k}" + (KTAG if k != "HDBSCAN" else ""), fn) for k, fn in fns.items()}


def section7(old, new_by_noll, idx):
    from sklearn.cluster import MiniBatchKMeans
    y_old_all = catify([tech(c) for c in old])
    y_old = catify([tech(old[i]) for i in idx])
    y_new = catify([tech(new_by_noll[i]) for i in idx])
    c_old = catify([cls(old[i]) for i in idx])
    c_new = catify([cls(new_by_noll[i]) for i in idx])
    rows, five = [], {}
    for name, f in ENCODERS:
        t0 = time.time()
        emb = l2n(np.load(os.path.join(EMB_DIR, f + ".npy")).astype(np.float32))
        for algo, lab in clusterings(emb, f).items():
            nc = len(set(lab.tolist()) - {-1})
            rows.append({"encoder": name, "clusterer": algo, "k": nc, "noise_pct": round(100 * float(np.mean(lab == -1)), 1),
                         "nwcd": nwcd_cos(emb, lab), "sil": sil(emb, lab),
                         "all4663_old": scores(y_old_all, lab),
                         "subset_old": scores(y_old, lab[idx]), "subset_new": scores(y_new, lab[idx]),
                         "subset_old_class": scores(c_old, lab[idx]), "subset_new_class": scores(c_new, lab[idx])})
            print(f"{name:18s} {algo:8s} k={nc:2d} all-old NMI={rows[-1]['all4663_old']['NMI']:.3f} "
                  f"| subset old AMI={rows[-1]['subset_old']['AMI']:.3f} new AMI={rows[-1]['subset_new']['AMI']:.3f}", flush=True)
        runs = defaultdict(list)
        for s in SEEDS5:
            lab = cached(f"{f}_kmeans_ninit10_seed{s}{KTAG}",
                         lambda: MiniBatchKMeans(n_clusters=K, random_state=s, n_init=10, batch_size=1024).fit_predict(emb))
            for key, y, l in (("all4663_old", y_old_all, lab), ("subset_old", y_old, lab[idx]), ("subset_new", y_new, lab[idx])):
                for m, v in scores(y, l).items():
                    runs[f"{key}|{m}"].append(v)
        five[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in runs.items()}
        print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    return {"rows": rows, "five_seed": five,
            "categories": {"all4663_old": int(len(set(y_old_all.tolist()))), "subset_old": int(len(set(y_old.tolist()))),
                           "subset_new": int(len(set(y_new.tolist())))}}


PAPER_MAP_SIZES = [968, 800, 800, 738, 503, 400, 182, 117, 104, 51]  # Table cluster-composition


def kmeans_auto(X, lo=8, hi=40):
    """compute_full.py: k-means on the 2-D map with k chosen by cosine silhouette over [8, 40]."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    best = None
    for k in range(lo, hi + 1):
        lab = MiniBatchKMeans(n_clusters=k, random_state=RS, n_init=3, batch_size=1024).fit_predict(X)
        s = silhouette_score(X, lab, metric="cosine", sample_size=min(4000, len(X)), random_state=RS)
        if best is None or s > best[0]:
            best = (s, k, lab)
    return best[2]


def map_composition(old, ds, new_by_noll):
    """Regenerate the steered PCA-50 + UMAP map and its k-means clusters (App. C, Table
    cluster-composition) and report each cluster's class shares under the old and the new labels."""
    from sklearn.decomposition import PCA
    import umap
    emb = l2n(np.load(os.path.join(EMB_DIR, "f2llm_steer_noll.npy")).astype(np.float32))
    xy = cached("steer_pca50_umap_xy", lambda: umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.0, metric="euclidean",
                                                          random_state=RS).fit_transform(PCA(n_components=50, random_state=RS).fit_transform(emb)))
    lab = cached("steer_map_kmeans_auto", lambda: kmeans_auto(xy))
    sizes = sorted(Counter(lab.tolist()).values(), reverse=True)
    rows = []
    for c in sorted(set(lab.tolist()), key=lambda c: -int(np.sum(lab == c))):
        m = np.where(lab == c)[0]
        mn = [i for i in m if i in new_by_noll]
        src = Counter(s for i in m for s in {source_name(x) for x in ds[i]})
        rows.append({"cluster": int(c), "n": int(len(m)), "n_new": len(mn),
                     "old": {k: round(100 * float(np.mean([k in cls(old[i]) for i in m])), 0) for k in CLASSES},
                     "new": {k: round(100 * float(np.mean([k in cls(new_by_noll[i]) for i in mn])), 0) if mn else None
                             for k in CLASSES},
                     "sources": [(s, round(100 * n / len(m))) for s, n in src.most_common(2)]})
    return {"sizes": sizes, "matches_paper_sizes": sizes == PAPER_MAP_SIZES, "rows": rows}


# ---------------------------------------------------------------- section 6
def class_agreement(old, new_by_noll, idx):
    """Per class, how often the old and new labels agree on its presence (Cohen's kappa)."""
    from sklearn.metrics import cohen_kappa_score
    out = {}
    for c in CLASSES:
        a = [c in cls(old[i]) for i in idx]
        b = [c in cls(new_by_noll[i]) for i in idx]
        out[c] = {"old_pct": round(100 * float(np.mean(a)), 1), "new_pct": round(100 * float(np.mean(b)), 1),
                  "agree_pct": round(100 * float(np.mean(np.array(a) == np.array(b))), 1),
                  "kappa": round(float(cohen_kappa_score(a, b)), 3)}
    return out


def section6(old, ds, new_by_noll, idx):
    out = {"class_agreement": class_agreement(old, new_by_noll, idx)}
    for ver, labels in (("old", {i: old[i] for i in idx}), ("new", {i: new_by_noll[i] for i in idx})):
        C = {i: set(cls(labels[i])) for i in idx}
        T = {i: set(tech(labels[i])) for i in idx}
        by_src = defaultdict(list)
        for i in idx:
            for s in {source_name(x) for x in ds[i]}:
                by_src[s].append(i)
        matrix = {}
        for s in sorted(by_src, key=lambda s: SOURCE_ORDER.index(s) if s in SOURCE_ORDER else 99):
            m = by_src[s]
            pct = {c: 100 * sum(c in C[i] for i in m) / len(m) for c in CLASSES}
            matrix[s] = {"n": len(m), **{c: round(pct[c], 1) for c in CLASSES}, "cov": sum(pct[c] >= 5 for c in CLASSES)}
        pct = {c: 100 * sum(c in C[i] for i in idx) / len(idx) for c in CLASSES}
        matrix["All labeled"] = {"n": len(idx), **{c: round(pct[c], 1) for c in CLASSES}, "cov": sum(pct[c] >= 5 for c in CLASSES)}
        ncls = np.array([len(C[i]) for i in idx]); ntech = np.array([len(T[i]) for i in idx])
        pairs = Counter()
        for i in idx:
            cs = sorted(C[i])
            pairs.update(f"{a}+{b}" for k, a in enumerate(cs) for b in cs[k + 1:])
        alone = lambda c: (sum(C[i] == {c} for i in idx), sum(c in C[i] for i in idx))
        out[ver] = {"matrix": matrix,
                    "finding1": {"median_classes": float(np.median(ncls)), "median_techniques": float(np.median(ntech)),
                                 "mean_classes": round(float(ncls.mean()), 2), "mean_techniques": round(float(ntech.mean()), 2),
                                 "single_class_pct": round(100 * float(np.mean(ncls == 1)), 1),
                                 "one_or_two_classes_pct": round(100 * float(np.mean(ncls <= 2)), 1),
                                 "max_classes": int(ncls.max()), "F3_alone_of_total": alone("F3"), "F6_alone_of_total": alone("F6"),
                                 "F5_and_F6": int(sum({"F5", "F6"} <= C[i] for i in idx)),
                                 "top_pairs": pairs.most_common(6)},
                    "finding2_class_pct": {c: round(pct[c], 1) for c in CLASSES},
                    "technique_pct": {t: round(100 * n / len(idx), 1) for t, n in
                                      sorted(Counter(t for i in idx for t in T[i]).items(), key=lambda x: -x[1])}}
    return out


def fmt3(x):
    return f"{x:.3f}"


def write_md(res):
    L = ["# Paper experiments rerun with the new PIAtlas technique labels", "",
         f"Labeled records: {res['n_subset']:,} of the paper's 4,663 curated templates (those whose new label has "
         "status ok). Old = the July labels used in the "
         "paper (31-technique labeling vocabulary); new = the current listing (34 techniques). Both are scored on the "
         "same records and the same clusterings.", ""]
    s7 = res["section7"]
    L += ["## Section 7 / App. C: cluster recovery", "",
          f"Reference categories (distinct class+technique sets): paper, all 4,663 = {s7['categories']['all4663_old']}; "
          f"labeled records, old = {s7['categories']['subset_old']}, new = {s7['categories']['subset_new']}.", "",
          "### Step 1: reproduction on all 4,663 records with the old labels (paper's App. C in parentheses)", "",
          "| encoder | clusterer | k | NWCD | sil. | NMI | ARI | AMI |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in s7["rows"]:
        # the paper's values are for k=31; for another k (and for HDBSCAN, which ignores k) show them only when comparable
        p = PAPER[(r["encoder"], r["clusterer"])] if (K == 31 or r["clusterer"] == "HDBSCAN") else None
        a = r["all4663_old"]
        q = (lambda j, v: f"{fmt3(v)} ({p[j]:.3f})") if p else (lambda j, v: fmt3(v))
        k = f"{r['k']}" + (f" ({r['noise_pct']:.0f}% n)" if r["clusterer"] == "HDBSCAN" else "") + (f" ({p[0]})" if p else "")
        L.append(f"| {r['encoder']} | {r['clusterer']} | {k} | {q(1, r['nwcd'])} | {q(2, r['sil'])} "
                 f"| {q(3, a['NMI'])} | {q(4, a['ARI'])} | {q(5, a['AMI'])} |")
    L += ["", "### Step 2: the same clusterings on the labeled records, old vs new class+technique sets", "",
          "| encoder | clusterer | NMI old | NMI new | ARI old | ARI new | AMI old | AMI new | class-set AMI old | class-set AMI new |",
          "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in s7["rows"]:
        o, n, co, cn = r["subset_old"], r["subset_new"], r["subset_old_class"], r["subset_new_class"]
        L.append(f"| {r['encoder']} | {r['clusterer']} | {fmt3(o['NMI'])} | {fmt3(n['NMI'])} | {fmt3(o['ARI'])} | {fmt3(n['ARI'])} "
                 f"| {fmt3(o['AMI'])} | {fmt3(n['AMI'])} | {fmt3(co['AMI'])} | {fmt3(cn['AMI'])} |")
    L += ["", "### Step 3: five k-means seeds (n_init=10), mean (sd)", "",
          "| encoder | all 4,663 old NMI | all 4,663 old ARI | labeled old NMI | labeled new NMI | labeled old ARI | labeled new ARI "
          "| labeled old AMI | labeled new AMI |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for name, _ in ENCODERS:
        f = s7["five_seed"][name]
        g = lambda k: f"{f[k][0]:.3f} ({f[k][1]:.3f})"
        L.append(f"| {name} | {g('all4663_old|NMI')} | {g('all4663_old|ARI')} | {g('subset_old|NMI')} | {g('subset_new|NMI')} "
                 f"| {g('subset_old|ARI')} | {g('subset_new|ARI')} | {g('subset_old|AMI')} | {g('subset_new|AMI')} |")
    L.append(f"\nPaper (Finding G1, steered, all 4,663): NMI {PAPER_FIVE_SEED['NMI'][0]} ± {PAPER_FIVE_SEED['NMI'][1]}, "
             f"ARI {PAPER_FIVE_SEED['ARI'][0]} ± {PAPER_FIVE_SEED['ARI'][1]}.")
    mp = res.get("map")
    if mp:
        L += ["", "### The ten-cluster steered map (Table cluster-composition), class shares old → new", "",
              f"Regenerated map cluster sizes: {mp['sizes']} "
              f"({'identical to' if mp['matches_paper_sizes'] else 'different from'} the paper's {PAPER_MAP_SIZES}).", "",
              "| n | F1 | F2 | F3 | F4 | F5 | F6 | dominant sources |", "|--:|--:|--:|--:|--:|--:|--:|---|"]
        for r in mp["rows"]:
            cells = [f"{r['old'][c]:.0f} → {r['new'][c]:.0f}" if r["new"][c] is not None else f"{r['old'][c]:.0f} → –"
                     for c in CLASSES]
            L.append(f"| {r['n']} | " + " | ".join(cells) + " | " + ", ".join(f"{s} ({p}%)" for s, p in r["sources"]) + " |")
    s6 = res["section6"]
    L += ["", "## Section 6: mapping and findings, on the labeled records", "",
          "### Benchmark-by-class matrix (percent of a source's labeled templates carrying each class), old → new", "",
          "| source | n | F1 | F2 | F3 | F4 | F5 | F6 | Cov. |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s, o in s6["old"]["matrix"].items():
        n = s6["new"]["matrix"][s]
        cells = [f"{o[c]:.0f} → {n[c]:.0f}" for c in CLASSES]
        L.append(f"| {s} | {o['n']:,} | " + " | ".join(cells) + f" | {o['cov']}/6 → {n['cov']}/6 |")
    L += ["", "### Findings 1 and 2, old → new", "", "| statistic | old | new |", "|---|--:|--:|"]
    for k in ("median_classes", "median_techniques", "mean_classes", "mean_techniques", "single_class_pct",
              "one_or_two_classes_pct", "max_classes", "F3_alone_of_total", "F6_alone_of_total", "F5_and_F6"):
        L.append(f"| {k} | {s6['old']['finding1'][k]} | {s6['new']['finding1'][k]} |")
    L.append(f"| top class pairs | {', '.join(f'{p} {n}' for p, n in s6['old']['finding1']['top_pairs'])} "
             f"| {', '.join(f'{p} {n}' for p, n in s6['new']['finding1']['top_pairs'])} |")
    L.append("| class prevalence % | " + ", ".join(f"{c} {s6['old']['finding2_class_pct'][c]}" for c in CLASSES)
             + " | " + ", ".join(f"{c} {s6['new']['finding2_class_pct'][c]}" for c in CLASSES) + " |")
    L += ["", "### Class presence, old vs new labels on the same records", "",
          "| class | old % | new % | agreement % | Cohen's kappa |", "|---|--:|--:|--:|--:|"]
    for c, a in s6["class_agreement"].items():
        L.append(f"| {c} | {a['old_pct']} | {a['new_pct']} | {a['agree_pct']} | {a['kappa']} |")
    L += ["", "### Technique prevalence with the new labels (percent of labeled records)", "",
          ", ".join(f"{t} {p}" for t, p in s6["new"]["technique_pct"].items()), ""]
    open(os.path.join(HERE, f"RERUN{KTAG}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    old, ds, new_by_noll, idx = load_inputs()
    print(f"labeled records: {len(idx)}", flush=True)
    res = {"n_subset": int(len(idx)), "section6": section6(old, ds, new_by_noll, idx)}
    if "--no-s7" not in sys.argv:
        res["section7"] = section7(old, new_by_noll, idx)
        res["map"] = map_composition(old, ds, new_by_noll)
    json.dump(res, open(os.path.join(HERE, f"rerun{KTAG}.json"), "w"), indent=1)
    if "section7" in res:
        write_md(res)


if __name__ == "__main__":
    main()
