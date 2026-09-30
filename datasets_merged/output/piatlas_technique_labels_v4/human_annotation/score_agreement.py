#!/usr/bin/env python3
"""Numbers for the single-author verification study in PIAtlas_final.

The production labels are checked on the 300-record verification sample
against the first author's class and technique sets in ``sheet_author_A.csv``.
There was one human annotator, so this script does not compute human--human
inter-rater agreement and does not perform adjudication.

The script computes:
  1. Agreement between the production and human label sets: per class, the share of records on which they agree about
     the class and Cohen's kappa; both pooled over all record-class decisions; the exact match of
     class sets; and Krippendorff's alpha with the MASI distance over class sets and technique sets.
  2. Precision, recall and F1 of the production labels against the reference: per class and
     micro-averaged on the sample as drawn, and micro-averaged with every record weighted by the
     inverse of its inclusion probability, which estimates agreement on the whole labeled set and
     on each partition. Inclusion probabilities come from rerunning the design of make_sample.py.
  3. Swaps on four boundary pairs: records whose production labels carry the first class of a pair
     but not the second while the reference carries the second but not the first (over-assigned),
     and the reverse (missed).
  4. Class shares corrected for the measured disagreements: the production share of the whole
     partition plus the weighted mean difference between the reference and the production labels on
     that partition's sample records (a difference estimator, as in prediction-powered inference),
     and the LLMail-Inject minus curated gaps of Finding 4. Their 95% intervals come from independent
     Beta posteriors (Jeffreys prior) on the weighted rates at which the reference adds and removes
     the class, with the effective sample size of the weighted partition sample, so that a class on
     which the sample shows no disagreement still gets an interval of honest width.
The other intervals are 95% percentile intervals of a bootstrap that resamples records within the
strata of the design (2,000 replicates, seed 2027).

Writes ``agreement.json``.  The paper reports production-versus-human
agreement, precision, recall, F1, corrected class shares, and uncertainty.
"""
import csv, json, os, random, re, sys
from collections import Counter, defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_sample as MS  # noqa: E402

SECTIONS = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "paper_draft", "PIAtlas_final", "sections"))
C = ["F1", "F2", "F3", "F4", "F5", "F6"]
PARTS = (("cur", "curated"), ("ll", "LLMail-Inject"))
PAIRS = (("F4", "F2"), ("F1", "F4"), ("F4", "F5"), ("F4", "F6"))
RARE = (("F3", 17), ("F5", 17), ("F6", 16))  # as in make_sample.draw
SHARES = C + ["single", "le2"]  # class shares, single-class records, records with one or two classes
GAPS = ("F2", "F3", "F4", "F5")  # the comparisons of Finding 4
N_MC, N_BOOT, N_DRAW, SEED = 50000, 2000, 20000, 2027
TECH_COL = "techniques (optional, e.g. F1.T2;F2.T4)"
TECH_RE = re.compile(r"^F[1-6]\.T\d+$")
MARKS = {"1", "x", "y", "yes"}


def tech(codes):
    return frozenset(".".join(c.strip().upper().split(".")[:2]) for c in codes if c.strip())


def classes(tset):
    return frozenset(t.split(".")[0] for t in tset)


def share_vector(cset):
    return [float(c in cset) for c in C] + [float(len(cset) == 1), float(1 <= len(cset) <= 2)]


# ---------------------------------------------------------------- agreement measures
def kappa(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    po, pa, pb = float(np.mean(a == b)), a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe >= 1 else (po - pe) / (1 - pe)


def masi(a, b):
    if a == b:
        return 0.0
    inter, union = len(a & b), len(a | b)
    m = 2 / 3 if (a <= b or b <= a) else (1 / 3 if inter else 0.0)
    return 1 - inter / union * m


def alpha_masi(A, B):
    """Krippendorff's alpha for two coders and set-valued labels, with the MASI distance."""
    n = len(A)
    d_o = float(np.mean([masi(a, b) for a, b in zip(A, B)]))
    counts = Counter(list(A) + list(B))
    keys = list(counts)
    tot = sum(2 * counts[s] * counts[t] * masi(s, t) for i, s in enumerate(keys) for t in keys[i + 1:])
    d_e = tot / (2 * n * (2 * n - 1))
    return float("nan") if d_e == 0 else 1 - d_o / d_e


def raters(A, B, At=None, Bt=None):
    """Agreement between two raters on class sets A, B (and technique sets At, Bt)."""
    out = {"per_class": {}}
    for c in C:
        a, b = [c in x for x in A], [c in x for x in B]
        out["per_class"][c] = {"agree": 100 * float(np.mean([x == y for x, y in zip(a, b)])), "kappa": kappa(a, b)}
    va = [c in x for x in A for c in C]
    vb = [c in x for x in B for c in C]
    out["pooled"] = {"agree": 100 * float(np.mean([x == y for x, y in zip(va, vb)])), "kappa": kappa(va, vb)}
    out["exact_class"] = 100 * float(np.mean([x == y for x, y in zip(A, B)]))
    out["alpha_class"] = alpha_masi(A, B)
    ks = [v["kappa"] for v in out["per_class"].values() if v["kappa"] == v["kappa"]]
    out["kappa_min"], out["kappa_max"] = min(ks), max(ks)
    if At is not None and Bt is not None:
        out["exact_tech"] = 100 * float(np.mean([x == y for x, y in zip(At, Bt)]))
        out["alpha_tech"] = alpha_masi(At, Bt)
    return out


def prf_counts(P, R):
    """Per-record true positives, false positives and false negatives of sets P against R."""
    return np.array([[len(p & r), len(p - r), len(r - p)] for p, r in zip(P, R)], float)


def prf(counts, w=None):
    tp, fp, fn = (counts if w is None else counts * np.asarray(w)[:, None]).sum(0)
    return {"p": tp / (tp + fp) if tp + fp else float("nan"), "r": tp / (tp + fn) if tp + fn else float("nan"),
            "f": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")}


def swaps(P, R):
    out = {}
    for x, y in PAIRS:
        over = sum(x in p and y not in p and y in r and x not in r for p, r in zip(P, R))
        missed = sum(y in p and x not in p and x in r and y not in r for p, r in zip(P, R))
        out[f"{x}{y}"] = {"over": int(over), "missed": int(missed)}
    return out


# ---------------------------------------------------------------- design: weights and strata
def inclusion_probabilities(recs, ids):
    cache = os.path.join(HERE, "inclusion_probabilities.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("n_mc") == N_MC and set(d["pi"]) == set(ids):
            return d["pi"]
    by_src = defaultdict(list)
    for k, r in sorted(recs.items()):
        if r["partition"] == "curated":
            by_src[r["source"]].append(k)
    _, alloc = MS.draw(recs, random.Random(0))  # the allocation does not depend on the seed
    ll = sorted(k for k, r in recs.items() if r["partition"] == "LLMail-Inject")
    rare = [(n, sorted(k for k, r in recs.items() if any(x.startswith(c) for x in r["production"]))) for c, n in RARE]

    def draw(rng):  # make_sample.draw call for call, without the bookkeeping
        sel = set()
        for s in sorted(by_src):
            sel.update(rng.sample(by_src[s], alloc[s]))
        sel.update(rng.sample(ll, MS.N_LL))
        for n, pool in rare:
            sel.update(rng.sample([k for k in pool if k not in sel], n))
        return sel

    assert draw(random.Random(MS.SEED)) == set(ids), "the design replica does not reproduce sample.json"
    idset, hits = set(ids), Counter()
    for i in range(N_MC):
        hits.update(draw(random.Random(10 ** 6 + i)) & idset)
    pi = {k: hits[k] / N_MC for k in ids}
    json.dump({"n_mc": N_MC, "pi": pi}, open(cache, "w"), indent=0)
    return pi


class Design:
    def __init__(self, sample, recs):
        self.ids = [r["id"] for r in sample]
        pi = inclusion_probabilities(recs, self.ids)
        self.w = np.array([1 / pi[i] for i in self.ids])
        self.part = np.array(["cur" if r["partition"] == "curated" else "ll" for r in sample])
        keys = [r["stratum"] + ("|" + r["source"] if r["stratum"].startswith("curated") else "") for r in sample]
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        self.groups = [np.array(g) for g in groups.values()]
        self.pop_share = {}
        for p, name in PARTS:
            vec = np.array([share_vector(classes(tech(r["production"]))) for r in recs.values() if r["partition"] == name])
            self.pop_share[p] = 100 * vec.mean(0)
        self.pop_n = {p: sum(r["partition"] == name for r in recs.values()) for p, name in PARTS}

    def boot(self):
        rng = np.random.default_rng(SEED)
        for _ in range(N_BOOT):
            yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in self.groups])


def ci(values):
    v = np.asarray([x for x in values if x == x])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def against_reference(D, prodT, refC, refT=None):
    """Production labels against a reference (class sets refC, technique sets refT if labeled):
    precision/recall/F1, swaps, corrected shares."""
    prodC = [classes(t) for t in prodT]
    cc = prf_counts(prodC, refC)
    out = {"per_class": {}}
    for c in C:
        k = prf_counts([p & {c} for p in prodC], [r & {c} for r in refC])
        out["per_class"][c] = dict(prf(k), support=int(sum(c in r for r in refC)))
    out["micro"] = prf(cc)
    out["weighted"] = prf(cc, D.w)
    for p, _ in PARTS:
        m = D.part == p
        out[p] = prf(cc[m], D.w[m])
    tc = prf_counts(prodT, refT) if refT is not None else None
    if tc is not None:
        out["tech"] = prf(tc)
    out["swaps"] = swaps(prodC, refC)
    # corrected shares: production share of the partition + weighted mean of (reference - production),
    # i.e. + (rate at which the reference adds the statistic) - (rate at which it removes it)
    sr, sp = np.array([share_vector(r) for r in refC]), np.array([share_vector(p) for p in prodC])
    add, rem = (sr > sp).astype(float), (sr < sp).astype(float)
    rng = np.random.default_rng(SEED)
    full, draws, n_eff = {}, {}, {}
    for p, _ in PARTS:
        m = D.part == p
        W = D.w[m]
        a_hat, r_hat = W @ add[m] / W.sum(), W @ rem[m] / W.sum()
        n_eff[p] = float(W.sum() ** 2 / (W ** 2).sum())
        full[p] = D.pop_share[p] + 100 * (a_hat - r_hat)
        a = rng.beta(n_eff[p] * a_hat + 0.5, n_eff[p] * (1 - a_hat) + 0.5, size=(N_DRAW, len(SHARES)))
        r = rng.beta(n_eff[p] * r_hat + 0.5, n_eff[p] * (1 - r_hat) + 0.5, size=(N_DRAW, len(SHARES)))
        draws[p] = D.pop_share[p] + 100 * (a - r)
    reps = defaultdict(list)
    for idx in D.boot():
        reps["f_micro"].append(prf(cc[idx])["f"])
        reps["f_weighted"].append(prf(cc[idx], D.w[idx])["f"])
        for p, _ in PARTS:
            m = D.part[idx] == p
            reps[("f", p)].append(prf(cc[idx][m], D.w[idx][m])["f"])
        if tc is not None:
            reps["f_tech"].append(prf(tc[idx])["f"])
    out["micro"]["ci"] = ci(reps["f_micro"])
    out["weighted"]["ci"] = ci(reps["f_weighted"])
    for p, _ in PARTS:
        out[p]["ci"] = ci(reps[("f", p)])
    if tc is not None:
        out["tech"]["ci"] = ci(reps["f_tech"])
    out["n_eff"] = n_eff
    out["shares"] = {}
    for p, _ in PARTS:
        clipped = np.clip(draws[p], 0, 100)
        out["shares"][p] = {s: {"production": float(D.pop_share[p][j]), "corrected": float(np.clip(full[p][j], 0, 100)),
                                "ci": ci(clipped[:, j])} for j, s in enumerate(SHARES)}
    gap = np.clip(draws["ll"], 0, 100) - np.clip(draws["cur"], 0, 100)
    out["gaps"] = {s: {"production": float(D.pop_share["ll"][j] - D.pop_share["cur"][j]),
                       "corrected": out["shares"]["ll"][s]["corrected"] - out["shares"]["cur"][s]["corrected"],
                       "ci": ci(gap[:, j])} for j, s in enumerate(SHARES) if s in GAPS}
    shifts = [abs(out["shares"][p][c]["corrected"] - out["shares"][p][c]["production"]) for p, _ in PARTS for c in C]
    out["max_shift"] = float(max(shifts))
    return out


# ---------------------------------------------------------------- first-author sheet
def read_sheet(name):
    """Rows the annotator has filled: any class mark, technique, or note ('none' for no class)."""
    rows, problems = {}, []
    with open(os.path.join(HERE, name), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cells = {c: (r.get(c) or "").strip().lower() for c in C}
            techs = [t.strip().upper() for t in re.split(r"[;,\s]+", r.get(TECH_COL) or "") if t.strip()]
            if not any(cells.values()) and not techs and not (r.get("notes") or "").strip():
                continue
            cls = {c for c in C if cells[c] in MARKS}
            bad = [t for t in techs if not TECH_RE.match(".".join(t.split(".")[:2]))]
            tset = tech([t for t in techs if t not in bad])
            if bad:
                problems.append(f"{name} {r['id']}: unreadable technique codes {bad}")
            if tset and not classes(tset) <= cls:
                problems.append(f"{name} {r['id']}: techniques {sorted(tset)} name classes not marked {sorted(cls)}")
            rows[r["id"]] = {"classes": frozenset(cls), "techniques": tset}
    return rows, problems


# ---------------------------------------------------------------- output
def f2(x):
    return "--" if x != x else f"{x:.2f}"


def f1(x):
    return "--" if x != x else f"{x:.1f}"


def interval(v, fmt=f1):
    return f"{fmt(v['corrected'])} [{fmt(v['ci'][0])}, {fmt(v['ci'][1])}]"


def main():
    if len(sys.argv) > 1:
        raise SystemExit("This single-author scorer takes no command-line options.")
    sample = json.load(open(os.path.join(HERE, "sample.json")))
    recs = MS.load_records()
    D = Design(sample, recs)
    prodT = [tech(r["production"]) for r in sample]
    author, problems = read_sheet("sheet_author_A.csv")
    if problems:
        raise ValueError("Problems in sheet_author_A.csv:\n" + "\n".join(problems))
    missing = [record_id for record_id in D.ids if record_id not in author]
    if missing:
        raise ValueError(f"sheet_author_A.csv is missing {len(missing)} records: {missing[:10]}")
    humanT = [author[record_id]["techniques"] for record_id in D.ids]
    humanC = [author[record_id]["classes"] for record_id in D.ids]
    if any(bool(cset) != bool(tset) for cset, tset in zip(humanC, humanT)):
        raise ValueError("Every labeled class set must have technique-level labels")

    agreement = raters(humanC, [classes(t) for t in prodT], humanT, prodT)
    reference = against_reference(D, prodT, humanC, humanT)
    res = {"n": len(D.ids), "sample_by_partition": dict(Counter(D.part.tolist())), "population": D.pop_n,
           "weights": {"min": float(D.w.min()), "max": float(D.w.max()),
                       "sum_by_partition": {p: float(D.w[D.part == p].sum()) for p, _ in PARTS}},
           "human": {
               "annotator": {"count": 1, "role": "first author", "sample_size": len(D.ids),
                             "blinded_to": ["production labels", "source dataset"],
                             "inter_rater_agreement": None, "adjudication": None},
               "agreement_with_production": agreement,
               "production_against_human_reference": reference,
               "production_support": {c: int(sum(c in classes(t) for t in prodT)) for c in C}}}

    r, g = agreement, reference
    print(f"Sample: {res['sample_by_partition']}; weights {res['weights']['min']:.1f}-{res['weights']['max']:.1f}, "
          f"sums {({p: round(v) for p, v in res['weights']['sum_by_partition'].items()})} vs population {D.pop_n}")
    print("\nFirst-author reference vs production labels: agreement% / kappa per class")
    print("  " + "  ".join(f"{c} {f1(r['per_class'][c]['agree'])}/{f2(r['per_class'][c]['kappa'])}" for c in C))
    print(f"  pooled {f1(r['pooled']['agree'])}/{f2(r['pooled']['kappa'])}; exact class sets {f1(r['exact_class'])}%, "
          f"alpha(MASI) class {f2(r['alpha_class'])}; exact technique sets {f1(r['exact_tech'])}%, alpha(MASI) technique {f2(r['alpha_tech'])}")
    print("Production against the first-author human reference: P/R/F1")
    for c in C:
        v = g["per_class"][c]
        print(f"  {c} support {v['support']:3d}  {f2(v['p'])} {f2(v['r'])} {f2(v['f'])}")
    for key in ("micro", "weighted", "cur", "ll", "tech"):
        v = g[key]
        print(f"  {key:8s} {f2(v['p'])} {f2(v['r'])} {f2(v['f'])}  CI {f2(v['ci'][0])}-{f2(v['ci'][1])}")
    print("  swaps:", g["swaps"])
    print("Corrected shares (production -> corrected [CI]):")
    for p, name in PARTS:
        print(f"  {name}: " + "; ".join(f"{s} {f1(v['production'])}->{interval(v)}" for s, v in g["shares"][p].items()))
    print("  Finding 4 gaps (LLMail - curated): " + "; ".join(f"{s} {f1(v['production'])}->{interval(v)}" for s, v in g["gaps"].items()))
    print(f"  largest shift of a class share: {f1(g['max_shift'])} points")

    with open(os.path.join(HERE, "agreement.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    print("\nWrote the single-author results to agreement.json")


if __name__ == "__main__":
    main()
