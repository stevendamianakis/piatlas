#!/usr/bin/env python3
"""Recompute the labeling-derived numbers of the paper (composition depth, Findings 1-4, the
benchmark-by-class shares) from the four released label files only:

    piatlas_v4_curated_4663.json   curated partition: noll_pos, sources, template, piatlas (technique codes)
    piatlas_v4_llmail_4000.json    adaptive partition: llmail_pos, merged_index, template, piatlas
    inputs/datasets_noll.json      per curated record, the list of source datasets (position = noll_pos)
    inputs/datasets_llmail.json    per LLMail record, its source (always LLMail-Inject)

Usage:  python3 label_numbers.py ..   (the labeling folder holding the four files) [--out results.json]
Prints a check table (paper value, recomputed value, match) and writes the full results as JSON.
Record texts are never printed. Rounding: integers in the paper are rounded half up on the exact
share; one-decimal values use Python's round(), as the paper's own scripts do.
"""
import argparse
import json
import os
from collections import Counter
from fractions import Fraction

C = ["F1", "F2", "F3", "F4", "F5", "F6"]
NORM = {"Agentdojo": "AgentDojo", "BrowseSafeBench": "BrowseSafe", "PIArena-refined": "PIArena",
        "SEP_train": "SEP", "SEP_validation": "SEP", "AgentSafetyBench": "Agent-SafetyBench",
        "OpenPromptInjection": "Open-Prompt-Injection"}
AGENT_BENCH = ["AgentDojo", "InjecAgent", "ASB", "AgentDyn"]  # Finding 3


def half_up(x):
    """Round a Fraction or float half up to an integer."""
    x = Fraction(x)
    return int((x + Fraction(1, 2)) // 1)


def pct(k, n):
    return Fraction(100 * k, n)


def dp1(k, n):
    """One decimal, computed as the paper's scripts do (100 * mean of floats); exact ties such as
    66.65 or 4.05 then fall to the lower digit, so '(tie)' is appended when the exact share is a tie."""
    tie = (Fraction(1000 * k, n) - Fraction(1000 * k, n) // 1) == Fraction(1, 2)
    return f"{100 * (k / n):.1f}" + (" (tie)" if tie else "")


def classes(codes):
    return frozenset(c.split(".")[0] for c in codes)


def techniques(codes):
    return frozenset(".".join(c.split(".")[:2]) for c in codes)


def load(d):
    cur = json.load(open(os.path.join(d, "piatlas_v4_curated_4663.json")))
    ll = json.load(open(os.path.join(d, "piatlas_v4_llmail_4000.json")))
    ds = json.load(open(os.path.join(d, "inputs", "datasets_noll.json")))
    dl = json.load(open(os.path.join(d, "inputs", "datasets_llmail.json")))
    assert all(r["piatlas_status"] == "ok" for r in cur + ll)
    assert len(ds) == len(cur) and len(dl) == len(ll)
    for r in cur:  # the release carries the same sources as datasets_noll.json
        assert sorted(r["sources"]) == sorted(ds[r["noll_pos"]])
    return cur, ll, ds, dl


def composition(rows):
    ncls = sorted(len(classes(r["piatlas"])) for r in rows)
    ntech = sorted(len(techniques(r["piatlas"])) for r in rows)
    n = len(rows)
    med = lambda v: (v[(n - 1) // 2] + v[n // 2]) / 2
    pairs = Counter()
    for r in rows:
        cs = sorted(classes(r["piatlas"]))
        pairs.update(f"{a}+{b}" for i, a in enumerate(cs) for b in cs[i + 1:])
    return {"n": n, "median_classes": med(ncls), "median_techniques": med(ntech),
            "single": sum(x == 1 for x in ncls), "one_or_two": sum(x <= 2 for x in ncls),
            "max_classes": max(ncls), "more_than_five": sum(x > 5 for x in ncls),
            "pairs": dict(pairs.most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="label_numbers.json")
    a = ap.parse_args()
    cur, ll, ds, dl = load(a.dir)
    src = {r["noll_pos"]: {NORM.get(s, s) for s in ds[r["noll_pos"]]} for r in cur}
    Ccur = {r["noll_pos"]: classes(r["piatlas"]) for r in cur}
    Tcur = {r["noll_pos"]: techniques(r["piatlas"]) for r in cur}
    Cll = [classes(r["piatlas"]) for r in ll]
    res, checks = {}, []

    def check(where, what, paper, got):
        ok = str(paper) == str(got)
        checks.append((where, what, paper, got, ok))

    # ---------------------------------------------------------------- partitions (III-C, App. D)
    sources = sorted({s for v in src.values() for s in v})
    res["partitions"] = {"curated": len(cur), "llmail": len(ll), "total": len(cur) + len(ll),
                         "curated_sources": sources, "llmail_sources": sorted({s for x in dl for s in x})}
    check("03_methodology.tex:88", "labeled records", "8663", len(cur) + len(ll))
    check("03_methodology.tex:89-90", "curated templates / benchmarks", "4663/13", f"{len(cur)}/{len(sources)}")
    check("03_methodology.tex:91", "adaptive sample", "4000", len(ll))

    # ---------------------------------------------------------------- Finding 1
    comp = composition(cur)
    res["finding1"] = comp
    n = comp["n"]
    check("06_findings.tex:29", "median classes / techniques", "2/3",
          f"{comp['median_classes']:g}/{comp['median_techniques']:g}")
    check("06_findings.tex:30", "single-class % (1 dp)", "38.3", round(float(pct(comp["single"], n)), 1))
    exact_le2 = f"{float(pct(comp['one_or_two'], n)):.2f}"
    check("06_findings.tex:30", f"one or two classes % (integer; exact {exact_le2})",
          "77", half_up(pct(comp["one_or_two"], n)))
    check("06_findings.tex:31", "records with more than five classes", "0", comp["more_than_five"])
    alone = {c: (sum(v == {c} for v in Ccur.values()), sum(c in v for v in Ccur.values())) for c in C}
    res["finding1"]["alone_of_total"] = alone
    check("06_findings.tex:39-40", "F3 alone of F3 records", "4 of 252", "%d of %d" % alone["F3"])
    check("06_findings.tex:39-40", "F6 alone of F6 records", "3 of 391", "%d of %d" % alone["F6"])
    f56 = [i for i, v in Ccur.items() if {"F5", "F6"} <= v]
    f56_by_src = Counter(s for i in f56 for s in src[i])
    f56_not_asb = sum("Agent-SafetyBench" not in src[i] for i in f56)
    res["finding1"]["F5_and_F6"] = {"n": len(f56), "by_source": dict(f56_by_src), "outside_Agent-SafetyBench": f56_not_asb}
    check("06_findings.tex:43-44", "F5+F6 records / outside Agent-SafetyBench", "75/61", f"{len(f56)}/{f56_not_asb}")
    top = list(comp["pairs"].items())[:5]
    check("06_findings.tex:45-46", "top five class pairs",
          "F1+F2 1633, F1+F4 1360, F1+F5 626, F4+F5 482, F2+F4 428", ", ".join(f"{p} {k}" for p, k in top))

    # ---------------------------------------------------------------- Finding 2 (and Table tab:shares, production)
    share = {c: round(float(pct(sum(c in v for v in Ccur.values()), n)), 1) for c in C}
    res["finding2_curated_share"] = share
    check("06_findings.tex:61-63", "curated class shares F1..F6 (1 dp)", "99.2 35.3 5.4 29.5 13.6 8.4",
          " ".join(str(share[c]) for c in C))
    share_ll = {c: dp1(sum(c in v for v in Cll), len(ll)) for c in C}
    res["llmail_share"] = share_ll
    check("app_d_annotation.tex:174-179", "LLMail production shares F1..F6 (1 dp)",
          "98.5 75.6 68.7 66.6 (tie) 4.3 11.8", " ".join(str(share_ll[c]) for c in C))
    single_ll = sum(len(v) == 1 for v in Cll)
    le2_ll = sum(len(v) <= 2 for v in Cll)
    check("app_d_annotation.tex:181-182", "single / one-or-two, curated (1 dp)", "38.3/76.5",
          f"{round(float(pct(comp['single'], n)), 1)}/{round(float(pct(comp['one_or_two'], n)), 1)}")
    check("app_d_annotation.tex:181-182", "single / one-or-two, LLMail (1 dp)", "4.0 (tie)/22.9",
          f"{dp1(single_ll, len(ll))}/{dp1(le2_ll, len(ll))}")

    # ---------------------------------------------------------------- benchmark matrix and Finding 3
    by_src = {}
    for s in sources:
        idx = [i for i in Ccur if s in src[i]]
        cnt = {c: sum(c in Ccur[i] for i in idx) for c in C}
        by_src[s] = {"n": len(idx), "count": cnt,
                     "pct_int": {c: half_up(pct(cnt[c], len(idx))) for c in C},
                     "cov": sum(pct(cnt[c], len(idx)) >= 5 for c in C)}
    res["by_source"] = by_src
    for s in AGENT_BENCH:
        check("06_findings.tex:87-92", f"{s}: Cov. / F5 templates", {"ASB": "2/0", "InjecAgent": "2/0",
              "AgentDojo": "4/0", "AgentDyn": "4/0"}[s], f"{by_src[s]['cov']}/{by_src[s]['count']['F5']}")
    f3_agent = sorted(i for i in Ccur if "F3" in Ccur[i] and src[i] & {"AgentDojo", "AgentDyn"})
    res["finding3_F3_in_AgentDojo_AgentDyn"] = {"n_distinct": len(f3_agent),
                                                 "F3_techniques": dict(Counter(t for i in f3_agent for t in Tcur[i] if t.startswith("F3")))}
    check("06_findings.tex:90-91", "distinct F3 templates in AgentDojo or AgentDyn", "4", len(f3_agent))
    check("06_findings.tex:94", "Open-Prompt-Injection Cov.", "2", by_src["Open-Prompt-Injection"]["cov"])
    no_f6 = sorted(s for s in sources if by_src[s]["count"]["F6"] == 0)
    res["sources_without_F6"] = no_f6
    check("06_findings.tex:95", "sources with no F6 template", "6", len(no_f6))
    six = sorted((s, by_src[s]["n"]) for s in sources if by_src[s]["cov"] == 6)
    check("06_findings.tex:98-99", "Cov. 6/6 sources (n)", "Agent-SafetyBench 24, CyberSecEval 154, Greshake 20",
          ", ".join(f"{s} {k}" for s, k in six))

    # ---------------------------------------------------------------- Finding 4 and Section IX
    ll_int = {c: half_up(pct(sum(c in v for v in Cll), len(ll))) for c in C}
    cur_int = {c: half_up(pct(sum(c in v for v in Ccur.values()), n)) for c in C}
    res["finding4"] = {"llmail_pct_int": ll_int, "curated_pct_int": cur_int}
    check("06_findings.tex:112-122", "F2, F3, F4, F5: LLMail vs curated (%)", "76/35 69/5 67/29 4/14",
          " ".join(f"{ll_int[c]}/{cur_int[c]}" for c in ("F2", "F3", "F4", "F5")))
    check("09_related.tex:75-76", "LLMail templates with F2 (three of every four)", "75.6",
          round(float(pct(sum("F2" in v for v in Cll), len(ll))), 1))

    # ---------------------------------------------------------------- technique coverage (App. D)
    t_all = {t for r in cur + ll for t in techniques(r["piatlas"])}
    t_cur = {t for r in cur for t in techniques(r["piatlas"])}
    res["techniques_used"] = {"all": len(t_all), "curated": len(t_cur),
                              "missing_in_curated": sorted(t_all - t_cur)}
    check("app_d_annotation.tex:201", "techniques occurring: all / curated", "34/33", f"{len(t_all)}/{len(t_cur)}")

    # ---------------------------------------------------------------- output
    res["checks"] = [dict(zip(("paper_location", "quantity", "paper", "recomputed", "match"), c)) for c in checks]
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    w = max(len(c[1]) for c in checks)
    for where, what, paper, got, ok in checks:
        print(f"{'OK  ' if ok else 'DIFF'} {where:30s} {what:{w}s}  paper={paper}  recomputed={got}")
    print(f"\n{sum(c[4] for c in checks)} of {len(checks)} checks match; wrote {a.out}")


if __name__ == "__main__":
    main()
