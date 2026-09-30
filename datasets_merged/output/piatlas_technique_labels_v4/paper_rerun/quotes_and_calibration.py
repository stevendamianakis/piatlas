#!/usr/bin/env python3
"""Recompute, without any model call, the labeling numbers that need the raw label files rather than
the two released JSON files:

  * the evidence-quote statistics of Section III-C and Appendix D ("over 99.5% of these quotes occur
    verbatim in the record, and nearly all others differ only in whitespace"; "every production
    technique label carries an evidence quote");
  * the calibration numbers of Table tab:agreement / tab:stability (micro-F1 and exact-set match per
    comparison, averaged over both B=1 runs for B=4 and B=8) and the techniques per record (3.3 vs 3.5);
  * a consistency check that piatlas_v4_curated_4663.json and piatlas_v4_llmail_4000.json are exactly
    the export of the raw label files (the rule of run_labeling.py: the F6-recheck label replaces the
    first-pass label; for a record labeled twice, the last row counts).

Inputs, all under the labeling folder <dir>: labels.jsonl, labels_curated_rest.jsonl,
labels_f6_recheck.jsonl, labels_llmail.jsonl, calib/{b1a,b1b,b4,b8}.jsonl, inputs/merged_texts_noll.json,
piatlas_v4_curated_4663.json, piatlas_v4_llmail_4000.json; and --wof, the file whose
template_instruction field labels.jsonl is keyed by (prompt_injections_whole_only_false.json).
Usage:  python3 quotes_and_calibration.py .. --wof ../../prompt_injections_whole_only_false.json [--out results.json]
Record texts and quotes are never printed.
"""
import argparse
import json
import os
from collections import Counter


def rows(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["idx"]] = r  # the last row for an index counts, as in run_labeling.load()
    return out


def norm(s):
    return " ".join(s.split())


def micro(a, b, level):
    cut = (lambda c: c.split(".")[0]) if level == "class" else (lambda c: c)
    tp = na = nb = exact = n = 0
    for i in set(a) & set(b):
        if a[i]["status"] != "ok" or b[i]["status"] != "ok":
            continue
        sa, sb = {cut(c) for c in a[i]["codes"]}, {cut(c) for c in b[i]["codes"]}
        tp += len(sa & sb); na += len(sa); nb += len(sb); exact += sa == sb; n += 1
    return 2 * tp / (na + nb), 100 * exact / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--wof", required=True)
    ap.add_argument("--out", default="quotes_and_calibration.json")
    a = ap.parse_args()
    D = a.dir
    checks, res = [], {}

    def check(where, what, paper, got):
        checks.append((where, what, str(paper), str(got), str(paper) == str(got)))

    # ------------------------------------------------------------ production rows per record
    noll = json.load(open(os.path.join(D, "inputs", "merged_texts_noll.json"), encoding="utf-8"))
    pos = {t: i for i, t in enumerate(noll)}
    wof = [r["template_instruction"] for r in json.load(open(a.wof, encoding="utf-8"))]
    cur, provenance = {}, {}
    for i, r in rows(os.path.join(D, "labels.jsonl")).items():
        cur[pos[wof[i]]] = r
        provenance[pos[wof[i]]] = "whole_only_false run" + (" (reused calibration b1a)" if r.get("reused_from") else "")
    for i, r in rows(os.path.join(D, "labels_curated_rest.jsonl")).items():
        cur[i] = r
        provenance[i] = "curated_rest run"
    for i, r in rows(os.path.join(D, "labels_f6_recheck.jsonl")).items():
        cur[i] = r
        provenance[i] = "F6 recheck (prompt v1.1)"
    llm = rows(os.path.join(D, "labels_llmail.jsonl"))
    res["curated_label_provenance"] = dict(Counter(provenance.values()))
    res["runs_in_final_curated_rows"] = dict(Counter(r.get("run") for r in cur.values()))
    res["runs_in_final_llmail_rows"] = dict(Counter(r.get("run") for r in llm.values()))

    # ------------------------------------------------------------ release == export of the raw files
    rel_c = json.load(open(os.path.join(D, "piatlas_v4_curated_4663.json"), encoding="utf-8"))
    rel_l = json.load(open(os.path.join(D, "piatlas_v4_llmail_4000.json"), encoding="utf-8"))
    bad_c = sum(sorted(cur[r["noll_pos"]]["codes"]) != sorted(r["piatlas"]) or cur[r["noll_pos"]]["status"] != r["piatlas_status"]
                for r in rel_c)
    bad_l = sum(sorted(llm[r["llmail_pos"]]["codes"]) != sorted(r["piatlas"]) or llm[r["llmail_pos"]]["status"] != r["piatlas_status"]
                for r in rel_l)
    check("release", "records whose released codes differ from the raw label files (curated/LLMail)", "0/0", f"{bad_c}/{bad_l}")
    check("release", "prompt tags in release (curated)", "{'v1.0': 4185, 'v1.1 (F6 recheck)': 478}",
          dict(Counter(r["prompt"] for r in rel_c)))

    # ------------------------------------------------------------ evidence quotes
    text_c = {r["noll_pos"]: r["template"] for r in rel_c}
    text_l = {r["llmail_pos"]: r["template"] for r in rel_l}
    q = Counter()
    for part, R, T in (("curated", cur, text_c), ("llmail", llm, text_l)):
        for i, r in R.items():
            for t in r["techniques"]:
                s, txt = t.get("quote") or "", T[i]
                exact = bool(s) and s in txt
                ws = bool(norm(s)) and norm(s) in norm(txt)
                q[(part, "labels")] += 1
                q[(part, "empty")] += not s
                q[(part, "verbatim")] += exact
                q[(part, "whitespace_only")] += (not exact) and ws
                q[(part, "stored_flag_verbatim")] += bool(t.get("quote_exact"))
    tot = {k: q[("curated", k)] + q[("llmail", k)] for k in ("labels", "empty", "verbatim", "whitespace_only", "stored_flag_verbatim")}
    res["quotes"] = {"curated": {k: q[("curated", k)] for k in tot}, "llmail": {k: q[("llmail", k)] for k in tot}, "all": tot}
    vpct = 100 * tot["verbatim"] / tot["labels"]
    other = tot["labels"] - tot["verbatim"]
    res["quotes"]["verbatim_pct"] = vpct
    check("03_methodology.tex:102, app_d:202-203", "quotes verbatim in the record: over 99.5%", True, vpct > 99.5)
    res["quotes"]["verbatim_pct_str"] = f"{vpct:.2f}% ({tot['verbatim']} of {tot['labels']})"
    check("app_d_annotation.tex:203-204", "non-verbatim quotes that differ only in whitespace",
          "nearly all", f"{tot['whitespace_only']} of {other}")
    # characterize the rest without printing them: case-only, ellipsis, or other
    rest = Counter()
    for part, R, T in (("curated", cur, text_c), ("llmail", llm, text_l)):
        for i, r in R.items():
            for t in r["techniques"]:
                s, txt = t.get("quote") or "", T[i]
                if s in txt or norm(s) in norm(txt):
                    continue
                if norm(s).lower() in norm(txt).lower():
                    rest["case only"] += 1
                elif "..." in s or "\u2026" in s:
                    rest["contains an ellipsis"] += 1
                else:
                    rest["other (paraphrase, joined spans, or edited text)"] += 1
    res["quotes"]["neither_verbatim_nor_whitespace"] = dict(rest)
    check("app_d_annotation.tex:202", "technique labels without a quote", "0", tot["empty"])

    # ------------------------------------------------------------ calibration
    L = {k: rows(os.path.join(D, "calib", f"{k}.jsonl")) for k in ("b1a", "b1b", "b4", "b8")}
    comp = {f"{x}_vs_{y}": {lvl: micro(L[x], L[y], lvl) for lvl in ("class", "technique")}
            for x, y in (("b1a", "b1b"), ("b4", "b1a"), ("b4", "b1b"), ("b8", "b1a"), ("b8", "b1b"))}
    res["calibration_pairs"] = comp

    def cell(keys, lvl):
        f = sum(comp[k][lvl][0] for k in keys) / len(keys)
        e = sum(comp[k][lvl][1] for k in keys) / len(keys)
        return f"{f:.2f} ({e:.0f}%)"

    tab = {"B=1": (["b1a_vs_b1b"],), "B=4": (["b4_vs_b1a", "b4_vs_b1b"],), "B=8": (["b8_vs_b1a", "b8_vs_b1b"],)}
    paper = {"B=1": ("0.99 (94%)", "0.97 (83%)"), "B=4": ("0.96 (84%)", "0.94 (73%)"), "B=8": ("0.96 (83%)", "0.94 (73%)")}
    for b, (keys,) in tab.items():
        check("app_d_annotation.tex:364-366 / 03_methodology.tex:125-126", f"{b} vs B=1, class", paper[b][0], cell(keys, "class"))
        check("app_d_annotation.tex:364-366 / 03_methodology.tex:125-126", f"{b} vs B=1, technique", paper[b][1], cell(keys, "technique"))
    tpr = {k: sum(len(r["codes"]) for r in v.values()) / len(v) for k, v in L.items()}
    res["techniques_per_record"] = tpr
    check("app_d_annotation.tex:350-351", "techniques per record, B=4 and B=8 against B=1", "3.3, 3.3 against 3.5",
          f"{tpr['b4']:.1f}, {tpr['b8']:.1f} against {tpr['b1a']:.1f}")
    check("app_d_annotation.tex:147", "calibration records", "64", len(L["b1a"]))

    res["checks"] = [dict(zip(("paper_location", "quantity", "paper", "recomputed", "match"), c)) for c in checks]
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    w = max(len(c[1]) for c in checks)
    for where, what, p, g, ok in checks:
        print(f"{'OK  ' if ok else 'DIFF'} {what:{w}s}  paper={p}  recomputed={g}   [{where}]")
    print(f"\n{sum(c[4] for c in checks)} of {len(checks)} checks match; wrote {a.out}")


if __name__ == "__main__":
    main()
