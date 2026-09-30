#!/usr/bin/env python3
"""
Regenerate the rows of sections/tab_benchmark_matrix.tex from the new labels, in the paper's
format: percent of a source's labeled templates carrying each class, shading blue!N with
N = floor(0.6 x unrounded percent), "$<$1" for shares below 0.5% but above 0, and Cov. = classes
present in at least 5% of templates before rounding. A template listed under several sources counts
in each. Writes bench_matrix_rows.tex and bench_matrix.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C = ["F1", "F2", "F3", "F4", "F5", "F6"]
ROWS = [("ASB", "ASB~\\cite{zhang2024asb}"), ("BrowseSafe", "BrowseSafe~\\cite{zhang2025browsesafe}"),
        ("TaskTracker", "TaskTracker~\\cite{abdelnabi2024tasktracker}"), ("SEP", "SEP train/val.~\\cite{zverev2025sep}"),
        ("PIArena", "PIArena~\\cite{geng2026piarena}"),
        ("CyberSecEval", "CyberSecEval 2/3~\\cite{meta2024cyberseceval,meta2024cyberseceval3}"),
        ("InjecAgent", "InjecAgent~\\cite{zhan2024injecagent}"), ("AgentDojo", "AgentDojo~\\cite{debenedetti2024agentdojo}"),
        ("Agent-SafetyBench", "Agent-SafetyBench~\\cite{zhang2024safetybench}"),
        ("Greshake", "Greshake et al.~\\cite{greshake2023notwhat}"),
        ("Open-Prompt-Injection", "Open-Prompt-Injection~\\cite{liu2024openpi}"),
        ("AgentDyn", "AgentDyn~\\cite{li2026agentdyn}"), ("PromptInject", "PromptInject~\\cite{perez2022ignore}")]
NORM = {"SEP_train": "SEP", "SEP_validation": "SEP", "PIArena-refined": "PIArena", "Agentdojo": "AgentDojo",
        "BrowseSafeBench": "BrowseSafe", "AgentSafetyBench": "Agent-SafetyBench", "OpenPromptInjection": "Open-Prompt-Injection"}


def cls(codes):
    return {c.upper().split(".")[0] for c in codes}


def row(label, sets):
    n = len(sets)
    pct = {c: 100 * sum(c in s for s in sets) / n for c in C}
    cells = []
    for c in C:
        p = pct[c]
        shown = "$<$1" if 0 < p < 0.5 else f"{int(np.floor(p + 0.5))}"
        cells.append(f"\\cellcolor{{blue!{int(0.6 * p)}}}{shown}")
    cov = sum(pct[c] >= 5 for c in C)
    return f"{label} & {n:,} & " + " & ".join(cells) + f" & {cov}/6 \\\\", {"n": n, "pct": pct, "cov": cov}


def main():
    new = {int(k): v for k, v in json.load(open(os.path.join(HERE, "new_labels_noll.json"))).items() if v["status"] == "ok"}
    ds = json.load(open(os.path.join(HERE, "datasets_noll.json")))
    out, lines = {}, []
    for key, label in ROWS:
        sets = [cls(new[i]["codes"]) for i in sorted(new) if key in {NORM.get(s, s) for s in ds[i]}]
        line, stats = row(label, sets)
        lines.append(line)
        out[key] = stats
    lines.append("\\midrule")
    ll_path = os.path.join(HERE, "..", "piatlas_v4_llmail_4000.json")
    if os.path.exists(ll_path):
        ll = [cls(r["piatlas"]) for r in json.load(open(ll_path)) if r["piatlas_status"] == "ok"]
        line, stats = row("LLMail-Inject (4,000 sample)~\\cite{abdelnabi2025llmail}", ll)
        lines.append(line.replace(f"& {len(ll):,} &", "& 4,000 &") if len(ll) != 4000 else line)
        out["LLMail-Inject"] = stats
        lines.append("\\midrule")
    line, stats = row("All curated (non-LLMail)", [cls(new[i]["codes"]) for i in sorted(new)])
    lines.append(line)
    out["All curated"] = stats
    open(os.path.join(HERE, "bench_matrix_rows.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(HERE, "bench_matrix.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
