#!/usr/bin/env python3
"""Check that every category name in Appendix F's card data occurs in the local copy of its
source under literature/ (see literature/INDEX.md). Usage, from PIAtlas_final/:
    python3 tools/check_crosswalk_sources.py
Prints, per scheme, how many category names occur verbatim, how many occur only as separate
words (usually a table cell split or a grouping phrase), and which are not found."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[3]  # repository root
LIT = PROJ / "literature"
DATA = PROJ / "paper_draft/PIAtlas_final/tools/crosswalk_data.json"
COPY = False

# key -> list of (source path, name to use in literature/)
M = {
 "perez2022ignore": [(LIT / "perez2022ignore.pdf", None)],
 "greshake2023notwhat": [(LIT / "greshake2023notwhat.pdf", None)],
 "liu2023jailbreaking": [(LIT / "liu2023jailbreaking.pdf", None),
                         (LIT / "liu2023jailbreaking_v1.pdf", None)],
 "wei2023jailbroken": [(LIT / "wei2023jailbroken.pdf", None)],
 "schulhoff2023hackaprompt": [(LIT / "schulhoff2023hackaprompt.pdf", None)],
 "kang2024exploiting": [(LIT / "kang2024exploiting.pdf", None)],
 "rao2024tricking": [(LIT / "rao2024tricking.pdf", None)],
 "toyer2024tensortrust": [(LIT / "toyer2024tensortrust_iclr.pdf", None),
                          (LIT / "toyer2024tensortrust.pdf", None)],
 "rossi2024early": [(LIT / "rossi2024early.pdf", None)],
 "liu2024houyi": [(LIT / "liu2024houyi.pdf", None), (LIT / "liu2024houyi_v2.pdf", None)],
 "liu2024openpi": [(LIT / "liu2024openpi.pdf", None)],
 "zeng2024johnny": [(LIT / "zeng2024johnny.pdf", None),
                    (LIT / "zeng2024johnny_taxonomy.jsonl", None)],
 "wallace2024hierarchy": [(LIT / "wallace2024hierarchy.pdf", None)],
 "meta2024cyberseceval": [(LIT / "meta2024cyberseceval.pdf", None),
                          (LIT / "meta2024cyberseceval_prompt_injection.json", None)],
 "rawat2024attackatlas": [(LIT / "rawat2024attackatlas.pdf", None)],
 "rehberger2024trustno": [(LIT / "rehberger2024trustno.pdf", None)],
 "yi2023bipia": [(LIT / "yi2023bipia.pdf", None),
                 (LIT / "yi2023bipia_text_attack_test.json", None),
                 (LIT / "yi2023bipia_text_attack_train.json", None),
                 (LIT / "yi2023bipia_code_attack_test.json", None)],
 "nist2025aml": [(LIT / "nist2025aml.pdf", None)],
 "pfister2025gandalf": [(LIT / "pfister2025gandalf.pdf", None),
                        (LIT / "pfister2025gandalf_icml.pdf", None)],
 "abdelnabi2025llmail": [(LIT / "abdelnabi2025llmail.pdf", None)],
 "hong2025sokeval": [(LIT / "hong2025sokeval_v3.pdf", None),
                     (LIT / "hong2025sokeval_v1.pdf", None)],
 "hiddenlayer2025ape": [(LIT / "hiddenlayer2025ape_ape.json", None),
                        (LIT / "hiddenlayer2025ape_README.md", None),
                        (LIT / "hiddenlayer2025ape_about.html", None),
                        (LIT / "hiddenlayer2025ape_blog.html", None)],
 "wang2026guardrails": [(LIT / "wang2026guardrails.pdf", None)],
 "xu2026sokjailbreak": [(LIT / "xu2026sokjailbreak.pdf", None)],
 "wang2026landscape": [(LIT / "wang2026landscape.pdf", None)],
 "maloyan2026coding": [(LIT / "maloyan2026coding.pdf", None)],
 "promptware2026killchain": [(LIT / "promptware2026killchain.pdf", None)],
 "giarrusso2026guardrails": [(LIT / "giarrusso2026guardrails.pdf", None)],
 "mlcommons2026jailbreak": [(LIT / "mlcommons2026jailbreak.pdf", None),
                            (LIT / "mlcommons2026jailbreak_taxonomy.yaml", None),
                            (LIT / "mlcommons2026jailbreak_attacks.yaml", None),
                            (LIT / "mlcommons2026jailbreak_overview.md", None)],
 "iyer2026talk": [(LIT / "iyer2026talk.pdf", None)],
 "owasp2026llm01": [(LIT / "owasp2026llm01.pdf", None)],
 "mitreatlas": [(LIT / "mitreatlas_ATLAS-2026.08.yaml", None),
                (LIT / "mitreatlas_ATLAS-data.yaml", None)],
 "debenedetti2024satml": [(LIT / "debenedetti2024satml.pdf", None)],
 "trustauth2025bip": [(LIT / "trustauth2025bip.pdf", None)],
 "yu2024dontlisten": [(LIT / "yu2024dontlisten.pdf", None)],
 "li2024mhj": [(LIT / "li2024mhj.pdf", None)],
 "inie2025summon": [(LIT / "inie2025summon.pdf", None)],
 "lin2025achilles": [(LIT / "lin2025achilles.pdf", None)],
 "dziemian2026vulnerable": [(LIT / "dziemian2026vulnerable.pdf", None)],
 "khodayari2026wild": [(LIT / "khodayari2026wild.pdf", None)],
 "paloalto2025securing": [(LIT / "paloalto2025securing.pdf", None)],
 "unit42_2026fooling": [(LIT / "unit42_2026fooling.html", None)],
 "microsoft2024promptshields": [(LIT / "microsoft2024promptshields.html", None)],
 "haddix2025arcanum": [(LIT / "haddix2025arcanum_taxonomy.json", None),
                       (LIT / "haddix2025arcanum_README.md", None),
                       (LIT / "haddix2025arcanum_CHANGELOG.md", None)],
 "cisco2025framework": [(LIT / "cisco2025framework.pdf", None),
                        (LIT / "cisco2025framework_taxonomy_source.py", None)],
 # native-label card
 "NATIVE": [(LIT / f, None) for f in ("zhan2024injecagent.pdf", "debenedetti2024agentdojo.pdf",
            "zhang2024asb.pdf", "zverev2025sep.pdf", "zhang2025browsesafe.pdf",
            "geng2026piarena.pdf", "li2026agentdyn.pdf")],
 # cited in Appendix F's introduction only (no card)
 "microsoft2026failuremodes": [(LIT / "microsoft2026failuremodes.pdf", None)],
 "owasp2025agenticthreats": [(LIT / "owasp2025agenticthreats_v1.0.pdf", None),
                             (LIT / "owasp2025agenticthreats_v1.1.pdf", None)],
 "owasp2025agentic": [(LIT / "owasp2025agentic.pdf", None)],
 "shi2025gemini": [(LIT / "shi2025gemini.pdf", None)],
 "chu2024comprehensive": [(LIT / "chu2024comprehensive.pdf", None)],
 "samvelyan2024rainbow": [(LIT / "samvelyan2024rainbow.pdf", None)],
 "shen2024dan": [(LIT / "shen2024dan.pdf", None)],
}


def text_of(p):
    if p.suffix == ".pdf":
        return subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True).stdout
    t = p.read_text(encoding="utf-8", errors="ignore")
    if p.suffix in (".html", ".htm"):
        t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
        t = re.sub(r"<[^>]+>", " ", t)
        t = re.sub(r"&amp;", "&", t)
    return t


def norm(s):
    s = s.replace("-\n", "").replace("\u00ad", "")
    s = s.lower()
    s = re.sub(r"\\[_&%#]", lambda m: m.group(0)[1], s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = s.replace("``", "").replace("''", "").replace("’", "'").replace("‘", "'")
    s = re.sub(r"[-_/–—]", " ", s)
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def names(cat):
    cat = re.sub(r"\([^)]*\)", " ", cat)            # drop parenthetical annotations
    cat = re.sub(r"^[A-Za-z0-9 .\-]+:\s", "", cat)   # 'BrowseSafe: ...' prefixes
    return [c.strip() for c in re.split(r";", cat) if c.strip()]


def main():
    data = json.loads(DATA.read_text())
    report, missing_files, copied = [], [], []
    texts = {}
    for key, files in M.items():
        t = ""
        for src, dest in files:
            if not src.exists():
                missing_files.append((key, str(src)))
                continue
            if COPY and dest:
                target = LIT / dest
                if not target.exists():
                    shutil.copy2(src, target)
                    copied.append((key, dest, src.stat().st_size))
            t += "\n" + text_of(src)
        texts[key] = norm(t)
    for s in data["schemes"]:
        key = "NATIVE" if s.get("native") else s["key"]
        src = texts.get(key, "")
        exact, weak, miss = [], [], []
        for r in s["rows"]:
            if r["kind"] != "unit" or r["place"].get("see"):
                continue
            for n in names(r["category"]):
                nn = norm(n)
                if not nn:
                    continue
                variants = {nn, nn.rstrip("s"), nn + "s", nn.replace(" attacks", " attack"),
                            nn.replace(" attack", " attacks")}
                if any(v and v in src for v in variants):
                    exact.append(n)
                else:
                    words = [w for w in nn.split() if len(w) >= 4]
                    if words and all(w in src for w in words):
                        weak.append(n)
                    else:
                        miss.append(n)
        report.append((s.get("title"), key, len(exact), len(weak), miss))
    print("MISSING LOCAL FILES:", missing_files or "none")
    if COPY:
        print("COPIED into literature/: %d files, %.1f MB" % (len(copied), sum(c[2] for c in copied) / 1e6))
        for c in copied:
            print("   ", c[1])
    tot = [0, 0, 0]
    for title, key, e, w, m in report:
        tot[0] += e; tot[1] += w; tot[2] += len(m)
        print("%-34s exact %3d  words-only %3d  not found %2d  %s" % (title[:34], e, w, len(m), m[:6]))
    print("TOTAL names: exact %d, all words present %d, not found %d" % tuple(tot))


if __name__ == "__main__":
    main()
