#!/usr/bin/env python3
"""Export the PIAtlas listing (taxonomy_listing.tex) as the JSON specification
that the Open Science section promises.

Usage:  python3 tools/export_taxonomy_json.py   (from paper_draft/PIAtlas_final/)
Writes: artifact/piatlas_taxonomy.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LISTING = ROOT / "taxonomy_listing.tex"
OUT = ROOT / "artifact"

# 30 Sep 2026: the listing now writes "F1 Instruction Manipulation", "Its subversion
# mechanism is ...", "It exploits ...", and "F1.T1.S1 Name." (no dashes or colons);
# the older forms are still accepted.
CLASS_RE = re.compile(r"\\subsection\*\{(F\d)(?: ---)? (.+?)\}")
MECH_RE = re.compile(r"(?:\\textit\{Subversion mechanism:\}|Its subversion mechanism is)\s*(.+?)\\\\")
EXPL_RE = re.compile(r"^(?:\\textit\{Mechanism exploited:\}|It exploits)\s*(.+)")
TECH_RE = re.compile(r"\\item \\textbf\{(F\d\.T\d+) (.+?)\.\}\s*(.*)")
SUB_RE = re.compile(r"\\item \\textit\{(F\d\.T\d+\.S\d+) (.+?)[:.]\}\s*(.*)")
# 28 Sep 2026: the listing marks entries with no labeled instance by \nolab{} (a dagger).
MARK_RE = re.compile(r"\s*(?:\\textit\{\[(reserved[^\]]*|added in v3\.1|added in v4\.0)\]\}|\\nolab\{\})\s*$")


def detex(s: str) -> str:
    """Turn the few LaTeX constructs used in the listing into plain text."""
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d").replace("~", " ")
    s = re.sub(r"\\(?:textit|emph|textbf|texttt|tech|cls)\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    s = s.replace("\\ldots", "...").replace("\\,", " ")
    s = re.sub(r"\\[a-zA-Z]+\s?", "", s)
    return re.sub(r"\s+", " ", s).replace("{", "").replace("}", "").strip()


def status_of(text: str):
    m = MARK_RE.search(text)
    if not m:
        return text, "labeled"
    if m.group(1) is None:
        return text[: m.start()], "no_labeled_instance"
    status = "reserved" if m.group(1).startswith("reserved") else m.group(1).replace(" ", "_")
    return text[: m.start()], status


def parse_listing():
    classes, cur, tech = [], None, None
    for raw in LISTING.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        # Drop the worked example (\exa{...} to end of line) and any
        # {\raggedright ...\par} line-wrap wrapper before matching.
        line = re.sub(r"\s*\\exa\{.*$", "", line)
        line = line.replace("{\\raggedright", "").strip()
        if m := CLASS_RE.search(line):
            cur = {"code": m.group(1), "name": detex(m.group(2)),
                   "subversion_mechanism": "", "mechanism_exploited": "",
                   "techniques": []}
            classes.append(cur)
            tech = None
        elif (m := MECH_RE.search(line)) and cur is not None:
            cur["subversion_mechanism"] = detex(m.group(1)[:1].upper() + m.group(1)[1:])
        elif (m := EXPL_RE.search(line)) and cur is not None:
            cur["mechanism_exploited"] = detex(m.group(1)[:1].upper() + m.group(1)[1:])
        elif m := TECH_RE.match(line):
            definition, status = status_of(m.group(3))
            tech = {"code": m.group(1), "name": detex(m.group(2)),
                    "definition": detex(definition), "sub_techniques": []}
            if status != "labeled":
                tech["status"] = status
            cur["techniques"].append(tech)
        elif m := SUB_RE.match(line):
            definition, status = status_of(m.group(3))
            tech["sub_techniques"].append({
                "code": m.group(1), "name": detex(m.group(2)),
                "definition": detex(definition), "status": status})
    return classes


def main():
    classes = parse_listing()
    n_tech = sum(len(c["techniques"]) for c in classes)
    subs = [s for c in classes for t in c["techniques"] for s in t["sub_techniques"]]
    assert len(classes) == 6 and n_tech == 34 and len(subs) == 131, (len(classes), n_tech, len(subs))
    for s in subs:  # the labeled/unlabeled marks came from the earlier labels; the paper no longer uses them
        s.pop("status", None)
    spec = {
        "name": "PIAtlas",
        "axis": "construction: how the injected instruction is composed to subvert the model",
        "excluded_axes": ["delivery vector or channel", "attacker goal",
                          "search or optimization method (black-box or white-box)",
                          "rendering-level tricks that hide text from humans or parsers without changing the model's input",
                          "turn structure of multi-turn attacks",
                          "input modality (text in images or audio is classified once extracted)"],
        "codes_are_permanent": True,
        "counts": {"classes": 6, "techniques": n_tech, "sub_techniques": len(subs)},
        "labeling_rule": "multi-label: a record receives every class whose surface signature it matches",
        "span_rule": ("the span, the shortest stretch of text that performs one construction, is the unit: "
                      "when definitions in two classes fit one span, the class whose part of the text the span "
                      "changes applies (Appendix G gives the recurring cases); within a class, the more specific "
                      "sub-technique applies"),
        "classes": classes,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "piatlas_taxonomy.json").write_text(
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(spec["counts"]))


if __name__ == "__main__":
    main()
