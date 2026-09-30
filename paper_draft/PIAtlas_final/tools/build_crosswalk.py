#!/usr/bin/env python3
"""Generate the prior-taxonomy comparison from one data file.

Reads tools/crosswalk_data.json and writes
  sections/app_f_crosswalk.tex    Appendix F: one card per scheme
  sections/crosswalk_counts.tex   macros: \\xwTableRows (the body rows of Table
                                  tab:tax-compare) and every count the text uses
  tools/crosswalk_stats.json      the same counts and marks, machine-readable

The six class marks and the "Mixed-in axes" cell of each table row are computed
from the card rows, so the table cannot disagree with its card:
  full (filled circle) : some row of kind "unit" (a category of the scheme's own
                         classification) contains the class in every alternative;
  half (half circle)   : the class appears anywhere else (examples, description
                         text, or only some alternatives of a row);
  none (open circle)   : the class does not appear.
A scheme enters the counts (\\xwNschemes and the rest) only if it defines at
least one category of its own, i.e. has a row of kind "unit". A source whose
card lists only examples (the SaTML CTF report) is printed after the card-only
schemes, named in the appendix's opening paragraph, and left out of the counts.
Technique names are looked up in taxonomy_listing.tex, and every code is checked
against it. Usage (from PIAtlas_final/):
    python3 tools/build_crosswalk.py
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
LISTING = HERE / "taxonomy_listing.tex"
DATA = HERE / "tools" / "crosswalk_data.json"
OUT = HERE / "sections" / "app_f_crosswalk.tex"
COUNTS = HERE / "sections" / "crosswalk_counts.tex"
STATS = HERE / "tools" / "crosswalk_stats.json"

# ---------------------------------------------------------------- listing
NAMES = {}
for _line in LISTING.read_text(encoding="utf-8").splitlines():
    if _line.lstrip().startswith("%"):
        continue
    for m in re.finditer(r"\\textbf\{(F\d\.T\d+) ([^}]*)\.\}", _line):
        NAMES[m.group(1)] = m.group(2)
    for m in re.finditer(r"\\textit\{(F\d\.T\d+\.S\d+) ([^}:]*?)[:.]\}", _line):
        NAMES[m.group(1)] = m.group(2)
CLASS = {"F1": "instruction manipulation", "F2": "structural spoofing",
         "F3": "obfuscation", "F4": "social engineering",
         "F5": "scenario framing", "F6": "output coercion"}
KEEP_UPPER = ["ASCII", "API", "JSON", "XML", "HTML", "URL", "AI"]

# Axes other than construction. The first three are the ones Section II-E sets
# aside; the other five are properties of an attack or a system that PIAtlas
# does not model. Only "delivery" takes a qualifier.
AXES = ["delivery", "goal", "search method", "lifecycle", "trust", "target",
        "failure mode", "model access"]
DELIVERY_QUALIFIERS = {"position", "channel", "rendering", "modality", "turn structure"}

ERRORS = []


def err(msg):
    ERRORS.append(msg)


def check_code(code, where):
    if code in CLASS or code in NAMES:
        return True
    err("%s: unknown code %s" % (where, code))
    return False


def base_axis(tag, where):
    m = re.fullmatch(r"([a-z ]+?)(?: \(([a-z ]+)\))?", tag.strip())
    if not m or m.group(1) not in AXES:
        err("%s: unknown axis tag '%s'" % (where, tag))
        return None
    if m.group(2) and (m.group(1) != "delivery" or m.group(2) not in DELIVERY_QUALIFIERS):
        err("%s: bad qualifier in '%s'" % (where, tag))
    return m.group(1)


# ---------------------------------------------------------------- rendering
def nice(code):
    """Sentence-case technique name for a code such as F2.T4.S1."""
    name = NAMES[code].lower()
    for w in KEEP_UPPER:
        name = re.sub(r"\b%s\b" % w.lower(), w, name)
    return name


def render_code(code, with_badge):
    cls = code[:2]
    if code == cls:
        return "\\cb{%s}\\,%s" % (cls[1], CLASS[cls])
    text = "%s\\,\\tcode{%s}" % (nice(code), code[3:])
    return ("\\cb{%s}\\,%s" % (cls[1], text)) if with_badge else text


def show_axis(tag):
    """'delivery (modality)' is shown as 'delivery modality'."""
    m = re.fullmatch(r"([a-z ]+?) \(([a-z ]+)\)", tag.strip())
    return "%s %s" % (m.group(1), m.group(2)) if m else tag.strip()


def show_category(r):
    """The scheme's category names, separated by commas. The data separates them by
    semicolons, which tools/check_crosswalk_sources.py relies on; a row whose names
    contain commas of their own carries a "display" form instead."""
    if r.get("display"):
        return r["display"]
    return ", ".join(p.strip() for p in r["category"].split(";") if p.strip())


def render_place(place):
    """Alternatives joined by 'or'; a composition joined by '+' (in parentheses
    when the row has several alternatives); codes of one class that follow each
    other share a badge; the non-construction part comes last."""
    alts = place.get("any_of", [])
    parts, prev = [], None
    for alt in alts:
        items = []
        for code in alt:
            cls = code[:2]
            items.append(render_code(code, with_badge=(cls != prev or code == cls)))
            prev = cls
        comp = " \\,+\\, ".join(items)
        if len(alts) > 1 and len(alt) > 1:
            comp = "(%s)" % comp
        parts.append(comp)
    text = " or ".join(parts)
    note = place.get("note", "")
    outs = place.get("outside", [])
    if text and note:
        text += " (%s)" % note
    if outs:
        axes = ", ".join(show_axis(t) for t in outs)
        # a placed category that also lies on another axis is "partly outside"
        text = (text + ", \\xwalso{%s}" % axes) if text else "\\outside{%s}" % axes
        if not parts and note:
            text += ", " + note
    return text


def tex_escape_cat(s):
    return s


# ---------------------------------------------------------------- marks
def row_classes(place):
    """Classes present in every alternative, and in some alternative. A placement marked
    "partial" covers only some instances of the category, so no class is in every one."""
    alts = [set(c[:2] for c in alt) for alt in place.get("any_of", [])]
    if not alts:
        return set(), set()
    every = set.intersection(*alts)
    some = set.union(*alts)
    if place.get("partial"):
        every = set()
    return every, some


def scheme_marks(s):
    full, half = set(), set()
    for r in s["rows"]:
        every, some = row_classes(r["place"])
        if r["kind"] == "unit":
            full |= every
            half |= some - every
        else:
            half |= some
    half -= full
    return ["full" if c in full else "half" if c in half else "none"
            for c in ("F1", "F2", "F3", "F4", "F5", "F6")]


def scheme_axes(s):
    found = []
    for r in s["rows"]:
        if r["kind"] != "unit":
            continue
        for tag in r["place"].get("outside", []):
            b = base_axis(tag, s["key"])
            if b and b not in found:
                found.append(b)
    return [a for a in AXES if a in found]


MARK = {"full": "\\CIRCLE", "half": "\\LEFTcircle", "none": "$\\circ$"}


def has_categories(s):
    """True if the scheme defines a category of its own (a row of kind unit)."""
    return any(r.get("kind") == "unit" for r in s.get("rows", []))


# ---------------------------------------------------------------- validation
def validate(data):
    keys = set()
    for s in data["schemes"]:
        k = s.get("key", "")
        if not s.get("native"):
            if k in keys:
                err("duplicate key %s" % k)
            keys.add(k)
            for f in ("title", "organizes_by", "summary", "table", "rows"):
                if f not in s:
                    err("%s: missing field %s" % (k, f))
            if s.get("table", {}).get("in_table") and not has_categories(s):
                err("%s: a row of Table tab:tax-compare needs a category of the "
                    "scheme's own (a row of kind unit)" % k)
        for i, r in enumerate(s.get("rows", [])):
            where = "%s row %d" % (k or "native", i + 1)
            if r.get("kind") not in ("unit", "example"):
                err("%s: kind must be unit or example" % where)
            p = r.get("place", {})
            for alt in p.get("any_of", []):
                for code in alt:
                    check_code(code, where)
            for tag in p.get("outside", []):
                base_axis(tag, where)
            if not p.get("any_of") and not p.get("outside") and not p.get("see"):
                err("%s: empty placement" % where)
            if r.get("kind") == "unit" and not isinstance(r.get("n"), int):
                err("%s: unit row needs an integer n" % where)


# ---------------------------------------------------------------- output
HEAD = r"""% AUTO-GENERATED by tools/build_crosswalk.py from tools/crosswalk_data.json -- edit the data there.
\section{Where Every Prior Category Sits in PIAtlas}
\label{app:crosswalk}
\newcommand{\xwalso}[1]{\textcolor{black!55}{\itshape partly outside PIAtlas (#1)}}

This appendix places the categories of \xwNschemes{} prior schemes in PIAtlas,
one card per scheme, first the \xwNtable{} of Table~\ref{tab:tax-compare} and then
\xwNcardonly{} more that we also checked.@EXAMPLE_ONLY@ The last card places the
native labels of the ingested datasets. The title bar gives the scheme and what it organizes attacks by, and
a one-sentence summary follows. Each row pairs a category with its place in
PIAtlas, given as a class badge and the technique name, with the code in grey (see
Appendix~\ref{app:fulltax}). A category in \textit{italics} is one of the
scheme's own categories, named as the source names it. An entry in upright type
is an example, a published attack that the scheme files under one of its
categories, or description text, and never sets a filled circle in
Table~\ref{tab:tax-compare}. ``Or'' means that the placement depends on the
instance, and ``+'' means a composition. A grey \outside{goal} marks a category
defined on an axis other than construction, either one of the three that PIAtlas sets
aside (delivery, goal, and search method in Section~\ref{sec:scope}) or a
lifecycle stage, trust level, target, failure mode, or kind of model access,
which PIAtlas does not model. A category that is placed in PIAtlas but also
depends on such an axis is marked \xwalso{goal}. No category needed a seventh class. We also
read schemes that classify something other than attack text, namely agent
failure modes~\cite{microsoft2026failuremodes}, risks of agentic
applications~\cite{owasp2025agentic,owasp2025agenticthreats}, and ways of
generating attacks~\cite{shi2025gemini}. Their categories lie on other axes, so
they have no card. Rainbow Teaming's attack styles~\cite{samvelyan2024rainbow},
JailbreakRadar's obfuscation class~\cite{chu2024comprehensive}, and the
in-the-wild prompt communities of Shen et al.~\cite{shen2024dan} do include
construction categories. We have not carded them, but each falls within the
six classes. In the \emph{Multi-label} column of Table~\ref{tab:tax-compare}, \checkmark{}
means that an attack may carry several of the scheme's categories, and
\emph{facets} and \emph{stages} mean one category per facet or per stage of an
attack. The entries \emph{comb.}, \emph{comp.}, and \emph{fact.} mean that the scheme
combines its categories, builds an attack from its categories as parts, or
crosses them as factors, and a blank means one category per attack.

\paragraph*{Boundary cases}
Four kinds of category sit on the line between construction and another
axis. Rendering-level stealth (HTML comments, white or tiny text, text in
images) is delivery, because the model still receives ordinary text. Hiding
done with the characters themselves (invisible Unicode, ASCII smuggling)
reaches the tokenizer and is F3. A conditional or delayed directive
(``when the user says thanks, \ldots'') is an F1 construction that also plays a
lifecycle role. A multi-stage attack is a lifecycle pattern whose first stage,
the pointer to the rest of the payload, is fetch and follow (F1). A multi-turn
attack is placed turn by turn, and its turn structure is delivery.

\newtcolorbox{xwcard}[2]{enhanced, breakable, colback=white, colframe=black!28,
  boxrule=0.45pt, arc=1.5pt, outer arc=1.5pt,
  left=3.5pt, right=3.5pt, top=2.5pt, bottom=2.5pt,
  colbacktitle=hdrnavy, coltitle=white, toptitle=1.3pt, bottomtitle=1.3pt,
  fonttitle=\sffamily\bfseries\footnotesize,
  title={#1\unskip\nobreak\hfil\penalty50\hskip1em\hbox{}\nobreak\hfil\mbox{\mdseries\scriptsize\color{white!78!hdrnavy}#2}\parfillskip=0pt\finalhyphendemerits=0\relax},
  before skip=7pt, after skip=2pt, fontupper=\footnotesize}
\newcommand{\xwnote}[1]{{\raggedright\noindent\scriptsize\itshape\color{black!70}#1\par}\nopagebreak\vspace{2.5pt}}
\newcommand{\xwex}[1]{{\upshape #1}}
\newenvironment{xwrows}{\noindent\rowcolors{1}{black!4}{white}%
  \renewcommand{\arraystretch}{1.14}\setlength{\tabcolsep}{2.5pt}%
  \begin{tabular}{@{}>{\raggedright\arraybackslash\itshape}p{0.37\linewidth}%
  >{\raggedright\arraybackslash}p{0.60\linewidth}@{}}}{\end{tabular}}
"""

CHUNK = 6   # rows per tabular, so that long cards can break across columns


def card(s):
    title = s["title"] + ("~\\cite{%s}" % s["key"] if s.get("key") else "")
    lines = ["\\needspace{7\\baselineskip}",
             "\\begin{xwcard}{%s}{organizes by %s}" % (title, s["organizes_by"]),
             "\\xwnote{%s}" % s["summary"]]
    rows = s["rows"]
    for i in range(0, len(rows), CHUNK):
        lines.append("\\begin{xwrows}")
        for r in rows[i:i + CHUNK]:
            cat = show_category(r)
            if r["kind"] == "example":
                cat = "\\xwex{%s}" % cat
            place = r["place"].get("see") or render_place(r["place"])
            lines.append("%s & %s \\\\" % (cat, place))
        lines.append("\\end{xwrows}\\par")
    lines += ["\\end{xwcard}", ""]
    return "\n".join(lines)


def table_row(s):
    """One body row of Table tab:tax-compare: label, axis, scope, Der., ML,
    six class marks, and one column per axis other than construction."""
    t = s["table"]
    marks = " & ".join(MARK[m] for m in scheme_marks(s))
    used = set(scheme_axes(s))
    axes = " & ".join("\\textbullet" if a in used else "" for a in AXES)
    der = "\\checkmark" if t["der"] else ""
    ml = "\\checkmark" if t["ml"] == "✓" else t["ml"]
    return "%s~\\cite{%s} & %s & %s & %s & %s & %s & %s \\\\" % (
        t.get("label", s["title"]), s["key"], t["axis"].replace("; ", ", "),
        t["scope"].replace("; ", ", "), der, ml, marks, axes)


WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
         6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
         12: "twelve"}


def example_only_sentence(cards):
    """Sentences of the opening paragraph on the cards that list only examples."""
    if not cards:
        return ""
    names = ["%s~\\cite{%s}" % (s["title"], s["key"]) for s in cards]
    if len(names) == 1:
        return ("\nThe card for %s follows them. That source defines no categories "
                "of its own, so its card lists only examples, and we do not count it "
                "among the \\xwNschemes{} schemes." % names[0])
    listed = ", ".join(names[:-1]) + (", and " if len(names) > 2 else " and ") + names[-1]
    return ("\nThe cards for %s follow them. Those sources define no categories of "
            "their own, so their cards list only examples, and we do not count them "
            "among the \\xwNschemes{} schemes." % listed)


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    validate(data)
    if ERRORS:
        print("\n".join(ERRORS))
        sys.exit(1)
    carded = [s for s in data["schemes"] if not s.get("native")]
    native = [s for s in data["schemes"] if s.get("native")]
    # only a scheme with a category of its own is compared and counted
    schemes = [s for s in carded if has_categories(s)]
    example_only = [s for s in carded if not has_categories(s)]
    order = {k: i for i, k in enumerate(data.get("table_order", []))}
    in_table = [s for s in schemes if s["table"]["in_table"]]
    missing = [s["key"] for s in in_table if s["key"] not in order]
    if missing:
        print("not in table_order:", missing)
        sys.exit(1)
    in_table.sort(key=lambda s: order[s["key"]])
    card_only = [s for s in schemes if not s["table"]["in_table"]]
    card_only.sort(key=lambda s: order.get(s["key"], 10**6))
    example_only.sort(key=lambda s: order.get(s["key"], 10**6))

    # counts over every compared scheme (not the example-only or native-label cards)
    ncat = sum(r["n"] for s in schemes for r in s["rows"] if r["kind"] == "unit")
    nplaced = sum(r["n"] for s in schemes for r in s["rows"]
                  if r["kind"] == "unit" and r["place"].get("any_of"))
    named = {c: [s["key"] for s in in_table if scheme_marks(s)[i] == "full"]
             for i, c in enumerate(("F1", "F2", "F3", "F4", "F5", "F6"))}
    anymark = {c: [s["key"] for s in in_table if scheme_marks(s)[i] != "none"]
               for i, c in enumerate(("F1", "F2", "F3", "F4", "F5", "F6"))}
    stats = {"schemes": len(schemes), "in_table": len(in_table),
             "card_only": len(card_only),
             "example_only": [s["key"] for s in example_only], "categories": ncat,
             "placed": nplaced, "outside_only": ncat - nplaced,
             "named_in_table": named, "present_in_table": anymark,
             "marks": {s["key"]: scheme_marks(s) for s in carded},
             "axes": {s["key"]: scheme_axes(s) for s in carded}}

    head = HEAD.replace("@EXAMPLE_ONLY@", example_only_sentence(example_only))
    body = [head] + [card(s) for s in in_table] + [card(s) for s in card_only] \
        + [card(s) for s in example_only] + [card(s) for s in native]
    OUT.write_text("\n".join(body), encoding="utf-8")
    def num(n):
        """1448 -> 1{,}448, as the paper writes numbers."""
        return "{:,}".format(n).replace(",", "{,}")
    macros = {"xwNschemes": len(schemes), "xwNtable": len(in_table),
              "xwNcardonly": len(card_only), "xwNcats": num(ncat),
              "xwNplaced": num(nplaced), "xwNoutside": num(ncat - nplaced)}
    for c, word in zip(("F1", "F2", "F3", "F4", "F5", "F6"),
                       ("one", "two", "three", "four", "five", "six")):
        macros["xwNamedF" + word] = len(named[c])
    rows = "\n".join(table_row(s) for s in in_table)
    COUNTS.write_text(
        "% AUTO-GENERATED by tools/build_crosswalk.py from tools/crosswalk_data.json.\n"
        + "".join("\\providecommand{\\%s}{}\\renewcommand{\\%s}{%s}\n" % (k, k, v)
                  for k, v in macros.items())
        + "% Body rows of Table tab:tax-compare (a macro, because \\input cannot start a table row).\n"
        + "\\providecommand{\\xwTableRows}{}\\renewcommand{\\xwTableRows}{%\n" + rows + "}\n",
        encoding="utf-8")
    STATS.write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print("schemes %d (table %d, card only %d); example-only cards %d, native %d; "
          "categories %d, placed %d, outside only %d"
          % (len(schemes), len(in_table), len(card_only), len(example_only), len(native),
             ncat, nplaced, ncat - nplaced))
    for c in named:
        print("%s named (full) in %d table rows, present in %d" % (c, len(named[c]), len(anymark[c])))


if __name__ == "__main__":
    main()
