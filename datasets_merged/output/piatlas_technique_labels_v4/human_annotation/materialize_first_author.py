#!/usr/bin/env python3
"""Materialize the first-author verification labels in release-friendly files.

The first author confirmed that the class and technique sets are identical to
the labels in ``labels_quality.jsonl``.  That file is retained unchanged so its
LLM provenance remains explicit.  This script joins those certified-identical
sets to the shuffled verification sample and writes separate human-reference
files without copying the model, effort, timing, cost, or evidence metadata.
"""

import csv
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = [f"F{i}" for i in range(1, 7)]
TECH_COL = "techniques (optional, e.g. F1.T2;F2.T4)"
HEADER = ["id", "template", *CLASSES, TECH_COL, "notes"]
TECH_RE = re.compile(r"^(F[1-6]\.T\d+)")


def technique_level(code):
    match = TECH_RE.match(code.strip().upper())
    if not match:
        raise ValueError(f"Unrecognized technique code: {code!r}")
    return match.group(1)


def main():
    with open(os.path.join(HERE, "sample.json"), encoding="utf-8") as fh:
        sample = json.load(fh)

    quality = {}
    with open(os.path.join(HERE, "labels_quality.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("status") != "ok":
                raise ValueError(f"Incomplete source label for {record.get('idx')}")
            quality[record["idx"]] = sorted({technique_level(c) for c in record["codes"]})

    sample_ids = [record["id"] for record in sample]
    if len(sample_ids) != 300 or len(set(sample_ids)) != 300:
        raise ValueError("Expected 300 distinct records in sample.json")
    if set(sample_ids) != set(quality):
        missing = sorted(set(sample_ids) - set(quality))
        extra = sorted(set(quality) - set(sample_ids))
        raise ValueError(f"Sample/source ID mismatch; missing={missing}, extra={extra}")

    rows = []
    jsonl_rows = []
    for record in sample:
        codes = quality[record["id"]]
        classes = {code.split(".")[0] for code in codes}
        rows.append(
            [record["id"], record["template"]]
            + ["1" if c in classes else "" for c in CLASSES]
            + [";".join(codes), "first-author label"]
        )
        jsonl_rows.append(
            {
                "id": record["id"],
                "annotator": "first author",
                "classes": sorted(classes),
                "techniques": codes,
            }
        )

    for name in ("sheet_author_A.csv", "sheet_human_reference.csv"):
        with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(HEADER)
            writer.writerows(rows)

    with open(os.path.join(HERE, "labels_first_author.jsonl"), "w", encoding="utf-8") as fh:
        for record in jsonl_rows:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Wrote 300 first-author records to sheet_author_A.csv, "
          "sheet_human_reference.csv, and labels_first_author.jsonl")


if __name__ == "__main__":
    main()
