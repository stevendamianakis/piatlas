#!/usr/bin/env python3
"""
Quality-oriented LLM labeling of the 300-record verification sample: Claude Opus 5.5 at high effort
(the production run used medium effort), one record per call, the production prompt (v1.1) plus
an instruction to test every signature and boundary rule before answering. This is a second LLM
labeling, not a human annotation; its agreement with the production labels is an LLM-LLM check.

Writes labels_quality.jsonl, calls_quality.jsonl, progress_quality.json, quality_config.json.
"""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.argv = [sys.argv[0]]
import run_labeling as R

QUALITY_SUFFIX = """
QUALITY MODE
Work through each record slowly. First test the six class signatures in their fixed order. Then, for
every technique you are considering, check each boundary rule that could apply and the definitions of
its neighbors, and assign it only if a quote from the record shows its construction. Do not assign a
technique on the strength of a single ambiguous word.
"""


def main():
    R.EFFORT = "high"
    R.CALLS, R.PROGRESS = HERE / "calls_quality.jsonl", HERE / "progress_quality.json"
    out = HERE / "labels_quality.jsonl"
    system_prompt = (HERE.parent / "system_prompt.txt").read_text(encoding="utf-8").rstrip() + "\n" + QUALITY_SUFFIX
    valid = set(json.loads((HERE.parent / "valid_codes.json").read_text()))
    sample = json.loads((HERE / "sample.json").read_text(encoding="utf-8"))
    texts = {r["id"]: r["template"] for r in sample}
    (HERE / "quality_config.json").write_text(json.dumps({
        "model": R.MODEL, "effort": R.EFFORT, "records": len(sample), "one_record_per_call": True,
        "system_prompt": "system_prompt.txt (v1.1) + QUALITY MODE suffix",
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(), "started_at": R.now_iso()}, indent=2))
    done = R.load(out)
    todo = [k for k in texts if k not in done]
    print(f"START {R.now_iso()} quality pass: {len(todo)} of {len(texts)} records, effort={R.EFFORT}", flush=True)
    R.run_batches([[k] for k in todo], texts, system_prompt, valid, "quality", out, workers=8, progress_every=50,
                  total=len(todo))
    final = R.load(out)
    from collections import Counter
    print(f"DONE {R.now_iso()} labeled={len(final)}/{len(texts)} {dict(Counter(r['status'] for r in final.values()))}", flush=True)


if __name__ == "__main__":
    main()
