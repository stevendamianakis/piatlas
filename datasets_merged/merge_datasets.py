#!/usr/bin/env python3
"""
Merge all prompt-injection datasets into two JSON files:

  output/prompt_injections_whole_only_false.json
      -> rows where the injection separates into template + task.
         Deduplicated GLOBALLY by `template_instruction` (one row per unique template).

  output/prompt_injections_whole_only_true.json
      -> rows where only the whole instruction is available.
         Deduplicated GLOBALLY by `whole_instruction` (one row per unique whole string).

Each output row keeps ONLY these 10 fields:
  id, dataset, generation, pi_type, environment, pi_technique,
  whole_only, template_instruction, task_instruction, whole_instruction

`id` is reassigned sequentially (1..N) within each output file.
"""
import json
import os
import hashlib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")

# (filename, wrapper_key_for_the_list_or_None)
# Order = dedup priority: curated / smaller / human datasets first so that when
# two rows share a dedup key, the more descriptive source becomes the survivor.
# "successful" subsets are placed before their giant "full" counterparts.
SRC_ORDER = [
    ("greshake_injections_dataset.json", "injections"),
    ("agentdojo_prompt_injections.json", "rows"),
    ("agentdyn_prompt_injections.json", None),
    ("asafetybench_prompt_injections.json", None),
    ("injecagent_injections.json", None),
    ("cyberseceval_injection_instructions.json", None),
    ("openprompt_injection_instructions.json", "instructions"),
    ("asecbench_prompt_injections.json", "data"),
    ("piarena_injection_instructions.json", None),
    ("piarena_refined_injections.json", None),
    ("sep_prompt_injections.json", None),
    ("tasktracker_prompt_injections.json", None),
    ("ctf_satml24_prompt_injections_successful.json", None),
    ("ctf_satml24_prompt_injections.json", None),
    ("llmailinject_successful_prompt_injection_instructions.json", None),
    ("llmailinject_prompt_injection_instructions.json", None),
    ("gandalf_injection_instructions.json", None),
]

BIG_THRESHOLD = 50_000_000  # bytes; above this + plain array we stream line-by-line


def iter_rows(path, key):
    """Yield row dicts. Huge plain-array files are parsed one-object-per-line
    (they are written that way) to keep memory flat; everything else json.load."""
    size = os.path.getsize(path)
    if size > BIG_THRESHOLD and key is None:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s in ("[", "]", ""):
                    continue
                if s.endswith(","):
                    s = s[:-1]
                yield json.loads(s)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if key:
            data = data[key]
        for r in data:
            yield r


def infer_whole_only(r):
    """Use explicit flag when present; otherwise infer: false only when BOTH
    template_instruction and task_instruction are present and not 'none'."""
    wo = r.get("whole_only")
    if isinstance(wo, bool):
        return wo
    ti, ta = r.get("template_instruction"), r.get("task_instruction")
    if ti and ti != "none" and ta and ta != "none":
        return False
    return True


def khash(s):
    return hashlib.sha1((s if s is not None else "").encode("utf-8")).digest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    false_path = os.path.join(OUT_DIR, "prompt_injections_whole_only_false.json")
    true_path = os.path.join(OUT_DIR, "prompt_injections_whole_only_true.json")

    seen_false, seen_true = set(), set()
    id_false = id_true = 0
    contrib_false = Counter()   # dataset -> unique templates contributed
    contrib_true = Counter()    # dataset -> unique whole strings contributed
    seen_rows = dropped = 0

    with open(false_path, "w", encoding="utf-8") as ff, \
         open(true_path, "w", encoding="utf-8") as ft:
        ff.write("[\n")
        ft.write("[\n")

        for fname, key in SRC_ORDER:
            path = os.path.join(HERE, fname)
            for r in iter_rows(path, key):
                seen_rows += 1
                dataset = r.get("dataset")
                generation = r.get("generation")
                pi_type = r.get("pi_type")
                environment = r.get("environment")
                pi_technique = r.get("pi_technique", "none")
                if pi_technique is None:
                    pi_technique = "none"
                whole_instruction = r.get("whole_instruction")
                whole_only = infer_whole_only(r)

                if whole_only is False:
                    template_instruction = r.get("template_instruction")
                    task_instruction = r.get("task_instruction")
                    k = khash(template_instruction)
                    if k in seen_false:
                        dropped += 1
                        continue
                    seen_false.add(k)
                    id_false += 1
                    contrib_false[dataset] += 1
                    out = {
                        "id": id_false,
                        "dataset": dataset,
                        "generation": generation,
                        "pi_type": pi_type,
                        "environment": environment,
                        "pi_technique": pi_technique,
                        "whole_only": False,
                        "template_instruction": template_instruction if template_instruction is not None else "none",
                        "task_instruction": task_instruction if task_instruction is not None else "none",
                        "whole_instruction": whole_instruction,
                    }
                    ff.write((",\n" if id_false > 1 else "") + json.dumps(out, ensure_ascii=False))
                else:
                    k = khash(whole_instruction)
                    if k in seen_true:
                        dropped += 1
                        continue
                    seen_true.add(k)
                    id_true += 1
                    contrib_true[dataset] += 1
                    out = {
                        "id": id_true,
                        "dataset": dataset,
                        "generation": generation,
                        "pi_type": pi_type,
                        "environment": environment,
                        "pi_technique": pi_technique,
                        "whole_only": True,
                        "template_instruction": "none",
                        "task_instruction": "none",
                        "whole_instruction": whole_instruction,
                    }
                    ft.write((",\n" if id_true > 1 else "") + json.dumps(out, ensure_ascii=False))

        ff.write("\n]\n")
        ft.write("\n]\n")

    print(f"rows read           : {seen_rows}")
    print(f"duplicates dropped  : {dropped}")
    print(f"whole_only=FALSE out: {id_false}  -> {false_path}")
    print(f"whole_only=TRUE  out: {id_true}  -> {true_path}")
    print("\nunique TEMPLATES contributed per dataset (whole_only=false):")
    for ds, n in contrib_false.most_common():
        print(f"  {ds:28s} {n}")
    print("\nunique WHOLE strings contributed per dataset (whole_only=true):")
    for ds, n in contrib_true.most_common():
        print(f"  {ds:28s} {n}")


if __name__ == "__main__":
    main()
