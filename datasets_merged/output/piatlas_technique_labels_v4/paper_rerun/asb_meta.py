# Wrapper, environment and tool of every ASB template, keyed by position in the 4,663 curated
# templates (writes asb_meta_noll.json). Backs Finding G3: ASB's three map clusters hold the same
# 400 tool names and differ only by wrapper.  Usage: python3 asb_meta.py
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..")  # datasets_merged/output
d = json.load(open(os.path.join(OUT, 'prompt_injections_whole_only_false.json')))
tg = json.load(open(os.path.join(HERE, 'eval_targets.json')))
noll = {r['pos']: r['noll_pos'] for r in tg}
meta = {}
for i, r in enumerate(d):
    if r['dataset'] != 'ASB': continue
    m = re.search(r'using the "([^"]+)" tool', r['template_instruction'])
    meta[noll[i]] = {'wrapper': r['pi_technique'], 'environment': r['environment'], 'tool': m.group(1) if m else None}
print(len(meta), 'ASB templates; tools found:', sum(v['tool'] is not None for v in meta.values()), '; distinct tools', len({v['tool'] for v in meta.values()}))
json.dump(meta, open(os.path.join(HERE, 'asb_meta_noll.json'), 'w'))
