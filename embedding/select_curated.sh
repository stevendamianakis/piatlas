#!/bin/bash
# Keep the rows of the 4,663 curated templates (drop LLMail-Inject), as used in Section VII.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY - <<'PYEOF'
import json, os, numpy as np
idx = np.array(json.load(open("keep_idx_noll.json")))
for src, dst in [("out_f2llm_plain_full","out_f2llm_plain_noll"),
                 ("out_f2llm_steer_full","out_f2llm_steer_noll"),
                 ("out_qwen_full","out_qwen_noll")]:
    os.makedirs(dst, exist_ok=True)
    e = np.load(os.path.join(src, "emb_full.npy"))
    np.save(os.path.join(dst, "emb_full.npy"), e[idx])
    print(f"  {src} {e.shape} -> {dst} {e[idx].shape}", flush=True)
PYEOF
