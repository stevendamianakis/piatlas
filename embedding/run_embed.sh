#!/bin/bash
# Embed the templates with the three encoders of Section VII (no truncation, bf16, one GPU).
# Input: merged_texts.json, a JSON list with the template_instruction of each template.
# F2LLM serves the plain and the steered configuration with shared weights.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STEER="Represent the rhetorical strategy of this text, not its topic or task."
CUDA_VISIBLE_DEVICES=0 $PY -u embed_full_notrunc.py --model codefuse-ai/F2LLM-v2-14B \
  --config "out_f2llm_plain_full::" \
  --config "out_f2llm_steer_full::${STEER}" > embed_f2llm_nt.log 2>&1 &
FP=$!
CUDA_VISIBLE_DEVICES=0 $PY -u embed_full_notrunc.py --model Qwen/Qwen3-Embedding-8B \
  --config "out_qwen_full::" > embed_qwen_nt.log 2>&1 &
QP=$!
wait $FP $QP
