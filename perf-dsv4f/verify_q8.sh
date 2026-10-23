#!/bin/bash
# full Q8_K_XL check with FP8 dense weights: no draft (450K) with pp, then DSpark n3/p0.7 at 450K, 5 prompts x 700 tokens each
# usage: perf-dsv4f/verify_q8.sh; compare the md5s and draft counts with the tables in FP8-HANDOFF.md; logs go to ~/dsv4f-work
export GGML_OP_OFFLOAD_MIN_BATCH=768 OUT=$HOME/dsv4f-work
R=~/llama.cpp/perf-dsv4f/spec_run.sh
OT='blk\.[0-9]+\.(attn_output_[ab]|attn_q_[ab]|attn_kv|indexer\.attn_q_b|ffn_(gate|up|down)_shexp)\.weight=CUDA0_FP8'
T="-lv 4 -lm dio --threads-batch 20 --cpu-range-batch 0-19"
D="-md models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0.7 --spec-draft-n-ubatch 512 -ngld 999 -ncmoed 999 -td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1"
$R fp8_q8 -c 450000 $T -ot "$OT"
SKIP_PP=1 $R fp8_q8dspark450 -c 450000 $T -ot "$OT" $D
