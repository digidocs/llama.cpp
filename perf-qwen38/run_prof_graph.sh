#!/bin/bash
# nsys + GGML_CPU_PROFILE of steady-state MTP decode: server under nsys, capture starts after DELAY s, lasts DUR s
. ~/llama.cpp-qwen38/perf-qwen38/common.sh
cd ~/llama.cpp-qwen38
tag=${1:-prof}; shift
DELAY=${DELAY:-75}; DUR=${DUR:-20}
GGML_OP_OFFLOAD_MIN_BATCH=1400 nsys profile -o ~/qwen38-work/prof/$tag --force-overwrite true \
  --trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none -y $DELAY -d $DUR \
  build/bin/llama-server -m $M --port 8013 --jinja $OFFLOAD -b 4096 -ub 4096 -c 131072 -np 1 -lm dio \
  --threads 24 --cpu-range 0-23 --cpu-strict 1 "$@" > ~/qwen38-work/prof/$tag.server.log 2>&1 &
until curl -sf localhost:8013/health >/dev/null; do sleep 2; done
echo "ready $(date +%T)"
# long greedy generation that spans the capture window
curl -s localhost:8013/completion -d '{"prompt":"Write a detailed, step-by-step explanation of how a CPU cache hierarchy works, with examples.","n_predict":1500,"temperature":0,"ignore_eos":true}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("gen %d tok %.2f t/s, draft %s/%s" % (t["predicted_n"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n")))'
pkill -INT -x llama-server; wait
