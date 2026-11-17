#!/bin/bash
# perf record of steady-state MTP decode: server, long greedy generation, sample all threads for DUR s after DELAY s
. ~/llama.cpp-qwen38/perf-qwen38/common.sh
cd ~/llama.cpp-qwen38
tag=${1:-perf}; shift
DUR=${DUR:-15}
GGML_OP_OFFLOAD_MIN_BATCH=1400 build/bin/llama-server -m $M --port 8013 --jinja $OFFLOAD -b 4096 -ub 4096 -c 131072 -np 1 -lm dio \
  --threads 24 --cpu-range 0-23 --cpu-strict 1 "$@" > ~/qwen38-work/prof/$tag.server.log 2>&1 &
PID=$!
until curl -sf localhost:8013/health >/dev/null; do sleep 2; done
curl -s localhost:8013/completion -d '{"prompt":"Write a detailed, step-by-step explanation of how a CPU cache hierarchy works, with examples.","n_predict":1200,"temperature":0,"ignore_eos":true}' \
  | python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print("gen %d tok %.2f t/s" % (t["predicted_n"], t["predicted_per_second"]))' &
C=$!
sleep 10
perf record -q -F 1999 -g --call-graph lbr -p $PID -o ~/qwen38-work/prof/$tag.perf.data -- sleep $DUR 2>&1 | tail -2
wait $C
kill -INT $PID; wait $PID
echo "server pid $PID"
