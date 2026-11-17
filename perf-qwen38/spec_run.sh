#!/bin/bash
# server A/B for speculative decoding: ~8K-token pp check (spec_pp.py) + 5 decode prompts with md5s (spec_prompts.py)
# usage: [OUT=<logdir>] [CTX=200000] spec_run.sh <tag> <extra server args...>   (SKIP_PP=1 / PP_ONLY=1, MAX_TOKENS, GREEDY=1, DEEP=<chars> near-full-context prompt)
HERE=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
cd "$HERE/.." && . perf-qwen38/common.sh
S=${OUT:-$W}; mkdir -p "$S"
export GGML_OP_OFFLOAD_MIN_BATCH=1400
./build/bin/llama-server -a qwen38 -m $M --host 127.0.0.1 --port 8011 --jinja $OFFLOAD \
  --batch-size ${UB:-4096} --ubatch-size ${UB:-4096} -c ${CTX:-131072} -np 1 --temp ${TEMP:-1.0} --top-p 0.95 --top-k 20 --min-p 0.0 -lv 4 -lm dio \
  ${SRV_CPU:---threads 24 --threads-batch 24 --cpu-range 0-23 --cpu-range-batch 0-23 --cpu-strict 1 --cpu-strict-batch 1} \
  "$@" > $S/server_$tag.log 2>&1 &
PID=$!
until curl -sf localhost:8011/health >/dev/null; do sleep 3; kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -5 $S/server_$tag.log; exit 1; }; done
echo "$tag vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
[ -n "$SKIP_PP" ] || python3 -I $HERE/spec_pp.py $tag
[ -z "$DEEP" ] || python3 -I $HERE/deep_pp.py $tag $DEEP
[ -n "$PP_ONLY" ] || REPLY_DIR=$S GREEDY=$GREEDY MAX_TOKENS=$MAX_TOKENS python3 -I $HERE/spec_prompts.py $tag
kill -INT $PID; wait $PID 2>/dev/null
