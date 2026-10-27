#!/bin/bash
# usage: [MODEL=gguf] TAG=name [LM=none] md5_q8.sh [extra server args]; logs to ~/dsv4f-work/<TAG>.log
# Q8_K_XL no draft, one 300-token request, prints t/s and md5 of the reply (reference 1129672c4d2a)
B=$(cd "$(dirname "$0")/.." && pwd) # the worktree this script is in
cd ~
export GGML_OP_OFFLOAD_MIN_BATCH=768
L=~/dsv4f-work/${TAG:-why8}.log; : > $L
$PREFIX $B/build/bin/llama-server -a ds4f \
 --model ${MODEL:-models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf} \
 --host 127.0.0.1 --port 8011 --jinja --n-gpu-layers 999 -ncmoe 999 --batch-size 4096 --ubatch-size 4096 -c 450000 -np 1 \
 --flash-attn on --temp 1.0 --top-p 1.0 --min-p 0.01 -lv 4 -lm ${LM:-dio} \
 --threads 20 --threads-batch 20 --cpu-range 0-19 --cpu-range-batch 0-19 --cpu-strict 1 --cpu-strict-batch 1 "$@" >> $L 2>&1 &
PID=$!
until curl -sf localhost:8011/health >/dev/null; do sleep 3; kill -0 $PID 2>/dev/null || { echo DIED; tail $L; exit 1; }; done
curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into total seconds, with tests."}],"max_tokens":300,"seed":1000,"chat_template_kwargs":{"enable_thinking":false}}' \
 | python3 -c 'import json,sys,hashlib; r=json.load(sys.stdin); t=r["timings"]; c=r["choices"][0]["message"]["content"]; print("%s: tg %.2f t/s, %d tok, md5 %s" % (sys.argv[1], t["predicted_per_second"], t["predicted_n"], hashlib.md5(c.encode()).hexdigest()[:12]))' "${TAG:-why8}"
kill -INT $PID; wait $PID 2>/dev/null
