#!/bin/bash
# usage: TAG=name [WHYENV=VAR=1] md5_q4_dspark.sh [extra server args]; logs to ~/dsv4f-work/<TAG>.log
# one 600-token request with Q4_K_XL + DSpark n3/p0.7, prints t/s, draft counts and md5 of the reply (reference 349af7bedf9c)
cd ~
export GGML_OP_OFFLOAD_MIN_BATCH=768 ${WHYENV:-MD5_TEST=1}
L=~/dsv4f-work/${TAG:-why}.log; : > $L
./llama.cpp/build/bin/llama-server -a ds4f \
 --model models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-0731-UD-Q4_K_XL-00001-of-00005.gguf \
 --host 127.0.0.1 --port 8011 --jinja --n-gpu-layers 999 -ncmoe 999 \
 --batch-size 6144 --ubatch-size 6144 -c 450000 -np 1 --flash-attn on --temp 1.0 --top-p 1.0 --min-p 0.01 -lv 4 -lm dio \
 --threads 20 --threads-batch 20 --cpu-range 0-19 --cpu-range-batch 0-19 --cpu-strict 1 --cpu-strict-batch 1 \
 -md models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
 --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0.7 --spec-draft-n-ubatch 512 -ngld 999 -ncmoed 999 \
 -td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1 "$@" >> $L 2>&1 &
PID=$!
until curl -sf localhost:8011/health >/dev/null; do sleep 3; kill -0 $PID 2>/dev/null || { echo DIED; tail $L; exit 1; }; done
echo READY >> $L
curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into total seconds, with tests."}],"max_tokens":600,"seed":1000,"chat_template_kwargs":{"enable_thinking":false}}' \
 | python3 -c 'import json,sys,hashlib; r=json.load(sys.stdin); t=r["timings"]; c=r["choices"][0]["message"]["content"]; print("tg %.2f t/s, %d tok, draft %s/%s, text md5 %s" % (t["predicted_per_second"], t["predicted_n"], t.get("draft_n"), t.get("draft_n_accepted"), hashlib.md5(c.encode()).hexdigest()[:12]))'
kill -INT $PID; wait $PID 2>/dev/null
