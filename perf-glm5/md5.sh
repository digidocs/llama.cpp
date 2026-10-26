#!/bin/bash
# usage: TAG=name [LM=dio] md5.sh [extra server args]; logs to ~/glm5-work/<TAG>.log
# one 300-token greedy-ish request (seed 1000, reasoning_effort low; md5 covers reasoning + content), prints load time, pp/tg t/s and md5 of the reply
cd ~/llama.cpp-glm5 && . perf-glm5/common.sh
export GGML_OP_OFFLOAD_MIN_BATCH=768
L=~/glm5-work/${TAG:-md5}.log; : > $L
T0=$(date +%s)
$PREFIX ./build/bin/llama-server -a glm5 -m $M --host 127.0.0.1 --port 8011 --jinja $OFFLOAD \
 --batch-size 2048 --ubatch-size 2048 -c 250000 -np 1 --temp 1.0 --top-p 1.0 --min-p 0.01 -lv 4 -lm ${LM:-dio} \
 --threads 20 --threads-batch 20 --cpu-range 0-19 --cpu-range-batch 0-19 --cpu-strict 1 --cpu-strict-batch 1 "$@" >> $L 2>&1 &
PID=$!
until curl -sf localhost:8011/health >/dev/null; do sleep 2; kill -0 $PID 2>/dev/null || { echo DIED; tail $L; exit 1; }; done
echo "${TAG:-md5}: ready in $(( $(date +%s)-T0 )) s"
curl -s localhost:8011/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Write a Python function that parses an ISO 8601 duration string like P3DT4H12M into total seconds, with tests."}],"max_tokens":300,"seed":1000,"chat_template_kwargs":{"reasoning_effort":"low"}}' \
 | python3 -c 'import json,sys,hashlib; r=json.load(sys.stdin); t=r["timings"]; m=r["choices"][0]["message"]; c=(m.get("reasoning_content") or "")+"\n----\n"+(m.get("content") or ""); open(sys.argv[2],"w").write(c); print("%s: pp %.2f t/s, tg %.2f t/s, %d tok, md5 %s" % (sys.argv[1], t["prompt_per_second"], t["predicted_per_second"], t["predicted_n"], hashlib.md5(c.encode()).hexdigest()[:12]))' "${TAG:-md5}" ~/glm5-work/${TAG:-md5}.reply.txt
kill -INT $PID; wait $PID 2>/dev/null
