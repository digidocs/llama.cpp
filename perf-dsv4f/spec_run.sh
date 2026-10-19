#!/bin/bash
# usage: OUT=<logdir> spec_run.sh <tag> <extra server args...>
S=${OUT:-/tmp}; HERE=$(cd "$(dirname "$0")" && pwd)
tag=$1; shift
cd ~
M=models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
./llama.cpp/build/bin/llama-server -a ds4f --model $M --host 127.0.0.1 --port 8011 --jinja --n-gpu-layers 999 -ncmoe 999 \
  --batch-size 4096 --ubatch-size 4096 -np 1 --flash-attn on --temp 1.0 --top-p 1.0 --min-p 0.01 -lm none \
  --threads 20 --threads-batch 16 --cpu-range 0-19 --cpu-range-batch 0-23 --cpu-strict 1 --cpu-strict-batch 1 \
  "$@" > $S/server_$tag.log 2>&1 &
PID=$!
until curl -sf localhost:8011/health >/dev/null; do sleep 3; kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -20 $S/server_$tag.log; exit 1; }; done
echo "vram_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
python3 -I $HERE/spec_pp.py $tag
[ -n "$PP_ONLY" ] || python3 -I $HERE/spec_prompts.py $tag
echo "vram_after_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
kill -INT $PID; wait $PID 2>/dev/null
