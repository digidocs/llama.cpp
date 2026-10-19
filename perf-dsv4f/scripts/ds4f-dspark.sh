# experts run on the CPU for batches below this size instead of being copied to the GPU (faster for short prompts)
export GGML_OP_OFFLOAD_MIN_BATCH=768

./llama.cpp/build/bin/llama-server  \
-a Deepseek-V4-Flash-0731 --model models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf \
--host 0.0.0.0 --port 8001 --jinja --n-gpu-layers 999 -ncmoe 999 \
--batch-size 4096 --ubatch-size 4096 -c 200000 -np 1 \
--flash-attn on --temp 1.0 --top-p 1.0 --min-p 0.01 -v -lv 4 -lm none \
--threads 20 --threads-batch 20 --cpu-range 0-19 \
--cpu-range-batch 0-19 --cpu-strict 1 --cpu-strict-batch 1 \
-md models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
--spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-ubatch 512 -ngld 999 -ncmoed 999 \
-td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1
