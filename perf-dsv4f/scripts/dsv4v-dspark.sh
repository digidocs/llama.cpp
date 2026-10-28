# DeepSeek-V4-Flash-Vision-Exp + its own DSpark drafter, same setup as ds4f-dspark.sh
# the dense weights DeepSeek ships in FP8 are kept in VRAM as FP8 (exact, see perf-dsv4f/FP8-HANDOFF.md)
# experts run on the CPU for batches below this size instead of being copied to the GPU (faster for short prompts)
export GGML_OP_OFFLOAD_MIN_BATCH=768

MODEL_DIR=models/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF

# unsloth publishes no drafter for this model; build it from DeepSeek's checkpoint (fetches only the 3 mtp.* shards, ~11 GB):
#   python3 llama.cpp/convert_hf_to_gguf.py --remote deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --dspark \
#     --target-model-dir <snapshot dir it prints> --outtype q8_0 --outfile $MODEL_DIR/dspark-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf

./llama.cpp-common/build/bin/llama-server  \
-a Deepseek-V4-Flash-Vision-Exp --model $MODEL_DIR/UD-Q8_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q8_K_XL-00001-of-00005.gguf --mmproj $MODEL_DIR/mmproj-BF16.gguf \
--host 0.0.0.0 --port 8001 --jinja --n-gpu-layers 999 -ncmoe 999 \
--batch-size 4096 --ubatch-size 4096 -c 450000 -np 1 \
--flash-attn on --temp 1.0 --top-p 1.0 --min-p 0.01 -v -lv 4 -lm dio \
-ot 'blk\.[0-9]+\.(attn_output_[ab]|attn_q_[ab]|attn_kv|indexer\.attn_q_b|ffn_(gate|up|down)_shexp)\.weight=CUDA0_FP8' \
--threads 20 --threads-batch 20 --cpu-range 0-19 \
--cpu-range-batch 0-19 --cpu-strict 1 --cpu-strict-batch 1 \
-md $MODEL_DIR/dspark-DeepSeek-V4-Flash-Vision-Exp-Q8_0.gguf \
--spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0.7 --spec-draft-n-ubatch 512 -ngld 999 -ncmoed 999 \
-td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1
