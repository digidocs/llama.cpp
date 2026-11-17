# shared settings for the Qwen3.8-Flash-Next Q8_0 checks (worktree ~/llama.cpp-qwen38)
M=${MODEL:-$HOME/models/unsloth/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
MTP_DIR=$HOME/models/unsloth/Qwen3.8-Flash-Next-GGUF/MTP
W=$HOME/qwen38-work
# GPU_EXPS: -ot pattern for experts kept on the GPU (default: none, all routed experts on the CPU)
GPU_EXPS=${GPU_EXPS:-}
OFFLOAD="-ngl 999 ${GPU_EXPS:+-ot $GPU_EXPS} -ncmoe 999 -fa on"
CPU=${CPU:-"-t 24 -C 0xFFFFFF --cpu-strict 1"}
