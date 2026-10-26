# shared settings for the GLM-5.3-Flash UD-Q4_K_XL checks
# experts are ~190 GB vs 187 GiB RAM: non-expert weights + the experts of MoE layer 3 go to the GPU (layer 4 too does not leave room for 250K context)
M=${MODEL:-$HOME/models/unsloth/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}
GPU_EXPS=${GPU_EXPS:-'blk\.3\.ffn_(gate|up|down)_exps\.weight=CUDA0'}
OFFLOAD="-ngl 999 -ot $GPU_EXPS -ncmoe 999 -fa on"
CPU="-t 20 -C 0xFFFFF --cpu-strict 1"
