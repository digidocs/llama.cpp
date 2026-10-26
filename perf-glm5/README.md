# GLM-5.3-Flash UD-Q4_K_XL decode profile

Xeon Gold 6212U (24 cores, 6ch DDR4-2933, ~122 GB/s measured read) + RTX 3090, 187 GiB RAM.
Non-expert weights (Q8_0) and the experts of MoE layer 3 on the GPU, the experts of the other 41 MoE layers in
pinned host memory (`-ncmoe 999`), `-c 250000`, `-lm dio`. Scripts: `md5.sh`, `kld_check.sh` (settings in `common.sh`).

## Baseline (branch glm5-base: upstream d854d98ba + parallel loading/pinning)

- ready in 88 s, md5 `3241ffcdeb80` (server, seed 1000, reasoning_effort low, 300 tokens), tg 10.70 t/s at 20 threads
- KLD reference `~/glm5-work/base250k-ub1.kld`, PPL 3.2347; self-compare: mean KLD 0, max 5.5e-5, same top 100%
- moving one layer's experts between CPU and GPU: mean KLD ~0.012-0.013, same top ~96.5%, max ~0.5, PPL +-1% with
  no consistent direction (3.279 layers 3+4 on GPU, 3.235 layer 3, 3.269 none). All 18979 test-backend-ops pass.
  Use ~0.012 as the size of a CPU <-> GPU path change, and ~0 for pure reorderings.

## Threads and the per-thread src1 copy (llama-bench tg64 / pp512, t/s)

| threads | shared src1 (upstream) | per-thread src1 (45f01bc86) |
|--:|--|--|
| 16 | 11.39 / 65.9 | 12.59 / 66.4 |
| 20 | 10.79 / 76.4 | 13.47 / 76.3 |
| 24 | 12.46 / 80.6 | **13.68** / 80.5 |

`GGML_MMID_NO_PRIVATE_SRC1=1` restores the shared copy (branch glm5-prof).

## Where a token goes (24 threads, per-thread src1, 13.6 t/s = 73 ms)

| part | ms/token | notes |
|--|--:|--|
| CPU expert matmul loops | ~41.6 | 41 layers x 8 experts x 15.3 MB = 5.0 GB at ~120 GB/s: at the DRAM ceiling |
| CPU per-op overhead | ~7.5 | src1 quantize ~3, thread imbalance wait in q4_K MUL_MAT_ID ~3, GLU ~0.75 |
| GPU kernels | ~18.6 | mul_mat_vec_q 11.9 (~9.6 GB dense Q8_0 at ~800 GB/s of 936), quantize_q8_1 0.9, rms_norm 0.9, bin_bcast 0.7, ... |
| handoffs / launch gaps | ~5 | |

CPU and GPU run strictly in turn: under nsys (85 ms/token with tracing) the GPU is busy 19.8 ms and idle 65 ms per
token, in 82 gaps > 200 us (two per CPU layer, median 1.2 ms). Transfers are small: H2D 5.3 MB, D2H 1.2 MB per token.

Both sides are bandwidth bound, so the levers are overlap (GPU work that does not depend on the CPU experts,
e.g. the shared expert), less CPU overhead, fewer bytes per token, and speculative decoding (the model has one MTP layer).

Commands:
```
GGML_CPU_PROFILE=1 ./build/bin/llama-bench -m $M -ngl 999 -ot $GPU_EXPS -ncmoe 999 -fa on -lm dio -C 0xFFFFFF --cpu-strict 1 -t 24 -p 0 -n 64 -r 1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o tg ./build/bin/llama-bench ... -p 0 -n 32
```
