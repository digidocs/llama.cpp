# DeepSeek-V4-Flash-0731 decode performance notes

Branch: `perf/dsv4-flash-profile` (based on master `ff5a23376`)

Goal: more decode tokens/s for `~/ds4f.sh` (UD-Q8_K_XL, experts MXFP4 on CPU via `-ncmoe 999`, dense BF16/Q8 on the RTX 3090) without losing precision, prompt processing speed, or VRAM (need >= 100K context with MTP).

## Current status

Recommended: **`~/ds4f-q4-dspark.sh`** (UD-Q4_K_XL + DSpark Q8_0 draft with **1 drafted token**, 450K context, ub 6144, 20 threads, `GGML_OP_OFFLOAD_MIN_BATCH=768`). `~/ds4f.sh` points to it; `~/ds4f-q4.sh` is the same without a draft.

| | context | VRAM | pp (9K prompt) | tg |
|---|---|---|---|---|
| start: Q8_K_XL, master | 450K | 23.0 GB | 207 t/s | 11.0 (bench) |
| `ds4f-q4.sh` (server, no draft) | 450K | 20.7 GB | 271 t/s | 18.19 / 18.55 t/s (two runs) |
| **`ds4f-q4-dspark.sh`** (server, n-max 1) | 450K | 21.9 GB | **269 t/s** | **19.50 t/s** (84% acceptance) |

llama-bench (Q4_K_XL, 20 threads, `-lm none`): pp4096 290 t/s at ub 4096 and 6144; tg128 18.48 +- 0.51 (ub 4096) / 18.38 +- 0.51 (ub 6144). With `-lm mmap`, tg128 is 18.86 +- 0.03.

Draft settings (server, same harness, after the GPU fusions):

| draft | n-max | p-min | tg t/s | acceptance |
|---|---|---|---|---|
| none (controls) | - | - | 18.19, 18.55 | - |
| DSpark Q8_0 | **1** | 0 | **19.50** | 84% |
| DSpark Q8_0 | 2 | 0 | 18.31 | 75% |
| DSpark Q8_0 | 2 | 0.5 | 18.89 | 77% |
| DSpark Q8_0 | 3 | 0.7 | 16.51 | 86% |
| DSpark Q8_0 | 4 | 0.8 | 16.66 | 91% |

- Each drafted token costs a full set of CPU expert reads when it is verified (tokens mostly route to different experts), and a rejected one wastes them; only the GPU part is shared across the verify batch. After the GPU fusions shrank that part, 1 drafted token is the sweet spot.
- A p-min cutoff raises acceptance but was much slower at n-max 3 and 4, likely because the varying draft length changes the verify batch size every step (graph rebuilds / CUDA graph recapture). Not investigated.
- The Q2K-Q4K draft runs and the closing no-draft control were stopped before they finished. 19.50 is a single run; the run-to-run noise of tg is about +-0.4 t/s.
- No MTP head in this GGUF (no `mtp`/`nextn` tensors), so `draft-mtp` is not an option.

Decode progress, all without precision loss beyond the model quant: 11.04 (master, Q8_K_XL) -> 16.10 (CPU expert fixes) -> 17.8 (Q4_K_XL) -> 18.86 t/s (bit-exact GPU fusions + quantize reuse), llama-bench; 19.5 t/s with the server and DSpark n-max 1.

Remaining options, by size: overlap of the shared expert on the GPU with the CPU routed experts (~4.5 ms/token, ~+8%, scheduler work); rms_norm fusions (~0.6 ms total, three separate bit-exact fusions of about 0.4% each); everything else is small.

## Hardware

- Xeon Gold 6212U, 24C/48T, 1 socket, 6x 32 GB DDR4 (all channels populated)
  - measured DRAM read: ~122 GB/s (16+ threads, AVX-512 stream read)
  - Intel license clocks: AVX-512 2.6 GHz @16 cores / 3.0 GHz @12; AVX2 3.0 GHz @16 / 3.3 GHz @12
- RTX 3090 24 GB, PCIe 3.0 x16
- Build: CUDA, `GGML_NATIVE=ON` (AVX-512 BW/DQ/VL/VNNI), OpenMP ON (libgomp is linked and used; cpu mask/strict is applied inside the omp region)

## Results so far

`llama-bench -p 4096 -n 128 -b 4096 -ub 4096 -ngl 999 -ncmoe 999 -fa on -t N -C 0xFFFFFFFFFFFF --cpu-strict 1 -lm none -r 3`

| build | threads | pp4096 t/s | tg128 t/s |
|---|---|---|---|
| master baseline | 16 | 284.2 +- 0.8 | 11.04 +- 0.02 |
| + ymm VNNI kernel + prefetch | 16 | 283.8 +- 0.8 | 12.85 +- 0.34 |
| + private src1 in MUL_MAT_ID | 16 | 283.7 +- 0.8 | 15.08 +- 0.30 |
| + private src1 in MUL_MAT_ID | **20** | 284.2 +- 0.7 | **16.10 +- 0.06** |
| + private src1 in MUL_MAT_ID | 24 | 284.1 +- 1.1 | 16.18 +- 0.04 |

**+46% decode** at 20-24 threads, prompt processing unchanged, no VRAM or precision change (integer dot products are exact; only float accumulation order differs, as with any SIMD width change).

`~/ds4f.sh` now uses `--threads 20 --cpu-range 0-19` (24 is the same within noise and leaves no cores free). Prompt-processing threads are unchanged (pp is PCIe bound).

Through `llama-server` (ds4f.sh settings, 450K context, 5 chat prompts x 700 tokens, temp 1.0): **15.47 t/s**. That's a bit below llama-bench's 16.1 because of sampling and server overhead.

### 256-bit vs 512-bit kernel (A/B, only libggml-cpu swapped)

| | real-weight layer, t=16 | t=20 | t=24 | full model tg128 t=16 | t=20 | t=24 |
|---|---|---|---|---|---|---|
| ymm (default) | 855 us | 785 us | 781 us | 15.08 +- 0.30 | 16.10 | 16.18 |
| zmm (`-DGGML_MXFP4_ZMM`) | 870 us | 802 us | 784 us | 15.58 +- 0.05 | 16.12 | 16.06 |

zmm is ~4% faster single-threaded, but at 16+ threads the kernel is DRAM bound (~102 GB/s of ~122) and the two are equal within noise. ymm stays the default: no AVX-512 license transitions (relevant when other work shares the cores), and it is the smaller change.

## Q4_K_XL (recommended model)

UD-Q4_K_XL keeps the experts in MXFP4 (lossless for this model, same bytes and kernels as Q8_K_XL). Only the dense weights change: BF16 -> Q8_0 (7.9 GB instead of 14.7 GB). UD-IQ4_XS re-quantizes the gate/up experts to IQ3_S: it decoded *slower* (11.95 t/s at 20 threads; the IQ3_S CPU kernel is compute bound) and was dropped.

llama-bench, ub 4096: pp4096 288 t/s, tg128 17.8 (20 threads) / 18.1 (24 threads), vs Q8_K_XL 284 / 16.1.

Server runs (`spec_run.sh` with `MODEL=` override, `GGML_OP_OFFLOAD_MIN_BATCH=768`, 20 threads everywhere; pp = 9,094-token prompt, tg = 5 chat prompts x 700 tokens). Run-to-run noise on tg is about +-0.4 t/s.

| config | context | VRAM | pp t/s | tg t/s | acceptance |
|---|---|---|---|---|---|
| ub 4096 | 450K | 17.2 GB | 210 | 17.13 | - |
| ub 4096, 24 threads | 450K | 17.2 GB | 211 | 17.15 | - |
| ub 4096 | 250K | - | - | 16.58 | - |
| **ub 6144** | 450K | 20.7 GB | **270** | 16.70 | - |
| ub 8192 | 450K | OOM (13.5 GB compute buffer) | | | |
| ub 8192 | 250K | 18.6 GB | 285 | 16.51 / 16.63 | - |
| DSpark n3, ub 4096 | 450K | 18.4 GB | 206 | 18.28 | 65% |
| DSpark n3, 24 threads | 450K | 18.4 GB | - | 18.25 | 65% |
| DSpark n2, ub 4096 | 450K | 18.4 GB | - | 18.62 | 75% |
| DSpark n1, ub 4096 | 450K | 18.4 GB | - | 18.58 | 84% |
| DSpark n3, ub 8192 | 200K | 18.6 GB | 276 | 17.45 | 65% |
| DSpark n3, ub 6144 | 450K | 21.9 GB | 261 | 18.47 | 65% |
| **DSpark n2, ub 6144** | 450K | 21.9 GB | **259** | 17.85 | 75% |

Notes:
- The 9K pp test is 2 x 4096 + a 902-token tail at ub 4096 (the tail costs about as much as a full chunk); ub 6144 makes it 6144 + 2950, which is why pp jumps from 210 to 270. Long prompts approach the llama-bench numbers (288 at ub 4096, ~413 at ub 8192).
- ub 8192 at 250K and the 250K ub 4096 control decode at the same speed, so the ubatch size does not cost tg; the lower 250K numbers vs the first 450K runs look like run-to-run drift.
- DSpark n-max 1/2/3 and ub 4096/6144 all land within the noise band (17.9-18.6). n-max 2 is the middle choice.
- 24 threads gives no gain over 20 in server runs.

Launch scripts (copies in `scripts/`):
- `~/ds4f-q4.sh`: Q4_K_XL, 450K context, ub 6144. ~17 t/s, ~270 t/s pp, 20.7 GB.
- `~/ds4f-q4-dspark.sh`: the same plus DSpark Q8_0 (n-max 2, draft experts on CPU). ~18-18.5 t/s, ~260 t/s pp, 21.9 GB.

## Bit-exact GPU fusions

Rule: a fusion must give **bitwise identical** results to the unfused CUDA kernels, not just "close". Reason: this model amplifies float differences (MoE routing flips), so an end-to-end check cannot tell a bug from legitimate reordering:

| vs reference logits (`kld_check.sh`, 2 x 512 tokens) | mean KLD | max KLD | same top-1 |
|---|---|---|---|
| identical rerun | 0.000000 | 0.000039 | 100% |
| `GGML_CUDA_DISABLE_FUSION=1` (upstream fusions reorder fp) | 0.0039 | 0.11 | 98.8% |

The nonzero floor of an identical rerun comes from the 16-bit logit storage of `--kl-divergence-base`. Each fusion is verified three ways:
1. `perf-dsv4f/bitexact/*`: fused vs unfused CUDA outputs compared with memcmp/cmp, random data incl. -0, denormals and saturation, plus a negative control that must fail (the CPU implementation for comp_pool; a deliberately FMA-contracted kernel for the HC gate), and an nsys check that the fused kernel actually ran
2. `test-backend-ops` (CUDA vs CPU reference within tolerance)
3. `kld_check.sh` on the decode path (UB=1) and the prefill path (UB=512): must equal the identical-rerun floor exactly

Iteration: `LOAD_MODE=mmap` keeps the model in the page cache, ~74 s per model-level check (vs ~3.8 min with `-lm none`).

| fusion | kernels/token | Q4_K_XL tg128 |
|---|---|---|
| baseline | 4398 | 18.03 |
| `DSV4_COMP_POOL` op (compressor gather/softmax/weighted sum; `LLAMA_DSV4_COMP_POOL=0` disables) | | 18.61 |
| CUDA fusion MUL -> ADD -> SIGMOID -> SCALE (hyper-connection gates) | 3258 | 18.78 |
| reuse of the q8_1 quantization of src1 across mul_mat_vec_q calls (`GGML_CUDA_NO_MMVQ_SRC1_REUSE=1` disables) | 3091 | **18.86** |

GPU busy per token 18.25 -> 15.10 ms. How exactness was achieved:
- comp_pool: one warp per output replays soft_max_f32 and reduce_rows_f32 exactly: same block sizes (32/128 for the softmax, 32 or 512 for sum_rows depending on rows vs SM count), same per-thread partial sums and butterfly / inter-warp trees, `vals*(1/sum)` form, and the appended zero/-inf rows computed as `0*x` like the scale kernel. A first version that emulated this with one thread per output was exact but slower than the unfused chain (17.80 vs 17.96).
- hc_gate: `__fmul_rn`/`__fadd_rn` to stop contraction, the same sigmoid expression, and `__fmaf_rn` because `scale_f32` compiles to FFMA.FTZ (checked in the SASS). The first matcher never fired because a view node sits between MUL and ADD in the node list; the nsys kernel check caught that (the "identical" result was vacuous).

- mmvq src1 reuse: quantize_q8_1 per token 511 -> 344. The cache holds one src1 per graph evaluation, keyed by tensor, data pointer, shape/strides, padding and stream; any node writing memory overlapping src1 invalidates it. Bitwise identical by construction (same quantization); the harness rewrites src1 in place between the calls, and a build with invalidation disabled is detected. A multi-entry cache would only remove 85 more per token (~0.12 ms), not worth it.

Next candidates: rms_norm (365/token, 0.91 ms), concat/get_rows/set_rows/rope (~780/token, ~1.3 ms).

## Where the time goes now (~62 ms/token at 20 threads)

| part | ms/token | notes |
|---|---|---|
| CPU routed experts | ~34 | 30.3 ms in the matmul loops at ~114 GB/s (DRAM ceiling ~122), plus ~2.6 ms of per-op overhead (quantize, routing, barriers, GLU) |
| GPU | ~26 | unchanged |
| sync / handoffs | ~2-3 | |

Per layer at 20 threads (`moe_real`, `GGML_CPU_PROFILE=1`): matmul loops 704 us, quantize 11 us, routing + barrier 10 us, barrier waits 34 us, GLU 5 us.

## Where the time went (baseline, ~90 ms/token)

| part | ms/token | notes |
|---|---|---|
| CPU routed experts (`MUL_MAT_ID`, MXFP4) | ~56 | 6 experts x 3 mats x 43 layers = 3.45 GB/token, only ~61 GB/s |
| GPU (CUDA graphs) | ~26 | 44 graph launches/token, ~3900 kernels/token |
| sync / transfers / gaps | ~8 | 88 splits/token; H2D ~4.3 MB/token (small) |

CPU and GPU work is strictly serial; neither overlaps the other.

GPU kernel split per token (nsys, node-level graph trace): `mul_mat_vec_f` BF16 15.3 ms, `mul_mat_f` BF16 3.7 ms, many small ops ~1-3 us each, flash attn 0.6 ms. 13 GB of dense weights on GPU. (The early "~470 GB/s" estimate was total GPU time over total bytes and includes the small kernels; per-weight bandwidth is in "Where to go next" item 4.)

Why the experts don't use the CPU repack path: with `-ncmoe` the experts live in a `CUDA_Host` pinned buffer so the GPU can stream them during prompt processing (pp is PCIe bound: ~150 GB of experts per 4096-token ubatch). Repacked weights can't be offloaded, so repack would kill pp. Decode therefore runs the generic `ggml_vec_dot_mxfp4_q8_0`. `--no-host 1` (THP) gave tg 11.45 but pp dropped to 225, so it's not worth it.

## Changes on this branch

0. **The big one, `ggml_compute_forward_mul_mat_id` in `ggml/src/ggml-cpu/ggml-cpu.c`: private src1 for small batches.**
   Before, src1 was quantized to q8_0 once into the shared work buffer (each thread did a slice) and all threads then re-read that shared copy for every weight row. Under heavy weight streaming those shared lines kept getting evicted (my working theory is the Skylake-SP snoop filter back-invalidating them; not verified), and the expert matmul ran ~1.5x slower than the identical loop with thread-local data (66 vs 104 GB/s). The slowdown grew with thread count (none at 1 thread, 445 vs 285 us at 16).
   Now, when the quantized src1 is <= 32 KB (`GGML_MMID_PRIVATE_SRC1_MAX`, which covers decode and small speculative batches), every thread quantizes all of src1 into its own cache-line aligned slice. That is ~1 us of redundant work per call. The work size in `ggml_graph_plan` accounts for it, and the tiled path is skipped in this mode because it expects the shared layout. Larger batches keep the old path.
   Expert time per token: ~51 ms -> ~34 ms.

1. `ggml/src/ggml-cpu/arch/x86/quants.c`, `ggml_vec_dot_mxfp4_q8_0`:
   - New 256-bit AVX-512VL + VNNI path: abs(e2m1) LUT as u8, sign bit moved onto the q8 side with a masked subtract, then `vpdpbusd`.
   - A 512-bit (zmm) variant also exists (`-DGGML_MXFP4_ZMM`, with the same prefetch); see the A/B above.
   - Software prefetch `GGML_MXFP4_PF_DIST` (default 2048 bytes ahead). The sweep gave 0: 59.8, 256: 62.4, 512: 65.6, 1024: 66.0, 2048: 67.2 GB/s.
2. `ggml/src/ggml-cpu/ggml-cpu.c`: env-gated profiler, `GGML_CPU_PROFILE=1`, which prints at exit:
   - per-op wall / busy / barrier-wait time (thread 0), keyed by op and src0 type
   - `MUL_MAT_ID` phase breakdown (quantize, route+barrier, experts) and per-thread chunk time
   - no overhead when unset (one cached env check per graph)
3. `tests/test-backend-ops.cpp`: perf cases with the DSV4F expert shapes (256 experts, 6 used, 4096x2048 and 2048x4096, n = 1/2/4):
   `./build/bin/test-backend-ops perf -o MUL_MAT_ID -b CPU -p "type_a=mxfp4.*n_mats=256"`

Verified: all 162 MXFP4 `MUL_MAT` / `MUL_MAT_ID` cases pass vs the CPU reference and vs CUDA, and the full CPU `MUL_MAT_ID` (945) and `MUL_MAT` (2506) suites pass after the private-src1 change.

## Tools in this directory

- `moe_real.cpp`: real-weight expert harness. Loads `ffn_{gate,up,down}_exps` for N consecutive layers straight from the GGUF splits (about 9 GB per layer, a few seconds), then runs gate/up MUL_MAT_ID -> swiglu -> down on the CPU backend with random routing. It reproduces the in-model expert time exactly, without the 150 s model load.
  ```
  g++ -O2 -march=native -fopenmp -std=c++17 -Iggml/include perf-dsv4f/moe_real.cpp \
      -Lbuild/bin -lggml -lggml-cpu -lggml-base -Wl,-rpath,$PWD/build/bin -o moe_real
  ./moe_real <model-00001-of-00005.gguf> threads cpu_first layer tokens gap_us iters n_layers mode
  #   e.g. ./moe_real $M 16 0 10 1 0 400 4 0
  # mode: 0 full layer, 1 gate only (ggml), 2 down only, 3 gate only direct vec_dot loop (no ggml),
  #       4 direct loop with ggml-style 16-row blocking, 5 alternate ggml/direct each iteration
  ```
- `spec_run.sh` + `spec_prompts.py` + `spec_pp.py`: server A/B for speculative decoding. `spec_pp.py` first runs a ~9K-token prompt twice (prompt cache off) to measure pp; set `PP_ONLY=1` to skip the decode prompts. Starts `llama-server` with the ds4f.sh settings plus whatever args you pass, records VRAM, runs 5 chat prompts x 700 tokens (temp 1.0, fixed seeds, thinking off), prints t/s and draft acceptance, then stops the server with SIGINT so it prints its memory summary.
  ```
  OUT=/tmp/spec perf-dsv4f/spec_run.sh base -c 450000 -lv 4
  OUT=/tmp/spec perf-dsv4f/spec_run.sh ds3 -c 200000 -lv 4 -md models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
      --spec-type draft-dspark -ngld 999 -ncmoed 999 --spec-draft-n-max 3 --spec-draft-n-ubatch 512 \
      --threads-batch 20 --cpu-range-batch 0-19 -td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1
  ```
- `ana.py`: per-token GPU busy/idle from an nsys sqlite export.
  ```
  nsys profile --trace=cuda --sample=none --cpuctxsw=none -o tg ./build/bin/llama-bench ... -p 0 -n 32
  nsys export --type sqlite -o tg.sqlite tg.nsys-rep
  python3 perf-dsv4f/ana.py tg.sqlite 44 20      # 44 graph launches per token, last 20 tokens
  # add --cuda-graph-trace=node to nsys to get per-kernel times inside the graphs
  ```
- perf works (`perf_event_paranoid=1`); `sudo -n pcm-memory` works for DRAM bandwidth. During decode the whole-token average is ~41 GB/s read, because the CPU is idle while the GPU runs.

## Where to go next

Status of each option, with what's been measured:

| option | effort | est. win | status |
|---|---|---|---|
| ggml MUL_MAT_ID gap | - | - | **done**, the private-src1 change |
| speculative decoding (DSpark) | low-medium | measured **+14%** | works at 200K context, not at 450K; pp unchanged; see below |
| short-prompt pp (batched MXFP4 kernel + offload threshold) | - | 2x CPU pp for 32-1000 token prompts | **done**, see "Short prompts" |
| fuse up + gate + swiglu on CPU | medium | ~1 ms (~2%) | measured, not built |
| GPU small-kernel fusion | high | up to ~6 ms | not started |
| overlap shared expert with CPU experts | high | ~4.5 ms | not started |
| fewer CPU-GPU handoffs | medium | 1-3 ms | not started |
| non-OpenMP build retest | very low | 0-5% | not retested since the private-src1 fix |

### 1. (done) ggml MUL_MAT_ID ran well below the kernel's speed
Fixed by the private-src1 change above. The debugging path, for reference: per-op profiler -> real-weight harness reproducing the in-model time -> direct loop vs ggml (same buffers, same kernel) -> gap is zero at 1 thread and grows with threads -> not frequency (`core_power.lvl2_turbo_license` = 0) -> identical instruction counts but 2.3x more L2 demand misses -> thread-private copy of y removes it.
The same pattern (one quantized src1 shared by all threads at small batch) exists in plain `ggml_compute_forward_mul_mat`. It doesn't matter here (dense weights are on the GPU), but it likely affects CPU-only decode of any model.

### 2. Speculative decoding with DSpark: works at 200K context, +14% decode
The target model verifies every drafted token, so the output is exactly what plain decode would produce and precision is unaffected.

Draft files (`architecture = dflash`, 3 layers, 256 experts):
- `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (used below): 10.9 GB; experts MXFP4 (10.3 GB), dense q8_0 + markov weights
- `DeepSeek-V4-Flash-0731-DSpark-Q2K-Q4K.gguf`: 8.4 GB; experts q2_K/q4_K, no markov weights (not tested)

With `-ncmoed 999` the draft's experts go to system RAM and its VRAM weights are only 594 MB.

**Results** (`spec_run.sh`, ds4f.sh settings, 5 chat prompts x 700 tokens, temp 1.0; pp on a 9,094-token prompt):

| config | context | VRAM | tg t/s | acceptance | pp t/s |
|---|---|---|---|---|---|
| baseline | 450K | 23.0 GB | 15.47 | - | - |
| baseline | 200K | - | - | - | 207.5 |
| DSpark n-max 1 | 200K | 20.4 GB | 17.28 | 82.5% | - |
| DSpark n-max 2 | 200K | 20.4 GB | 17.50 | 73.5% | - |
| DSpark n-max 3 | 200K | 20.4 GB | **17.60** | 66.3% | **204.6** |

Required flags beyond ds4f.sh:
```
-c 200000 -md models/unsloth/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
--spec-type draft-dspark -ngld 999 -ncmoed 999 --spec-draft-n-max 3 --spec-draft-n-ubatch 512 \
--threads-batch 20 --cpu-range-batch 0-19 -td 20 -tbd 20 --cpu-range-draft 0-19 -Cbd 0xFFFFF --cpu-strict-draft 1
```

**Gotcha: every thread pool must match.** With the default thread settings DSpark ran at **3.05 t/s** (5x slower than no draft), 80% of CPU time spinning in libgomp. In speculative mode every target call is a 2+ token batch, so it uses the *batch* thread settings (ds4f.sh: 16 threads on 0-23), while plain decode uses 20 on 0-19, and the draft has its own pools. With OpenMP, alternating team sizes and pinnings made libgomp tear down and recreate threads constantly (75 threads, thread IDs vanishing mid-scan). Setting target decode, target batch, draft decode and draft batch all to 20 threads on cores 0-19 fixed it.

**VRAM:** it does not fit at 450K. The draft needs its 594 MB of weights plus a ~1.1-2.3 GB compute buffer that barely shrinks with `--spec-draft-n-ubatch` (default 2,340 MiB, 512 -> 1,241, 64 -> 1,107), and only ~1.4 GB is free at 450K. At 200K the target's KV and compute buffers shrink (compute 6.2 -> 3.35 GB), leaving 3.7 GB free with the draft loaded, so ~300K probably fits too (not tested).

**Why the gain is modest:** with experts on the CPU, verifying k tokens reads ~k x the expert bytes, because tokens mostly route to different experts (`moe_real`, 20 threads: 1 token 0.80 ms/layer, 3 tokens 2.25 ms, 4 tokens 2.90 ms). Only the GPU part is shared across the batch. Each extra drafted token adds about as much CPU time as plain decode of a token, and acceptance falls (82% -> 66%), so n-max 1-3 all land at 17.3-17.6.

**Tried and dropped:** making the private-src1 copy a `memcpy` from the shared quantized buffer (to cover bigger verify batches) was 7-11% *slower* than each thread quantizing redundantly. Every thread reading lines that 20 different cores just wrote, all at once, is costly. Raising the private-src1 limit from 32 KB to 128 KB made no measurable difference. Both reverted.

Not yet tried: the Q2K-Q4K draft, `--spec-type draft-mtp` (built-in MTP head), 300K context.

### Short prompts (done): batched MXFP4 kernel + offload threshold
Time to first token for chat follow-ups was dominated by prompt processing of *short* prompts. There was no other lag: in a timed two-turn chat (`chat_timing.py` style client), wall time = server prompt time + generation time to within 0.05 s. A 197-token follow-up took 7.1 s (28 t/s).

Two paths, both slow for short prompts:
- **GPU offload** (default for batches >= 32 tokens): the used expert weights are copied over PCIe per ubatch, 3-9 s even for 32-500 tokens. Efficient only for long prompts.
- **CPU path**: was a flat ~40 t/s, compute bound (a 256-token batch reads ~140 GB of experts, ~1.3 s at full bandwidth, but took 6.3 s). `vec_dot` unpacked every 4-bit block again for every token.

Fixes:
1. **`ggml_vec_dot_mxfp4_q8_0_rows`** (x86 AVX-512 VNNI, `arch/x86/quants.c`; generic fallback in `quants.c`, mapped in `arch-fallback.h` for other archs): one weight row against up to 8 tokens. Each pair of mxfp4 blocks is unpacked once and reused for all tokens. Tokens are repacked once per call (q8 values contiguous, scales pre-broadcast) so the per-token inner loop is load, sign, `vpdpbusd`, convert, scale. Uses zmm, since this path is compute bound. Called from `ggml_compute_forward_mul_mat_id_one_chunk` for experts with 2+ tokens when the batch has >= `GGML_MXFP4_ROWS_MIN_BATCH` (16) tokens. Decode and speculative verify batches keep the plain ymm `vec_dot`: with only 1-3 tokens per expert the new path was 1-5% slower there. MXFP4 also uses 64-row chunks so the repack is amortized.
2. **`GGML_OP_OFFLOAD_MIN_BATCH=768`** in both launch scripts: batches below 768 tokens run the experts on the CPU instead of copying them to the GPU.

| prompt tokens | GPU offload | CPU before | CPU after |
|---|---|---|---|
| 32 | 9.6 t/s | 35 | **48** |
| 64 | 13.7 | 36 | **59** |
| 128 | 20.9 | 40 | **69** |
| 256 | 34.2 | 41 | **74** |
| 512 | 58 | 39 | **78** |
| 1024 | **102** | 40 | 79 |
| 2048 | **177** | 40 | 79 |

(llama-bench, 20 threads; the CPU columns use `GGML_OP_OFFLOAD_MIN_BATCH=100000`.) Real-weight harness, expert part only: 128 tokens 72.7 -> 41.4 ms/layer, 256 tokens 139.7 -> 71.3 ms/layer; 2-16 tokens unchanged within ~1%. Decode A/B: unchanged (15.3 vs 15.3 t/s on the same, somewhat noisy, runs). All 945 CPU `MUL_MAT_ID` tests and 74 MXFP4 CUDA comparisons pass. Long prompts are unaffected, because 4096-token ubatches still offload.

`scripts/` holds copies of the launch scripts `~/ds4f.sh` and `~/ds4f-dspark.sh`.

### Long prompts: measured, overlap in progress
pp4096 = 284 t/s = ~14.4 s per 4096-token ubatch. nsys of one ubatch:

| | time |
|---|---|
| expert weight copies, 125.7 GB H2D in 3946 copies | 10.20 s (12.33 GB/s while active) |
| GPU compute (flash attn 1.48 s, mul_mat_q 1.00 s, ...) | 3.85 s |
| overlap between them | **0.00 s** |

- PCIe runs Gen3 x16 under load (it idles at Gen1); pinned H2D peaks at 12.38 GB/s, so the copies are already link-bound.
- Why there's no overlap: `llama_context::sched_copy_experts` (the `ggml_backend_sched` copy callback) uploads only the experts a batch uses. It reads the router ids back with a synchronize, so a layer's experts can't start uploading before its router has run. The scheduler also waits for the previous split before overwriting a split's input copy, since the allocator reuses that memory.
- ubatch 8192: 413 t/s (+46%) at small context, but the compute buffer doubles (2.66 -> 5.32 GB at 8K ctx; ~12 GB at 450K). Dead end: out of memory.
- `--moe-cache-mib 1024` (existing LRU device cache for host experts) at 450K: out of memory on the first prompt (VRAM 24.0 GB after load). It would hold < 1% of the experts anyway.
- The partial last ubatch of a long prompt costs nearly a full copy. With `GGML_OP_OFFLOAD_MIN_BATCH=768`, tails under 768 tokens run on the CPU at ~78 t/s instead.

Remaining options: overlap uploads with compute (prefetch the next expert matrix into a dedicated device buffer; needs ~1.2 GB+ VRAM, so it fits the 200K config but not 450K; up to ~+36%), or hybrid prefill (the idle CPU computes a share of the experts while the GPU does the rest; no VRAM cost, ~+20-25%, very high effort).

### 3. CPU expert fusion: measured, small
The matmul loops already run at ~114 GB/s of ~122, and re-sweeping prefetch after the private-src1 fix confirms 2048 is still best (1-6 KB all within 2%). Fusing up + gate + swiglu into one pass would remove one quantize, one routing + barrier, one barrier wait and the GLU pass: ~25 us/layer, **~1 ms/token**. The barrier after down stays. Worth more under speculative decoding (batches of 2+ share the fused pass), but still small. The fusion hook is `ggml_cpu_try_fuse_ops` in `ggml-cpu.c`. The graph order per layer is `MUL_MAT_ID(up)`, `MUL_MAT_ID(gate)`, `GLU(swiglu_clamp)`, `MUL_MAT_ID(down)`. The clamp limit comes from `swiglu_clamp_exp`, see `build_moe_ffn` in `src/llama-graph.cpp`.

### 4. GPU segment (~26 ms/token)
- The 85 us `mul_mat_f` per layer (3.7 ms/token) is **not** a wrong-kernel problem. It's `wo_a`, the grouped attention output projection (4096 x 1024 x 8 groups, BF16, 64 MiB/layer), running at ~790 GB/s (~84% of the 3090's 936 GB/s). At most ~0.5 ms to gain.
- `mul_mat_vec_f` BF16 (15.3 ms/token): per-weight bandwidth not measured yet. Do that first (~30 min) to see if it has headroom like `wo_a` or not.
- ~3900 kernels/token, mostly 1-3 us ops (rms_norm, bin_bcast, concat, get_rows, scale, rope, `dsv4_hc_*`): ~6 ms total, the main fusion target. Constraint: no extra VRAM, no lower precision for the dense weights.

### 5. Overlap GPU and CPU work (~4.5 ms/token)
The shared expert (3 x 4096x2048 BF16 per layer, ~2.2 GB/token) could run on the GPU while the CPU does the routed experts. This needs either graph reordering in `src/models/deepseek4.cpp` plus a scheduler change (record the event for the CPU split's inputs before the shared-expert part of the GPU split), or an async CPU split. Only the shared expert can overlap; everything else in a layer depends on the previous step.

### 6. Spin-wait while the GPU runs
33% of CPU samples are `ggml_graph_compute_secondary_thread` spinning during GPU segments, plus 5% in `ggml_barrier`. Now that decode is DRAM bound, freeing turbo headroom is unlikely to help; low priority.

### Ruled out
- More experts in VRAM: no room.
- Huge pages for the expert buffer: dTLB walk cycles are < 1%.
- Memory speed: already ~87% of the theoretical DDR4-2933 rate (122 of 140.8 GB/s).
- Lower precision for dense weights: against the precision requirement.

## Housekeeping
- `build-noomp/`, `build-swap/` (non-OpenMP libggml-cpu), `build-zmm/` and `build-swap-zmm/` (zmm libggml-cpu, used via `LD_LIBRARY_PATH=$PWD/build-swap-zmm`) are scratch; delete when done. Non-OpenMP has not been retested since the private-src1 fix.
- The `-lm none` load takes ~150 s; use `moe_real` and `test-backend-ops perf` for iteration and the full `llama-bench` only to confirm.
