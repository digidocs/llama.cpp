# DeepSeek-V4-Flash-0731 decode performance notes

Branch: `perf/dsv4-flash-profile` (based on master `ff5a23376`)

Goal: more decode tokens/s for `~/ds4f.sh` (UD-Q8_K_XL, experts MXFP4 on CPU via `-ncmoe 999`, dense BF16/Q8 on the RTX 3090) without losing precision, prompt processing speed, or VRAM (need >= 100K context with MTP).

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
