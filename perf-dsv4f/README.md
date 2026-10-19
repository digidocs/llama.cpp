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

Recommended for ds4f.sh: `--threads 20 --cpu-range 0-19` (24 is the same within noise and leaves no cores free).

### 256-bit vs 512-bit kernel (A/B, only libggml-cpu swapped)

| | real-weight layer, t=16 | t=20 | t=24 | full model tg128 t=16 | t=20 | t=24 |
|---|---|---|---|---|---|---|
| ymm (default) | 855 us | 785 us | 781 us | 15.08 +- 0.30 | 16.10 | 16.18 |
| zmm (`-DGGML_MXFP4_ZMM`) | 870 us | 802 us | 784 us | 15.58 +- 0.05 | 16.12 | 16.06 |

zmm is ~4% faster single-threaded, but at 16+ threads the kernel is DRAM bound (~102 GB/s of ~122) and the two are equal within noise. ymm stays the default: no AVX-512 license transitions (relevant when other work shares the cores), and it is the smaller change.

## Where the time goes (baseline, ~90 ms/token)

| part | ms/token | notes |
|---|---|---|
| CPU routed experts (`MUL_MAT_ID`, MXFP4) | ~56 | 6 experts x 3 mats x 43 layers = 3.45 GB/token, only ~61 GB/s |
| GPU (CUDA graphs) | ~26 | 44 graph launches/token, ~3900 kernels/token |
| sync / transfers / gaps | ~8 | 88 splits/token; H2D ~4.3 MB/token (small) |

CPU and GPU work is strictly serial; neither overlaps the other.

GPU kernel split per token (nsys, node-level graph trace): `mul_mat_vec_f` BF16 15.3 ms, `mul_mat_f` BF16 3.7 ms, many small ops ~1-3 us each, flash attn 0.6 ms. 13 GB of dense weights on GPU; BF16 matvec runs at ~470 GB/s, roughly half of 3090 bandwidth.

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
- `ana.py`: per-token GPU busy/idle from an nsys sqlite export.
  ```
  nsys profile --trace=cuda --sample=none --cpuctxsw=none -o tg ./build/bin/llama-bench ... -p 0 -n 32
  nsys export --type sqlite -o tg.sqlite tg.nsys-rep
  python3 perf-dsv4f/ana.py tg.sqlite 44 20      # 44 graph launches per token, last 20 tokens
  # add --cuda-graph-trace=node to nsys to get per-kernel times inside the graphs
  ```
- perf works (`perf_event_paranoid=1`); `sudo -n pcm-memory` works for DRAM bandwidth. During decode the whole-token average is ~41 GB/s read, because the CPU is idle while the GPU runs.

## Where to go next (ordered by expected payoff)

### 1. (done) ggml MUL_MAT_ID ran well below the kernel's speed
Fixed by the private-src1 change above. The debugging path, for reference: per-op profiler -> real-weight harness reproducing the in-model time -> direct loop vs ggml (same buffers, same kernel) -> gap is zero at 1 thread and grows with threads -> not frequency (`core_power.lvl2_turbo_license` = 0) -> identical instruction counts but 2.3x more L2 demand misses -> thread-private copy of y removes it.
The same pattern (one quantized src1 shared by all threads at small batch) exists in plain `ggml_compute_forward_mul_mat`. It doesn't matter here (dense weights are on the GPU), but it likely affects CPU-only decode of any model.

### 2. Overlap GPU and CPU work (~4-5 ms/token)
The shared expert (3 x 4096x2048 BF16 per layer, ~2.2 GB/token) could run on the GPU while the CPU does the routed experts. This needs either graph reordering in `src/models/deepseek4.cpp` plus a scheduler change (record the event for the CPU split's inputs before the shared-expert part of the GPU split), or an async CPU split. More generally, CPU and GPU are each idle about 30-65% of every token.

### 3. GPU segment efficiency (~26 ms/token)
BF16 `mul_mat_vec_f` reaches ~470 GB/s; check per-matrix achieved bandwidth (small N, launch shape) and fusion opportunities. About 3900 kernels per token, mostly 1-3 us ops (rms_norm, bin_bcast, concat, get_rows, scale, rope); hyper-connection ops (`dsv4_hc_*`) and concat could be fused further. Constraint: no extra VRAM.

### 4. Spin-wait while the GPU runs
33% of CPU samples are `ggml_graph_compute_secondary_thread` spinning during GPU segments, plus 5% in `ggml_barrier`. That costs no time directly, but it uses package power and turbo headroom that the expert phase could have. Try a short spin then sleep (poll level) in the CPU threadpool, measured with full-model tg.

### 5. MTP / speculative decoding
With experts on CPU, verifying 2 tokens means reading up to 2x the unique experts. Check `test-backend-ops perf` with n = 2/4 (gate n=2: 302 GFLOPS vs 195 at n=1, so the batched kernel already amortizes well). Repeat the profiling with the MTP setup once the plain-decode work settles.

## Housekeeping
- `build-noomp/`, `build-swap/` (non-OpenMP libggml-cpu), `build-zmm/` and `build-swap-zmm/` (zmm libggml-cpu, used via `LD_LIBRARY_PATH=$PWD/build-swap-zmm`) are scratch; delete when done. Non-OpenMP has not been retested since the private-src1 fix.
- The `-lm none` load takes ~150 s; use `moe_real` and `test-backend-ops perf` for iteration and the full `llama-bench` only to confirm.
