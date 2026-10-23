# FP8 dense weights for DeepSeek-V4-Flash: handoff

Branch: `feat/fp8-dense` (off `perf/dsv4-flash-profile` at `eddc42ce0`). Fork remote: `mine` = `git@github.com:digidocs/llama.cpp.git` (ask before pushing).

## Morning summary (2026-10-24)

All output byte-identical to the previous build (md5 of every reply; Q4 + DSpark also still `349af7bedf9c`). Q8_K_XL with the FP8-origin dense weights stored as FP8 in VRAM, final build `3a7caae37`:

| | before | now |
|---|---|---|
| Q8, no draft, 450K, 5 prompts x 700 tokens | 17.24 t/s | **18.59-18.68 t/s** (+8%) |
| Q8 + DSpark n3/p0.7 | 19.65 t/s at 350K | **20.23 t/s at 450K** |
| pp, 9K prompt | 208.8 | 208.2 |
| dense weights in VRAM | 13.0 GB | 7.6 GB (~19 GB used with DSpark at 450K) |
| load to ready | ~76 s | ~82 s |

- llama-bench, original command (pp4096 / tg128, Q8_K_XL, 20 threads): master at the start 284.2 / 11.04 (16 threads); now BF16 dense 284.80 / 17.56, FP8 dense 283.79 / 18.69. 22 and 24 threads are within noise of 20.
- The Q4_K_XL model and its `~` runners were deleted (2026-10-24); `kld_check.sh` now defaults to Q8_K_XL.
- `~/ds4f.sh` and `~/ds4f-dspark.sh` use the FP8 storage (`-ot ...=CUDA0_FP8`); the DSpark script is now at 450K. Backups: `~/old-scripts/*.pre-fp8`.
- Q8_K_XL + FP8 now reads about the same dense bytes as Q4_K_XL (whose dense weights are lossy Q8_0) and is within ~3% of its speed (Q4: 19.25 no draft, 20.91 DSpark), losslessly. The rest of the gap is mostly tensors that are true BF16 in Q8 (`output` head 1.0 GB vs 0.5 GB Q8_0), which cannot be stored exactly in 8 bits.
- Commits on `feat/fp8-dense` (not pushed): FP8 buffer type + BF16 expansion path (`3d610ed49`), native FP8 in mmvf/mmf (`18d0034f5`), fast exact decode + one-warp-per-row kernel (`8e4eb969f`), two rows per warp (`09dba3342`), head/tail graph capture (`3a7caae37`).
- Measured and dropped: more warps per row / more rows per warp for long rows; register double-buffering of FP8 loads in `mul_mat_f` (57 -> 71 us, occupancy); `cuStreamWaitValue32` to pre-launch graphs (stream memory ops are disabled on this driver: the attribute is 0 and the wait hangs).
- Head/tail graph capture cut the GPU wait after each per-layer launch from ~55 to ~26 us (2.7 -> 1.1 ms/token of idle GPU) but only gave +0.5% end to end.
- With DSpark the GPU is only ~18% of a verify step (CPU experts dominate, at the DRAM limit), so further GPU work mostly helps the no-draft path. Remaining GPU time per token (no draft, ~19 ms): FP8 matmuls ~8 ms, BF16 `output` head 1.2 ms (at bandwidth), ~2,100 small kernels 5.7 ms (rms_norm 1.0, flash attn 0.7, concat 0.5, ...), `attn_output_a` (mmf) 2.5 ms.
- Most frequent op pairs still run as separate kernels (decode, per token, after existing fusions): RMS_NORM(+MUL) -> ROPE 178 (`kv`, `q`, `csa/hca/lid_state_compress`), RMS_NORM(+MUL) -> MUL_MAT 130, GET_ROWS -> SET_ROWS 125 (the compressor state persist copies; source and destination are the same state cache, so a one-kernel copy is unsafe and the fusion memory check would refuse it), ADD -> CONCAT 83. A norm -> rope fusion would save ~0.3 ms/token but must reproduce each rope variant bit for bit.
- Possible next steps: the norm -> rope fusion; the `attn_output_a` tensor-core path; merging this branch back into `perf/dsv4-flash-profile` and pushing.

## Is it bit-exact?

Yes: by construction, by md5 of the generated text, and at the logit level (`kld_check.sh`, below).

**By construction**
- **Weights.** Only tensors whose every value is exactly an e4m3 number times a power-of-two scale per 32 values are stored as FP8. The scan found this all-or-nothing per tensor kind: every block of the 9 FP8-origin kinds passes, every block of the true-BF16 kinds fails. `set_tensor` checks every block on every load and aborts the load if any value is not exact, so each stored weight decodes to exactly its BF16 value.
- **Decode.** Byte permute, masks and shifts, an FP16 -> FP32 conversion and a multiply by a power of two, none of which round. No float subnormals are produced, which matters because the build uses `-use_fast_math` (flush to zero).
- **Kernels.** Each FP8 kernel performs the same floating-point operations in the same order as the BF16 kernel it replaces.
  - `mul_mat_vec_f` (FP8): the same per-thread loop and reductions as BF16.
  - `mul_mat_vec_f_vt`: reproduces the 256-thread kernel's per-thread partial sums and its warp and second-stage butterfly reductions.
  - `mul_mat_f` (FP8): feeds the same BF16 pairs into the same MMA sequence.
  - Paths without an FP8 kernel (cuBLAS for batches over 16 columns): expand to BF16 and run unchanged code.
- **Graph head/tail split.** Same kernels, same order, same stream; only the submission is split, at an op boundary, so the fusion decisions are unchanged.

**Tested (md5 of every reply equal to the previous build)**
- Q8 no draft: 5 prompts x 700 tokens (several runs), plus many 300-token runs during development.
- Q8 + DSpark at 450K: 5 prompts x 700 tokens, identical draft counts (1860/2157).
- Q4 + DSpark: the 600-token reference `349af7bedf9c`.
- BF16 weights through `mul_mat_vec_f_vt`.

Sampling is at temperature 1.0, so a change in any logit would almost always change a token within a few hundred tokens.

**Logits (`kld_check.sh`, Q8_K_XL, 2 x 512 tokens, FP8 storage vs BF16 reference logits)**

FP8 matches Q8's identical-rerun floor exactly, in every printed figure. The floor is non-zero because `--kl-divergence-base` stores the reference logits in 16 bits.

| | BF16 rerun (floor) | FP8 vs BF16 |
|---|---|---|
| decode (UB=1): mean PPL / mean ln(PPL ratio) / max KLD | 4.130426 / 0.004877 / 0.000043 | 4.130426 / 0.004877 / 0.000043 |
| prefill (UB=512): mean PPL / mean ln(PPL ratio) / max KLD | 4.150460 / 0.005704 / 0.000044 | 4.150460 / 0.005704 / 0.000044 |
| same top-1 | 100% | 100% |

Command: `MODEL=<Q8_K_XL> LOAD_MODE=dio UB=1 perf-dsv4f/kld_check.sh base ref.kld`, then `... EXTRA="-ot <FP8 regex>" perf-dsv4f/kld_check.sh cmp ref.kld fp8`. Repeat both with `UB=512`.

**Not yet tested**
- Prompts far beyond 9K tokens. None of the changed kernels depends on context length.

## Goal

Store the dense weights that DeepSeek shipped as FP8 in 8 bits in VRAM instead of the 16-bit BF16 the GGUF holds them in. That halves the bytes the GPU reads per token for those weights and frees VRAM, with **bit-identical output**.

Expected payoff on Q8_K_XL:
- Decode reads ~5.3 GB less per token on the GPU: ~6-7 ms/token, so ~17.3 -> ~19.5 t/s without drafting and ~19.7 -> ~21.5 with DSpark.
- ~5.3 GB of VRAM freed: probably enough to run Q8 + DSpark at 450K instead of 350K.

## Hard requirements (from the user)

- **Bit-identical output.** md5 of the reply text must match the current build on the reference runs below. Lossless is not enough.
- Prompt processing must not drop: Q8 ~208 t/s and Q4 ~271 t/s on the 9K prompt.
- Context >= 150K. Single RTX 3090 (sm_86, no FP8 tensor cores), single socket.
- **Upstream-friendly local design.** This is not being submitted now, but keep the option open: no hard-coded tensor names, clean refusal paths, tests.
- No Co-Authored-By trailer in commits.

## Why it is exact

The model card says, and the scan confirms: "FP8 dequantizes into BF16 with no rounding". Every BF16 value of an FP8-origin tensor is `e4m3 value * 2^k`, with a power-of-two scale per block.

MXFP8 stores exactly that: 32 e4m3 bytes plus one E8M0 (power-of-two) scale byte per block of 32 along a row. That's 33 bytes per 32 weights, 8.25 bits each. Picking the scale per 32-block from the block max is never worse than DeepSeek's 128x128 blocks.

## Scan results (`perf-dsv4f/fp8scan.py`, all BF16 tensors of UD-Q8_K_XL)

Each tensor kind is all-or-nothing: every block passes, or every block fails. Confirmed on all 43 layers (summary in `~/dsv4f-work/fp8scan.txt`).

| tensor kind | exact | size (43 layers unless noted) |
|---|---|---|
| `attn_output_a`, `attn_output_b`, `attn_q_b` | yes | 8.06 GiB |
| `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` | yes | 2.02 GiB |
| `attn_q_a`, `attn_kv`, `indexer.attn_q_b` (21 layers) | yes | 0.83 GiB |
| `attn_compressor_{gate,kv}`, `indexer_compressor_{gate,kv}`, `indexer.proj`, `ffn_gate_inp`, `output`, `token_embd` | no, 100% of blocks fail (true BF16) | ~2.6 GiB |

That gives 10.9 GiB to convert, which becomes ~5.6 GiB as MXFP8.

## Design

### 1. File stays BF16; convert at load time

A new CUDA buffer type, e.g. `CUDA0_FP8`, modelled on the CPU "repack" extra buffer types:
- It is registered as an extra buffer type of the CUDA device (`ggml_backend_dev_get_extra_bufts`), so `-ot` / `--override-tensor` can target it by regex. Example: `-ot "blk\..*\.(attn_output_[ab]|attn_q_[ab]|attn_kv|indexer\.attn_q_b|ffn_(gate|up|down)_shexp)\.weight=CUDA0_FP8"`.
- The tensor type stays `GGML_TYPE_BF16`, so the graph and model code are unchanged. Only the storage differs.
- `get_alloc_size`: MXFP8 size (rows x ne0/32 x 33 bytes).
- `set_tensor`: converts the BF16 bytes to MXFP8 and verifies every block. If any value is not exactly representable, it aborts the load with a clear message naming the tensor. Never round silently. Parallelise the conversion, since the loader already reads with 8 threads.
- `get_tensor`: converts back to BF16, for anything that reads the weights (state save, debugging).
- Op support: accept only `MUL_MAT` with the tensor as src0, contiguous, on this CUDA device. Refuse everything else (`GET_ROWS`, `CPY`, CPU fallback, `MUL_MAT_ID`) so llama keeps such tensors in the normal buffer. Check how the CPU repack buffer types do this in `ggml/src/ggml-cpu/` (`repack.cpp`, `traits`, `extra_buffer_type`) and in llama's weight buffer selection (`src/llama-model.cpp`, search `select_weight_buft` / `extra_bufts`).

### 2. Kernels: bit-identical by construction

- **Every path except plain decode, the safety net:** expand MXFP8 -> BF16 into a scratch buffer from the CUDA pool, then call the **exact same** BF16 code path as today. This covers verify batches, small prompts below `GGML_OP_OFFLOAD_MIN_BATCH` that still run dense matmuls on the GPU, and cuBLAS prefill. The values and code are identical, so the output is identical. Correct before fast.
- **Decode (the hot path):** a native kernel that reads e4m3 bytes, converts them exactly to float, multiplies by the power-of-two scale (also exact), and accumulates in **exactly the same order** as today's BF16 `mul_mat_vec_f` (`ggml/src/ggml-cuda/mmvf.cu`). That includes per-thread partial sums, the warp reduction, and the fused variants that fire on these tensors (e.g. the shared expert's gate+up+GLU fusion). If a variant can't be matched, route that case to the safety net.
- Do not quantize activations to q8_1 the way mmvq does: that changes the numbers.

### 3. Things to find out first (step 1)

- Which kernel each batch size uses today for BF16 src0 on sm_86. That means `ggml_cuda_mul_mat` dispatch in `ggml-cuda.cu`, `mmvf.cu` (`ggml_cuda_should_use_mmvf`, max columns), `mmf.cu` (tensor-core path for small batches?), and cuBLAS for large ones. Verify batches are 2-4 tokens with DSpark n-max 3.
- Which fusions fire on the convertible tensors. Search `ggml_cuda_try_fuse` and the `mmvf` fusion args: gate/up/GLU, bias, `ggml_cuda_match_shared_expert` (`MUL_MAT_ID` + shared expert fusion: note it uses mmvq for the shared part when quantized, so check what happens for BF16).
- How the mmvq src1 reuse cache and CUDA graphs interact: buffer-type dispatch must be stable across CUDA graph replays.
- Where CUDA decides "is this tensor's buffer one of mine" (`ggml_backend_buft_is_cuda`, split buffers). The new type must count as CUDA memory for scheduling and splits, and must report as not host.

### 4. Step 1 findings (read 2026-10-23)

Dispatch for a BF16 src0 on the 3090 (sm_86), from `ggml_cuda_mul_mat` in `ggml-cuda.cu`:
- `ggml_cuda_should_use_mmvf` (`mmvf.cu`): for BF16 on Ampere it returns true only when src0 is "small" (`ne[1] <= 512` or `ne[2]*ne[3] == 1`) and `ne11 == 1`. Otherwise `ggml_cuda_should_use_mmf` (`mmf.cu`) is true for `src1_ncols <= 16` (with `ne0 % 64 == 0` and rows a multiple of the rows-per-block). Above 16 columns: cuBLAS.
- **2D weights at decode (1 token): `mul_mat_vec_f`.** Per thread it loops `col2 = tid; col2 < ncols2; col2 += block_size` over pairs, converting each bf16 to float and doing `acc += w*x` (`ggml_cuda_mad(float&, float, float)`, `common.cuh:752`), then a warp/block reduction. An FP8 variant that yields the same float per weight in the same per-thread order performs the same operations, so it is bit-identical. `block_size` is chosen on the host (`mmvf.cu` ~line 439) and must be chosen the same way.
- **Fused shared expert at decode:** `ffn_gate_shexp` + `ffn_up_shexp` + GLU (swiglu with clamp) go through `mul_mat_vec_f` with `fusion.gate` (`ggml-cuda.cu` ~4446). So the FP8 mmvf must support the gate pointer, and both gate and up live in the FP8 buffer.
- **`attn_output_a` (wo_a) is created 3D** (`deepseek4.cpp:122`, `{n_head*n_embd_head/o_groups, o_lora_rank, o_groups}`, 8 groups). It is not small, so on Ampere it uses **`mul_mat_f` (tensor cores) even at batch 1**. That is the "85 us `mul_mat_f` per layer" noted in the README, 2.7 GiB of the 10.9 GiB. Native FP8 is therefore needed in **`mul_mat_f` too**: convert e4m3 -> bf16 exactly where src0 tiles are loaded, so the MMA inputs are identical. `mul_mat_f` also serves DSpark verify batches (2-16 tokens) for every weight.
- **Large batches (>16 columns): cuBLAS**, through the BF16-expansion safety net. That covers prefill and prompts below `GGML_OP_OFFLOAD_MIN_BATCH`, which still run the dense part on the GPU.
- **`-ot` resolution** (`common/arg.cpp`, `parse_tensor_buffer_overrides`) only lists each device's default buffer type. It must also list the devices' extra buffer types (via the `ggml_backend_dev_get_extra_bufts` proc address, as `src/llama-model.cpp:1229` does).
- **Overrides skip the support check:** `src/llama-model-loader.cpp` ~1243 uses the override buft directly. For an extra GPU buffer type, check support for the tensor's op and fall back with a warning, so a too-broad regex can't put e.g. `token_embd` (used by `GET_ROWS`) into FP8 storage.
- GPU extra buffer types are appended **after** the default one in `make_gpu_buft_list` (`llama-model.cpp:1229`), so they are only used when requested through `-ot`. That is the desired opt-in.

### 5. What is implemented (steps 2-3)

- `ggml/src/ggml-cuda/fp8.cuh`, `fp8.cu`: storage layout (e4m3 bytes for all elements, then one E8M0 scale per 32), exact host `ggml_cuda_fp8_pack` (returns false if any value is inexact), `ggml_cuda_fp8_unpack`, the device decode `ggml_cuda_fp8_e4m3_scaled`, and the expansion kernel `ggml_cuda_fp8_to_bf16`. Scales are limited to 2^-111..2^103 so the device decode stays in the normal float range.
- `ggml-cuda.cu`: `CUDA<n>_FP8` buffer type (`ggml_backend_cuda_fp8_buffer_type`). `set_tensor` converts the BF16 bytes; partial sets are staged until complete, because the loader uploads in chunks; any inexact value aborts. `get_tensor` converts back. memset and cpy are refused. Registered via `ggml_backend_dev_get_extra_bufts`. `supports_buft` accepts it. `supports_op` allows an FP8 weight only as src0 of a `MUL_MAT` with F32 src1/dst. Async set/get fall back to the synchronous buffer path.
- Safety net: `ggml_cuda_fp8_as_bf16` expands an FP8 weight into a pool buffer at the top of `ggml_cuda_mul_mat` and at the three fused `mul_mat_vec_f` call sites in `ggml_cuda_try_fuse` (gate/GLU twice, bias once).
- `common/arg.cpp`: `-ot` also lists the extra buffer types of non-CPU devices.
- `src/llama-model-loader.cpp`: an override to a GPU extra buffer type is checked with `weight_buft_supported`; if the weight's op isn't supported there, it keeps the default with a warning.
- Usage: `-ot 'blk\.[0-9]+\.(attn_output_[ab]|attn_q_[ab]|attn_kv|indexer\.attn_q_b|ffn_(gate|up|down)_shexp)\.weight=CUDA0_FP8'`

### 6. Performance findings (2026-10-23 night)

- First native FP8 kernels gave **no speedup**: per-call times equal to BF16. nsys with `--cuda-graph-trace=node` (without it the kernels inside CUDA graphs are invisible) showed the BF16 kernels at only 620-750 GB/s for these shapes, so they are latency/issue bound, not bandwidth bound; the FP8 decode (several integer ops per weight) then made them ALU bound.
- **Fix 1, cheaper exact decode** (`ggml_cuda_fp8x2_e4m3_scaled`): an e4m3 byte placed in FP16 exponent/mantissa bits is exactly the value times 2^-8 (subnormals too), so two weights decode with a byte permute, two masks/shifts, one `__half22float2` and one multiply by the power-of-two scale. Note the build uses `-use_fast_math` (float subnormals flush to zero), so decodes must never produce float subnormals; FP16 subnormal inputs convert exactly.
- **Fix 2, `mul_mat_vec_f_vt`**: the block-wide `mul_mat_vec_f` kernel (one row per 256-thread block) is replayed with fewer real threads: lane l of warp r does the work of the original threads l + 32*(r + R*j) and the same per-warp butterflies plus the second-stage butterfly over the zero-padded warp sums are replayed, so the result is bit-identical. R=1 (one warp per row) is faster for short rows (`attn_q_b`: 32768 rows x 1024; 108 -> 81 us BF16, 72 us FP8) and slower for long rows; it is used when each original thread has <= 2 iterations. R=2/4 for long rows is being tuned (`TMP_MMVF_VT_LONG_R`, temporary).
- Per-call times after both fixes (us, BF16 -> FP8): attn_output_b + ffn_down_shexp 49.6 -> 30.3, shexp gate+up fused 41.6 -> 25.8, attn_output_a (mmf) 85.4 -> 57.4, attn_q_b 108 -> 72. GPU busy over 300 tokens 7.30 s (original BF16) -> 5.76 s (FP8).
- Profiling recipe: `PREFIX="nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o <file>" TAG=x perf-dsv4f/md5_q8.sh [-ot ...]`, then `nsys export -t sqlite` and the per-kernel scripts `perf-dsv4f/kern.py` (totals) / `perf-dsv4f/kern2.py` (by template instance and grid).

## Verification references (current build, `perf/dsv4-flash-profile` `eddc42ce0`)

The scripts are in `perf-dsv4f/`. Logs and outputs go to `~/dsv4f-work/`. Every load uses `-lm dio` (about 76 s to ready).

Quick checks:
- `TAG=x perf-dsv4f/md5_q8.sh`: Q8 no draft, 300 tokens, md5 **`1129672c4d2a`**
- `TAG=x perf-dsv4f/md5_q4_dspark.sh`: Q4 + DSpark, 600 tokens, md5 **`349af7bedf9c`**, draft 449/422

Full runs: `OUT=~/dsv4f-work SKIP_PP=1 GGML_OP_OFFLOAD_MIN_BATCH=768 perf-dsv4f/spec_run.sh <tag> <args>`, 5 prompts x 700 tokens. Drop `SKIP_PP` to also measure pp.

| config | args | md5s (p0..p4) | t/s |
|---|---|---|---|
| Q8 no draft | `-c 450000 -lv 4 -lm dio --threads-batch 20 --cpu-range-batch 0-19` | 80a5ad2bca35 0ab7ba258774 6ae213c65435 3537ad67bbc3 e734764a3de3 | 17.24-17.27, pp 208.8 |
| Q8 + DSpark n3/p0.7 | `-c 350000` + the DSpark args from `~/ds4f-dspark.sh` | d9411523e933 534d038bf3d2 47c19460fe86 de9a583a9d6c ea8996530224 | 19.65, accepted 1860/2157 |
| Q4 no draft | `MODEL=<Q4 path>`, `-c 450000 --batch-size 6144 --ubatch-size 6144` | 26b25e492067 8d2d9ecc0e98 41975bc585ca 66714584a9da 6db50b5286a8 | 19.25, pp 271.6 |
| Q4 + DSpark | same + DSpark args | f3fbdbc276ad 48418ba7171f fd894925da3c 2bee19024663 2f4f6cca667f | 20.91, accepted 1923/2230 |

Q4_K_XL quantizes these dense tensors itself (lossy), so FP8 storage only applies to Q8_K_XL. Q4 runs are still useful as a regression check, since the code must not change Q4 output.

## Plan and status

| # | step | status |
|---|---|---|
| 0 | Finish the scan, confirm the table | running (`~/dsv4f-work/fp8scan.txt`) |
| 1 | Read the current BF16 paths (section 3), write findings here | done, see section 4 |
| 2 | `CUDA0_FP8` buffer type: alloc size, exact `set_tensor`, `get_tensor`, op refusal, `-ot` registration (`common/arg.cpp`), support check for overrides in the loader | done |
| 3 | Safety net: MXFP8 -> BF16 expansion feeding the existing paths (all of mmvf, mmf and cuBLAS at first) | done: Q8 md5 `1129672c4d2a` identical, weights 13,004 -> 7,592 MiB, 6.3 GB free at 450K, 12.0 t/s (expansion on every call) |
| 4a | Native FP8 in `mul_mat_vec_f` (2D weights at 1 token, including the gate/GLU fusion) | done (`18d0034f5`), bit-identical |
| 4b | Native FP8 in `mul_mat_f` (wo_a at every batch size, all weights at 2-16 tokens) | done (`18d0034f5`), bit-identical |
| 4c | Faster exact FP8 decode (FP16-bits trick) + one-warp-per-row mat-vec for short rows (`mul_mat_vec_f_vt`) | done (`see git log`), bit-identical, GPU time ~5 ms/token lower than BF16 |
| 5 | Bitwise tests: FP8 vs BF16 weights, batch 1-4 and large; `test-backend-ops` if it fits | |
| 6 | Full verification: md5s (all 4 configs), pp, VRAM, speed | |
| 7 | Use the freed VRAM (Q8 + DSpark at 450K?), scripts, README, commit | |

## Gotchas learned on this project

- **Don't rebuild while a benchmark server is running.** It replaces shared libraries under a live process.
- **`pkill -f <pattern>` can kill the calling shell** if the pattern appears in its own command line. Use exact PIDs.
- **A CUDA fusion that gets split changes the output.** That's how the early-copy overlap first broke md5: the top-k MoE fusion needs `ARGSORT -> VIEW -> GET_ROWS` contiguous. Expect the same with any change to node order or graph splits.
- **Debugging md5 differences:** with `GGML_CUDA_DISABLE_GRAPHS=1`, a temporary per-node output checksum in the eager loop of `ggml_cuda_graph_evaluate_and_capture` finds the first differing node quickly. Remember that intermediates of fused groups are not written.
- **The ggml thread pool (non-OpenMP) collapses with DSpark** (two 20-thread pools spinning on the same cores). Stay on OpenMP.
