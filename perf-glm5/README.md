# GLM-5.3-Flash UD-Q4_K_XL on Xeon 6212U + RTX 3090

Xeon Gold 6212U (24 cores, 6ch DDR4-2933, ~122 GB/s measured read) + RTX 3090 (24 GB), 187 GiB RAM.
Model: `unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL, 200 GB (experts Q4_K/Q5_K/Q6_K 190 GB, everything else Q8_0).
Launch scripts: `~/glm53.sh` (no MTP, 200K) and `~/glm53-mtp.sh` (MTP, 185K), port 8001. Logs, references and
traces: `~/glm5-work`.

## Status

| setting | value |
|--|--|
| branch | `glm5-prof` (in `~/llama.cpp-glm5`, a worktree of `~/llama.cpp`) |
| placement | non-expert weights + MoE layer 3 experts on the GPU, the other 41 MoE layers in pinned host memory (`-ncmoe 999`) |
| context / ubatch | 200K / 4096 without MTP, 185K / 4096 with MTP (pp >= 200 t/s needs ubatch 4096; its compute buffer limits the context) |
| threads | 24 (`--cpu-range 0-23`), `--poll` default 50 (0: -0.6%, 100: same) |
| sampling | GLM defaults: temperature 1.0, top_p 0.95, min_p 0 (DeepSWE: 0.95 / 1.0) |
| decode | **14.1 t/s** without MTP, **16.5 t/s** with MTP (upstream at 20 threads: 10.7); with MTP at 64K depth 15.3 (13.45 before the k-pool layout fix), 9.9 at 175K before it |
| prompt | ~210 t/s on an 8.3K-token prompt, 186 t/s over a 175K-token prompt; < 1400 tokens runs the experts on the CPU, 80-100 t/s |
| load | ready in ~88 s with `-lm dio` |

## Branches

- `glm5-base`: upstream master `d854d98ba` (has GLM-5.3-Flash, #27773) + `3ccb12065` parallel loading/pinning + scripts.
  The `unsloth/glm5next/upstream` branch from the model card is outdated: it names the arch `glm5next`, the GGUF says `glm5-next`.
- `glm5-prof` (current): glm5-base +
  - `45f01bc86` per-thread src1 in CPU MUL_MAT_ID + `GGML_CPU_PROFILE` (`GGML_MMID_NO_PRIVATE_SRC1=1` turns it off)
  - unsloth `aba4a4c1d` chunked indexer re-pooling (no effect here, see below; harmless, md5 unchanged)
  - `82d47401e` shared expert overlap (scheduler) + the same hook for glm5-next
  - `73fbcf3d4` mmvq src1 reuse, `0ba9fd8b1` / `c6dc3e41b` CUDA graphs per shape / batch size
  - two bug fixes: the mmvq src1 cache breaking the VMM pool's LIFO order in the synchronous mul_mat_id fallback
    (abort in `test-backend-ops -o MUL_MAT_ID`), and the CUDA graph property check not comparing source tensor types
    (f16 graph replayed on bf16 data, exposed by `c6dc3e41b`). Both are also on `~/llama.cpp` perf/dsv4-flash-profile
    (`bdf564559`, `6e6d0a539`; 18719/18719 ops pass, DSV4 Q8 md5 `1129672c4d2a` unchanged).
  - `d5680bda6` AVX2 `quantize_row_q8_K` (x86 used the scalar reference): bit-exact (0 mismatched blocks of 1.28M,
    incl. ties and zero blocks), 27 -> 4.7 us per 16K floats, MUL_MAT_ID quantize 3 -> 0.8 ms per token.
    Not on the main repo branch yet: DSV4 Q4_K_XL (q4_K experts) would gain from it too.
- `glm5-ports`: obsolete (the ports are on glm5-prof).
- `glm5-opt` (in `~/llama.cpp-glm5opt`, 2026-10-27, not in production yet): glm5-prof + the MUL_MAT_ID work queue and
  the HC gate fusion, +2.8% decode, bit-exact. See "glm5-opt" below.

Every glm5-prof change is bit-exact: md5 `3241ffcdeb80`, greedy replies byte-identical, KLD vs the ub 1 reference 0.

## Correctness references

- `md5.sh` (server, one request, seed 1000, reasoning_effort low, 300 tokens, temp 1.0 / top_p 1.0 / min_p 0.01):
  `3241ffcdeb80` with layer 3 experts on the GPU. Unchanged by the per-thread src1 copy, the chunked re-pooling,
  ubatch 512/2048/4096 and context 200K/250K. GLM's template has no thinking switch, only reasoning_effort
  (low/high/max), so the md5 covers reasoning + content.
- `kld_check.sh` (2 x 512 tokens, ub 1): reference `~/glm5-work/base250k-ub1.kld`, PPL 3.2347;
  self-compare: mean KLD 0, max 5.5e-5, same top 100%.
- Moving one layer's experts between CPU and GPU: mean KLD ~0.012-0.013, same top ~96.5%, max ~0.5, PPL +-1% with
  no consistent direction (3.279 layers 3+4 on GPU, 3.235 layer 3, 3.269 none), most likely the different activation
  quantization (CPU q8_K blocks of 256, CUDA q8_1 blocks of 32). All 18979 test-backend-ops pass.
  Use ~0.012 as the size of a CPU <-> GPU path change, ~0 for pure reorderings.
- Speculative decoding: compare at temperature 0 (`GREEDY=1 spec_run.sh`); at temperature > 0 it consumes the RNG
  differently, so md5s differ from the non-speculative run by design.

## Threads and the per-thread src1 copy (llama-bench tg64 / pp512, t/s)

| threads | shared src1 (upstream) | per-thread src1 (45f01bc86) |
|--:|--|--|
| 16 | 11.39 / 65.9 | 12.59 / 66.4 |
| 20 | 10.79 / 76.4 | 13.47 / 76.3 |
| 24 | 12.46 / 80.6 | **13.68** / 80.5 |

Server, 20 threads: 10.70 -> 13.38 t/s, md5 unchanged.

Later commits (llama-bench tg64, 24 threads): 13.62 (per-thread src1) -> 14.45 t/s (+ shared expert overlap, mmvq
src1 reuse, CUDA graph commits, AVX2 q8_K). Server, 5 prompts x 400 tokens: 13.67 -> 14.07 t/s before the q8_K change.

## Where a token goes (24 threads, per-thread src1, 13.6 t/s = 73 ms; before the shared expert overlap and AVX2 q8_K)

| part | ms/token | notes |
|--|--:|--|
| CPU expert matmul loops | ~41.6 | 41 layers x 8 experts x 15.3 MB = 5.0 GB at ~120 GB/s: at the DRAM ceiling |
| CPU per-op overhead | ~7.5 | src1 quantize ~3 (now 0.8, AVX2 q8_K), wait in q4_K MUL_MAT_ID ~3 (threads arriving after the GPU handoff; `--poll 100` does not help), GLU ~0.75 |
| GPU kernels | ~18.6 | mul_mat_vec_q 11.9 (~9.6 GB dense Q8_0 at ~800 GB/s of 936), quantize_q8_1 0.9, rms_norm 0.9, bin_bcast 0.7, ... |
| handoffs / launch gaps | ~5 | |

CPU and GPU run strictly in turn: under nsys (85 ms/token with tracing) the GPU is busy 19.8 ms and idle 65 ms per
token, in 82 gaps > 200 us (two per CPU layer, median 1.2 ms). Transfers are small: H2D 5.3 MB, D2H 1.2 MB per token.
Both sides are bandwidth bound, so a faster Q4_K/Q5_K CPU kernel would gain almost nothing; the levers are overlap,
less CPU overhead, and speculative decoding (the model has one MTP layer).

## VRAM, context and ubatch

KV cache is ~19 KiB/token (11 of 45 layers have attention, the rest are KDA linear attention with a 146 MiB recurrent
state). The compute buffer grows with ubatch x context (indexer scores and the attention mask) and sets the limit:

| context | ubatch | KV + indexer cache | compute buffer | fits | pp (t/s) |
|--:|--:|--:|--:|--|--|
| 32K | 2048 | 616 MiB | 2830 MiB | layers 3+4 experts on GPU | |
| 250K | 2048 | 4702 MiB | 3893 MiB | layer 3 only (3+4 is 1.8 GB short) | 140 (pp4096) |
| 250K | 512 | 4702 MiB | 837 MiB | layer 3 only | 43 (pp4096) |
| 200K | 4096 | 3763 MiB | 5607 MiB | layer 3 only, 1.6 GB free | 242 (pp8192) |

Long prompts are PCIe bound (~173 GB of host experts streamed per ubatch), so pp scales with ubatch: 512: 43, 1024: 77,
2048: 140, 4096: 242 t/s.

`GGML_OP_OFFLOAD_MIN_BATCH` (batches below it run the experts on the CPU) crosses over at ~1400 tokens
(llama-bench, ub 4096, t/s):

| prompt | 512 | 768 | 1024 | 1280 | 1536 | 2048 |
|--|--:|--:|--:|--:|--:|--:|
| GPU (stream experts over PCIe) | 42 | 60 | 77 | 94 | **111** | **139** |
| CPU | **80** | **89** | **96** | **100** | 104 | 108 |

The launch scripts and harness use 1400 (was 768). It also applies to the last, partial ubatch of a long prompt.

unsloth's indexer compute-buffer fix (`aba4a4c1d`, chunked re-pooling) does not help a single slot: 3893 MiB with and
without it at ub 2048, 863 -> 837 MiB at ub 512. Its gains are for `-np 4` / `--kv-unified` setups.

## MTP speculative decoding (`--spec-type draft-mtp`)

Needs a small `--spec-draft-n-ubatch` (the default sizes the draft compute buffer like the main context: 5.2 GB).
Draft threads: `-td 24 -Crd 0-23 --cpu-strict-draft 1` (the batch-draft CPU options are not enabled in llama-server).
MTP adds ~1.5 GB of VRAM (MTP weights 191 MiB, recurrent state copies for rollback +291 MiB, draft KV 342 MiB, draft
compute 680 MiB at draft ub 512 / 180K) and ~4 GB of pinned host memory.

### Final setup (`~/glm53-mtp.sh`): 185K, draft ub 256, q8_0 draft cache, n-max 2, p-min 0.9

Fitting it: `-ctkd q8_0 -ctvd q8_0` (draft cache only; the main cache stays f16) halves the draft KV (342 -> 182 MiB),
draft ub 256 shrinks the draft compute buffer (939 MiB at ub 512 / 200K). The CUDA pool also needs ~224 MiB above
the idle level during a 4096 ubatch (sampled with nvidia-smi; the same peak for an 8K and a 175K-token prompt):

| context | draft | idle VRAM | peak in pp | |
|--|--|--:|--:|--|
| 200K | ub 512, q8_0 | | | draft compute buffer alloc fails |
| 200K | ub 256, q8_0, n-max 1 | 24000 | | CUDA OOM in the first prompt |
| 190K | ub 256, q8_0, n-max 1 | 23568 | 23792 | ok, also with a 174779-token prompt (186 t/s pp, 9.89 t/s tg) |
| 190K | ub 256, q8_0, n-max 2 | 23714 | 24022 | works, 93 MiB margin: too tight |
| **185K** | ub 256, q8_0, n-max 2, p-min 0.9 | 23490 | 23798 | chosen, 317 MiB margin |

Draft length (190K, GLM sampling, 5 prompts x 400 tokens, current glm5-prof):

| draft | decode t/s | acceptance |
|--|--:|--:|
| no MTP | 14.07 | |
| n-max 1 | 16.18 | 84.3% |
| n-max 2 | 15.74 | 71.1% |
| n-max 3 | 14.07 | 57.0% |
| n-max 2, p-min 0.6 | 16.31 | 82.3% |
| n-max 2, p-min 0.8 | 16.52 | 89.0% |
| **n-max 2, p-min 0.9** | **16.53** | 94.3% |
| n-max 3, p-min 0.8 | 16.18 | 84.8% |
| n-max 3, p-min 0.9 | 16.52 | 91.0% |

`--spec-draft-p-min` drafts the next token only when the draft's top probability is at least p-min, so the second
token is drafted only when it is likely to be accepted. (n-max 1/2/3 rows at 180K with draft ub 512 and f16 draft
cache; the draft cache type and size did not change acceptance measurably.)

Batched verification numerics: greedy replies with and without MTP diverge after 70-400 characters. KLD vs the ub 1
reference: ub 2 mean 0.0126 (max 0.50, same top 97.1%), ub 3 mean 0.0106 (same top 96.5%), PPL unchanged within
noise: the same size as moving one layer between CPU and GPU, i.e. different kernels for 2-3 token batches, not a bug.

### Earlier runs (before the fixes above)

`spec_run.sh`, 5 prompts x 400 tokens, temp 1.0 / top_p 1.0 / min_p 0.01, 180K:

| run | decode t/s | acceptance | per prompt |
|--|--:|--:|--|
| no MTP | 13.74 | | 13.73-13.75 |
| MTP n-max 2 | **15.44** | 70.6% | 14.6-16.7 |
| MTP n-max 3 | 14.23 | 59.9% | 12.4-16.6 |

Every drafted token reads a different set of 8 experts on the CPU, so longer drafts stop paying off sooner than on a
dense model. pp of the 8.3K-token prompt is unchanged (~210 t/s) except the first prompt after load (~181 t/s).

## MTP at depth: k-pool layout rebuild (2026-10-26)

With MTP, decode fell with depth (server, `~/prof/depthprof.py`, 600 tokens: 15.39 t/s at 0, 13.45 at 64K) while
llama-bench without MTP stayed flat (13.53 -> 13.37 at 131K). CPU expert time was constant (~39 ms/token); the extra
~9 ms was the other 23 threads spinning in the libgomp barrier. nsys (`~/prof/nsyscmp.py n0 n64k`) put only ~1.9
ms/token of it on the GPU (lightning_indexer 18 -> 95 us per launch, flash_attn 16 -> 64 us, sparse index/top-k,
draft cache q8_0 -> f16 dequant 118 us). The rest was the main thread in `std::_Rb_tree_increment` (~8% of its time at
64K, none at depth 0).

Cause: the server drops rejected drafts with a `seq_rm` after every verify step, which marks the sequence stale from
p0, and `llama_memory_hybrid_idx::kpool_layout_update()` rebuilt the whole layout on any stale mark: it walked the
sequence's whole `seq_pos` set and rescanned every pool. Without MTP there is no `seq_rm`. Fix: for an edit at
p0 > pos_min, keep the cells below p0 and the pools that end before it, resume the scan after the last kept pool and
append the tail from the set. An edit at pos_min or a cell count mismatch still rebuilds fully.

| depth | before | after |
|--|--:|--:|
| 0 | 15.39 t/s (264/290 drafts) | 15.20 (259/281) |
| 64K | 13.45 (265/277) | **15.30** (269/286) |

Checked with a temporary assertion that rebuilt the layout the old way after every update and compared cells, pools,
scan position and pos_min: 1024+ checks, 652 through the trimmed path (MTP decode at 16K plus prompt cache reuse that
trims at ~12K and re-prefills ~4K), no mismatch. Greedy output cannot be compared token for token: two runs of the same
build diverge after ~70 tokens (batched verification numerics, see above). Not exercised: order mode (qwen4exp),
`-np` > 1, `seq_add`.

## CPU/GPU overlap: measured, no gain (2026-10-26)

Profile of the current build (nsys, `nsys_tg.py`; 79.8 ms/token traced vs ~69 untraced): per CPU MoE layer the GPU
runs attention/KDA + router (73 kernels, 0.41 ms), hands the router output to the CPU, runs the shared expert during
the CPU window (the overlap works), then sits idle for the routed experts (~1.1 ms) until the 128 KB result comes
back. GPU busy 19.8 ms per token. Layer N+1's GPU work needs layer N's expert output, so the only way to keep both
busy is splitting each layer's routed experts between CPU and GPU, with the GPU reading its share from pinned RAM.

`bw_concurrent.cu` (CPU AVX2 read of 16 GB, optionally while the GPU streams pinned memory):

| | CPU | GPU over PCIe | total |
|--|--:|--:|--:|
| CPU alone, 24 threads | 122.5 | | 122.5 GB/s |
| + H2D copies | 110.3 | 12.3 | 122.6 |
| + kernel reading mapped host memory | 108.5 | 12.3 | 120.8 |

DRAM is the shared ceiling (~122 GB/s of the 140.8 theoretical): every byte the GPU pulls over PCIe is taken from
the CPU, so a CPU/GPU expert split cannot speed up decode on this machine.

## glm5-opt: MUL_MAT_ID work queue + HC gate fusion (2026-10-27)

Both bit-exact: md5 `3241ffcdeb80`, KLD vs the ub 1 reference and vs a new ub 512 reference (`base-ub512.kld`, made
with glm5-prof) at the self-compare floor (max 4.0e-5 / 4.8e-5, same top 100%), `test-backend-ops -o MUL_MAT_ID`
931/931 (CUDA vs CPU), `perf-dsv4f/bitexact/hc_gate.cpp` fused vs `GGML_CUDA_DISABLE_FUSION=1` identical (`cmp`).
MTP server run 2: all 5 reply md5s identical between glm5-prof and glm5-opt.

`ab_bench.sh` (llama-bench tg128 x 3 per round, configs alternated round by round, 3 rounds; t/s):

| build | tg128 | |
|--|--:|--|
| glm5-prof | 14.60 | |
| + HC gate fusion (`GGML_MMID_QUEUE=0`) | 14.74 | +1.0% |
| + MUL_MAT_ID queue (glm5-opt) | **15.01** | +2.8% |
| MTP server, production settings at 185K (2 runs) | 16.65 / 16.53 -> **16.78 / 16.70** | +0.9% |

- **HC gate fusion** (cherry-pick of the DSV4 branch's `50a6407a7`): each hyper-connection gate was 4 kernels
  (mul, add, sigmoid, scale; 4 gates per layer); now one `dsv4_hc_gate_f32`. ~0.6 ms/token.
- **MUL_MAT_ID queue**: at decode each expert has one token, and the per-expert path splits it into one fixed chunk per
  thread (no work stealing), so each op waited for the slowest thread. Experts that would take that path now get a
  fixed per-thread share in whole 16-row blocks (gate/up 80 of 2048 rows, down 160 of 4096 at 24 threads) and the
  rest (128 / 256 rows) goes to one queue shared by all threads in 16-row chunks. Per-thread chunk time is now even
  (2776-2810 ms per 64 tokens across the 24 threads). `GGML_MMID_QUEUE=0` turns it off.
  The first try (everything in 32-row queue chunks) was **5% slower** than the fixed split (13.94 vs 14.70): short
  chunks break the long per-thread weight streams. Shares of 75/85/90/95% before the queue: 14.82/14.89/14.90/14.95.
  The MTP gain is smaller: verify batches give an expert up to 3 tokens and those already had 128 chunks to balance.

Measured, no gain:
- **swiglu / swiglu_clamp split into column blocks** (8 rows -> 24 threads; bit-exact, 64-float aligned blocks):
  GLU 47 -> 21 ms per 64 tokens, but the down MUL_MAT_ID's per-thread src1 copy then gathers the GLU output from 24
  cores instead of 8: src1 quantize 58 -> 92 ms, route+barrier 33 -> 52 ms, tg 14.94 -> 14.84. Reverted. Same
  effect as behind the per-thread src1 copy: rows written by other cores are expensive to read.
- `--poll 100`: 14.96 vs 14.94 (noise).
- CUDA graph launch: nsys shows ~70 us per `cudaGraphLaunch` and a ~80 us GPU gap after every expert handoff, but an
  `LD_PRELOAD` timer on `cudaGraphLaunch` without nsys gives 4.7 us mean (84 launches per token). Profiler artifact.
- The shared expert overlap works: the router output goes to the host right after `topk_moe` (early copy), the
  shared expert runs during the CPU window.

Where a token goes now (15.0 t/s = 66.6 ms): CPU MUL_MAT_ID 45.9 ms (q4_K gate/up at ~108 GB/s, q5_K down at ~112
GB/s: ~90% of the DRAM ceiling; of it quantize 0.9, route+barrier 0.5), GLU 0.7, GPU ~15 (mul_mat_vec_q ~10,
~2,400 kernels), handoffs ~5. What is left is small: single-block `rms_norm_f32` over the 4 x 4096 HC rows
(~0.6 ms/token, splitting it across blocks changes the summation order), `quantize_q8_1` 0.56, `concat_cont` 0.2.

## Layer 3 experts back on the CPU for more context: does not fit in RAM (2026-10-27)

`perf-glm5/l3cpu.sh` (paths hard-coded to `~/llama.cpp-glm5opt`; glm5-opt, `GPU_EXPS` matching nothing, production MTP settings, `-c 280000`, MemAvailable /
VRAM logged every 5 s to `l3cpu-mem.txt`). Not bit-exact by design (a CPU <-> GPU path change, KLD ~0.012).

- VRAM: fits. 23466 MiB after load, peak 23760 during a 266,734-token prompt (CUDA0: model 8741, context 5701,
  compute 7323 MiB). The deep prompt ran at 166 t/s pp, 9.87 t/s tg at that depth.
- RAM: does not fit. Host buffers 184.7 GB (model 181.7 + compute 3.0); MemAvailable fell to ~3 GB at the end of
  the load and then to 0 for most of the run (210 of 387 samples), with the system swapping. Short-prompt MTP decode
  15.25 t/s vs 16.7 with layer 3 on the GPU, most likely the swapping, not the extra 4.2 GB of CPU expert reads
  (expected ~1.5%). No OOM kill in the journal.
- With layer 3 on the GPU there is ~4 GB of headroom left; moving a 4.2 GB layer to the host needs that plus the
  3 GB host compute buffer. Would need e.g. a smaller ubatch for the host compute buffer (costs pp) or no `-lm dio`
  pinning of some layer, neither tried.

## Next candidates (paused 2026-10-26; 1 and part of 2 done on glm5-opt)

Remaining token budget (~69 ms at 14.45 t/s): CPU experts ~42 (at the DRAM ceiling), GPU `mul_mat_vec_q` 11.9 (~800
GB/s, near practical peak), GPU small kernels ~7, CPU thread wait in q4_K MUL_MAT_ID ~3.4, quantize + GLU + handoffs ~4.

1. CPU wait in gate/up MUL_MAT_ID (~3.4 ms/token, 220 ms per 64 tokens in `GGML_CPU_PROFILE`): uneven thread finish
   at the end of each op (thread 23 does ~15% less chunk work). Try finer final chunks / a different chunk size for
   small src1. Expect 1-2 ms (+2%); should stay bit-exact.
2. GPU small-kernel fusion (~2,000 launches per token: rms_norm, k_bin_bcast, quantize_q8_1, scale, unary,
   dsv4_hc_*): fuse norm -> quantize and add/scale chains. Expect 2-3 ms (+3-4%); several CUDA kernels to touch.

Both together: ~15.2 t/s without MTP, ~17.3 with MTP. Beyond that needs more DRAM bandwidth (see the Rome notes).

## Not done

- q8_0 KV cache for the main model (ruled out).
- `--prio 2`: needs root.
- Reusing the gate/up src1 quantization: < 0.2 ms per token after the AVX2 q8_K.

## Commands

```
GGML_CPU_PROFILE=1 ./build/bin/llama-bench -m $M -ngl 999 -ot $GPU_EXPS -ncmoe 999 -fa on -lm dio -C 0xFFFFFF --cpu-strict 1 -t 24 -p 0 -n 64 -r 1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o tg ./build/bin/llama-bench ... -p 0 -n 32
CTX=180000 MAX_TOKENS=400 perf-glm5/spec_run.sh mtp2 --spec-type draft-mtp --spec-draft-n-ubatch 512 -td 24 -Crd 0-23 --cpu-strict-draft 1 --spec-draft-n-max 2
GREEDY=1 SKIP_PP=1 perf-glm5/spec_run.sh g-mtp2 ...   # replies saved as ~/glm5-work/<tag>.p<i>.txt
SKIP_PP=1 PP_ONLY=1 DEEP=620000 perf-glm5/spec_run.sh deep ...   # one ~175K-token prompt (deep_pp.py)
UB=3 perf-glm5/kld_check.sh cmp ~/glm5-work/base250k-ub1.kld ub3
```
