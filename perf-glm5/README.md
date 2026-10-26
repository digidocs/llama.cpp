# GLM-5.3-Flash UD-Q4_K_XL on Xeon 6212U + RTX 3090

Xeon Gold 6212U (24 cores, 6ch DDR4-2933, ~122 GB/s measured read) + RTX 3090 (24 GB), 187 GiB RAM.
Model: `unsloth/GLM-5.3-Flash-GGUF` UD-Q4_K_XL, 200 GB (experts Q4_K/Q5_K/Q6_K 190 GB, everything else Q8_0).
Launch script: `~/glm53.sh` (port 8001). Logs, references and traces: `~/glm5-work`.

## Status

| setting | value |
|--|--|
| branch | `glm5-prof` (in `~/llama.cpp-glm5`, a worktree of `~/llama.cpp`) |
| placement | non-expert weights + MoE layer 3 experts on the GPU, the other 41 MoE layers in pinned host memory (`-ncmoe 999`) |
| context / ubatch | 200K / 4096 (pp >= 200 t/s needs ubatch 4096; its compute buffer limits the context) |
| threads | 24 (`--cpu-range 0-23`) |
| sampling | GLM defaults: temperature 1.0, top_p 0.95, min_p 0 (DeepSWE: 0.95 / 1.0) |
| decode | 13.7 t/s without MTP (upstream at 20 threads: 10.7) |
| prompt | ~210 t/s on an 8.3K-token prompt (server), 242 t/s pp8192 (llama-bench); < 768 tokens stays on the CPU path, ~80 t/s |
| load | ready in ~88 s with `-lm dio` |

## Branches

- `glm5-base`: upstream master `d854d98ba` (has GLM-5.3-Flash, #27773) + `3ccb12065` parallel loading/pinning + scripts.
  The `unsloth/glm5next/upstream` branch from the model card is outdated: it names the arch `glm5next`, the GGUF says `glm5-next`.
- `glm5-prof`: glm5-base +
  - `45f01bc86` per-thread src1 in CPU MUL_MAT_ID + `GGML_CPU_PROFILE` (`GGML_MMID_NO_PRIVATE_SRC1=1` turns it off)
  - unsloth `aba4a4c1d` chunked indexer re-pooling (no effect here, see below; harmless, md5 unchanged)
  - `82d47401e` shared expert overlap (scheduler) + the same hook for glm5-next (**built and tested next**)
- `glm5-ports`: the other ports (CUDA graph per shape/batch size, mmvq src1 reuse) plus two bug fixes that are still
  missing from the main repo branches: the mmvq src1 cache breaking the VMM pool's LIFO order in the synchronous
  mul_mat_id fallback (abort in `test-backend-ops -o MUL_MAT_ID`), and the CUDA graph property check not comparing
  source tensor types (f16 graph replayed on bf16 data, exposed by `c6dc3e41b`).

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

## Where a token goes (24 threads, per-thread src1, 13.6 t/s = 73 ms)

| part | ms/token | notes |
|--|--:|--|
| CPU expert matmul loops | ~41.6 | 41 layers x 8 experts x 15.3 MB = 5.0 GB at ~120 GB/s: at the DRAM ceiling |
| CPU per-op overhead | ~7.5 | src1 quantize ~3, thread imbalance wait in q4_K MUL_MAT_ID ~3, GLU ~0.75 |
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

unsloth's indexer compute-buffer fix (`aba4a4c1d`, chunked re-pooling) does not help a single slot: 3893 MiB with and
without it at ub 2048, 863 -> 837 MiB at ub 512. Its gains are for `-np 4` / `--kv-unified` setups.

## MTP speculative decoding (`--spec-type draft-mtp`)

Needs `--spec-draft-n-ubatch 512` (the default sizes the draft compute buffer like the main context: 5.2 GB).
Draft threads: `-td 24 -Crd 0-23 --cpu-strict-draft 1` (the batch-draft CPU options are not enabled in llama-server).
MTP adds ~1.5 GB of VRAM (MTP weights 191 MiB, recurrent state copies for rollback +291 MiB, draft KV 342 MiB, draft
compute 680 MiB) and ~4 GB of pinned host memory, which does not fit at 200K / ub 4096: the runs below use 180K.

`spec_run.sh`, 5 prompts x 400 tokens, temp 1.0 / top_p 1.0 / min_p 0.01, 180K:

| run | decode t/s | acceptance | per prompt |
|--|--:|--:|--|
| no MTP | 13.74 | | 13.73-13.75 |
| MTP n-max 2 | **15.44** | 70.6% | 14.6-16.7 |
| MTP n-max 3 | 14.23 | 59.9% | 12.4-16.6 |

Every drafted token reads a different set of 8 experts on the CPU, so longer drafts stop paying off sooner than on a
dense model. pp of the 8.3K-token prompt is unchanged (~210 t/s) except the first prompt after load (~181 t/s).

Pending: greedy correctness (no MTP vs MTP n-max 2), speed with the GLM sampling defaults (no MTP, n-max 1, 2),
the shared expert overlap, fitting MTP at 200K, variable draft length (`--spec-draft-p-min`, needs the CUDA graph
per batch size commits from glm5-ports).

## Commands

```
GGML_CPU_PROFILE=1 ./build/bin/llama-bench -m $M -ngl 999 -ot $GPU_EXPS -ncmoe 999 -fa on -lm dio -C 0xFFFFFF --cpu-strict 1 -t 24 -p 0 -n 64 -r 1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -o tg ./build/bin/llama-bench ... -p 0 -n 32
CTX=180000 MAX_TOKENS=400 perf-glm5/spec_run.sh mtp2 --spec-type draft-mtp --spec-draft-n-ubatch 512 -td 24 -Crd 0-23 --cpu-strict-draft 1 --spec-draft-n-max 2
GREEDY=1 SKIP_PP=1 perf-glm5/spec_run.sh g-mtp2 ...   # replies saved as ~/glm5-work/<tag>.p<i>.txt
```
