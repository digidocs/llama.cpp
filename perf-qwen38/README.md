# Qwen3.8-Flash-Next Q8_0 on Xeon 6212U + RTX 3090

Xeon Gold 6212U (24 cores, 6ch DDR4-2933, ~122 GB/s read) + RTX 3090 (24 GB, PCIe 3.0 x16), 187 GiB RAM.
Model: `unsloth/Qwen3.8-Flash-Next-GGUF` Q8_0, 188 GB in 6 shards (routed experts ~128 GB, PLE n-gram table 51 GB lazy-read),
MTP head `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` (borrows the target's embeddings and lm head).
Launch script: `~/qwen4exp-mtp.sh` (port 8001). Experiment log with every run: `LOG.md`. Logs and traces: `~/qwen38-work`.

## Status

| setting | value |
|--|--|
| branch | `qwen38-flash-next` in `~/llama.cpp-qwen38` (upstream master `a946eb541` + the commits below) |
| placement | non-expert weights + MTP head on the GPU, all routed experts in pinned host memory (`-ncmoe 999`, `-lm dio`) |
| context / ubatch | 262144 (native maximum) / 4096, draft ubatch 512: VRAM 22.1 of 24 GB |
| threads | 24 (`--cpu-range 0-23 --cpu-strict 1`) |
| sampling | model card thinking mode: temperature 1.0, top_p 0.95, top_k 20, min_p 0 |
| decode | **~31.5 t/s** with MTP at temperature 1.0 (33.2 greedy), 24.8 without MTP; upstream master 16.1 without MTP, crashes with MTP |
| prompt | ~295 t/s on a 9K-token prompt (b = ub = 4096, `GGML_OP_OFFLOAD_MIN_BATCH=1400`) |
| reference | Qwen3.8-27B Q5_K_M MTP on the same 5 prompts: 67.3 t/s (dense model, fits in VRAM) |

## Branch `qwen38-flash-next`

- `020bbcb6b` MTP draft borrows the target's embeddings and lm head (unsloth 78ede59e3)
- `0d3f7180a` parallel model loading + faster pinned allocation (moe-hybrid adddcd17a); `~/llama.cpp-qwen38-base` is a
  frozen build of this commit for A/B
- moe-hybrid ports: `d949e0e00` per-thread src1 + work queue in CPU MUL_MAT_ID + `GGML_CPU_PROFILE`, `41ca29ee4` / `4a4765d58`
  mmvq src1 reuse, `a8de3d3d6` / `846f02e67` CUDA graphs per shape / batch size, `9705f4e75` graph check compares src types,
  `2c6f64cbc` k-pool layout tail rebuild, `862fc39ac` shared expert overlap (+ qwen4exp hook)
- `47af058eb` test-backend-ops perf cases for the expert shapes
- `b41230589` **fix**: the MTP block is dense (compress ratio 0) but its graph built the k-pool input; k_idxs got no buffer and
  the server aborted at start (`GGML_ASSERT(buffer)`). Upstream bug, also in master.
- `589e6576f` scheduler queues host -> device input copies (`set_tensor_async`) instead of a blocking copy: +1.8%, bit-exact
  (`GGML_SCHED_SYNC_H2D=1` restores the old path)
- `28335ed9b` `scripts/moe-cache-tune.py` respects `--max-mib` in its calibration runs

## Results (5 prompts of `spec_prompts.py`, server, details in LOG.md)

| tree / config | decode t/s | notes |
|--|--:|--|
| master, llama-bench tg128 | 16.12 | |
| guide branch danielhanchen `qwen4exp/mtp` (Sep 21), llama-bench tg128 | 18.96 | |
| ours, llama-bench tg128 | 25.76 | ports help 1-token decode only |
| ours, server, no MTP | 24.80 | |
| guide branch, MTP Q8_0 n3 | 31.67 | same replies (md5) as ours |
| ours, MTP Q8_0 n3 | 31.37 | acceptance 64% |
| ours, MTP Q8_0 n2 | 31.68 | acceptance 74%, same tokens per ms |
| ours, MTP + MoE cache 6000 MiB, CPU share 0.95 | 31.83 | within noise, takes the VRAM context needs |
| ours, MTP n3, greedy, async H2D | 33.23 | 32.63 before, md5 identical |
| ours, MTP n3, greedy, 262K context | 32.82 | md5 identical to 131K |

With MTP the branches converge: every step verifies 4 tokens, and that batch is limited by reading the experts from DRAM.

## Where a step goes (MTP n3, ~93 ms, ~2.9 tokens)

| part | ms | |
|--|--:|--|
| CPU expert matmul (4-token verify) | ~63 | ~112 GB/s, near the DRAM limit |
| GPU graphs (attention / GDN / shared expert / router, 49 per step) | ~14 | ~10 ms kernels (dense weights near GPU bandwidth) + gaps between small kernels |
| CPU -> GPU hand-off | ~4 | cudaGraphLaunch 54-58 us per layer (was ~130 us before the async H2D copy) |
| CPU GET_ROWS q8_0 | ~3 | 0.46 ms per call, not page faults (0 measured); not yet explained |
| MTP drafts (3 per step, GPU) | ~6 | |

## Dead ends

- MoE cache: PCIe 3.0 uploads (~13 GB/s) are ~9x slower than the CPU reading the experts; only ~5% of experts fit.
- Draft length: n-max 2 and 3 give the same tokens per ms.
- CUDA graphs off: 32.37 vs 33.23 t/s.
- Thread affinity applied once per OpenMP thread instead of per graph (`~/qwen38-work/affinity-once.patch`): +0.4%, noise.

## Next candidates

1. Queue the next GPU graph before the CPU split finishes, gated on a host flag (hides cudaGraphLaunch, up to ~3%, deadlock risk).
2. Find what the CPU GET_ROWS q8_0 is waiting on (~3%).
3. Fewer, fused GPU kernels per layer (~4 ms of gaps per step).
4. KLD check of the current tree against the base tree (`kld_check.sh`).

## Commands

```bash
perf-qwen38/ab_bench.sh 2 base=$HOME/llama.cpp-qwen38-base new=$HOME/llama.cpp-qwen38   # llama-bench tg A/B
GREEDY=1 perf-qwen38/spec_run.sh <tag> -md $MTP_DIR/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp -ngld 99 --spec-draft-n-max 3
CTX=262144 SKIP_PP=1 DEEP=830000 perf-qwen38/spec_run.sh <tag> ...   # ~230K-token prompt
perf-qwen38/run_prof.sh <tag> ...    # nsys (node trace) + GGML_CPU_PROFILE
perf-qwen38/run_perf.sh <tag> ...    # perf LBR call graphs (needs kernel.perf_event_paranoid <= 1)
perf-qwen38/run_prof_graph.sh <tag> ...  # nsys graph-level trace; then: nsys export --type sqlite -o x.sqlite x.nsys-rep
python3 -I perf-qwen38/timeline.py x.sqlite 60                          # per-layer CPU/GPU hand-off timeline
```
