# Qwen3.8-Flash-Next Q8_0 on Xeon 6212U + RTX 3090: experiment log

Started 2026-11-16 00:30 CST (overnight, unattended). Newest entries at the bottom.

Worktrees (all share ~/llama.cpp's object store):
- `~/llama.cpp-qwen38`, branch `qwen38-flash-next`: working tree, one commit per kept experiment
- `~/llama.cpp-qwen38-base`, detached at `0d3f7180a`: frozen baseline build for A/B (`ab_bench.sh ... base=~/llama.cpp-qwen38-base`)

Harness: `perf-qwen38/` (copied from `~/llama.cpp/perf-glm5`): `common.sh` (model, offload), `ab_bench.sh` (llama-bench A/B,
alternated rounds), `spec_run.sh` (server: 8K pp check + 5 decode prompts), `md5.sh`, `kld_check.sh`. Logs and references: `~/qwen38-work`.

Goals (from the user): accuracy first (no quality-losing shortcuts: keep Q8_0 weights, check KLD for every numeric change), then
context length, tg as high as possible with pp >= 200 t/s on long prompts. Target to beat: Qwen3.8-27B Q5_K_M MTP.
Serving sampling: temperature 1.0, top_p 0.95, top_k 20, min_p 0 (model card, thinking mode).

## Model

qwen4exp, 48 layers (every 4th full attention with QSA sparse attention, compress ratio 4, indexer top-k 2048; the rest
GDN-style linear attention, ssm inner 6144), n_embd 2560, 4 hyper-connection streams, 512 experts / 10 used, expert ff 640,
1 shared expert (ff 640), PLE n-gram embeddings on layer 1 (16 heads x ~20M rows x 160), 262K native context.
Q8_0 files: 188.2 GB in 6 shards; shard 3 (54.4 GB) is the PLE table, routed experts ~132 GB.
MTP heads in `MTP/`: shared-Q8_0 2.79 GB, shared-Q4_K_M 1.91 GB (borrow token_embd/output from the target; need `020bbcb6b`).

## Branch so far

- `a946eb541` upstream master (2026-11-16)
- `020bbcb6b` cherry-pick unsloth 78ede59e3: MTP draft borrows the target's embeddings and lm head (conflict in
  common/speculative.cpp resolved: kept master's common_model_load + set mparams.model_shared)
- `0d3f7180a` cherry-pick moe-hybrid adddcd17a: parallel model loading + faster pinned allocation (conflict: master
  rewrote split loading; kept master's code and added file_paths.emplace_back next to files.emplace_back)

## Reference: Qwen3.8-27B Q5_K_M MTP (~/llama.cpp moe-hybrid, qwen-27b-mtp.sh settings, temp 0.6), spec_prompts.py

| prompt | t/s | drafts accepted |
|--|--:|--|
| p0..p4 | 68.5 / 71.2 / 68.0 / 59.9 / 67.6 | |
| **total** | **67.27 t/s** | 73.8% (1960/2657) |

## Experiments

### 00:40-01:10 ports from moe-hybrid (before the model finished downloading)

| commit | what | notes |
|--|--|--|
| `d949e0e00` | per-thread src1 in CPU mul_mat_id + MUL_MAT_ID work queue + GGML_CPU_PROFILE (d5195792c ggml-cpu.c part, cfa901777, 437374652) | applied cleanly; `test-backend-ops -o MUL_MAT_ID -b CUDA0` OK with and without `GGML_MMID_NO_PRIVATE_SRC1=1` |
| `41ca29ee4`, `4a4765d58` | mmvq q8_1 src1 reuse (496ef6d3d, c968fb2aa) | conflict: upstream dropped the ids/type args of quantize_row_q8_1_cuda |
| `a8de3d3d6`, `846f02e67` | one CUDA graph per shape / per batch size (9dc2284b0, 0973e9379) | clean |
| `9705f4e75` | CUDA graph property check compares src types (d82816598) | node_properties moved to graph.cuh upstream; hand port |
| `2c6f64cbc` | k-pool layout tail rebuild after seq_rm (ba992442f) | qwen4exp uses order mode (pools = every kpool cells from 0): the trimmed path resumes at pools.back()+kpool, same as a full scan. To verify at runtime with MTP. |
| `862fc39ac` | shared expert overlap (85cad9a0d) + qwen4exp hook (shexp incl. its sigmoid gate built before build_moe_ffn) | merged with upstream's new ggml_backend_sched_prefetch_inputs (inputs computed before the split; ours: inputs computed by the split; both kept) |

Not ported: HC gate fusion (3ead29ca8): qwen4exp already uses upstream's fused dsv4_hc_pre_gated / dsv4_hc_post, different op chain.
AVX2 q8_K quantize (d5680bda6): irrelevant for Q8_0 experts (src1 is quantized to q8_0).

### 00:50 checks of the ported tree
- `test-backend-ops -b CUDA0` (all ops, CUDA vs CPU): all pass, 0 FAIL (log `~/qwen38-work/tbo-full.log`).
- CPU MUL_MAT_ID micro-bench (`test-backend-ops perf -o MUL_MAT_ID -b CPU -p "type_a=q8_0,...,n_mats=512"`, new perf cases
  commit `47af058eb`): weights partly stay in the 35.75 MB L3 between runs, so absolute GB/s are optimistic (~250 GB/s), but A/B holds:

| case (us/run) | ported | NO_PRIVATE_SRC1=1 | QUEUE=0 |
|--|--:|--:|--:|
| gate_up 1280x2560, n=1 | **138.6** | 243.1 | 151.4 |
| down 2560x640, n=1 | **71.2** | 264.7 | 71.2 |
| gate_up n=3 | 740.8 | 732.6 | 763.5 |
| down n=3 | 256.1 | 283.4 | 253.9 |

- Temporary k-pool order-mode checker saved as `~/qwen38-work/kpool-check.patch` (env LLAMA_KPOOL_CHECK, aborts on mismatch).

### 06:27 download incident
- The original `hf download` exited around 01:54 with shards 1, 2, 5 and 6 complete, but shards 3 (PLE, 54.4 GB) and 4 missing
  (no .incomplete left behind). My watcher never fired: its `pgrep -f "hf download ..."` matched its own bash command line.
- Restarted at 06:27 for only shards 3 and 4 (`~/qwen38-work/dl-resume.log`), ~103 GB at ~20 MB/s, so ETA ~08:00.
  New watcher matches the python `hf` process only. No model benchmarks have run yet.
- 07:45 download finished. All 6 Q8_0 shards and both shared MTP heads match the repo sizes, and shards 3/4 match their sha256.

### 07:50 smoke test, ported tree (`47af058eb`), all experts on CPU, 24 threads
- Loads fine: `per_layer_token_embd.weight (51880 MiB) lazy read enabled` under `-lm dio`. Peak RSS 127 GB, peak VRAM 6.3 GB.
- llama-bench, default ub 512: pp512 128.40 +- 1.52, tg128 25.94 +- 0.06 t/s (`~/qwen38-work/smoke-new.log`).

### 07:55 baseline A/B, tg128, all experts on CPU (`~/qwen38-work/ab-base-new.log`)
| tree | round 1 | round 2 | mean |
|--|--:|--:|--:|
| base `0d3f7180a` (master + borrow + parallel load) | 15.67 | 16.57 | 16.12 |
| new `47af058eb` (moe-hybrid ports) | 25.67 | 25.85 | **25.76 (+60%)** |

### 08:00 prompt processing, ported tree, b = ub = 4096
- `GGML_OP_OFFLOAD_MIN_BATCH=1400`: pp8192 **560.45 t/s** (well over the 200 floor).
- `GGML_OP_OFFLOAD_MIN_BATCH=0`: context creation fails (compute buffers don't fit in VRAM). Not useful anyway: it would also offload decode.
- Expert bytes per layer (computed): 512 x (1280x2560 + 2560x640) x 34/32 = 2.49 GiB. ~17 GB of VRAM is free at 6.3 GB used,
  so about 6 whole layers (12.5%) or a MoE cache of about that size.

### 08:05 server baseline, no MTP (`spec_run.sh new-nomtp`, c 131072, ub 4096, temp 1.0 / top-p 0.95 / top-k 20)
- VRAM 11.9 GB. pp of a 9145-token prompt: 303.6 / 306.9 t/s.
- Decode: p0..p4 24.82 / 24.81 / 24.74 / 24.88 / 24.78, **total 24.80 t/s** (3332 tokens). 27B reference: 67.27.

### 18:05 MTP crash: GGML_ASSERT(buffer) at server start, base and ported trees alike
- Backtrace (`~/qwen38-work/gdb-mtp.log`): llm_graph_input_kpool::set_input -> set_input_k_idxs on a tensor with no buffer,
  during common_context_can_seq_rm's test decode of the draft context.
- Cause: the MTP block (layer 48) has compress ratio 0 (dense attention), but graph_mtp built the k-pool input anyway;
  no op read k_idxs, so it got no buffer. Upstream master bug, not a port.
- Fix (commit below): build the k-pool input only when dsv4_compress_ratios[il] > 0. set_input_kpool is const (layout is
  updated in kpool_layout_update), so skipping it changes no state. The guide's branch (danielhanchen qwen4exp/mtp, Sep 21)
  also uses a plain dense KV input (build_attn_inp_kv) for the MTP block.

### 18:10 first MTP run, Q8_0 shared head on GPU, n-max 3 (`spec_run.sh mtp-q8-n3`)
- VRAM 17.1 GB. pp 9145 tok: 292.2 / 298.0 t/s.
- p0..p4: 35.10 / 29.24 / 34.51 / 27.14 / 30.74, **total 31.37 t/s**, acceptance 64.27% (2148/3342). No MTP: 24.80.
- commit b41230589 qwen4exp: build the MTP k-pool input only for a QSA block
