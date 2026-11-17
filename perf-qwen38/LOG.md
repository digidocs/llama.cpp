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

### 18:20 reference: guide branch danielhanchen/llama.cpp qwen4exp/mtp (6fcaa16f4, Sep 21), worktree ~/llama.cpp-dh-mtp
- MTP Q8_0 n-max 3 (`spec_run.sh dh-mtp-q8-n3`): total 31.67 t/s, acceptance 64.54%, md5 of all 5 replies identical to ours.
- llama-bench tg128 (no MTP): 18.96 t/s. Ours 25.76, master 16.12.
- So the ports help n=1 decode a lot but nothing with MTP: the verify batch (up to 4 tokens) runs the CPU mul_mat_id
  path the ports don't speed up (micro-bench gate_up n=3: 741 us ported vs 733 no-private). Next target: small-batch MUL_MAT_ID.

### 18:45 why do the branches differ at n=1 but not with MTP?
- llama-bench batch times (ms per batch, -p 1,2,3,4 -r 20): ours 40.0 / 58.4 / 72.3 / 87.3, guide 60.8 / 76.6 / 87.5 / 99.5.
- Server MTP steps (p4, from print_timing): ~3.0 drafted per step, so 4-token verify batches. 93.0 ms/step ours,
  92.4 ms/step guide. A step = verify + 3 MTP draft passes on the GPU (~6 ms), so verify is ~87 ms on both trees.
- So in steady-state decode both trees verify 4 tokens in ~87 ms (near DRAM bandwidth). The guide's llama-bench pp
  numbers carry ~8-12 ms of per-run overhead its tg128 doesn't (guide pp1 60.8 ms vs its tg128 52.7 ms; ours 40.0 vs 38.8).
  Inference, not a direct measurement: GGML_CPU_PROFILE isn't in the guide tree.
- The ports' real win is only 1-token decode (CPU mul_mat_id at n=1); n>=2 was already efficient upstream (micro-bench agrees).
- Unexplained: master 16.12 vs guide 18.96 at tg128 (newer master, possible decode regression). Matters only without MTP.
- Use server step time, not llama-bench pp, to compare verify cost across trees.

### 18:55 MoE cache: dead end on this machine (PCIe 3.0 x16, ~13 GB/s vs ~112 GB/s CPU reads)
- Fixed scripts/moe-cache-tune.py: both calibration sizes now respect --max-mib (the first run used 10% of the experts,
  12224 MiB, and ran out of VRAM with the MTP draft loaded, which the tuner doesn't measure).
- Tuner run 1 (2944 MiB, CPU share 0, every miss uploaded): 4.8 t/s. Killed by my 590 s timeout; the full tuner doesn't
  fit a foreground call at this speed.
- Direct test, MTP Q8_0 n-max 3 + `--moe-cache-mib 6000 --moe-cache-cpu-share 0.95`: total 31.83 t/s (vs 31.37 without),
  acceptance 63.0%, VRAM 23.1 GB. Within noise, eats the VRAM that context needs, and changes the replies (hits run on
  the GPU), so it would need a KLD check too. Not pursued.

### 19:00 draft length: MTP Q8_0 n-max 2 (`spec_run.sh mtp-q8-n2`)
- p0..p4 30.58 / 32.82 / 32.79 / 28.47 / 33.30, **total 31.68 t/s**, acceptance 74.3% (1971/2654). n-max 3: 31.37.
- Per step: n2 ~2.49 tokens in ~79 ms, n3 ~2.93 tokens in ~93 ms; the same tokens per ms. Draft length isn't a lever.
- Note: at temperature 1.0 the replies differ between configs, so per-prompt speeds wobble ~+-10%; use GREEDY=1 for tighter A/B.

### 19:10 profile of steady-state MTP decode (Q8_0 head, n-max 3, greedy), `~/qwen38-work/prof/run_prof.sh mtp-n3`
- Drafter is fully on the GPU (50/50 layers, 2.6 GB CUDA0 weights + its KV). Its compute buffer is 1.77 GB because the draft
  context inherits ub 4096: `--spec-draft-n-ubatch` should free ~1.5 GB of VRAM.
- Run: 1500 tokens, 30.07 t/s, acceptance 68.1%, mean accepted length 3.04. Capture: 20 s, ~205 steps, ~97 ms/step under nsys.
- CPU (GGML_CPU_PROFILE, whole run): MUL_MAT_ID 95.6% of CPU op time (~63 ms/step), GET_ROWS q8_0 4.0% (0.46 ms/call,
  ~2.7 ms/step; which table? likely the lazy PLE), everything else < 0.5%.
- GPU (nsys, 20 s): kernels 1.99 s total, so the GPU is busy ~10% (~9.7 ms/step). Top: mmvq q8_0 ncols 4 (44.6%, the 4-token
  verify), mmvq ncols 1 (13.6%, drafts), quantize_q8_1 4.5%, GDN 3.4%. Memcpy H2D 179 ms + D2H 60 ms (~1.2 ms/step).
- CUDA API (host): cudaGraphLaunch 10276 calls (~50/step) avg 97 us (inflated by --cuda-graph-trace=node), cudaStreamSynchronize
  29336 calls (~143/step) avg 24 us, cudaEventSynchronize ~26/step avg 225 us (waiting on GPU work).
- Per-step budget (~93 ms unprofiled): CPU experts ~63, GPU kernels ~10, copies ~1, CPU GET_ROWS ~3, launch+sync overhead
  ~5-8, unexplained host time ~10 (graph build / sched / sampling?; needs perf).

### 19:20 perf (LBR call graphs, 15 s of MTP decode, all threads; `~/qwen38-work/prof/run_perf.sh mtp-n3`): 32.34 t/s
- On-CPU samples: ggml_vec_dot_q8_0_q8_0 66.6%, libgomp 31.1% (mostly the OpenMP team spinning between parallel
  regions while the main thread drives the GPU; some ggml_barrier -> GOMP_barrier), kernel 0.9%, libcuda 0.9%.
- Host-side llama/ggml work (graph build, sched split, allocator, sampling, seq_rm) is < 0.2% in total: not a target.
- Combined with GGML_CPU_PROFILE (CPU graph wall 68.7 ms/step) and nsys (GPU kernels ~10 ms/step): of a ~93 ms step,
  ~24 ms is outside the CPU graphs; ~10 ms of that is GPU compute, the other ~14 ms is CPU<->GPU hand-off latency
  (~50 graph launches, ~143 stream syncs, ~26 event syncs per step, i.e. ~2 hand-offs per layer). At n=1 the same
  non-CPU time is ~17.6 ms, so it is mostly a fixed per-layer cost, not proportional to batch size.

### 19:30 hand-off latency, step 1: async host -> device input copies
- nsys graph-level timeline (`~/qwen38-work/prof/g-n3.sqlite`, `timeline.py`): a verify layer is ~2.4 ms: CPU experts ~1840 us,
  GPU graph 250-410 us, CPU->GPU hand-off ~130 us (H2D 400 KB 35 us + blocking stream sync 37 us + cudaGraphLaunch 50-90 us),
  GPU->CPU ~20 us. Verify graph: 98 splits (49 GPU/CPU round trips); draft graphs 2 splits.
- Cause of the blocking sync: CPU -> CUDA copies fail cpy_tensor_async (device-to-device only) and fall back to
  ggml_backend_tensor_copy. Now queued with set_tensor_async when the source is host memory (the CPU compute buffer is
  pinned CUDA_Host). GGML_SCHED_SYNC_H2D=1 restores the old path.
- Greedy 5 prompts, MTP Q8_0 n3: sync 32.63 t/s, async **33.23 t/s (+1.8%)**, md5 of all 5 replies identical.
- commit 589e6576f ggml-backend : sched queues host -> device input copies instead of waiting

### 19:45 dead ends: CUDA graphs off, PLE page faults, per-graph thread affinity
- GGML_CUDA_DISABLE_GRAPHS=1 (greedy): 32.37 t/s vs 33.23 with graphs (cudaGraphLaunch costs 54-58 us per layer, but
  per-kernel launches cost more). Graphs stay on.
- PLE GET_ROWS: 0 major and 0 minor page faults over 600 tokens (/proc/PID/stat), so the slow CPU GET_ROWS q8_0
  (0.46 ms/call) is not disk reads of the lazy 51 GB table. Not yet explained.
- perf kernel samples (0.9% of all): 66% native_queued_spin_lock_slowpath, from pthread_setaffinity_np in
  ggml_graph_compute's OpenMP region (affinity re-applied for all 24 threads on every CPU graph, ~50 per step).
  Patch to apply it only when a thread's mask changes (`~/qwen38-work/affinity-once.patch`, env
  GGML_CPU_AFFINITY_EVERY_GRAPH=1 for the old path): greedy 33.34 vs 33.20 t/s, md5 identical. Within noise: reverted.
- Server hang: llama-server ignored SIGINT for ~5 min after a /completion request; SIGTERM worked. Scripts use SIGTERM.
