// real-weight MoE expert benchmark: gate/up MUL_MAT_ID -> swiglu -> down MUL_MAT_ID on the CPU backend
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model-00001-of-N.gguf [threads=16] [cpu_first=0] [layer=10] [tokens=1] [gap_us=0] [iters=200] [n_layers_rot=1]\n", argv[0]); return 1; }
    std::string first = argv[1];
    int n_threads = argc > 2 ? atoi(argv[2]) : 16;
    int cpu_first = argc > 3 ? atoi(argv[3]) : 0;
    int layer     = argc > 4 ? atoi(argv[4]) : 10;
    int n_tokens  = argc > 5 ? atoi(argv[5]) : 1;
    int gap_us    = argc > 6 ? atoi(argv[6]) : 0;
    int iters     = argc > 7 ? atoi(argv[7]) : 200;
    int n_rot     = argc > 8 ? atoi(argv[8]) : 1; // number of consecutive layers to cycle through (defeats L3 reuse)
    int mode      = argc > 9 ? atoi(argv[9]) : 0; // 0 full layer, 1 gate only, 2 down only

    ggml_cpu_init();
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    for (int i = 0; i < n_threads; i++) tpp.cpumask[cpu_first + i] = true;
    tpp.strict_cpu = true;
    ggml_threadpool * tp = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(be, tp);
    ggml_backend_cpu_set_n_threads(be, n_threads);

    const char * names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead()*4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    std::vector<std::vector<ggml_tensor *>> W(n_rot, std::vector<ggml_tensor *>(3));
    struct src_t { std::string path; size_t off; size_t sz; };
    std::vector<std::vector<src_t>> srcs(n_rot, std::vector<src_t>(3));

    // scan split files
    std::string base = first.substr(0, first.find("-00001-of-"));
    std::string tail = first.substr(first.find("-00001-of-") + 10);
    int n_split = atoi(tail.c_str());
    int found = 0;
    for (int s = 1; s <= n_split; s++) {
        char buf[32]; snprintf(buf, sizeof(buf), "-%05d-of-", s);
        std::string path = base + buf + tail;
        gguf_init_params gp = { true, nullptr };
        gguf_context * g = gguf_init_from_file(path.c_str(), gp);
        if (!g) { fprintf(stderr, "failed %s\n", path.c_str()); return 1; }
        ggml_context * mctx = nullptr;
        gguf_free(g);
        gguf_init_params gp2 = { true, &mctx };
        g = gguf_init_from_file(path.c_str(), gp2);
        for (int r = 0; r < n_rot; r++) for (int k = 0; k < 3; k++) {
            std::string tn = "blk." + std::to_string(layer + r) + "." + names[k] + ".weight";
            int64_t id = gguf_find_tensor(g, tn.c_str());
            if (id < 0) continue;
            ggml_tensor * mt = ggml_get_tensor(mctx, tn.c_str());
            W[r][k] = ggml_new_tensor(ctx, mt->type, 3, mt->ne);
            ggml_set_name(W[r][k], tn.c_str());
            srcs[r][k] = { path, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id), ggml_nbytes(mt) };
            found++;
        }
        ggml_free(mctx);
        gguf_free(g);
    }
    if (found != 3*n_rot) { fprintf(stderr, "found %d/%d tensors\n", found, 3*n_rot); return 1; }
    const int64_t n_embd = W[0][0]->ne[0], n_ff = W[0][0]->ne[1], n_expert = W[0][0]->ne[2];
    const int n_used = 6;
    printf("layer %d..%d: %s %lld x %lld x %lld, threads %d @ cpu %d.., tokens %d, gap %d us\n", layer, layer + n_rot - 1,
           ggml_type_name(W[0][0]->type), (long long) n_embd, (long long) n_ff, (long long) n_expert, n_threads, cpu_first, n_tokens, gap_us);

    ggml_tensor * inp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * dinp = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_used, n_tokens);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    for (int r = 0; r < n_rot; r++) for (int k = 0; k < 3; k++) {
        int fd = open(srcs[r][k].path.c_str(), O_RDONLY);
        size_t done = 0; char * dst = (char *) W[r][k]->data;
        while (done < srcs[r][k].sz) { ssize_t n = pread(fd, dst + done, srcs[r][k].sz - done, srcs[r][k].off + done); if (n <= 0) { perror("pread"); return 1; } done += n; }
        close(fd);
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n_embd*n_tokens); for (auto & v : x) v = nd(rng);
    ggml_backend_tensor_set(inp, x.data(), 0, ggml_nbytes(inp));
    { std::vector<float> d(n_ff*n_used*n_tokens); for (auto & v : d) v = nd(rng); ggml_backend_tensor_set(dinp, d.data(), 0, ggml_nbytes(dinp)); }

    std::vector<ggml_cgraph *> gfs(n_rot);
    ggml_context * gctx = ggml_init({ 256*ggml_tensor_overhead() + ggml_graph_overhead()*n_rot, nullptr, true });
    std::vector<ggml_tensor *> outs(n_rot);
    for (int r = 0; r < n_rot; r++) {
        ggml_tensor * gate = ggml_mul_mat_id(gctx, W[r][0], inp, ids);
        ggml_tensor * up   = ggml_mul_mat_id(gctx, W[r][1], inp, ids);
        ggml_tensor * act  = ggml_swiglu_split(gctx, gate, up);
        ggml_tensor * down = (mode == 1 || mode == 5) ? gate : ggml_mul_mat_id(gctx, W[r][2], mode == 2 ? dinp : act, ids);
        gfs[r] = ggml_new_graph(gctx);
        ggml_build_forward_expand(gfs[r], down);
        outs[r] = down;
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    for (int r = 0; r < n_rot; r++) ggml_gallocr_reserve(ga, gfs[r]);

    std::vector<int32_t> idv(n_used*n_tokens);
    double t_tot = 0; int n_meas = 0; double t_alt[2] = {0, 0}; int n_alt[2] = {0, 0};
    double bytes_per = 0; for (int k = 0; k < 3; k++) if (mode == 0 || ((mode == 1 || mode >= 3) && k == 0) || (mode == 2 && k == 2)) bytes_per += (double) ggml_nbytes(W[0][k]) / n_expert * n_used;
    for (int it = 0; it < iters + 5; it++) {
        int r = it % n_rot;
        ggml_gallocr_alloc_graph(ga, gfs[r]);
        for (int t = 0; t < n_tokens; t++) {
            std::vector<int> perm(n_expert); for (int i = 0; i < n_expert; i++) perm[i] = i;
            for (int i = 0; i < n_used; i++) { int j = i + rng() % (n_expert - i); std::swap(perm[i], perm[j]); idv[t*n_used + i] = perm[i]; }
        }
        ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
        if (gap_us > 0) { double t0 = now_us(); while (now_us() - t0 < gap_us) {} } // emulate GPU segment with the caller spinning
        double t0 = now_us();
        const bool direct = mode == 3 || mode == 4 || (mode == 5 && (it & 1));
        if (direct) {
            // direct vec_dot over the gate weights, static row split, bypassing the graph
            const auto * tt = ggml_get_type_traits_cpu(W[r][0]->type);
            static std::vector<char> yq(ggml_row_size(GGML_TYPE_Q8_0, n_embd));
            static std::vector<float> o(n_ff*n_used);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x.data(), yq.data(), n_embd);
            const char * wb = (const char *) W[r][0]->data;
            const size_t rs = W[r][0]->nb[1], es = W[r][0]->nb[2];
            #pragma omp parallel num_threads(n_threads)
            {
                const int ith = omp_get_thread_num();
                cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu_first + ith, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
                if (getenv("H_BARRIER")) {
                    #pragma omp barrier
                    #pragma omp barrier
                }
                for (int u = 0; u < n_used; u++) {
                    const char * e = wb + idv[u]*es;
                    const int r0 = n_ff*ith/n_threads, r1 = n_ff*(ith + 1)/n_threads;
                    if (mode == 3) {
                        for (int row = r0; row < r1; row++) tt->vec_dot(n_embd, &o[u*n_ff + row], 0, e + row*rs, 0, yq.data(), 0, 1);
                    } else {
                        float tmp[16];
                        for (int b0 = r0; b0 < r1; b0 += 16) {
                            for (int row = b0; row < b0 + 16 && row < r1; row++) tt->vec_dot(n_embd, &tmp[row - b0], 0, e + row*rs, 0, yq.data(), 0, 1);
                            memcpy(&o[u*n_ff + b0], tmp, (std::min(b0 + 16, r1) - b0)*sizeof(float));
                        }
                    }
                }
            }
        } else {
            ggml_backend_graph_compute(be, gfs[r]);
        }
        double dt = now_us() - t0;
        if (it >= 5) { t_tot += dt; n_meas++; t_alt[direct] += dt; n_alt[direct]++; }
    }
    double us = t_tot / n_meas;
    if (mode == 5) printf("interleaved: ggml %.1f us, direct %.1f us\n", t_alt[0]/n_alt[0], t_alt[1]/n_alt[1]);
    printf("%.1f us/layer-call, %.1f GB/s (unique-expert bytes, %.2f MB/token-layer), %.2f ms/token for 43 layers\n",
           us, bytes_per*n_tokens/us/1e3, bytes_per/1e6, us*43/1e3/n_tokens*n_tokens);
    return 0;
}
