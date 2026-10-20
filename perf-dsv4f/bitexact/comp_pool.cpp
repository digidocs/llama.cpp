// bitwise check: ggml_dsv4_comp_pool vs the decomposed compressor pooling graph of deepseek4.cpp, both on CUDA
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "decomp_fn.inc"

static std::vector<float> run(ggml_backend_t be, bool fused, const std::vector<float> & kvd, const std::vector<float> & scd,
                              const std::vector<int32_t> & ixd, int64_t w, int64_t n_rows, int64_t E, int64_t ratio, int64_t n_blocks, bool overlap) {
    ggml_init_params ip = { 256*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * kv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w, n_rows);
    ggml_tensor * sc = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w, n_rows);
    ggml_tensor * ix = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ixd.size());
    ggml_set_input(kv); ggml_set_input(sc); ggml_set_input(ix);
    ggml_tensor * out = fused ? ggml_dsv4_comp_pool(ctx, kv, sc, ix, ratio, overlap) : decomposed(ctx, kv, sc, ix, ratio, E, n_blocks, overlap);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    ggml_backend_tensor_set(kv, kvd.data(), 0, kvd.size()*4);
    ggml_backend_tensor_set(sc, scd.data(), 0, scd.size()*4);
    ggml_backend_tensor_set(ix, ixd.data(), 0, ixd.size()*4);
    ggml_backend_graph_compute(be, gf);
    std::vector<float> r(ggml_nelements(out));
    ggml_backend_tensor_get(out, r.data(), 0, r.size()*4);
    ggml_gallocr_free(ga);
    ggml_free(ctx);
    return r;
}

int main() {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    std::mt19937 rng(1234);
    struct shape { const char * name; int64_t E, ratio; bool overlap; };
    const shape shapes[] = { {"csa", 512, 4, true}, {"lid", 128, 4, true}, {"hca", 512, 128, false} };
    int n_fail = 0, n_tot = 0;
    for (const auto & sh : shapes) {
        for (int64_t n_blocks : {1, 2, 5, 33, 160}) {
            for (int trial = 0; trial < 8; ++trial) {
                const int64_t w = sh.overlap ? 2*sh.E : sh.E;
                const int64_t n_rows = sh.overlap ? 2*sh.ratio*n_blocks + 3 : sh.ratio*n_blocks + 5;
                std::normal_distribution<float> nv(0.0f, trial % 2 ? 1.0f : 6.0f);
                std::vector<float> kvd(w*n_rows), scd(w*n_rows);
                for (auto & x : kvd) x = nv(rng);
                for (auto & x : scd) x = nv(rng);
                std::vector<int32_t> ixd((sh.overlap ? 2 : 1)*sh.ratio*n_blocks);
                for (auto & x : ixd) x = rng() % (n_rows + (sh.overlap ? 1 : 0));
                const auto a = run(be, true,  kvd, scd, ixd, w, n_rows, sh.E, sh.ratio, n_blocks, sh.overlap);
                const auto b = run(be, false, kvd, scd, ixd, w, n_rows, sh.E, sh.ratio, n_blocks, sh.overlap);
                n_tot++;
                if (a.size() != b.size() || memcmp(a.data(), b.data(), a.size()*4) != 0) {
                    size_t nd = 0; double md = 0;
                    for (size_t i = 0; i < a.size() && i < b.size(); ++i) { if (memcmp(&a[i], &b[i], 4)) { nd++; md = fmax(md, fabs(a[i]-b[i])/fmax(1e-30, fabs(b[i]))); } }
                    printf("MISMATCH %s n_blocks=%lld trial=%d: %zu/%zu elements differ, max rel %.3g\n", sh.name, (long long) n_blocks, trial, nd, b.size(), md);
                    n_fail++;
                }
            }
        }
    }
    printf("%d/%d cases bit-identical\n", n_tot - n_fail, n_tot);
    ggml_backend_free(be);
    return n_fail != 0;
}
