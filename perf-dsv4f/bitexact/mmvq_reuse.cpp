// bitwise check of the mul_mat_vec_q src1 reuse: run normally and with GGML_CUDA_NO_MMVQ_SRC1_REUSE=1, compare with cmp.
// Four q8_0 mat-vecs read the same x; between the 2nd and 3rd an in-place op rewrites x's memory, so the
// 3rd and 4th must see the new values (the cache must be invalidated)
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    FILE * f = fopen(argv[1], "wb");
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t K = 4096, N = 1024, nw = 4;
    for (int trial = 0; trial < 4; ++trial) {
        ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * wctx = ggml_init(ip);
        std::vector<ggml_tensor *> w(nw);
        for (auto & t : w) t = ggml_new_tensor_2d(wctx, GGML_TYPE_Q8_0, K, N);
        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
        std::vector<float> wf(K*N);
        std::vector<char>  wq(ggml_row_size(GGML_TYPE_Q8_0, K)*N);
        for (auto & t : w) {
            for (auto & v : wf) v = nd(rng);
            ggml_quantize_chunk(GGML_TYPE_Q8_0, wf.data(), wq.data(), 0, N, K, nullptr);
            ggml_backend_tensor_set(t, wq.data(), 0, wq.size());
        }

        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
        ggml_set_input(x);
        ggml_tensor * xs = ggml_scale(ctx, x, 1.0f);              // computed hidden state (not a graph input)
        ggml_tensor * o0 = ggml_mul_mat(ctx, w[0], xs);
        ggml_tensor * o1 = ggml_mul_mat(ctx, w[1], xs);
        ggml_tensor * xm = ggml_scale_inplace(ctx, xs, 3.0f);     // rewrites xs's memory
        ggml_tensor * o2 = ggml_mul_mat(ctx, w[2], xs);           // must see the rewritten values
        ggml_tensor * o3 = ggml_mul_mat(ctx, w[3], xs);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        for (ggml_tensor * t : {o0, o1, xm, o2, o3}) {
            ggml_set_output(t);
            ggml_build_forward_expand(gf, t);
        }
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
        ggml_gallocr_alloc_graph(ga, gf);
        std::vector<float> xd(K);
        for (auto & v : xd) v = nd(rng);
        ggml_backend_tensor_set(x, xd.data(), 0, K*4);
        for (int rep = 0; rep < 3; ++rep) { // repeated computes go through CUDA graph capture/replay
            ggml_backend_graph_compute(be, gf);
        }
        for (ggml_tensor * t : {o0, o1, o2, o3}) {
            std::vector<float> r(ggml_nelements(t));
            ggml_backend_tensor_get(t, r.data(), 0, r.size()*4);
            fwrite(r.data(), 4, r.size(), f);
        }
        ggml_gallocr_free(ga);
        ggml_free(ctx);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
    }
    fclose(f);
    ggml_backend_free(be);
    return 0;
}
