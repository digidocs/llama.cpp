// bitwise check of the hyper-connection gate chains: run once normally (fused) and once with
// GGML_CUDA_DISABLE_FUSION=1 (four kernels), each writing its outputs to argv[1]; compare the files with cmp
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_t be = ggml_backend_cuda_init(0);
    FILE * f = fopen(argv[1], "wb");
    std::mt19937 rng(42);
    const int64_t hc = 4, mix = (2 + hc)*hc;
    for (int64_t nt : {1, 3, 7, 64, 512, 4096}) {
        for (int trial = 0; trial < 6; ++trial) {
            ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * mixes = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, mix, nt);
            ggml_tensor * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
            ggml_tensor * base  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, mix);
            ggml_set_input(mixes); ggml_set_input(scale); ggml_set_input(base);
            auto chain = [&](int64_t off, int si, float s, float b) {
                ggml_tensor * x = ggml_view_2d(ctx, mixes, hc, nt, mixes->nb[1], off*sizeof(float));
                ggml_tensor * m = ggml_view_1d(ctx, scale, 1, si*sizeof(float));
                ggml_tensor * c = ggml_view_1d(ctx, base, hc, off*sizeof(float));
                ggml_tensor * t = ggml_add(ctx, ggml_mul(ctx, x, m), c);
                return ggml_scale_bias(ctx, ggml_sigmoid(ctx, t), s, b);
            };
            ggml_tensor * pre  = chain(0,  0, 1.0f, 1e-6f);  // hc_pre: sigmoid + eps
            ggml_tensor * post = chain(hc, 1, 2.0f, 0.0f);   // hc_post: 2*sigmoid
            ggml_set_output(pre); ggml_set_output(post);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, pre);
            ggml_build_forward_expand(gf, post);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            ggml_gallocr_alloc_graph(ga, gf);
            const float sd = trial % 3 == 0 ? 0.5f : trial % 3 == 1 ? 4.0f : 40.0f;
            std::normal_distribution<float> nd(0.0f, sd);
            std::vector<float> md(mix*nt), sc(3), bd(mix);
            for (auto & v : md) v = nd(rng);
            for (auto & v : sc) v = nd(rng);
            for (auto & v : bd) v = nd(rng);
            if (trial == 5) { md[0] = 1e-39f; md[1] = -0.0f; md[2] = 1e30f; md[3] = -1e30f; } // denormal, -0, saturation
            ggml_backend_tensor_set(mixes, md.data(), 0, md.size()*4);
            ggml_backend_tensor_set(scale, sc.data(), 0, sc.size()*4);
            ggml_backend_tensor_set(base, bd.data(), 0, bd.size()*4);
            ggml_backend_graph_compute(be, gf);
            for (ggml_tensor * t : {pre, post}) {
                std::vector<float> r(ggml_nelements(t));
                ggml_backend_tensor_get(t, r.data(), 0, r.size()*4);
                fwrite(r.data(), 4, r.size(), f);
            }
            ggml_gallocr_free(ga);
            ggml_free(ctx);
        }
    }
    fclose(f);
    ggml_backend_free(be);
    return 0;
}
