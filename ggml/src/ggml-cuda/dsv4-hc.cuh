#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);
// fused MUL -> ADD -> SIGMOID -> SCALE (hyper-connection gate), the result is written to scale
void ggml_cuda_op_dsv4_hc_gate(ggml_backend_cuda_context & ctx, const ggml_tensor * mul, const ggml_tensor * add, ggml_tensor * scale);
