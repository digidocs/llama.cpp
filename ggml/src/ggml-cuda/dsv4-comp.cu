#include "dsv4-comp.cuh"
#include "softmax.cuh"

// The decomposed compressor graph runs SOFT_MAX over the J entries of each (element, block) row, multiplies by the
// values (bin_bcast) and reduces with SUM_ROWS. To give bit-identical results, this kernel replays those kernels'
// arithmetic: one warp per output reproduces the per-thread partial sums and the warp butterfly / inter-warp trees
// of soft_max_f32 (block of sm_block threads) and reduce_rows_f32 (block of sr_block threads).
// Lane l of the warp holds the entries j = l + 32*k: in a block reduction, entry j belongs to virtual thread j, so
// virtual warp k of the reference block is exactly lane l's k-th entry.

#define DSV4_COMP_MAX_K 8 // J <= 256

template <bool overlap, int n_k>
static __global__ void dsv4_comp_pool_f32(
        const char * __restrict__ kv, const char * __restrict__ score, const int32_t * __restrict__ ids, float * __restrict__ dst,
        const int64_t n_embd, const int64_t n_blocks, const int64_t n_rows, const int32_t ratio,
        const size_t nb_kv1, const size_t nb_score1, const size_t nb_dst2,
        const int sm_block, const int sr_block) {
    const int     lane = threadIdx.x;
    const int64_t e    = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    const int64_t b    = blockIdx.y;
    if (e >= n_embd) {
        return;
    }

    const int64_t n_read = (int64_t) ratio*n_blocks;
    const int     J      = overlap ? 2*ratio : ratio;

    float v[n_k];
    float s[n_k];

    // gather, and the soft_max_f32 input transform val = x*scale + 0 with scale 1
    float max_val = -INFINITY;
#pragma unroll
    for (int k = 0; k < n_k; ++k) {
        const int j = lane + 32*k;
        v[k] = 0.0f;
        s[k] = -INFINITY;
        if (j < J) {
            const bool    second = overlap && j >= ratio;
            const int64_t row    = second ? ids[n_read + b*ratio + (j - ratio)] : ids[b*ratio + j];
            const int64_t col    = second ? n_embd + e : e;
            if (row >= n_rows) {
                // the appended row of the decomposed graph: scale(row 0, 0) for kv, scale_bias(row 0, 0, -inf) for score
                v[k] = 0.0f*((const float *) kv)[col] + 0.0f;
                s[k] = 0.0f*((const float *) score)[col] + (-INFINITY);
            } else {
                v[k] = ((const float *) (kv    + row*nb_kv1))[col];
                s[k] = ((const float *) (score + row*nb_score1))[col];
            }
            s[k] = s[k]*1.0f + 0.0f;
            max_val = max(max_val, s[k]);
        }
    }
    max_val = warp_reduce_max(max_val);

    // softmax sum: virtual thread j holds tmp = 0 + expf(val - max); virtual warp k = this lane's entry k
    float wsum[n_k];
#pragma unroll
    for (int k = 0; k < n_k; ++k) {
        const int j = lane + 32*k;
        float tmp = 0.0f;
        if (j < J) {
            s[k] = expf(s[k] - max_val);
            tmp += s[k];
        }
        wsum[k] = warp_reduce_sum(tmp);
    }
    float sum = wsum[0];
    if (sm_block > WARP_SIZE) {
        float t = 0.0f;
#pragma unroll
        for (int k = 0; k < n_k; ++k) {
            if (lane == k && k < sm_block/WARP_SIZE) {
                t = wsum[k];
            }
        }
        sum = warp_reduce_sum(t);
    }
    const float inv_sum = 1.0f/sum;

    // weights, product with the values, then reduce_row_f32
    float p[n_k];
#pragma unroll
    for (int k = 0; k < n_k; ++k) {
        p[k] = v[k]*(s[k]*inv_sum);
    }

    float res;
    if (sr_block == WARP_SIZE) {
        // one warp: thread `lane` accumulates entries lane + 32*q into sum_temp[q] (q < 8), then sums sum_temp in order
        float part = 0.0f;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            float st = 0.0f;
            if (q < n_k && lane + 32*q < J) {
                st += p[q < n_k ? q : 0];
            } else {
                st += 0.0f;
            }
            part += st;
        }
        res = warp_reduce_sum(part);
    } else {
        // sr_block > J: virtual thread j only has entry j, virtual warp k = this lane's entry k, then the inter-warp step
#pragma unroll
        for (int k = 0; k < n_k; ++k) {
            const int j = lane + 32*k;
            float part = 0.0f;
            float st0 = 0.0f;
            if (j < J) {
                st0 += p[k];
            }
            part += st0;
#pragma unroll
            for (int q = 1; q < 8; ++q) {
                part += 0.0f;
            }
            wsum[k] = warp_reduce_sum(part);
        }
        float t = 0.0f;
#pragma unroll
        for (int k = 0; k < n_k; ++k) {
            if (lane == k) {
                t = wsum[k];
            }
        }
        res = warp_reduce_sum(t);
    }

    if (lane == 0) {
        *(float *) ((char *) dst + b*nb_dst2 + e*sizeof(float)) = res;
    }
}

void ggml_cuda_op_dsv4_comp_pool(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kv    = dst->src[0];
    const ggml_tensor * score = dst->src[1];
    const ggml_tensor * idxs  = dst->src[2];

    GGML_ASSERT(kv->type == GGML_TYPE_F32 && score->type == GGML_TYPE_F32 && idxs->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t ratio   = ggml_get_op_params_i32(dst, 0);
    const bool    overlap = ggml_get_op_params_i32(dst, 1) != 0;

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_blocks = dst->ne[2];
    const int     J        = overlap ? 2*ratio : ratio;
    GGML_ASSERT(J <= 32*DSV4_COMP_MAX_K);
    if (n_blocks == 0) {
        return;
    }

    // block sizes the decomposed SOFT_MAX and SUM_ROWS kernels would use for this shape
    int sm_block = WARP_SIZE;
    while (sm_block < J && sm_block < CUDA_SOFT_MAX_BLOCK_SIZE) {
        sm_block *= 2;
    }
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int sr_block = (n_embd*n_blocks)/nsm < 2 ? 512 : (J < 1024 ? 32 : 128);
    GGML_ASSERT(sm_block <= 32*DSV4_COMP_MAX_K);
    GGML_ASSERT(sr_block == WARP_SIZE || (sr_block >= J && sr_block/WARP_SIZE <= WARP_SIZE));

    const int n_warps = 4;
    const dim3 block(WARP_SIZE, n_warps, 1);
    const dim3 grid((n_embd + n_warps - 1)/n_warps, n_blocks, 1);

    cudaStream_t stream = ctx.stream();
#define DSV4_COMP_LAUNCH(OV, NK) \
    dsv4_comp_pool_f32<OV, NK><<<grid, block, 0, stream>>>((const char *) kv->data, (const char *) score->data, \
            (const int32_t *) idxs->data, (float *) dst->data, n_embd, n_blocks, kv->ne[1], ratio, kv->nb[1], score->nb[1], dst->nb[2], \
            sm_block, sr_block)
    const int n_k = (J + WARP_SIZE - 1)/WARP_SIZE;
    if (overlap) {
        GGML_ASSERT(n_k == 1);
        DSV4_COMP_LAUNCH(true, 1);
    } else if (n_k <= 1) {
        DSV4_COMP_LAUNCH(false, 1);
    } else if (n_k <= 4) {
        DSV4_COMP_LAUNCH(false, 4);
    } else {
        DSV4_COMP_LAUNCH(false, 8);
    }
#undef DSV4_COMP_LAUNCH
}
