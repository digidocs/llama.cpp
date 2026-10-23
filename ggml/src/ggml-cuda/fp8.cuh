#pragma once

#include "common.cuh"

// FP8 storage for BF16 weights whose values are exactly e4m3 numbers times a power-of-two scale per 32 values, as for
// weights shipped in FP8 and expanded to BF16 without rounding. The tensor keeps type BF16; its buffer holds the e4m3
// bytes of all values followed by one E8M0 scale byte per 32 values, in the order of the BF16 elements.

#define GGML_CUDA_FP8_BLOCK 32

static inline size_t ggml_cuda_fp8_size(int64_t n) {
    return n + n/GGML_CUDA_FP8_BLOCK;
}

// packs n BF16 values (n a multiple of GGML_CUDA_FP8_BLOCK) into e4m3 values q[n] and scales e[n/32];
// returns false if a value is not exactly representable
bool ggml_cuda_fp8_pack(const uint16_t * src, int64_t n, uint8_t * q, uint8_t * e);

void ggml_cuda_fp8_unpack(const uint8_t * q, const uint8_t * e, int64_t n, uint16_t * dst);

// expands the FP8 storage of a BF16 tensor with n elements to BF16 on the GPU
void ggml_cuda_fp8_to_bf16(const void * fp8, int64_t n, nv_bfloat16 * dst, cudaStream_t stream);

// exact value of an e4m3 byte times 2^(scale - 127)
static __device__ __forceinline__ float ggml_cuda_fp8_e4m3_scaled(const uint8_t q, const uint8_t scale) {
    const uint32_t sign = (uint32_t) (q & 0x80) << 24;
    const int      ex   = (q >> 3) & 0x0F;
    const uint32_t m    = q & 0x07;
    if (ex == 0) {
        // subnormal e4m3: m * 2^-9 * 2^(scale - 127), exact as a product of a small integer and a power of two
        return __uint_as_float(sign | __float_as_uint((float) m * __uint_as_float((uint32_t) (scale - 9) << 23)));
    }
    return __uint_as_float(sign | ((uint32_t) (ex - 7 + scale) << 23) | (m << 20));
}
