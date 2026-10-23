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

// marks FP8 storage as the weight type of a kernel template; pointer arithmetic is per element (one byte)
struct ggml_cuda_fp8_t {
    uint8_t q;
};

// true if the tensor is a BF16 weight held in a CUDA FP8 buffer
bool ggml_cuda_tensor_is_fp8(const ggml_tensor * tensor);

// exact value of an e4m3 byte times 2^(scale - 127): an integer mantissa times a power of two, both normal floats
// (no subnormals, which -use_fast_math would flush) and no branches
static __device__ __forceinline__ float ggml_cuda_fp8_e4m3_scaled(const uint32_t q, const uint32_t scale) {
    const uint32_t ex = (q >> 3) & 0x0F;
    const uint32_t m  = q & 0x07;
    const float    mi = (float) (ex ? (8 | m) : m);
    const float    p2 = __uint_as_float(((ex ? ex : 1) + scale - 10) << 23);
    return __uint_as_float(__float_as_uint(mi*p2) | ((q & 0x80) << 24));
}
