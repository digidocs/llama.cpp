#include <atomic>
#include <thread>
#include <vector>

#include "fp8.cuh"

// e4m3 byte for a BF16 value with unbiased exponent eu, 7-bit mantissa m and sign s under the scale 2^k; -1 if inexact
static int fp8_encode(const int eu, const int m, const int s, const int k) {
    const int e = eu - k;
    if (e >= -6 && e <= 8) {
        if ((m & 0x0F) != 0 || (e == 8 && (m >> 4) == 7)) {
            return -1;
        }
        return (s << 7) | ((e + 7) << 3) | (m >> 4);
    }
    if (e >= -9 && e <= -7) {
        // subnormal e4m3: (128 + m) * 2^(e - 7) = j * 2^-9
        const int shift = -2 - e;
        const int v     = 128 + m;
        if ((v & ((1 << shift) - 1)) != 0) {
            return -1;
        }
        return (s << 7) | (v >> shift);
    }
    return -1;
}

static bool fp8_pack_blocks(const uint16_t * src, int64_t ib0, int64_t ib1, uint8_t * q, uint8_t * e) {
    for (int64_t ib = ib0; ib < ib1; ++ib) {
        const uint16_t * x = src + ib*GGML_CUDA_FP8_BLOCK;

        int emax = INT_MIN;
        for (int i = 0; i < GGML_CUDA_FP8_BLOCK; ++i) {
            const int ex = (x[i] >> 7) & 0xFF;
            const int m  = x[i] & 0x7F;
            if (ex == 0xFF || (ex == 0 && m != 0)) {
                return false; // inf, nan or a BF16 subnormal
            }
            if (ex != 0) {
                emax = std::max(emax, ex - 127);
            }
        }

        if (emax == INT_MIN) {
            e[ib] = 127;
            memset(q + ib*GGML_CUDA_FP8_BLOCK, 0, GGML_CUDA_FP8_BLOCK);
            continue;
        }

        // the scale that puts the block max at the top e4m3 exponent, or one step higher if the max does not fit there
        bool ok = false;
        for (int k = emax - 8; k <= emax - 7 && !ok; ++k) {
            ok = true;
            for (int i = 0; i < GGML_CUDA_FP8_BLOCK && ok; ++i) {
                const int ex = (x[i] >> 7) & 0xFF;
                const int m  = x[i] & 0x7F;
                const int s  = x[i] >> 15;
                int b;
                if (ex == 0) {
                    b = s << 7;
                } else {
                    b = fp8_encode(ex - 127, m, s, k);
                }
                if (b < 0) {
                    ok = false;
                } else {
                    q[ib*GGML_CUDA_FP8_BLOCK + i] = (uint8_t) b;
                }
            }
            // keep the decode in the normal float range (see ggml_cuda_fp8_e4m3_scaled)
            if (ok && (k + 127 < 16 || k + 127 > 230)) {
                ok = false;
            }
            if (ok) {
                e[ib] = (uint8_t) (k + 127);
            }
        }
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool ggml_cuda_fp8_pack(const uint16_t * src, int64_t n, uint8_t * q, uint8_t * e) {
    GGML_ASSERT(n % GGML_CUDA_FP8_BLOCK == 0);

    // blocks are independent: split them over threads, a model has GBs of such weights
    const int64_t nb        = n/GGML_CUDA_FP8_BLOCK;
    const int     n_threads = (int) std::min<int64_t>(std::max(1u, std::thread::hardware_concurrency()), std::max<int64_t>(1, nb/4096));
    std::atomic<bool> ok{true};
    std::vector<std::thread> threads;
    for (int t = 1; t < n_threads; ++t) {
        threads.emplace_back([&, t]() {
            if (!fp8_pack_blocks(src, nb*t/n_threads, nb*(t + 1)/n_threads, q, e)) {
                ok = false;
            }
        });
    }
    if (!fp8_pack_blocks(src, 0, nb/n_threads, q, e)) {
        ok = false;
    }
    for (auto & th : threads) {
        th.join();
    }
    return ok;
}

void ggml_cuda_fp8_unpack(const uint8_t * q, const uint8_t * e, int64_t n, uint16_t * dst) {
    for (int64_t i = 0; i < n; ++i) {
        const uint8_t b  = q[i];
        const int     k  = e[i/GGML_CUDA_FP8_BLOCK] - 127;
        const int     s  = b >> 7;
        const int     ex = (b >> 3) & 0x0F;
        const int     m  = b & 0x07;
        float v;
        if (ex == 0) {
            v = ldexpf((float) m, k - 9);
        } else {
            v = ldexpf(1.0f + m/8.0f, ex - 7 + k);
        }
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        dst[i] = (uint16_t) ((s << 15) | (bits >> 16));
    }
}

static __global__ void fp8_to_bf16(const uint8_t * __restrict__ q, const uint8_t * __restrict__ e, const int64_t n, nv_bfloat16 * __restrict__ dst) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float v = ggml_cuda_fp8_e4m3_scaled(q[i], e[i/GGML_CUDA_FP8_BLOCK]);
    dst[i] = __ushort_as_bfloat16((unsigned short) (__float_as_uint(v) >> 16));
}

void ggml_cuda_fp8_to_bf16(const void * fp8, int64_t n, nv_bfloat16 * dst, cudaStream_t stream) {
    const uint8_t * q = (const uint8_t *) fp8;
    const int block_size = 256;
    fp8_to_bf16<<<(n + block_size - 1)/block_size, block_size, 0, stream>>>(q, q + n, n, dst);
}
