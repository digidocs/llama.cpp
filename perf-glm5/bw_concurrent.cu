// DRAM headroom check: CPU read bandwidth (N threads, AVX2 loads over a large buffer) alone and while the GPU
// streams pinned host memory over PCIe (H2D copies, or a kernel reading mapped host memory)
// build: nvcc -O3 -arch=sm_86 -Xcompiler -fopenmp,-mavx2 bw_concurrent.cu -o bw_concurrent
// run:   ./bw_concurrent <cpu threads> [cpu GB] [pinned GB]
#include <cuda_runtime.h>
#include <immintrin.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double cpu_read(const char * buf, size_t n, int nth, int reps) {
    double t0 = now();
    __m256i acc_all = _mm256_setzero_si256();
#pragma omp parallel num_threads(nth)
    {
        __m256i acc = _mm256_setzero_si256();
        const int ith = omp_get_thread_num();
        const size_t chunk = n / nth / 4096 * 4096;
        for (int r = 0; r < reps; r++) {
            const __m256i * p = (const __m256i *) (buf + ith * chunk);
            for (size_t i = 0; i < chunk / 32; i += 4) {
                acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i));
                acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 1));
                acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 2));
                acc = _mm256_xor_si256(acc, _mm256_load_si256(p + i + 3));
            }
        }
#pragma omp critical
        acc_all = _mm256_xor_si256(acc_all, acc);
    }
    double t = now() - t0;
    volatile int sink = _mm256_extract_epi32(acc_all, 0); (void) sink;
    return (double) n * reps / t / 1e9;
}

__global__ void read_mapped(const int4 * p, size_t n, int * out) {
    int4 acc = make_int4(0, 0, 0, 0);
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        int4 v = p[i]; acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678) out[0] = 1;
}

int main(int argc, char ** argv) {
    const int nth = argc > 1 ? atoi(argv[1]) : 24;
    const size_t cpu_n = (size_t) (argc > 2 ? atof(argv[2]) : 16) * (1ull << 30);
    const size_t pin_n = (size_t) (argc > 3 ? atof(argv[3]) : 2) * (1ull << 30);

    char * buf = (char *) aligned_alloc(4096, cpu_n);
#pragma omp parallel for num_threads(nth)
    for (size_t i = 0; i < cpu_n; i += 4096) memset(buf + i, (int) i, 4096);

    char * pin; char * dev; int * dout;
    cudaHostAlloc((void **) &pin, pin_n, cudaHostAllocMapped);
    memset(pin, 1, pin_n);
    cudaMalloc((void **) &dev, pin_n);
    cudaMalloc((void **) &dout, 4);
    char * pin_dev; cudaHostGetDevicePointer((void **) &pin_dev, pin, 0);

    cpu_read(buf, cpu_n, nth, 1); // warm up
    printf("CPU alone, %d threads: %.1f GB/s\n", nth, cpu_read(buf, cpu_n, nth, 3));

    for (int mode = 0; mode < 2; mode++) {
        std::atomic<bool> stop{false};
        std::atomic<double> gpu_bytes{0}, gpu_time{0};
        std::thread g([&] {
            cudaStream_t s; cudaStreamCreate(&s);
            double t0 = now(); double bytes = 0;
            while (!stop) {
                if (mode == 0) cudaMemcpyAsync(dev, pin, pin_n, cudaMemcpyHostToDevice, s);
                else read_mapped<<<164, 512, 0, s>>>((const int4 *) pin_dev, pin_n / 16, dout);
                cudaStreamSynchronize(s);
                bytes += pin_n;
            }
            gpu_bytes = bytes; gpu_time = now() - t0;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        double c = cpu_read(buf, cpu_n, nth, 3);
        stop = true; g.join();
        double gbw = gpu_bytes / gpu_time / 1e9;
        printf("%s: CPU %.1f GB/s + GPU %.1f GB/s = %.1f GB/s\n", mode == 0 ? "with H2D copies   " : "with mapped reads ", c, gbw, c + gbw);
    }
    {
        cudaStream_t s; cudaStreamCreate(&s);
        double t0 = now();
        for (int i = 0; i < 5; i++) cudaMemcpyAsync(dev, pin, pin_n, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);
        printf("GPU H2D alone: %.1f GB/s\n", 5.0 * pin_n / (now() - t0) / 1e9);
    }
    return 0;
}
