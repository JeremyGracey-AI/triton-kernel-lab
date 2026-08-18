// Raw CUDA C++ baseline for the fused mul-add-relu kernel (fp32).
//
// Two variants of out = relu(a * b + c):
//   scalar  — grid-stride loop, one float per thread per step
//   float4  — 128-bit vectorized loads/stores, scalar tail
//
// This exists to measure what Triton's codegen is worth against the same
// operation written by hand: the float4 variant is the hand-rolled analogue
// of the vectorization Triton derives automatically from its block layout.
//
// Self-checking (host reference before any timing) and self-timing
// (cudaEvent, median of TIMED_ITERS). Emits one CSV row per variant:
//   impl,shape,numel,ms_median,gbps
// Usage: ./cuda_elementwise [numel]...     (default sweep matches bench/run.py)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                         cudaGetErrorString(err_), __FILE__, __LINE__);      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void fused_mul_add_relu_scalar(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          const float* __restrict__ c,
                                          float* __restrict__ out, long n) {
    long stride = (long)blockDim.x * gridDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = fmaxf(fmaf(a[i], b[i], c[i]), 0.0f);
    }
}

__global__ void fused_mul_add_relu_vec4(const float4* __restrict__ a,
                                        const float4* __restrict__ b,
                                        const float4* __restrict__ c,
                                        float4* __restrict__ out, long n4) {
    long stride = (long)blockDim.x * gridDim.x;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 av = a[i], bv = b[i], cv = c[i];
        float4 o;
        o.x = fmaxf(fmaf(av.x, bv.x, cv.x), 0.0f);
        o.y = fmaxf(fmaf(av.y, bv.y, cv.y), 0.0f);
        o.z = fmaxf(fmaf(av.z, bv.z, cv.z), 0.0f);
        o.w = fmaxf(fmaf(av.w, bv.w, cv.w), 0.0f);
        out[i] = o;
    }
}

__global__ void fused_mul_add_relu_tail(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        const float* __restrict__ c,
                                        float* __restrict__ out, long start, long n) {
    long i = start + (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaxf(fmaf(a[i], b[i], c[i]), 0.0f);
}

static constexpr int kBlock = 256;
static constexpr int kWarmup = 25;
static constexpr int kTimedIters = 100;

static int grid_for(long work) {
    long g = (work + kBlock - 1) / kBlock;
    return (int)std::min<long>(g, 65535L * 32);  // grid-stride handles the rest
}

struct Variant {
    const char* name;
    void (*launch)(const float*, const float*, const float*, float*, long);
};

static void launch_scalar(const float* a, const float* b, const float* c, float* o, long n) {
    fused_mul_add_relu_scalar<<<grid_for(n), kBlock>>>(a, b, c, o, n);
}

static void launch_vec4(const float* a, const float* b, const float* c, float* o, long n) {
    long n4 = n / 4;
    if (n4 > 0) {
        fused_mul_add_relu_vec4<<<grid_for(n4), kBlock>>>(
            (const float4*)a, (const float4*)b, (const float4*)c, (float4*)o, n4);
    }
    long tail = n - n4 * 4;
    if (tail > 0) {
        fused_mul_add_relu_tail<<<1, kBlock>>>(a, b, c, o, n4 * 4, n);
    }
}

int main(int argc, char** argv) {
    std::vector<long> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atol(argv[i]));
    if (sizes.empty()) sizes = {1L << 16, 1L << 18, 1L << 20, 1L << 22, 1L << 24, 1L << 26};

    std::printf("impl,shape,numel,ms_median,gbps\n");

    for (long n : sizes) {
        std::vector<float> ha(n), hb(n), hc(n), hout(n);
        std::srand(7);
        for (long i = 0; i < n; ++i) {
            ha[i] = (float)std::rand() / RAND_MAX - 0.5f;
            hb[i] = (float)std::rand() / RAND_MAX - 0.5f;
            hc[i] = (float)std::rand() / RAND_MAX - 0.5f;
        }

        float *da, *db, *dc, *dout;
        size_t bytes = (size_t)n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&dc, bytes));
        CUDA_CHECK(cudaMalloc(&dout, bytes));
        CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dc, hc.data(), bytes, cudaMemcpyHostToDevice));

        Variant variants[] = {{"cuda_scalar", launch_scalar}, {"cuda_float4", launch_vec4}};
        for (const Variant& v : variants) {
            // correctness gate before any timing
            v.launch(da, db, dc, dout, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));
            for (long i = 0; i < n; ++i) {
                float ref = std::max(ha[i] * hb[i] + hc[i], 0.0f);
                if (std::abs(hout[i] - ref) > 1e-5f) {
                    std::fprintf(stderr, "MISMATCH %s at %ld: %f vs %f\n", v.name, i,
                                 hout[i], ref);
                    return 1;
                }
            }

            for (int i = 0; i < kWarmup; ++i) v.launch(da, db, dc, dout, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            std::vector<float> ms(kTimedIters);
            for (int i = 0; i < kTimedIters; ++i) {
                CUDA_CHECK(cudaEventRecord(t0));
                v.launch(da, db, dc, dout, n);
                CUDA_CHECK(cudaEventRecord(t1));
                CUDA_CHECK(cudaEventSynchronize(t1));
                CUDA_CHECK(cudaEventElapsedTime(&ms[i], t0, t1));
            }
            std::sort(ms.begin(), ms.end());
            float med = ms[kTimedIters / 2];
            double gbps = 4.0 * bytes / (med * 1e-3) / 1e9;  // 3 loads + 1 store
            std::printf("%s,%ld,%ld,%.5f,%.2f\n", v.name, n, n, med, gbps);
            CUDA_CHECK(cudaEventDestroy(t0));
            CUDA_CHECK(cudaEventDestroy(t1));
        }

        CUDA_CHECK(cudaFree(da));
        CUDA_CHECK(cudaFree(db));
        CUDA_CHECK(cudaFree(dc));
        CUDA_CHECK(cudaFree(dout));
    }
    return 0;
}
