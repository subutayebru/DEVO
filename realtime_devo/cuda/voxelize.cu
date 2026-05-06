/*
 * voxelize.cu — CUDA event-to-voxel accumulation kernel.
 *
 * One CUDA thread per event; uses atomicAdd for conflict-free writes into
 * shared global memory.  A separate normalisation kernel runs afterwards.
 *
 * Build (via torch.utils.cpp_extension.load or setup.py):
 *   sources=["cuda/voxelize.cu"]
 *   extra_cuda_cflags=["-O3", "--use_fast_math"]
 *
 * Exported function:
 *   voxelize_cuda(x, y, t_norm, p, H, W, B) -> Tensor (B, H, W) float32
 *     x      : (N,) int32  on CUDA
 *     y      : (N,) int32  on CUDA
 *     t_norm : (N,) float32 on CUDA, values in [0, B-1]
 *     p      : (N,) float32 on CUDA, polarity in {-1, +1}
 *     H, W   : sensor dimensions
 *     B      : number of temporal bins
 *
 * Returns a z-score normalised (B, H, W) float32 tensor on CUDA.
 *
 * Parallelism
 * -----------
 * Kernel 1 — accumulate_kernel
 *   Grid: ceil(N / 256) blocks × 256 threads.
 *   Each thread handles one event, contributing polarity-weighted values
 *   to the left and/or right temporal bin via atomicAdd.
 *   atomicAdd on float32 is natively hardware-supported since sm_20.
 *
 * Kernel 2 — normalise_kernel
 *   Grid: B blocks × 256 threads.
 *   Each block normalises one temporal bin: parallel reduction for mean
 *   and variance, then element-wise z-score.
 *   Uses shared memory for the warp-level reductions.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(),  #x " must be a CUDA tensor")
#define CHECK_CONT(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONT(x)

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — accumulate (one thread per event)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void accumulate_kernel(
    const int32_t* __restrict__ x,
    const int32_t* __restrict__ y,
    const float*   __restrict__ t_norm,
    const float*   __restrict__ p,
    float*         __restrict__ grid,
    int64_t N, int64_t H, int64_t W, int64_t B
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float ti  = t_norm[i];
    int   b_l = static_cast<int>(ti);   // floor
    float w_r = ti - static_cast<float>(b_l);
    float w_l = 1.0f - w_r;
    float pi  = p[i];
    int64_t xy = static_cast<int64_t>(y[i]) * W + static_cast<int64_t>(x[i]);
    int64_t HW = H * W;

    if (b_l >= 0 && b_l < B)
        atomicAdd(grid + b_l * HW + xy, pi * w_l);

    int b_r = b_l + 1;
    if (b_r >= 0 && b_r < B)
        atomicAdd(grid + b_r * HW + xy, pi * w_r);
}


// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — per-bin z-score normalisation
//   Each block handles one bin.  HW elements per bin may exceed blockDim.x,
//   so threads stride over the bin in a grid-stride loop.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void normalise_kernel(
    float* __restrict__ grid,
    int64_t B, int64_t H, int64_t W
) {
    extern __shared__ float smem[];     // 2 × blockDim.x floats (sum + sq_sum)
    float* ssum = smem;
    float* ssq  = smem + blockDim.x;

    int b = blockIdx.x;
    if (b >= B) return;

    int64_t HW      = H * W;
    float*  bin     = grid + b * HW;
    int     tid     = threadIdx.x;
    int     nthreads = blockDim.x;

    // Thread-local partial sums
    float local_sum = 0.0f, local_sq = 0.0f;
    for (int64_t i = tid; i < HW; i += nthreads) {
        float v = bin[i];
        local_sum += v;
        local_sq  += v * v;
    }
    ssum[tid] = local_sum;
    ssq [tid] = local_sq;
    __syncthreads();

    // Tree reduction within block
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssq [tid] += ssq [tid + s];
        }
        __syncthreads();
    }

    // Compute mean and inv_std in thread 0, broadcast via __syncthreads
    __shared__ float mean_val, inv_std_val;
    if (tid == 0) {
        float mean = ssum[0] / static_cast<float>(HW);
        float var  = ssq[0]  / static_cast<float>(HW) - mean * mean;
        mean_val   = mean;
        inv_std_val = 1.0f / (sqrtf(var > 0.0f ? var : 0.0f) + 1e-6f);
    }
    __syncthreads();

    // Apply normalisation
    float m   = mean_val;
    float inv = inv_std_val;
    for (int64_t i = tid; i < HW; i += nthreads)
        bin[i] = (bin[i] - m) * inv;
}


// ─────────────────────────────────────────────────────────────────────────────
// Host entry point
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor voxelize_cuda(
    torch::Tensor x,       // (N,) int32 on CUDA
    torch::Tensor y,       // (N,) int32 on CUDA
    torch::Tensor t_norm,  // (N,) float32 on CUDA, in [0, B-1]
    torch::Tensor p,       // (N,) float32 on CUDA, in {-1, +1}
    int64_t H,
    int64_t W,
    int64_t B
) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(t_norm);
    CHECK_INPUT(p);

    const int64_t N = x.size(0);

    auto grid = torch::zeros({B * H * W}, x.options().dtype(torch::kFloat32));

    // Kernel 1 — accumulate
    const int threads = 256;
    const int blocks  = static_cast<int>((N + threads - 1) / threads);
    accumulate_kernel<<<blocks, threads>>>(
        x.data_ptr<int32_t>(),
        y.data_ptr<int32_t>(),
        t_norm.data_ptr<float>(),
        p.data_ptr<float>(),
        grid.data_ptr<float>(),
        N, H, W, B
    );

    // Kernel 2 — normalise (one block per bin, shared memory for reduction)
    const int norm_threads = 256;
    normalise_kernel<<<static_cast<int>(B), norm_threads,
                       2 * norm_threads * sizeof(float)>>>(
        grid.data_ptr<float>(),
        B, H, W
    );

    return grid.view({B, H, W});
}


// ─────────────────────────────────────────────────────────────────────────────
// Bindings
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_cuda", &voxelize_cuda,
          "CUDA voxelizer: one-thread-per-event with atomicAdd  "
          "(x, y, t_norm, p, H, W, B) -> Tensor(B,H,W)");
}
