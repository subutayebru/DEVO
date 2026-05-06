/*
 * voxelize.cpp — CPU event-to-voxel accumulation with OpenMP parallelism.
 *
 * Implements the same bilinear-temporal interpolation as the Python scatter_add
 * version, but uses one OS thread per event (via OpenMP) with GCC atomic
 * built-ins for conflict-free accumulation into shared grid memory.
 *
 * Build (via torch.utils.cpp_extension.load):
 *   extra_compile_args={"cxx": ["-O3", "-fopenmp", "-march=native"]}
 *   extra_link_args=["-fopenmp"]
 *
 * Exported function:
 *   voxelize(x, y, t_norm, p, H, W, B) -> Tensor (B, H, W) float32
 *     x      : (N,) int32  — pixel column
 *     y      : (N,) int32  — pixel row
 *     t_norm : (N,) float32 — timestamps already normalised to [0, B-1]
 *     p      : (N,) float32 — polarity in {-1, +1}
 *     H, W   : sensor dimensions
 *     B      : number of temporal bins
 *
 * Returns a z-score normalised (B, H, W) float32 tensor.
 *
 * Parallelism strategy
 * --------------------
 * Each event writes to at most 2 grid cells (left and right temporal bin).
 * Concurrent writes from different threads to the same cell are serialised
 * with __atomic_fetch_add on a raw float[] array using GCC's __ATOMIC_RELAXED
 * memory order — sufficient because the final reduction happens after the
 * parallel region (implicit barrier at end of omp parallel for).
 *
 * Note: hardware-level float atomic-add is available on x86 via lock xadd
 * (int) + bit-cast, exposed here through the __sync/__atomic GCC built-ins.
 * On platforms without hardware float atomics the compiler emits a CAS loop.
 */

#include <torch/extension.h>
#include <cmath>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Portable atomic float add
//   Uses a compare-and-swap loop so it works on all targets regardless of
//   whether the hardware has a native floating-point atomic instruction.
// ---------------------------------------------------------------------------
static inline void atomic_add_float(float* addr, float val) {
    // Reinterpret float* as int* for CAS
    static_assert(sizeof(float) == sizeof(int), "float/int size mismatch");
    int* iaddr = reinterpret_cast<int*>(addr);
    int old_bits = __atomic_load_n(iaddr, __ATOMIC_RELAXED);
    int new_bits;
    do {
        float old_val;
        memcpy(&old_val, &old_bits, sizeof(float));
        float new_val = old_val + val;
        memcpy(&new_bits, &new_val, sizeof(float));
    } while (!__atomic_compare_exchange_n(
                 iaddr, &old_bits, new_bits,
                 /*weak=*/true,
                 __ATOMIC_RELAXED, __ATOMIC_RELAXED));
}


// ---------------------------------------------------------------------------
// Per-bin z-score normalisation (serial, trivially fast)
// ---------------------------------------------------------------------------
static void normalise_inplace(float* grid, int B, int H, int W) {
    const int HW = H * W;
    for (int b = 0; b < B; ++b) {
        float* bin = grid + b * HW;
        // Mean
        double sum = 0.0;
        for (int i = 0; i < HW; ++i) sum += bin[i];
        float mean = static_cast<float>(sum / HW);
        // Variance
        double sq = 0.0;
        for (int i = 0; i < HW; ++i) {
            float d = bin[i] - mean;
            sq += d * d;
        }
        float std = static_cast<float>(std::sqrt(sq / HW + 1e-12));
        float inv_std = 1.0f / (std + 1e-6f);
        for (int i = 0; i < HW; ++i)
            bin[i] = (bin[i] - mean) * inv_std;
    }
}


// ---------------------------------------------------------------------------
// Main extension entry point
// ---------------------------------------------------------------------------
torch::Tensor voxelize(
    torch::Tensor x,       // (N,) int32
    torch::Tensor y,       // (N,) int32
    torch::Tensor t_norm,  // (N,) float32, in [0, B-1]
    torch::Tensor p,       // (N,) float32, in {-1, +1}
    int64_t H,
    int64_t W,
    int64_t B
) {
    TORCH_CHECK(x.is_cpu(),      "x must be a CPU tensor");
    TORCH_CHECK(t_norm.is_cpu(), "t_norm must be a CPU tensor");
    TORCH_CHECK(x.dtype()      == torch::kInt32,   "x must be int32");
    TORCH_CHECK(y.dtype()      == torch::kInt32,   "y must be int32");
    TORCH_CHECK(t_norm.dtype() == torch::kFloat32, "t_norm must be float32");
    TORCH_CHECK(p.dtype()      == torch::kFloat32, "p must be float32");

    const int64_t N  = x.size(0);
    const int64_t HW = H * W;

    // Output grid (zero-initialised)
    auto grid_t = torch::zeros({B * H * W}, torch::kFloat32);
    float* grid = grid_t.data_ptr<float>();

    const int32_t* xp = x.data_ptr<int32_t>();
    const int32_t* yp = y.data_ptr<int32_t>();
    const float*   tp = t_norm.data_ptr<float>();
    const float*   pp = p.data_ptr<float>();

    // Parallel accumulation — one thread per event
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < N; ++i) {
        float ti  = tp[i];
        int   b_l = static_cast<int>(ti);   // floor
        float w_r = ti - static_cast<float>(b_l);
        float w_l = 1.0f - w_r;
        float pi  = pp[i];
        int64_t xy = static_cast<int64_t>(yp[i]) * W + static_cast<int64_t>(xp[i]);

        if (b_l >= 0 && b_l < static_cast<int>(B)) {
            atomic_add_float(grid + b_l * HW + xy, pi * w_l);
        }
        int b_r = b_l + 1;
        if (b_r >= 0 && b_r < static_cast<int>(B)) {
            atomic_add_float(grid + b_r * HW + xy, pi * w_r);
        }
    }

    normalise_inplace(grid, static_cast<int>(B),
                             static_cast<int>(H),
                             static_cast<int>(W));

    return grid_t.view({B, H, W});
}


// ---------------------------------------------------------------------------
// Thread count helper (useful for diagnostics)
// ---------------------------------------------------------------------------
int64_t omp_thread_count() {
#ifdef _OPENMP
    return static_cast<int64_t>(omp_get_max_threads());
#else
    return 1;
#endif
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize",         &voxelize,         "Voxelize events (OpenMP atomic)");
    m.def("omp_thread_count", &omp_thread_count, "Max OpenMP threads available");
}
