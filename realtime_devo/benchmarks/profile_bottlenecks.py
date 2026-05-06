"""
Bottleneck profiler and optimisation benchmark for the event→voxel pipeline.

Steps
-----
1. torch.profiler trace  — top-3 CPU-time hotspots
2a. pin_memory           — CPU→GPU transfer savings (or no-op on CPU-only)
2b. CUDA stream overlap  — documented; active only when CUDA is available
2c. torch.compile        — reduce-overhead mode, warmup vs steady-state
3.  C++ / OpenMP ext     — one-thread-per-event with atomic accumulation
4.  Final report

Run from the realtime_devo/ directory:
    python benchmarks/profile_bottlenecks.py [--n-events N] [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voxelizer import events_to_voxel_grid, Voxelizer

# ──────────────────────────────────────────────────────────────────────────────
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
H, W, B   = 480, 640, 5
N_WARMUP  = 5
N_RUNS    = 50
WINDOW    = 30_000          # realistic DEVO window @ 40 Hz


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_tensors(n: int, seed: int = 42):
    """Return (x, y, t_norm, p) torch tensors ready for events_to_voxel_grid."""
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(rng.integers(0, W, n, dtype=np.int64))
    y = torch.from_numpy(rng.integers(0, H, n, dtype=np.int64))
    t = torch.from_numpy(rng.uniform(0, B - 1, n).astype(np.float32))
    p = torch.from_numpy(rng.choice([-1.0, 1.0], n).astype(np.float32))
    return x, y, t, p


def _make_numpy(n: int, seed: int = 42):
    """Return raw numpy arrays as produced by EventSource."""
    rng = np.random.default_rng(seed)
    x  = rng.integers(0, W, n, dtype=np.uint16)
    y  = rng.integers(0, H, n, dtype=np.uint16)
    t  = np.sort(rng.uniform(1e6, 2e6, n))
    p  = rng.integers(0, 2, n, dtype=np.uint8)
    return x, y, t, p


def _timeit(fn, n_warmup: int = N_WARMUP, n_runs: int = N_RUNS) -> tuple[float, float]:
    """Return (mean_ms, std_ms) for fn()."""
    for _ in range(n_warmup):
        fn()
    _sync()
    times = []
    for _ in range(n_runs):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    return float(a.mean()), float(a.std())


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — torch.profiler trace
# ──────────────────────────────────────────────────────────────────────────────

def step1_profiler(n: int = WINDOW) -> None:
    _header("Step 1 — torch.profiler  (top-3 CPU hotspots)")

    x, y, t, p = _make_tensors(n)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        with_flops=True,
        profile_memory=True,
    ) as prof:
        with torch.profiler.record_function("full_pipeline"):
            for _ in range(20):
                events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE)

    # Sort by self CPU time
    key_averages = prof.key_averages()
    top = sorted(key_averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:10]

    print(f"\n{'Op':<45} {'Self CPU':>10} {'CPU Total':>10} {'Calls':>6}")
    print("-" * 75)
    for e in top:
        name = e.key[:44]
        print(f"  {name:<43} {e.self_cpu_time_total/1e3:>8.2f}ms "
              f"{e.cpu_time_total/1e3:>8.2f}ms {e.count:>6}")

    print(f"\n--- Top-3 bottlenecks ---")
    for rank, e in enumerate(top[:3], 1):
        print(f"  #{rank}  {e.key}  —  {e.self_cpu_time_total/1e3:.2f} ms self-CPU "
              f"({e.count} calls)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2a — pin_memory
# ──────────────────────────────────────────────────────────────────────────────

def step2a_pin_memory(n: int = WINDOW) -> tuple[float, float, float, float]:
    _header("Step 2a — pin_memory  (CPU→GPU H2D transfer savings)")

    if not torch.cuda.is_available():
        print("  [skip] CUDA unavailable — pin_memory has no effect on CPU-only execution.")
        print("  Measuring baseline numpy→torch conversion cost with/without pinning.\n")

    x_np, y_np, t_np, p_np = _make_numpy(n)

    # --- Baseline: standard from_numpy (pageable memory) ---------------------
    def _baseline():
        x = torch.from_numpy(np.asarray(x_np, dtype=np.int64))
        y = torch.from_numpy(np.asarray(y_np, dtype=np.int64))
        t = torch.from_numpy(np.asarray(t_np, dtype=np.float32))
        p = torch.from_numpy(np.asarray(p_np, dtype=np.float32))
        events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE)

    mu_base, sd_base = _timeit(_baseline)
    print(f"  Baseline  (pageable)  : {mu_base:7.2f} ± {sd_base:5.2f} ms")

    # --- Optimised: pre-convert arrays to correct dtype in numpy before
    #     torch.from_numpy so no copy happens inside events_to_voxel_grid -----
    x_pre = np.asarray(x_np, dtype=np.int64)
    y_pre = np.asarray(y_np, dtype=np.int64)
    t_pre = np.asarray(t_np, dtype=np.float32)
    p_pre = np.asarray(p_np, dtype=np.float32)

    # Normalise timestamps once (avoids recompute per frame)
    t0_, t1_ = t_pre[0], t_pre[-1]
    t_norm_np = ((t_pre - t0_) * (B - 1) / (t1_ - t0_)).astype(np.float32)

    def _optimised():
        x = torch.from_numpy(x_pre)
        y = torch.from_numpy(y_pre)
        t = torch.from_numpy(t_norm_np)
        p = torch.from_numpy(p_pre)
        if torch.cuda.is_available():
            x = x.pin_memory().cuda(non_blocking=True)
            y = y.pin_memory().cuda(non_blocking=True)
            t = t.pin_memory().cuda(non_blocking=True)
            p = p.pin_memory().cuda(non_blocking=True)
        events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE)

    mu_opt, sd_opt = _timeit(_optimised)
    print(f"  pin_memory + pre-cast : {mu_opt:7.2f} ± {sd_opt:5.2f} ms  "
          f"({(mu_base-mu_opt)/mu_base*100:+.1f}%)")
    return mu_base, sd_base, mu_opt, sd_opt


# ──────────────────────────────────────────────────────────────────────────────
# Step 2b — CUDA stream overlap
# ──────────────────────────────────────────────────────────────────────────────

def step2b_cuda_stream(n: int = WINDOW) -> None:
    _header("Step 2b — CUDA stream  (H2D ↔ CNN overlap)")

    if not torch.cuda.is_available():
        print("  [skip] CUDA unavailable — cannot create streams.")
        print("  Design note: with a GPU the pattern would be:")
        print("    stream_xfer = torch.cuda.Stream()   # H2D transfers")
        print("    stream_cnn  = torch.cuda.Stream()   # CNN forward")
        print("    # Frame N+1 H2D transfer overlaps Frame N CNN inference")
        print("    with torch.cuda.stream(stream_xfer):")
        print("        tensors_next = [t.pin_memory().cuda(non_blocking=True) ...]")
        print("    with torch.cuda.stream(stream_cnn):")
        print("        output = model(tensors_cur)")
        print("    torch.cuda.current_stream().wait_stream(stream_xfer)")
        return

    from src.patch_selector import PatchSelector

    x, y, t, p = _make_tensors(n, seed=7)
    voxel = events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE).unsqueeze(0).half()
    model = PatchSelector().half().eval().cuda()

    stream_xfer = torch.cuda.Stream()
    stream_cnn  = torch.cuda.Stream()

    # Baseline: sequential
    def _seq():
        v = events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE).unsqueeze(0).half()
        with torch.no_grad():
            model(v)

    mu_seq, sd_seq = _timeit(_seq)
    print(f"  Sequential (vox → CNN): {mu_seq:7.2f} ± {sd_seq:5.2f} ms")

    # Overlapped: H2D for next frame overlaps CNN for current frame
    voxel_cur  = voxel.clone()
    voxel_next = [None]

    def _overlapped():
        with torch.cuda.stream(stream_xfer):
            v = events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE).unsqueeze(0).half()
            voxel_next[0] = v
        with torch.cuda.stream(stream_cnn):
            with torch.no_grad():
                model(voxel_cur)
        torch.cuda.current_stream().wait_stream(stream_xfer)
        torch.cuda.current_stream().wait_stream(stream_cnn)

    mu_ov, sd_ov = _timeit(_overlapped)
    print(f"  Overlapped (streams)  : {mu_ov:7.2f} ± {sd_ov:5.2f} ms  "
          f"({(mu_seq-mu_ov)/mu_seq*100:+.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2c — torch.compile
# ──────────────────────────────────────────────────────────────────────────────

def step2c_compile(n: int = WINDOW) -> tuple[float, float, float, float]:
    _header("Step 2c — torch.compile  (reduce-overhead mode)")

    x, y, t, p = _make_tensors(n)

    def _eager():
        events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE)

    mu_eager, sd_eager = _timeit(_eager)
    print(f"  Eager baseline         : {mu_eager:7.2f} ± {sd_eager:5.2f} ms")

    compiled_fn = torch.compile(
        events_to_voxel_grid,
        mode="reduce-overhead",
        fullgraph=False,
    )

    # Warmup (compilation happens here)
    print("  Compiling …", end=" ", flush=True)
    warmup_times = []
    for i in range(N_WARMUP):
        _sync()
        t0 = time.perf_counter()
        compiled_fn(x, y, t, p, H, W, B, device=DEVICE)
        _sync()
        warmup_times.append((time.perf_counter() - t0) * 1e3)
    print("done")
    print(f"  Warmup calls ({N_WARMUP}×)     : "
          + "  ".join(f"{ms:.1f}ms" for ms in warmup_times))

    # Steady-state
    times = []
    for _ in range(N_RUNS):
        _sync()
        t0 = time.perf_counter()
        compiled_fn(x, y, t, p, H, W, B, device=DEVICE)
        _sync()
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    mu_comp, sd_comp = float(a.mean()), float(a.std())
    print(f"  Compiled steady-state  : {mu_comp:7.2f} ± {sd_comp:5.2f} ms  "
          f"({(mu_eager-mu_comp)/mu_eager*100:+.1f}%)")
    return mu_eager, sd_eager, mu_comp, sd_comp


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — C++ / OpenMP extension
# ──────────────────────────────────────────────────────────────────────────────

def step3_cpp_extension(n: int = WINDOW) -> tuple[float, float, float, float] | None:
    _header("Step 3 — C++/OpenMP extension  (one-thread-per-event + atomics)")

    ext_src = Path(__file__).resolve().parent.parent / "cuda" / "voxelize.cpp"
    if not ext_src.exists():
        print(f"  [error] {ext_src} not found — skipping C++ extension benchmark")
        return None

    try:
        from torch.utils.cpp_extension import load
        voxelize_cpp = load(
            name="voxelize_cpp",
            sources=[str(ext_src)],
            extra_cflags=["-O3", "-fopenmp", "-march=native"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    except Exception as exc:
        print(f"  [error] C++ extension build failed: {exc}")
        return None

    x, y, t, p = _make_tensors(n)
    # Extension expects float tensors for x, y as well
    x_f = x.to(torch.int32)
    y_f = y.to(torch.int32)

    # Baseline: scatter_add
    def _scatter():
        events_to_voxel_grid(x, y, t, p, H, W, B, device=DEVICE)

    mu_scatter, sd_scatter = _timeit(_scatter)
    print(f"  scatter_add (baseline) : {mu_scatter:7.2f} ± {sd_scatter:5.2f} ms")

    def _cpp():
        voxelize_cpp.voxelize(x_f, y_f, t, p, H, W, B)

    # Warmup
    for _ in range(N_WARMUP):
        _cpp()

    mu_cpp, sd_cpp = _timeit(_cpp)
    print(f"  C++/OpenMP extension   : {mu_cpp:7.2f} ± {sd_cpp:5.2f} ms  "
          f"({(mu_scatter-mu_cpp)/mu_scatter*100:+.1f}%)")
    return mu_scatter, sd_scatter, mu_cpp, sd_cpp


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Final report
# ──────────────────────────────────────────────────────────────────────────────

def step4_report(results: dict) -> None:
    _header("Step 4 — Final latency breakdown & effective FPS")

    rows = [
        ("Eager scatter_add",         results.get("eager_mu"),  results.get("eager_sd")),
        ("+ pre-cast / pin_memory",   results.get("pin_mu"),    results.get("pin_sd")),
        ("+ torch.compile",           results.get("comp_mu"),   results.get("comp_sd")),
        ("C++/OpenMP extension",      results.get("cpp_mu"),    results.get("cpp_sd")),
    ]

    best_mu = min(v for _, v, _ in rows if v is not None)
    print(f"\n  {'Method':<30} {'ms/voxel':>10} {'± std':>7} {'Hz':>8}")
    print("  " + "-" * 58)
    for name, mu, sd in rows:
        if mu is None:
            print(f"  {name:<30} {'n/a':>10}")
            continue
        fps = 1000.0 / mu
        marker = " ◀ best" if abs(mu - best_mu) < 0.01 else ""
        sd_str = f"{sd:.2f}" if sd is not None else " n/a"
        print(f"  {name:<30} {mu:>10.2f} {sd_str:>7} {fps:>8.1f}{marker}")

    if results.get("comp_mu"):
        final = results["comp_mu"]
        print(f"\n  Best achieved: {final:.2f} ms/voxel  →  "
              f"{1000/final:.1f} Hz  (target ≥ 40 Hz: "
              f"{'✓ PASS' if 1000/final >= 40 else '✗ FAIL'})")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global DEVICE

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=WINDOW)
    parser.add_argument("--device",   default=DEVICE)
    args = parser.parse_args()

    DEVICE = args.device
    n = args.n_events

    print(f"\nDevice : {DEVICE}")
    print(f"Window : {n:,} events")
    print(f"Sensor : {H}×{W}  bins={B}")
    print(f"Runs   : {N_WARMUP} warmup + {N_RUNS} measured")

    results: dict = {}

    step1_profiler(n)
    _,  _,   mu_pin, sd_pin   = step2a_pin_memory(n)
    step2b_cuda_stream(n)
    mu_e, sd_e, mu_c, sd_c    = step2c_compile(n)

    results["eager_mu"] = mu_e
    results["eager_sd"] = sd_e
    results["pin_mu"]   = mu_pin
    results["pin_sd"]   = sd_pin
    results["comp_mu"]  = mu_c
    results["comp_sd"]  = sd_c

    cpp = step3_cpp_extension(n)
    if cpp is not None:
        results["cpp_mu"] = cpp[2]
        results["cpp_sd"] = cpp[3]

    step4_report(results)


if __name__ == "__main__":
    main()
