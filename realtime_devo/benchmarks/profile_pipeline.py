"""
Pipeline throughput benchmark.

Measures end-to-end latency of the event→voxel pipeline across different
window sizes.  Run with:

    python benchmarks/profile_pipeline.py [--n-events 1000000] [--device cpu]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.event_buffer import EventRingBuffer
from src.voxelizer import Voxelizer


def _synthetic_events(n: int, H: int, W: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, W, size=n, dtype=np.uint16)
    y = rng.integers(0, H, size=n, dtype=np.uint16)
    t = np.sort(rng.uniform(0, n * 1e3, size=n))   # ~1 µs per event
    p = rng.integers(0, 2, size=n, dtype=np.uint8)
    return x, y, t, p


def benchmark_voxelizer(
    n_events: int = 1_000_000,
    H: int = 480,
    W: int = 640,
    n_bins: int = 5,
    window_sizes: list[int] | None = None,
    device: str = "cpu",
    repeats: int = 3,
) -> None:
    if window_sizes is None:
        window_sizes = [10_000, 30_000, 100_000]

    print(f"\n{'='*60}")
    print(f"Voxelizer benchmark  |  H={H} W={W} bins={n_bins}  device={device}")
    print(f"{'='*60}")

    vox = Voxelizer(H=H, W=W, n_bins=n_bins, device=device)
    x_all, y_all, t_all, p_all = _synthetic_events(n_events, H, W)

    for ws in window_sizes:
        times_ms = []
        n_windows = n_events // ws
        for _ in range(repeats):
            t0 = time.perf_counter()
            for i in range(n_windows):
                sl = slice(i * ws, (i + 1) * ws)
                vox.build(x_all[sl], y_all[sl], t_all[sl], p_all[sl])
            if device != "cpu":
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(dt_ms)

        avg_ms = np.mean(times_ms)
        per_window_ms = avg_ms / n_windows
        fps = 1000 / per_window_ms
        print(
            f"  window={ws:>8,} events | "
            f"{per_window_ms:.2f} ms/voxel | {fps:,.1f} Hz | "
            f"({n_windows} windows)"
        )


def benchmark_ring_buffer(
    n_events: int = 1_000_000,
    batch_sizes: list[int] | None = None,
    capacity: int = 2_000_000,
    repeats: int = 3,
) -> None:
    if batch_sizes is None:
        batch_sizes = [1_000, 10_000, 100_000]

    print(f"\n{'='*60}")
    print(f"EventRingBuffer benchmark  |  capacity={capacity:,}")
    print(f"{'='*60}")

    rng = np.random.default_rng(1)
    x = rng.integers(0, 640, size=n_events, dtype=np.uint16)
    y = rng.integers(0, 480, size=n_events, dtype=np.uint16)
    t = np.arange(n_events, dtype=np.float64)
    p = (np.arange(n_events) % 2).astype(np.uint8)

    for bs in batch_sizes:
        buf = EventRingBuffer(capacity)
        n_batches = n_events // bs
        times_ms = []
        for _ in range(repeats):
            buf.clear()
            t0 = time.perf_counter()
            for i in range(n_batches):
                sl = slice(i * bs, (i + 1) * bs)
                buf.push(x[sl], y[sl], t[sl], p[sl])
                buf.pop(bs // 2)
            dt_ms = (time.perf_counter() - t0) * 1000
            times_ms.append(dt_ms)

        avg_ms = np.mean(times_ms)
        mevs = n_events / 1e6
        throughput = mevs / (avg_ms / 1000)
        print(
            f"  batch={bs:>8,} events | "
            f"{avg_ms:.1f} ms total | {throughput:.1f} Mev/s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-events", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--H", type=int, default=480)
    parser.add_argument("--W", type=int, default=640)
    args = parser.parse_args()

    benchmark_voxelizer(n_events=args.n_events, H=args.H, W=args.W, device=args.device)
    benchmark_ring_buffer(n_events=args.n_events)
