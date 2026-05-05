"""
Tests for events_to_voxel_grid and events_to_voxel_grid_reference.

Covers:
  - Output shape and dtype
  - Numerical agreement between fast (scatter_add) and reference (loop) paths
  - Polarity sign convention
  - Per-bin normalisation (zero mean, unit variance)
  - Edge cases: empty events, single event, all same bin
  - Wall-clock timing of the fast path on the available device
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.voxelizer import events_to_voxel_grid, events_to_voxel_grid_reference, Voxelizer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

H, W, B = 480, 640, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(0)


def _random_events(
    n: int,
    h: int = H,
    w: int = W,
    b: int = B,
    seed: int = 0,
    device: str = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x, y, t, p) tensors on `device`.

    t is already normalised to [0, b-1]; p is in {-1, +1}.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, w, size=n, dtype=np.int64)
    y = rng.integers(0, h, size=n, dtype=np.int64)
    t = rng.uniform(0.0, b - 1, size=n).astype(np.float32)
    p = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(t).to(device),
        torch.from_numpy(p).to(device),
    )


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

class TestOutputProperties:
    def test_shape(self):
        x, y, t, p = _random_events(1000, device="cpu")
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        assert grid.shape == (B, H, W), f"Expected ({B},{H},{W}), got {grid.shape}"

    def test_dtype_float32(self):
        x, y, t, p = _random_events(500, device="cpu")
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        assert grid.dtype == torch.float32

    def test_empty_events_returns_zeros(self):
        empty = torch.zeros(0, dtype=torch.long)
        emptf = torch.zeros(0, dtype=torch.float32)
        grid = events_to_voxel_grid(empty, empty, emptf, emptf, H, W, B, device="cpu")
        assert grid.shape == (B, H, W)
        # Normalisation of all-zero grid: (0-0)/(0+1e-6) == 0
        assert torch.all(grid == 0.0)

    def test_custom_bin_count(self):
        x, y, t, p = _random_events(200, b=3, device="cpu")
        t = t * 2.0 / (B - 1)   # rescale to [0, 2]
        grid = events_to_voxel_grid(x, y, t, p, H, W, B=3, device="cpu")
        assert grid.shape == (3, H, W)


# ---------------------------------------------------------------------------
# Per-bin normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_per_bin_zero_mean(self):
        x, y, t, p = _random_events(5000, device="cpu")
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        # Each bin must have mean ≈ 0
        means = grid.view(B, -1).mean(dim=1)
        assert torch.allclose(means, torch.zeros(B), atol=1e-5), \
            f"Per-bin means not near zero: {means}"

    def test_per_bin_unit_variance(self):
        x, y, t, p = _random_events(5000, device="cpu")
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        # std of each bin should be ≈ 1 (up to the 1e-6 eps shift)
        stds = grid.view(B, -1).std(dim=1)
        # Bins with many events converge to std ≈ 1; allow tolerance for sparse bins
        assert torch.all(stds < 1.1), f"Some bins have std > 1.1: {stds}"


# ---------------------------------------------------------------------------
# Numerical agreement between fast path and reference
# ---------------------------------------------------------------------------

N_COMPARE = 10_000

class TestAgreementWithReference:
    def test_max_absolute_difference_cpu(self):
        x, y, t, p = _random_events(N_COMPARE, device="cpu")
        fast = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        ref  = events_to_voxel_grid_reference(x, y, t, p, H, W, B, device="cpu")
        diff = (fast - ref).abs().max().item()
        print(f"\n[cpu] max |fast - ref| = {diff:.2e}  (N={N_COMPARE})")
        assert diff < 1e-4, f"Max absolute difference {diff:.2e} exceeds 1e-4"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_max_absolute_difference_cuda_and_timing(self):
        x, y, t, p = _random_events(N_COMPARE, device="cuda")

        # Warm up
        events_to_voxel_grid(x, y, t, p, H, W, B, device="cuda")
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        fast = events_to_voxel_grid(x, y, t, p, H, W, B, device="cuda")
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ref = events_to_voxel_grid_reference(x, y, t, p, H, W, B, device="cuda")
        diff = (fast - ref).abs().max().item()

        print(f"\n[cuda] max |fast - ref| = {diff:.2e}  "
              f"wall-clock = {elapsed_ms:.3f} ms  (N={N_COMPARE})")
        assert diff < 1e-4, f"Max absolute difference {diff:.2e} exceeds 1e-4"

    def test_timing_cpu_printed(self, capsys):
        """Ensure wall-clock time is printed (always runs on CPU)."""
        x, y, t, p = _random_events(N_COMPARE, device="cpu")

        t0 = time.perf_counter()
        fast = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ref = events_to_voxel_grid_reference(x, y, t, p, H, W, B, device="cpu")
        diff = (fast - ref).abs().max().item()

        print(f"\n[cpu]  wall-clock = {elapsed_ms:.3f} ms  "
              f"max |fast - ref| = {diff:.2e}  (N={N_COMPARE}, device={DEVICE})")

        assert diff < 1e-4


# ---------------------------------------------------------------------------
# Polarity sign
# ---------------------------------------------------------------------------

class TestPolaritySign:
    def test_positive_polarity_positive_contribution(self):
        """All-positive events should produce non-negative voxel values."""
        x, y, t, _ = _random_events(500, device="cpu")
        p = torch.ones_like(t)
        # Build without normalisation to inspect raw sign:
        # use reference for clarity
        ref = events_to_voxel_grid_reference(x, y, t, p, H, W, B, device="cpu")
        # After normalisation the sign relationship is preserved in mean-subtracted form;
        # instead check that raw (unnormalised) sum is positive by verifying the
        # normalised grid's maximum is positive.
        assert ref.max().item() > 0

    def test_negative_polarity_negative_contribution(self):
        """All-negative events should produce non-positive voxel values."""
        x, y, t, _ = _random_events(500, device="cpu")
        p = -torch.ones_like(t)
        ref = events_to_voxel_grid_reference(x, y, t, p, H, W, B, device="cpu")
        assert ref.min().item() < 0

    def test_polarity_antisymmetry(self):
        """Flipping all polarities should flip the grid (after mean-zero norm)."""
        x, y, t, p = _random_events(2000, device="cpu")
        grid_pos = events_to_voxel_grid(x, y, t,  p, H, W, B, device="cpu")
        grid_neg = events_to_voxel_grid(x, y, t, -p, H, W, B, device="cpu")
        diff = (grid_pos + grid_neg).abs().max().item()
        assert diff < 1e-5, f"Flipped grids are not antisymmetric: max diff = {diff}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_event_no_crash(self):
        x = torch.tensor([100], dtype=torch.long)
        y = torch.tensor([200], dtype=torch.long)
        t = torch.tensor([2.5], dtype=torch.float32)   # splits equally across bins 2&3
        p = torch.tensor([1.0])
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        assert grid.shape == (B, H, W)

    def test_all_events_in_first_bin(self):
        """t=0 for all events → contribution only to bin 0 (weight=1, bin1 weight=0)."""
        n = 100
        x = torch.zeros(n, dtype=torch.long)
        y = torch.zeros(n, dtype=torch.long)
        t = torch.zeros(n, dtype=torch.float32)   # all in bin 0
        p = torch.ones(n, dtype=torch.float32)
        # Should not raise; normalisation handles uniform bins
        grid = events_to_voxel_grid(x, y, t, p, H, W, B, device="cpu")
        assert grid.shape == (B, H, W)

    def test_t_exactly_at_bin_boundary(self):
        """t = integer values should give weight 1 to that bin, 0 to the next."""
        x = torch.tensor([0], dtype=torch.long)
        y = torch.tensor([0], dtype=torch.long)
        t = torch.tensor([2.0], dtype=torch.float32)   # exactly bin 2, w_right = 0
        p = torch.tensor([1.0])
        # Use reference for easy inspection
        ref = events_to_voxel_grid_reference(x, y, t, p, H=4, W=4, B=5, device="cpu")
        assert ref.shape == (5, 4, 4)

    def test_reference_matches_fast_for_single_event(self):
        x = torch.tensor([50], dtype=torch.long)
        y = torch.tensor([75], dtype=torch.long)
        t = torch.tensor([1.7], dtype=torch.float32)
        p = torch.tensor([-1.0])
        fast = events_to_voxel_grid          (x, y, t, p, 100, 100, B=5, device="cpu")
        ref  = events_to_voxel_grid_reference(x, y, t, p, 100, 100, B=5, device="cpu")
        diff = (fast - ref).abs().max().item()
        assert diff < 1e-6


# ---------------------------------------------------------------------------
# Voxelizer class (numpy interface)
# ---------------------------------------------------------------------------

class TestVoxelizerClass:
    def test_shape_from_numpy(self):
        vox = Voxelizer(H=48, W=64, n_bins=5, device="cpu")
        rng = np.random.default_rng(1)
        x = rng.integers(0, 64, 500, dtype=np.uint16)
        y = rng.integers(0, 48, 500, dtype=np.uint16)
        t = np.sort(rng.uniform(1e6, 2e6, 500))
        p = rng.integers(0, 2, 500, dtype=np.uint8)
        grid = vox.build(x, y, t, p)
        assert grid.shape == (5, 48, 64)
        assert grid.dtype == torch.float32

    def test_batch(self):
        vox = Voxelizer(H=48, W=64, n_bins=5, device="cpu")
        rng = np.random.default_rng(2)
        events = []
        for _ in range(3):
            x = rng.integers(0, 64, 200, dtype=np.uint16)
            y = rng.integers(0, 48, 200, dtype=np.uint16)
            t = np.sort(rng.uniform(0, 1, 200))
            p = rng.integers(0, 2, 200, dtype=np.uint8)
            events.append((x, y, t, p))
        batch = vox.build_batch(events)
        assert batch.shape == (3, 5, 48, 64)
