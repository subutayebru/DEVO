"""
Event voxel-grid builder.

Two implementations are provided:

  events_to_voxel_grid()           — vectorised, uses scatter_add (no event loops)
  events_to_voxel_grid_reference() — explicit Python loop over events (for validation)

Both apply bilinear temporal interpolation and per-bin z-score normalisation.

The Voxelizer class wraps events_to_voxel_grid for callers that use numpy arrays
and need batching support (used by the RealtimePipeline).
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Fast vectorised implementation
# ---------------------------------------------------------------------------

def events_to_voxel_grid(
    x: torch.Tensor,   # (N,) int   — pixel column, 0-based
    y: torch.Tensor,   # (N,) int   — pixel row,    0-based
    t: torch.Tensor,   # (N,) float — already normalised to [0, B-1]
    p: torch.Tensor,   # (N,) float — polarity in {-1, +1}
    H: int,
    W: int,
    B: int = 5,
    device: str = "cuda",
) -> torch.Tensor:
    """Build a voxel grid with bilinear temporal interpolation.

    Each event contributes to two adjacent time bins:
      left  bin  b      with weight  w_L = 1 - (t - floor(t))
      right bin  b + 1  with weight  w_R =     (t - floor(t))

    The contribution to each bin cell is  p * w  (polarity-weighted).
    Accumulation uses a single scatter_add over the flat grid — no Python
    loops over events.

    After accumulation the grid is normalised per bin: zero mean, unit
    variance (eps = 1e-6 prevents division by zero for silent bins).

    Parameters
    ----------
    x, y : (N,) integer tensors — pixel coordinates (not bounds-checked;
            callers should pre-clip or filter out-of-sensor events).
    t    : (N,) float tensor   — timestamps already mapped to [0, B-1].
    p    : (N,) float tensor   — polarity values in {-1.0, +1.0}.
    H, W : sensor height / width.
    B    : number of temporal bins.
    device : torch device string.

    Returns
    -------
    torch.Tensor, shape (B, H, W), dtype float32
        Normalised voxel grid; all-zero when N == 0.
    """
    x = x.to(device=device, dtype=torch.long)
    y = y.to(device=device, dtype=torch.long)
    t = t.to(device=device, dtype=torch.float32)
    p = p.to(device=device, dtype=torch.float32)

    N = x.shape[0]
    grid = torch.zeros(B * H * W, dtype=torch.float32, device=device)

    if N == 0:
        return grid.view(B, H, W)

    # ---- bilinear temporal weights ----------------------------------------
    b_left  = t.floor().long()              # (N,)  left bin index
    w_right = t - b_left.float()            # (N,)  right weight
    w_left  = 1.0 - w_right                 # (N,)  left  weight

    # ---- flat spatial index: row-major within each bin --------------------
    # flat index = b * H * W + y * W + x
    xy_flat = y * W + x                     # (N,)

    # ---- collect contributions for both bins into two arrays --------------
    # Left bin
    b_right = b_left + 1                    # (N,)

    mask_l = (b_left  >= 0) & (b_left  < B)
    mask_r = (b_right >= 0) & (b_right < B)

    idx_l = (b_left [mask_l] * (H * W) + xy_flat[mask_l])  # valid flat indices
    idx_r = (b_right[mask_r] * (H * W) + xy_flat[mask_r])

    val_l = p[mask_l] * w_left [mask_l]    # polarity * temporal weight
    val_r = p[mask_r] * w_right[mask_r]

    # Single scatter_add — accumulate both halves in one call
    all_idx = torch.cat([idx_l, idx_r])
    all_val = torch.cat([val_l, val_r])
    grid.scatter_add_(0, all_idx, all_val)

    # ---- per-bin z-score normalisation (vectorised, no Python loops) ------
    grid = grid.view(B, H * W)              # (B, H*W)
    mean = grid.mean(dim=1, keepdim=True)   # (B, 1)
    std  = grid.std (dim=1, keepdim=True)   # (B, 1)  unbiased, matches torch default
    grid = (grid - mean) / (std + 1e-6)

    return grid.view(B, H, W)


# ---------------------------------------------------------------------------
# Reference implementation (Python loop over events — for validation only)
# ---------------------------------------------------------------------------

def events_to_voxel_grid_reference(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    p: torch.Tensor,
    H: int,
    W: int,
    B: int = 5,
    device: str = "cuda",
) -> torch.Tensor:
    """Equivalent bilinear-temporal voxel grid built with an explicit event loop.

    Identical math to events_to_voxel_grid; used only for numerical validation.
    O(N) Python iterations — not suitable for production.
    """
    grid = torch.zeros(B, H, W, dtype=torch.float32, device=device)

    x_cpu = x.cpu().long().tolist()
    y_cpu = y.cpu().long().tolist()
    t_cpu = t.cpu().float().tolist()
    p_cpu = p.cpu().float().tolist()

    for xi, yi, ti, pi in zip(x_cpu, y_cpu, t_cpu, p_cpu):
        b_l = int(ti)                       # floor
        w_r = ti - b_l                      # right weight
        w_l = 1.0 - w_r                     # left  weight

        if 0 <= b_l < B:
            grid[b_l, yi, xi] += pi * w_l
        b_r = b_l + 1
        if 0 <= b_r < B:
            grid[b_r, yi, xi] += pi * w_r

    # Per-bin normalisation — same formula as the fast path
    grid = grid.view(B, H * W)
    mean = grid.mean(dim=1, keepdim=True)
    std  = grid.std (dim=1, keepdim=True)
    grid = (grid - mean) / (std + 1e-6)

    return grid.view(B, H, W)


# ---------------------------------------------------------------------------
# Voxelizer class — wraps events_to_voxel_grid for numpy callers / batching
# ---------------------------------------------------------------------------

class Voxelizer:
    """Convert raw event arrays to a batched voxel-grid tensor.

    Accepts numpy inputs (as produced by EventSource), converts them, and
    delegates to events_to_voxel_grid.

    Parameters
    ----------
    H, W   : sensor resolution
    n_bins : temporal bins (DEVO default: 5)
    device : torch device
    """

    def __init__(
        self,
        H: int = 480,
        W: int = 640,
        n_bins: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.H = H
        self.W = W
        self.n_bins = n_bins
        self.device = str(device)

    def build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> torch.Tensor:
        """Build a single normalised voxel grid from raw events.

        Parameters
        ----------
        x, y : (N,) integer coordinates
        t    : (N,) timestamps in any unit (normalised internally)
        p    : (N,) polarity in {0, 1} or {-1, +1}

        Returns
        -------
        torch.Tensor, shape (n_bins, H, W), float32
        """
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.float32)

        if len(x) == 0:
            return torch.zeros(self.n_bins, self.H, self.W,
                               dtype=torch.float32, device=self.device)

        # Remap {0,1} polarity to {-1,+1} if needed
        if p.min() >= 0:
            p = p.copy()
            p[p == 0] = -1.0

        # Normalise timestamps to [0, n_bins - 1]
        t0, t1 = t[0], t[-1]
        B = self.n_bins
        t_norm = (t - t0) * (B - 1) / (t1 - t0) if t1 != t0 else np.zeros_like(t)

        return events_to_voxel_grid(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(t_norm.astype(np.float32)),
            torch.from_numpy(p),
            H=self.H, W=self.W, B=B, device=self.device,
        )

    def build_batch(
        self,
        events_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        """Build a batch of voxel grids.

        Returns
        -------
        torch.Tensor, shape (batch, n_bins, H, W), float32
        """
        return torch.stack([self.build(*ev) for ev in events_list], dim=0)
