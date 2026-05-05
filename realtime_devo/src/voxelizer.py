"""
Event voxel-grid builder.

Implements the same bilinear temporal interpolation used by DEVO
(DEVO/utils/event_utils.py::to_voxel_grid).  The output is a float32
tensor of shape (B, N_BINS, H, W) ready for the feature extractors.
"""

from __future__ import annotations

import numpy as np
import torch


class Voxelizer:
    """Convert raw event arrays to a batched voxel-grid tensor.

    Parameters
    ----------
    H, W : int
        Sensor resolution.
    n_bins : int
        Number of temporal bins (DEVO default: 5).
    device : str or torch.device
        Target device for the output tensor.
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
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> torch.Tensor:
        """Build a single voxel grid from raw events.

        Replicates DEVO/utils/event_utils.py::to_voxel_grid exactly:
          - Polarity 0 is remapped to -1.
          - Each event is trilinearly interpolated into its 8 neighbouring
            (x_left/right, y_left/right, t_left/right) voxel corners.
          - The grid is accumulated via index_add_ for efficiency.

        Parameters
        ----------
        x, y : array-like, shape (N,) — pixel coordinates
        t    : array-like, shape (N,) — timestamps (any unit; only differences matter)
        p    : array-like, shape (N,) — polarities {0, 1}

        Returns
        -------
        torch.Tensor, shape (n_bins, H, W), dtype float32, on self.device
            Empty tensor (all zeros) when the event arrays are empty.
        """
        H, W, B = self.H, self.W, self.n_bins
        grid = torch.zeros(B, H, W, dtype=torch.float32, device=self.device)

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.int8)

        if len(x) == 0:
            return grid

        # Remap polarity: {0 → -1, 1 → +1}
        p = p.copy()
        p[p == 0] = -1

        # Normalise timestamps to [0, n_bins - 1]
        t_norm = (t - t[0]) * (B - 1) / (t[-1] - t[0]) if t[-1] != t[0] else np.zeros_like(t)

        x_t = torch.from_numpy(x).to(self.device)
        y_t = torch.from_numpy(y).to(self.device)
        t_t = torch.from_numpy(t_norm.astype(np.float64)).to(self.device)
        pol = torch.from_numpy(p.astype(np.float32)).to(self.device)

        grid_flat = grid.flatten()

        left_t, right_t = t_t.floor(), t_t.floor() + 1
        left_x, right_x = x_t.floor(), x_t.floor() + 1
        left_y, right_y = y_t.floor(), y_t.floor() + 1

        for lim_x in (left_x, right_x):
            for lim_y in (left_y, right_y):
                for lim_t in (left_t, right_t):
                    mask = (
                        (lim_x >= 0) & (lim_x <= W - 1)
                        & (lim_y >= 0) & (lim_y <= H - 1)
                        & (lim_t >= 0) & (lim_t <= B - 1)
                    )
                    lin_idx = (
                        lim_x.long()
                        + lim_y.long() * W
                        + lim_t.long() * W * H
                    )
                    weight = (
                        pol
                        * (1 - (lim_x - x_t).abs())
                        * (1 - (lim_y - y_t).abs())
                        * (1 - (lim_t - t_t).abs())
                    )
                    grid_flat.index_add_(0, lin_idx[mask], weight[mask].float())

        return grid  # (n_bins, H, W)

    def build_batch(
        self,
        events_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        """Build a batch of voxel grids.

        Parameters
        ----------
        events_list : list of (x, y, t, p) tuples

        Returns
        -------
        torch.Tensor, shape (B_batch, n_bins, H, W), float32
        """
        grids = [self.build(*ev) for ev in events_list]
        return torch.stack(grids, dim=0)
