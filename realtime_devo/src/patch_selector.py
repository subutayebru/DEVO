"""
Thin wrapper around DEVO's Scorer CNN.

The Scorer is a lightweight 4-layer CNN that maps a voxel-grid tensor
(B*N, 5, H, W) → (B*N, 1, H', W') score map used by DEVO's patch
selector to bias patch extraction toward event-rich regions.

This module provides a standalone wrapper so the scorer can be loaded
and run independently of the full eVONet, e.g. for profiling or ablation.

Reference: DEVO/devo/selector.py::Scorer
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


class Scorer(nn.Module):
    """Patch-score CNN.

    Identical architecture to DEVO/devo/selector.py::Scorer.

    Parameters
    ----------
    bins : int
        Number of voxel time-bins (input channels).  DEVO default: 5.
    """

    def __init__(self, bins: int = 5) -> None:
        super().__init__()
        self.bins = bins
        # Four conv layers followed by MaxPool — matches selector.py exactly.
        self.scorer = nn.Sequential(
            nn.Conv2d(bins, 8,  kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(8,   16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,  32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,  1,  kernel_size=3),
            nn.MaxPool2d(kernel_size=4, stride=4),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-location scores.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N, bins, H, W)
            Batched sequence of voxel grids.

        Returns
        -------
        torch.Tensor, shape (B, N, H', W')
            Score map.  H' = (H - 8) // 4,  W' = (W - 8) // 4
            for the default four 3×3-conv + MaxPool(4) configuration.
        """
        b, n, c, h, w = x.shape
        x_flat = x.view(b * n, c, h, w)
        scores = self.scorer(x_flat)           # (B*N, 1, H', W')
        _, _, h2, w2 = scores.shape
        return scores.view(b, n, h2, w2)


class ScorerWrapper:
    """Load a Scorer from a checkpoint and run inference.

    Only the scorer sub-module keys are extracted from full eVONet
    checkpoints that contain ``patchify.scorer.*`` weights.

    Parameters
    ----------
    bins : int
        Voxel time-bins (must match the checkpoint).
    device : str or torch.device
    """

    def __init__(self, bins: int = 5, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = Scorer(bins=bins).to(self.device)
        self.model.eval()

    def load_weights(self, checkpoint_path: str | os.PathLike) -> None:
        """Load scorer weights from a DEVO checkpoint or a bare state-dict.

        Handles three checkpoint formats:
          1. Full eVONet checkpoint with key ``model_state_dict``
          2. Legacy DataParallel checkpoint (``module.`` prefixed keys)
          3. Bare scorer state-dict

        Parameters
        ----------
        checkpoint_path : path-like
            Path to the ``.pth`` file (e.g. ``DEVO.pth``).
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        # Strip DataParallel prefix.
        state = {k.replace("module.", ""): v for k in state
                 for k in [k] if "update.lmbda" not in k}

        # Extract scorer sub-module keys (``patchify.scorer.*``).
        scorer_prefix = "patchify.scorer."
        scorer_state = {
            k[len(scorer_prefix):]: v
            for k, v in state.items()
            if k.startswith(scorer_prefix)
        }

        if scorer_state:
            self.model.scorer.load_state_dict(scorer_state)
        else:
            # Assume the file is already a bare scorer state-dict.
            self.model.load_state_dict(state)

    @torch.no_grad()
    def score(self, voxels: torch.Tensor) -> torch.Tensor:
        """Run the scorer on a voxel batch.

        Parameters
        ----------
        voxels : torch.Tensor, shape (B, N, bins, H, W) or (N, bins, H, W)
            Float32 voxel grids, already normalised.

        Returns
        -------
        torch.Tensor, shape (B, N, H', W') — score maps in [0, 1] after sigmoid.
        """
        if voxels.ndim == 4:
            voxels = voxels.unsqueeze(0)
        voxels = voxels.to(self.device)
        scores = self.model(voxels)
        return torch.sigmoid(scores)
