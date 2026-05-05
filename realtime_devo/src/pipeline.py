"""
Realtime event-to-voxel pipeline.

Ties together EventSource → EventRingBuffer → Voxelizer in a single
blocking or threaded loop.  Intended as the entry point for online
inference with DEVO.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterator

import numpy as np
import torch

from .event_buffer import EventRingBuffer
from .event_source import EventSource
from .voxelizer import Voxelizer


class RealtimePipeline:
    """Process streaming events into voxel grids.

    Parameters
    ----------
    source : EventSource
        Where events come from (file, camera driver, …).
    voxelizer : Voxelizer
        Converts event windows to voxel-grid tensors.
    buffer_capacity : int
        Ring-buffer size in number of events.
    window_size : int
        Number of events per voxel window.
    stride : int
        Events to advance between consecutive windows.  stride == window_size
        gives non-overlapping windows.
    on_voxel : callable or None
        Optional callback invoked with each ``(voxel, t_mid)`` pair where
        ``voxel`` is a ``(n_bins, H, W)`` float32 tensor and ``t_mid`` is
        the midpoint timestamp of the event window.
    """

    def __init__(
        self,
        source: EventSource,
        voxelizer: Voxelizer,
        buffer_capacity: int = 2_000_000,
        window_size: int = 30_000,
        stride: int | None = None,
        on_voxel: Callable[[torch.Tensor, float], None] | None = None,
    ) -> None:
        self.source = source
        self.voxelizer = voxelizer
        self.buffer = EventRingBuffer(buffer_capacity)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.on_voxel = on_voxel
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Blocking iterator interface
    # ------------------------------------------------------------------

    def run(self) -> Iterator[tuple[torch.Tensor, float]]:
        """Pull events from source and yield (voxel, t_mid) pairs.

        Yields
        ------
        voxel : torch.Tensor, shape (n_bins, H, W)
        t_mid : float — midpoint timestamp of the window
        """
        x, y, t, p = self.source.get_batch()
        self.buffer.push(x, y, t, p)

        while self.buffer.size >= self.window_size and not self._stop_event.is_set():
            x_w, y_w, t_w, p_w = self.buffer.pop(self.stride)

            # If stride < window_size, peek the remaining overlap.
            if self.stride < self.window_size:
                overlap = self.window_size - self.stride
                x_o, y_o, t_o, p_o = self.buffer.peek(overlap)
                x_w = np.concatenate([x_w, x_o])
                y_w = np.concatenate([y_w, y_o])
                t_w = np.concatenate([t_w, t_o])
                p_w = np.concatenate([p_w, p_o])

            voxel = self.voxelizer.build(x_w, y_w, t_w, p_w)
            t_mid = float((t_w[0] + t_w[-1]) / 2) if len(t_w) > 0 else 0.0

            if self.on_voxel is not None:
                self.on_voxel(voxel, t_mid)

            yield voxel, t_mid

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Simple single-shot helper
    # ------------------------------------------------------------------

    def run_once(self) -> list[tuple[torch.Tensor, float]]:
        """Consume all buffered events and return all voxel windows."""
        return list(self.run())


def run_from_source(
    source: EventSource,
    H: int = 480,
    W: int = 640,
    n_bins: int = 5,
    window_size: int = 30_000,
    stride: int | None = None,
    device: str = "cpu",
) -> list[tuple[torch.Tensor, float]]:
    """Convenience function: build voxels from an EventSource in one call.

    Parameters
    ----------
    source : EventSource
    H, W : int — sensor resolution
    n_bins : int — temporal bins per voxel
    window_size : int — events per window
    stride : int or None — events advanced per step (defaults to window_size)
    device : str — torch device for output tensors

    Returns
    -------
    list of (voxel, t_mid) tuples
    """
    voxelizer = Voxelizer(H=H, W=W, n_bins=n_bins, device=device)
    pipeline = RealtimePipeline(source, voxelizer, window_size=window_size, stride=stride)
    return pipeline.run_once()
