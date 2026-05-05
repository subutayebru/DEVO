"""
Event source abstractions and concrete readers.

H5 field layout matches DEVO's EventSlicer convention (utils/event_utils.py):
  Primary layout  — group "events/" containing datasets x, y, t, p
  Fallback layout — root-level datasets x, y, t, p
  Both layouts have a root-level "ms_to_idx" dataset and an optional "t_offset" scalar.

TXT field layout matches DEVO's ECD/FPV/RPG readers (utils/load_utils.py):
  Space-separated columns: t  x  y  p
  t is in seconds (float) or microseconds (int) depending on the dataset;
  the reader returns raw values without unit conversion so the caller decides.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np


class EventSource(ABC):
    """Abstract base for event data sources.

    Returns raw events as four parallel numpy arrays.
    """

    @abstractmethod
    def get_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all events as (x, y, t, p) numpy arrays.

        Returns
        -------
        x : np.ndarray, shape (N,), dtype uint16  — pixel column
        y : np.ndarray, shape (N,), dtype uint16  — pixel row
        t : np.ndarray, shape (N,), dtype float64 — timestamp (units: source-dependent)
        p : np.ndarray, shape (N,), dtype uint8   — polarity (0 or 1)
        """

    def close(self) -> None:
        """Release any held file handles. Override as needed."""


class H5Reader(EventSource):
    """Read events from an HDF5 file using DEVO's field layout.

    Supports two layouts:
      - Grouped:  h5["events/x"], h5["events/y"], h5["events/t"], h5["events/p"]
      - Flat:     h5["x"], h5["y"], h5["t"], h5["p"]

    Timestamps are offset-corrected when the file contains a "t_offset" scalar,
    matching EventSlicer behaviour in DEVO/utils/event_utils.py:41-44.
    """

    _FIELDS = ("x", "y", "t", "p")

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._h5 = h5py.File(self._path, "r")
        self._events = self._locate_events(self._h5)
        self._t_offset = int(self._h5["t_offset"][()]) if "t_offset" in self._h5 else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _locate_events(h5f: h5py.File) -> dict[str, h5py.Dataset]:
        """Resolve the grouped or flat event layout."""
        if "events/x" in h5f:
            return {k: h5f[f"events/{k}"] for k in ("x", "y", "t", "p")}
        # flat fallback
        missing = [k for k in ("x", "y", "t", "p") if k not in h5f]
        if missing:
            raise KeyError(
                f"HDF5 file has neither 'events/x' group nor root-level datasets. "
                f"Missing fields: {missing}"
            )
        return {k: h5f[k] for k in ("x", "y", "t", "p")}

    # ------------------------------------------------------------------
    # EventSource interface
    # ------------------------------------------------------------------

    def get_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all events from the file.

        Returns
        -------
        x : uint16 (N,)
        y : uint16 (N,)
        t : float64 (N,) — absolute timestamps in microseconds (t_offset applied)
        p : uint8 (N,)   — polarity 0 or 1
        """
        x = np.asarray(self._events["x"], dtype=np.uint16)
        y = np.asarray(self._events["y"], dtype=np.uint16)
        t = np.asarray(self._events["t"], dtype=np.float64) + self._t_offset
        p = np.asarray(self._events["p"], dtype=np.uint8)
        return x, y, t, p

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> "H5Reader":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"H5Reader({self._path})"


class TxtReader(EventSource):
    """Read events from a space-separated text file.

    Column order matches DEVO's ECD, FPV, and RPG readers
    (DEVO/utils/load_utils.py lines 803, 870, 1285):

        t  x  y  p

    where t is the first column (seconds as float or microseconds as int —
    the reader stores raw values; unit conversion is the caller's responsibility).
    Blank lines and lines starting with '#' are skipped.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(self._path)
        self._data: np.ndarray | None = None  # lazy load

    def _load(self) -> np.ndarray:
        if self._data is None:
            self._data = np.loadtxt(self._path, comments="#", delimiter=" ")
            if self._data.ndim == 1:
                self._data = self._data[np.newaxis, :]
            if self._data.shape[1] != 4:
                raise ValueError(
                    f"Expected 4 columns (t x y p) but got {self._data.shape[1]} "
                    f"in {self._path}"
                )
        return self._data

    def get_batch(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return all events from the text file.

        Returns
        -------
        x : uint16 (N,)
        y : uint16 (N,)
        t : float64 (N,) — raw timestamp from column 0 (units: source-dependent)
        p : uint8 (N,)   — polarity 0 or 1
        """
        data = self._load()
        t = data[:, 0].astype(np.float64)
        x = data[:, 1].astype(np.uint16)
        y = data[:, 2].astype(np.uint16)
        p = data[:, 3].astype(np.uint8)
        return x, y, t, p

    def __repr__(self) -> str:
        return f"TxtReader({self._path})"


def make_source(path: str) -> EventSource:
    """Factory that dispatches to the correct reader by file extension.

    Parameters
    ----------
    path : str
        Path to the event file.  Recognised extensions:
          .h5 / .hdf5  →  H5Reader
          .txt         →  TxtReader

    Raises
    ------
    ValueError
        For unrecognised extensions.
    FileNotFoundError
        When the file does not exist.
    """
    ext = Path(path).suffix.lower()
    if ext in {".h5", ".hdf5"}:
        return H5Reader(path)
    if ext == ".txt":
        return TxtReader(path)
    raise ValueError(
        f"Unsupported event file extension '{ext}'. "
        "Expected one of: .h5, .hdf5, .txt"
    )
