"""
Lock-free ring buffer for streaming events.

Events are stored as four parallel arrays (x, y, t, p).  Overflow wraps
around and overwrites the oldest data — caller is responsible for draining
the buffer fast enough.
"""

from __future__ import annotations

import numpy as np


class EventRingBuffer:
    """Fixed-capacity ring buffer storing (x, y, t, p) event tuples.

    Parameters
    ----------
    capacity : int
        Maximum number of events held simultaneously.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._x = np.empty(capacity, dtype=np.uint16)
        self._y = np.empty(capacity, dtype=np.uint16)
        self._t = np.empty(capacity, dtype=np.float64)
        self._p = np.empty(capacity, dtype=np.uint8)
        self._head = 0  # next write position
        self._size = 0  # number of valid events

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self._cap

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> int:
        """Insert events, wrapping around on overflow.

        Parameters
        ----------
        x, y, t, p : array-like, shape (N,)
            Event arrays.  All must have the same length.

        Returns
        -------
        int
            Number of events that overwrote existing data (0 when no wrap).
        """
        x = np.asarray(x, dtype=np.uint16)
        y = np.asarray(y, dtype=np.uint16)
        t = np.asarray(t, dtype=np.float64)
        p = np.asarray(p, dtype=np.uint8)

        n = len(x)
        if n == 0:
            return 0

        # When the batch itself is larger than capacity, keep only the tail.
        if n > self._cap:
            x, y, t, p = x[-self._cap:], y[-self._cap:], t[-self._cap:], p[-self._cap:]
            n = self._cap

        overwritten = max(0, self._size + n - self._cap)

        # Split into two contiguous writes if we wrap the ring.
        first_len = min(n, self._cap - self._head)
        second_len = n - first_len

        sl1 = slice(self._head, self._head + first_len)
        self._x[sl1] = x[:first_len]
        self._y[sl1] = y[:first_len]
        self._t[sl1] = t[:first_len]
        self._p[sl1] = p[:first_len]

        if second_len:
            sl2 = slice(0, second_len)
            self._x[sl2] = x[first_len:]
            self._y[sl2] = y[first_len:]
            self._t[sl2] = t[first_len:]
            self._p[sl2] = p[first_len:]

        self._head = (self._head + n) % self._cap
        self._size = min(self._size + n, self._cap)
        return overwritten

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def peek(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return up to *n* oldest events without consuming them.

        Parameters
        ----------
        n : int or None
            Number of events to return.  ``None`` returns all buffered events.

        Returns
        -------
        x, y, t, p : np.ndarray — copies of the requested events.
        """
        return self._read(n, consume=False)

    def pop(self, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Remove and return up to *n* oldest events.

        Parameters
        ----------
        n : int or None
            Number of events to consume.  ``None`` drains the buffer.

        Returns
        -------
        x, y, t, p : np.ndarray
        """
        return self._read(n, consume=True)

    def _read(
        self, n: int | None, consume: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = self._size if n is None else min(n, self._size)
        if count == 0:
            empty = np.empty(0, dtype=np.uint16)
            return empty, empty, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.uint8)

        tail = (self._head - self._size) % self._cap  # oldest element position
        first_len = min(count, self._cap - tail)
        second_len = count - first_len

        def _gather(arr: np.ndarray) -> np.ndarray:
            if second_len:
                return np.concatenate([arr[tail: tail + first_len], arr[:second_len]])
            return arr[tail: tail + first_len].copy()

        x = _gather(self._x)
        y = _gather(self._y)
        t = _gather(self._t)
        p = _gather(self._p)

        if consume:
            self._size -= count

        return x, y, t, p

    def clear(self) -> None:
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"EventRingBuffer(size={self._size}/{self._cap})"
