"""Tests for EventRingBuffer."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.event_buffer import EventRingBuffer


def _events(n: int, t_start: float = 0.0):
    x = np.arange(n, dtype=np.uint16)
    y = np.zeros(n, dtype=np.uint16)
    t = np.arange(n, dtype=np.float64) + t_start
    p = (np.arange(n) % 2).astype(np.uint8)
    return x, y, t, p


class TestRingBufferBasic:
    def test_empty_on_creation(self):
        buf = EventRingBuffer(100)
        assert buf.size == 0
        assert buf.is_empty()

    def test_push_and_size(self):
        buf = EventRingBuffer(100)
        buf.push(*_events(10))
        assert buf.size == 10

    def test_pop_returns_correct_count(self):
        buf = EventRingBuffer(100)
        buf.push(*_events(50))
        x, y, t, p = buf.pop(20)
        assert len(x) == len(y) == len(t) == len(p) == 20
        assert buf.size == 30

    def test_pop_all(self):
        buf = EventRingBuffer(100)
        buf.push(*_events(40))
        x, y, t, p = buf.pop()
        assert len(x) == 40
        assert buf.is_empty()

    def test_peek_does_not_consume(self):
        buf = EventRingBuffer(100)
        buf.push(*_events(30))
        buf.peek(10)
        assert buf.size == 30

    def test_order_preserved(self):
        """Events should come out oldest-first (FIFO)."""
        buf = EventRingBuffer(100)
        x_in, y_in, t_in, p_in = _events(10)
        buf.push(x_in, y_in, t_in, p_in)
        x_out, _, t_out, _ = buf.pop()
        np.testing.assert_array_equal(x_out, x_in)
        np.testing.assert_array_equal(t_out, t_in)

    def test_capacity_property(self):
        buf = EventRingBuffer(256)
        assert buf.capacity == 256


class TestRingBufferOverflow:
    def test_overflow_wraps(self):
        buf = EventRingBuffer(50)
        buf.push(*_events(60))   # exceeds capacity; should not raise
        assert buf.size == 50

    def test_overflow_returns_overwritten_count(self):
        buf = EventRingBuffer(50)
        buf.push(*_events(30))
        overwritten = buf.push(*_events(40))
        assert overwritten == 20  # 30+40 - 50

    def test_wrap_preserves_newest(self):
        """After overflow, the newest events survive."""
        buf = EventRingBuffer(10)
        buf.push(*_events(20, t_start=0.0))   # 0..19, newest = 10..19
        _, _, t_out, _ = buf.pop()
        # The oldest surviving event has t >= 10 (we kept the last 10).
        assert t_out[0] >= 10.0

    def test_batch_larger_than_capacity(self):
        buf = EventRingBuffer(10)
        buf.push(*_events(100))   # 10x capacity
        assert buf.size == 10

    def test_multi_push_wraps_correctly(self):
        buf = EventRingBuffer(10)
        for i in range(5):
            buf.push(*_events(4, t_start=float(i * 4)))
        # After 5 × 4 = 20 events into cap-10 buffer, 10 remain.
        assert buf.size == 10


class TestRingBufferClear:
    def test_clear_resets_size(self):
        buf = EventRingBuffer(50)
        buf.push(*_events(30))
        buf.clear()
        assert buf.size == 0
        assert buf.is_empty()

    def test_push_after_clear(self):
        buf = EventRingBuffer(50)
        buf.push(*_events(50))
        buf.clear()
        buf.push(*_events(10))
        assert buf.size == 10


class TestRingBufferEmpty:
    def test_pop_empty_returns_empty_arrays(self):
        buf = EventRingBuffer(50)
        x, y, t, p = buf.pop(10)
        assert len(x) == 0

    def test_pop_more_than_available(self):
        buf = EventRingBuffer(50)
        buf.push(*_events(5))
        x, y, t, p = buf.pop(20)
        assert len(x) == 5
