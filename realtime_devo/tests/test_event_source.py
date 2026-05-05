"""
Tests for H5Reader, TxtReader, and make_source.

H5 files are created synthetically in a tmp directory so the tests are
self-contained and require no external dataset.

The synthetic H5 layout matches the format used by DEVO's EventSlicer
(DEVO/utils/event_utils.py):
  - Primary layout: datasets under the group "events/" (x, y, t, p)
  - Root-level dataset: "ms_to_idx"
  - Optional root scalar: "t_offset"
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.event_source import H5Reader, TxtReader, make_source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_h5_grouped(path: Path, n: int = 1000, t_offset: int | None = None) -> dict:
    """Create an HDF5 file using DEVO's grouped layout (events/{x,y,t,p})."""
    rng = np.random.default_rng(42)
    x = rng.integers(0, 640, size=n, dtype=np.uint16)
    y = rng.integers(0, 480, size=n, dtype=np.uint16)
    t = np.sort(rng.uniform(0, 1_000_000, size=n)).astype(np.float64)
    p = rng.integers(0, 2, size=n, dtype=np.uint8)

    # ms_to_idx: one entry per millisecond from 0 to ceil(t_max / 1000)
    t_us = t.astype(np.uint64)
    ms_end = int(math.ceil(t_us.max() / 1000))
    ms_to_idx = np.searchsorted(t_us, np.arange(ms_end + 1) * 1000, side="left").astype(np.int64)

    with h5py.File(path, "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("x", data=x)
        grp.create_dataset("y", data=y)
        grp.create_dataset("t", data=t)
        grp.create_dataset("p", data=p)
        f.create_dataset("ms_to_idx", data=ms_to_idx)
        if t_offset is not None:
            f.create_dataset("t_offset", data=np.int64(t_offset))

    return dict(x=x, y=y, t=t, p=p, t_offset=t_offset or 0)


def _make_h5_flat(path: Path, n: int = 500) -> dict:
    """Create an HDF5 file using DEVO's flat fallback layout (root-level x,y,t,p)."""
    rng = np.random.default_rng(7)
    x = rng.integers(0, 640, size=n, dtype=np.uint16)
    y = rng.integers(0, 480, size=n, dtype=np.uint16)
    t = np.sort(rng.uniform(0, 500_000, size=n)).astype(np.float64)
    p = rng.integers(0, 2, size=n, dtype=np.uint8)

    t_us = t.astype(np.uint64)
    ms_end = int(math.ceil(t_us.max() / 1000))
    ms_to_idx = np.searchsorted(t_us, np.arange(ms_end + 1) * 1000, side="left").astype(np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("t", data=t)
        f.create_dataset("p", data=p)
        f.create_dataset("ms_to_idx", data=ms_to_idx)

    return dict(x=x, y=y, t=t, p=p, t_offset=0)


def _make_txt(path: Path, n: int = 200) -> dict:
    """Create a text file with DEVO's column layout: t x y p."""
    rng = np.random.default_rng(3)
    t = np.sort(rng.uniform(0.0, 10.0, size=n))
    x = rng.integers(0, 640, size=n, dtype=np.uint16)
    y = rng.integers(0, 480, size=n, dtype=np.uint16)
    p = rng.integers(0, 2, size=n, dtype=np.uint8)

    data = np.column_stack([t, x, y, p])
    np.savetxt(path, data, delimiter=" ", fmt=["%.9f", "%d", "%d", "%d"])

    return dict(x=x, y=y, t=t, p=p)


# ---------------------------------------------------------------------------
# H5Reader — grouped layout
# ---------------------------------------------------------------------------

class TestH5ReaderGrouped:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events_grouped.h5"
        self.ground_truth = _make_h5_grouped(self.path, n=1000)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_instantiation(self):
        reader = H5Reader(self.path)
        reader.close()

    def test_get_batch_returns_four_arrays(self):
        with H5Reader(self.path) as r:
            result = r.get_batch()
        assert len(result) == 4, "get_batch must return (x, y, t, p)"

    def test_output_shapes(self):
        n = 1000
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        assert x.shape == (n,), f"x.shape={x.shape}"
        assert y.shape == (n,), f"y.shape={y.shape}"
        assert t.shape == (n,), f"t.shape={t.shape}"
        assert p.shape == (n,), f"p.shape={p.shape}"

    def test_output_dtypes(self):
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        assert x.dtype == np.uint16, f"x.dtype={x.dtype}"
        assert y.dtype == np.uint16, f"y.dtype={y.dtype}"
        assert t.dtype == np.float64, f"t.dtype={t.dtype}"
        assert p.dtype == np.uint8,  f"p.dtype={p.dtype}"

    def test_values_match_written_data(self):
        gt = self.ground_truth
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        np.testing.assert_array_equal(x, gt["x"])
        np.testing.assert_array_equal(y, gt["y"])
        np.testing.assert_array_equal(p, gt["p"])

    def test_t_offset_applied(self):
        offset = 500_000
        path2 = Path(self.tmp.name) / "events_offset.h5"
        gt2 = _make_h5_grouped(path2, n=300, t_offset=offset)
        with H5Reader(path2) as r:
            _, _, t_out, _ = r.get_batch()
        # t_out should be raw t + offset
        np.testing.assert_allclose(t_out, gt2["t"] + offset, rtol=1e-9)

    def test_no_t_offset_by_default(self):
        with H5Reader(self.path) as r:
            _, _, t_out, _ = r.get_batch()
        gt = self.ground_truth
        np.testing.assert_allclose(t_out, gt["t"], rtol=1e-9)

    def test_polarity_values_in_range(self):
        with H5Reader(self.path) as r:
            _, _, _, p = r.get_batch()
        assert np.all((p == 0) | (p == 1)), "Polarities must be 0 or 1"

    def test_context_manager(self):
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        assert len(x) == 1000

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            H5Reader("/nonexistent/path/events.h5")


# ---------------------------------------------------------------------------
# H5Reader — flat layout (fallback)
# ---------------------------------------------------------------------------

class TestH5ReaderFlat:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events_flat.h5"
        self.ground_truth = _make_h5_flat(self.path, n=500)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_output_shapes_flat(self):
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        assert x.shape == (500,)
        assert y.shape == (500,)
        assert t.shape == (500,)
        assert p.shape == (500,)

    def test_output_dtypes_flat(self):
        with H5Reader(self.path) as r:
            x, y, t, p = r.get_batch()
        assert x.dtype == np.uint16
        assert y.dtype == np.uint16
        assert t.dtype == np.float64
        assert p.dtype == np.uint8


# ---------------------------------------------------------------------------
# TxtReader
# ---------------------------------------------------------------------------

class TestTxtReader:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.txt"
        self.ground_truth = _make_txt(self.path, n=200)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_output_shapes(self):
        reader = TxtReader(self.path)
        x, y, t, p = reader.get_batch()
        assert x.shape == (200,)
        assert y.shape == (200,)
        assert t.shape == (200,)
        assert p.shape == (200,)

    def test_output_dtypes(self):
        reader = TxtReader(self.path)
        x, y, t, p = reader.get_batch()
        assert x.dtype == np.uint16
        assert y.dtype == np.uint16
        assert t.dtype == np.float64
        assert p.dtype == np.uint8

    def test_values_match(self):
        gt = self.ground_truth
        reader = TxtReader(self.path)
        x, y, t, p = reader.get_batch()
        np.testing.assert_array_equal(x, gt["x"])
        np.testing.assert_array_equal(y, gt["y"])
        np.testing.assert_array_equal(p, gt["p"])

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            TxtReader("/nonexistent/events.txt")

    def test_wrong_column_count_raises(self):
        bad = Path(self.tmp.name) / "bad.txt"
        bad.write_text("1.0 2 3\n4.0 5 6\n")  # only 3 columns
        with pytest.raises(ValueError, match="4 columns"):
            TxtReader(bad).get_batch()


# ---------------------------------------------------------------------------
# make_source factory
# ---------------------------------------------------------------------------

class TestMakeSource:
    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_dispatch_h5(self):
        p = self.tmpdir / "events.h5"
        _make_h5_grouped(p, n=50)
        src = make_source(str(p))
        assert isinstance(src, H5Reader)
        src.close()

    def test_dispatch_hdf5_extension(self):
        p = self.tmpdir / "events.hdf5"
        _make_h5_grouped(p, n=50)
        src = make_source(str(p))
        assert isinstance(src, H5Reader)
        src.close()

    def test_dispatch_txt(self):
        p = self.tmpdir / "events.txt"
        _make_txt(p, n=50)
        src = make_source(str(p))
        assert isinstance(src, TxtReader)

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            make_source("/some/file.csv")

    def test_h5_get_batch_via_factory(self):
        p = self.tmpdir / "events.h5"
        _make_h5_grouped(p, n=100)
        with make_source(str(p)) as src:
            x, y, t, p_arr = src.get_batch()
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert t.shape == (100,)
        assert p_arr.shape == (100,)
