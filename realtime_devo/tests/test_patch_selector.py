"""
Smoke tests for PatchSelector.

Covers:
  - Architecture: output shape (1, 1, H//4, W//4) for fp16 input
  - Range:  all values in [0, 1]  (Sigmoid is fused)
  - TorchScript round-trip: save → load → identical output
  - Inference latency via CUDA events (or wall-clock on CPU):
      average over 100 runs, printed to stdout
  - Weight loading from a synthetic full-eVONet-style checkpoint
  - Checkpoint key inspection helper
"""

from __future__ import annotations

import time
import tempfile
from pathlib import Path
from collections import OrderedDict

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.patch_selector import (
    PatchSelector,
    DEVO_SCORER_KEYS,
    _extract_scorer_state,
    load_from_checkpoint,
    export_jit,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

H, W, B_BINS = 480, 640, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PT = Path(__file__).parent.parent / "models" / "patch_selector.pt"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_fp16(B: int = 1, bins: int = B_BINS, h: int = H, w: int = W,
                 device: str = DEVICE) -> torch.Tensor:
    return torch.randn(B, bins, h, w, device=device, dtype=torch.float16)


def _build_synthetic_checkpoint(tmp_dir: Path) -> Path:
    """Write a fake eVONet checkpoint whose scorer weights have the right shapes."""
    state: dict = {}
    shapes = {
        "patchify.scorer.scorer.0.weight": (8,  5, 3, 3),
        "patchify.scorer.scorer.0.bias":   (8,),
        "patchify.scorer.scorer.2.weight": (16, 8, 3, 3),
        "patchify.scorer.scorer.2.bias":   (16,),
        "patchify.scorer.scorer.4.weight": (32,16, 3, 3),
        "patchify.scorer.scorer.4.bias":   (32,),
        "patchify.scorer.scorer.6.weight": (1, 32, 3, 3),
        "patchify.scorer.scorer.6.bias":   (1,),
        # Dummy non-scorer key to verify filtering
        "patchify.fnet.conv1.weight": (32, 5, 7, 7),
    }
    for k, shape in shapes.items():
        state[k] = torch.randn(*shape)

    path = tmp_dir / "fake_DEVO.pth"
    torch.save({"model_state_dict": state}, path)
    return path


# ---------------------------------------------------------------------------
# 1. Architecture tests (PatchSelector module)
# ---------------------------------------------------------------------------

class TestPatchSelectorModule:
    def test_output_shape_fp32(self):
        model = PatchSelector().eval()
        x = torch.randn(1, 5, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, H // 4, W // 4), f"Got {out.shape}"

    def test_output_shape_fp16(self):
        model = PatchSelector().half().eval().to(DEVICE)
        x = _random_fp16(B=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 120, 160), f"Got {out.shape}"

    def test_output_dtype_fp16(self):
        model = PatchSelector().half().eval().to(DEVICE)
        x = _random_fp16(B=1)
        with torch.no_grad():
            out = model(x)
        assert out.dtype == torch.float16

    def test_output_range_in_0_1(self):
        """Sigmoid is fused — all outputs must be in [0, 1]."""
        model = PatchSelector().half().eval().to(DEVICE)
        x = _random_fp16(B=2)
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= 0.0, f"min = {out.min().item()}"
        assert out.max().item() <= 1.0, f"max = {out.max().item()}"

    def test_batch_dimension_preserved(self):
        model = PatchSelector().half().eval().to(DEVICE)
        for B in (1, 3, 8):
            x = _random_fp16(B=B)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (B, 1, H // 4, W // 4)

    def test_no_nan_or_inf(self):
        model = PatchSelector().half().eval().to(DEVICE)
        x = _random_fp16(B=1)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 2. TorchScript round-trip (models/patch_selector.pt)
# ---------------------------------------------------------------------------

class TestTorchScriptLoad:
    @pytest.mark.skipif(not MODEL_PT.exists(),
                        reason="models/patch_selector.pt not built yet")
    def test_load_and_run(self):
        """Step 4 smoke test: load .pt, run fp16 input, check shape and range."""
        scripted = torch.jit.load(str(MODEL_PT), map_location=DEVICE)
        scripted.eval()

        x = _random_fp16(B=1)
        with torch.no_grad():
            out = scripted(x)

        assert out.shape == (1, 1, 120, 160), f"Bad shape: {out.shape}"
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0
        print(f"\n[smoke] output shape {out.shape}  "
              f"min={out.min().item():.4f}  max={out.max().item():.4f}")

    @pytest.mark.skipif(not MODEL_PT.exists(),
                        reason="models/patch_selector.pt not built yet")
    def test_inference_latency(self):
        """Average latency over 100 runs via CUDA events (or wall-clock on CPU)."""
        scripted = torch.jit.load(str(MODEL_PT), map_location=DEVICE)
        scripted.eval()
        x = _random_fp16(B=1)
        N_RUNS = 100

        if DEVICE == "cuda":
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    scripted(x)
            torch.cuda.synchronize()

            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev   = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            for _ in range(N_RUNS):
                with torch.no_grad():
                    scripted(x)
            end_ev.record()
            torch.cuda.synchronize()
            avg_ms = start_ev.elapsed_time(end_ev) / N_RUNS
            method = "CUDA event"
        else:
            # Warm up
            for _ in range(3):
                with torch.no_grad():
                    scripted(x)
            t0 = time.perf_counter()
            for _ in range(N_RUNS):
                with torch.no_grad():
                    scripted(x)
            avg_ms = (time.perf_counter() - t0) * 1000 / N_RUNS
            method = "wall-clock"

        print(f"\n[latency] avg over {N_RUNS} runs ({method}, device={DEVICE}): "
              f"{avg_ms:.3f} ms/call")
        # Sanity: should complete in under 5 s total on any hardware
        assert avg_ms < 5000, f"Suspiciously slow: {avg_ms:.1f} ms/call"

    def test_eager_and_scripted_agree(self):
        """Eager fp16 and scripted fp16 outputs must agree to 1e-2.

        export_jit calls .half() in-place, so both eager and scripted
        models share fp16 weights; we compare them on the same fp16 input.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "ps.pt"
            model = PatchSelector().eval()
            export_jit(model, pt_path, use_fp16=True)
            # model is now fp16 after export_jit's in-place .half() call
            scripted = torch.jit.load(str(pt_path), map_location="cpu")

        x_fp16 = torch.randn(1, 5, 48, 64).half()

        with torch.no_grad():
            out_eager    = model(x_fp16).float()
            out_scripted = scripted(x_fp16).float()

        diff = (out_eager - out_scripted).abs().max().item()
        print(f"\n[agree] eager fp16 vs scripted fp16 max diff: {diff:.6f}")
        assert diff < 1e-2, f"Too large: {diff}"


# ---------------------------------------------------------------------------
# 3. Weight loading from synthetic checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointLoading:
    def test_extract_scorer_state_filters_keys(self):
        """_extract_scorer_state must only keep scorer keys, remapped to net.*"""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = _build_synthetic_checkpoint(Path(tmp))
            raw = torch.load(ckpt_path, map_location="cpu")
        state = _extract_scorer_state(raw)

        assert all(k.startswith("net.") for k in state), \
            f"Non-net keys leaked: {[k for k in state if not k.startswith('net.')]}"
        # Exactly 8 tensors (4 Conv2d × 2 params)
        assert len(state) == 8, f"Expected 8 params, got {len(state)}: {list(state)}"

    def test_load_from_checkpoint_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = _build_synthetic_checkpoint(Path(tmp))
            model = load_from_checkpoint(ckpt_path, device="cpu")
        assert isinstance(model, PatchSelector)
        x = torch.randn(1, 5, H, W)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, H // 4, W // 4)

    def test_loaded_weights_differ_from_random_init(self):
        """Loaded weights must equal the synthetic checkpoint values."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = _build_synthetic_checkpoint(Path(tmp))
            raw = torch.load(ckpt_path, map_location="cpu")
            state = _extract_scorer_state(raw)
            model = load_from_checkpoint(ckpt_path, device="cpu")

        loaded_w = model.net[0].weight.data   # Conv2d(5→8)
        ckpt_w   = state["net.0.weight"]
        assert torch.allclose(loaded_w, ckpt_w), "Conv0 weights did not load correctly"

    def test_devo_scorer_keys_constant_matches_source(self):
        """DEVO_SCORER_KEYS must list exactly the keys the synthetic checkpoint has."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = _build_synthetic_checkpoint(Path(tmp))
            raw = torch.load(ckpt_path, map_location="cpu")
        state: dict = raw.get("model_state_dict", raw)
        scorer_keys_in_ckpt = sorted(
            k for k in state if k.startswith("patchify.scorer.scorer.")
        )
        assert sorted(DEVO_SCORER_KEYS) == scorer_keys_in_ckpt

    def test_export_and_reload_round_trip(self):
        """export_jit → load → forward must be consistent."""
        with tempfile.TemporaryDirectory() as tmp:
            pt_path = Path(tmp) / "ps.pt"
            model = PatchSelector().eval()
            export_jit(model, pt_path, use_fp16=True)

            scripted = torch.jit.load(str(pt_path), map_location="cpu")
            x = torch.randn(1, 5, 48, 64).half()
            with torch.no_grad():
                out = scripted(x)
            assert out.shape == (1, 1, 12, 16)
            assert out.min().item() >= 0.0
            assert out.max().item() <= 1.0
