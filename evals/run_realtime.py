#!/usr/bin/env python3
"""
Realtime event-camera odometry with DEVO.

Bypasses DEVO's dataset-specific file loaders entirely.  Raw events from
any .h5 or .txt file are buffered, voxelized, and fed frame-by-frame into
the DEVO tracking loop via the realtime_devo pipeline.

Usage
-----
python DEVO/evals/run_realtime.py \\
    --data   path/to/sequence.h5 \\
    --calib  path/to/calib.txt   \\
    --weights DEVO.pth

Calibration file (whitespace-separated, one line or one value per line):
    fx  fy  cx  cy

H5 layout accepted (both variants used by DEVO datasets):
    grouped : h5["events/{x,y,t,p}"] + h5["ms_to_idx"]
    flat    : h5["{x,y,t,p}"]        + h5["ms_to_idx"]

TXT layout (ECD / RPG / FPV datasets):
    t_sec_or_us  x  y  p   (space-separated, one event per line)
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup – resolve roots without assuming a working directory
# ---------------------------------------------------------------------------
_EVAL_DIR  = Path(__file__).resolve().parent        # DEVO/evals/
_DEVO_ROOT = _EVAL_DIR.parent                       # DEVO/
_RT_ROOT   = _DEVO_ROOT / "realtime_devo"           # DEVO/realtime_devo/

sys.path.insert(0, str(_DEVO_ROOT))     # enables: from devo.xxx import ...
sys.path.insert(0, str(_RT_ROOT))       # enables: from src.xxx import ...

# DEVO internals
from devo.devo import DEVO
from devo.config import cfg
from devo.plot_utils import save_trajectory_tum_format

# realtime_devo components
from src.event_source import EventSource, make_source
from src.voxelizer import Voxelizer
from src.patch_selector import PatchSelector
from src.pipeline import RealtimePipeline


# ---------------------------------------------------------------------------
# In-memory EventSource
#   Wraps pre-loaded numpy arrays so the pipeline never re-reads the file.
# ---------------------------------------------------------------------------

class _ArraySource(EventSource):
    """EventSource backed by in-memory numpy arrays."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        p: np.ndarray,
    ) -> None:
        self._x, self._y, self._t, self._p = x, y, t, p

    def get_batch(self):
        return self._x, self._y, self._t, self._p


# ---------------------------------------------------------------------------
# Timed DEVO subclass
#   Records wall-clock time of every update() (= Dense Bundle Adjustment)
#   call without modifying the original DEVO source.
# ---------------------------------------------------------------------------

class _TimedDEVO(DEVO):
    """DEVO with per-call DBA timing."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dba_times_ms: list[float] = []
        self._record_dba: bool = True   # set False for post-loop refinement

    def update(self) -> None:
        if not self._record_dba:
            super().update()
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        super().update()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._dba_times_ms.append((time.perf_counter() - t0) * 1e3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_calib(path: str) -> torch.Tensor:
    """Return [fx, fy, cx, cy] from a whitespace-separated calibration file."""
    vals = np.loadtxt(path).flatten()
    if len(vals) < 4:
        raise ValueError(
            f"Calibration file needs ≥ 4 values (fx fy cx cy), found {len(vals)}: {path}"
        )
    return torch.tensor(vals[:4], dtype=torch.float32)


def _cuda_sync_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _stats(arr: list[float]) -> tuple[float, float]:
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std())


def _print_report(
    vox_ms:   list[float],
    score_ms: list[float],
    dba_ms:   list[float],
    e2e_ms:   list[float],
    fps_target: float,
) -> None:
    n = len(e2e_ms)
    line = "=" * 60
    print(f"\n{line}")
    print(f"  Latency report  ({n} frames)")
    print(line)
    rows = [
        ("Voxelization  ", vox_ms),
        ("Score map CNN ", score_ms),
        ("DBA update    ", dba_ms),
        ("End-to-end    ", e2e_ms),
    ]
    for label, data in rows:
        if data:
            mu, sd = _stats(data)
            print(f"  {label}: {mu:7.2f} ± {sd:5.2f} ms")
        else:
            print(f"  {label}: n/a")
    if e2e_ms:
        eff_hz = 1000.0 / _stats(e2e_ms)[0]
        print(f"  {'Effective Hz':<18}: {eff_hz:.1f} Hz   (target: {fps_target:.0f} Hz)")
    print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DEVO realtime event odometry — generic event file input",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True,
                        help=".h5 (grouped/flat layout) or space-separated .txt")
    parser.add_argument("--calib",      required=True,
                        help="Calibration file: fx fy cx cy")
    parser.add_argument("--weights",    required=True,
                        help="DEVO checkpoint (.pth)")
    parser.add_argument("--fps-target", type=float, default=40.0,
                        help="Target voxelization rate (Hz)")
    parser.add_argument("--config",     default=None,
                        help="YAML config override (default: DEVO/config/default.yaml)")
    parser.add_argument("--H",          type=int, default=480, help="Sensor height px")
    parser.add_argument("--W",          type=int, default=640,  help="Sensor width px")
    parser.add_argument("--n-bins",     type=int, default=5,
                        help="Temporal bins per voxel (must match weights)")
    parser.add_argument("--out",        default="trajectory.txt",
                        help="Output TUM-format trajectory path")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device  = args.device
    H, W, B = args.H, args.W, args.n_bins

    # ---------------------------------------------------------------- config
    cfg_yaml = args.config or str(_DEVO_ROOT / "config" / "default.yaml")
    cfg.merge_from_file(cfg_yaml)
    # The Voxelizer already applies per-bin z-score normalisation.
    # Disable DEVO's second normalisation pass to avoid double-normalising.
    cfg.defrost()
    cfg.NORM = "none"
    cfg.freeze()
    print(f"Config  : {cfg_yaml}  (NORM overridden → 'none')")

    # -------------------------------------------------------------- calib
    intrinsics = _load_calib(args.calib).to(device)
    print(f"Calib   : fx={intrinsics[0]:.2f}  fy={intrinsics[1]:.2f}  "
          f"cx={intrinsics[2]:.2f}  cy={intrinsics[3]:.2f}")

    # ----------------------------------------------- Step 1: load events
    print(f"\nLoading events  →  {args.data}")
    raw_src = make_source(args.data)
    x_all, y_all, t_all, p_all = raw_src.get_batch()
    raw_src.close()

    n_events   = len(x_all)
    t_span_us  = float(t_all[-1] - t_all[0])
    t_span_sec = t_span_us / 1e6

    # Compute event-count window that approximates the fps target.
    # Assumes roughly uniform event density; accurate enough for offline files.
    dt_us       = 1e6 / args.fps_target
    window_size = max(100, int(round(n_events * dt_us / t_span_us)))

    print(f"  {n_events:>12,} events  |  {t_span_sec:.2f} s  "
          f"|  window ≈ {window_size:,} events  "
          f"|  ~{n_events // window_size} frames @ {args.fps_target:.0f} Hz")

    # ----------------------------------------- Step 2: RealtimePipeline
    # Wrap pre-loaded arrays so the pipeline reads from memory, not disk.
    array_src = _ArraySource(x_all, y_all, t_all, p_all)

    # Monkey-patch a single Voxelizer instance to capture per-call build time.
    voxelizer   = Voxelizer(H=H, W=W, n_bins=B, device=device)
    vox_times_ms: list[float] = []
    _orig_build  = voxelizer.build

    def _timed_build(x, y, t, p):          # same signature as Voxelizer.build
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        vox = _orig_build(x, y, t, p)
        _cuda_sync_if_available()
        vox_times_ms.append((time.perf_counter() - t0) * 1e3)
        return vox

    voxelizer.build = _timed_build          # type: ignore[method-assign]

    pipeline = RealtimePipeline(
        source      = array_src,
        voxelizer   = voxelizer,
        window_size = window_size,
        stride      = window_size,          # non-overlapping windows
    )

    # ------- Score map CNN — separate instance for latency probe only ------
    # DEVO runs its own internal scorer.  We instantiate one here solely to
    # measure score-map forward-pass latency without touching DEVO internals.
    scorer = PatchSelector(bins=B).eval().to(device)
    if device != "cpu":
        scorer = scorer.half()
    score_times_ms: list[float] = []

    # ------------------------------------------- Step 3: DEVO tracking
    print(f"\nInitialising DEVO  (weights: {args.weights})")
    slam = _TimedDEVO(cfg, args.weights, evs=True, ht=H, wd=W)

    e2e_times_ms: list[float] = []
    print(f"Tracking  [device={device}]\n{'─'*60}")

    for frame_idx, (voxel, t_mid) in enumerate(pipeline.run()):
        # voxel : (B, H, W) float32 on `device`  — already z-score normalised
        # t_mid : midpoint timestamp of the event window (raw units from source)

        _cuda_sync_if_available()
        t_frame = time.perf_counter()

        # --- Score map CNN timing (parallel probe, no impact on DEVO) ------
        _cuda_sync_if_available()
        t0 = time.perf_counter()
        with torch.no_grad():
            scorer(voxel.unsqueeze(0).half() if device != "cpu" else voxel.unsqueeze(0))
        _cuda_sync_if_available()
        score_times_ms.append((time.perf_counter() - t0) * 1e3)

        # --- DEVO tracking step  -------------------------------------------
        # __call__(tstamp, image, intrinsics)
        # image must be (C, H, W); DEVO adds batch/seq dims internally.
        slam(t_mid, voxel, intrinsics)

        _cuda_sync_if_available()
        e2e_times_ms.append((time.perf_counter() - t_frame) * 1e3)

        if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
            print(f"  [{frame_idx+1:4d}]  "
                  f"vox {vox_times_ms[-1]:5.1f} ms  "
                  f"score {score_times_ms[-1]:5.1f} ms  "
                  f"e2e {e2e_times_ms[-1]:6.1f} ms")

    total_frames = frame_idx + 1 if e2e_times_ms else 0

    # --------------------------------------------- final BA refinement
    print(f"\nFinal refinement (12 BA steps)…")
    slam._record_dba = False        # keep post-loop DBA out of latency stats
    for _ in range(12):
        slam.update()

    # ------------------------------------------- trajectory extraction
    poses, tstamps = slam.terminate()
    # poses   : (N, 7)  [tx ty tz qx qy qz qw] world-to-camera, inverted
    # tstamps : (N,)    raw timestamps (same units as t_mid input)

    # ----------------------------------------------- Step 4: TUM output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_trajectory_tum_format((poses, tstamps), str(out_path))
    # save_trajectory_tum_format writes:  timestamp tx ty tz qx qy qz qw

    # ------------------------------------------------ Step 5: report
    _print_report(
        vox_ms   = vox_times_ms,
        score_ms = score_times_ms,
        dba_ms   = slam._dba_times_ms,
        e2e_ms   = e2e_times_ms,
        fps_target = args.fps_target,
    )

    print(f"\nTrajectory → {out_path.resolve()}  ({total_frames} poses)")


if __name__ == "__main__":
    main()
