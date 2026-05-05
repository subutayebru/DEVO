"""
Patch-selection score-map CNN for DEVO event-camera odometry.

Architecture (Section 3.2 of the DEVO paper, with padding=1 to preserve
spatial resolution through the conv stack):

    Conv2d(5→8,   k=3, pad=1) + ReLU
    Conv2d(8→16,  k=3, pad=1) + ReLU
    Conv2d(16→32, k=3, pad=1) + ReLU
    Conv2d(32→1,  k=3, pad=1) + MaxPool2d(4, 4) + Sigmoid

Input  : (B, 5, H, W)   float16 voxel grid
Output : (B, 1, H//4, W//4)   float16 score map in [0, 1]

Checkpoint key mapping from the full eVONet checkpoint
------------------------------------------------------
The DEVO.pth checkpoint stores the scorer weights under the path:

    patchify.scorer.scorer.<idx>.<weight|bias>

where <idx> is the Sequential index of each Conv2d layer:
    0  → Conv2d(5,  8,  3)   weight (8,  5, 3, 3)  bias (8,)
    2  → Conv2d(8,  16, 3)   weight (16, 8, 3, 3)  bias (16,)
    4  → Conv2d(16, 32, 3)   weight (32,16, 3, 3)  bias (32,)
    6  → Conv2d(32,  1, 3)   weight (1, 32, 3, 3)  bias (1,)

(ReLU at 1,3,5 and MaxPool2d at 7 have no parameters.)

Our PatchSelector uses the same Sequential index layout (indices 0,2,4,6
for Conv2d) stored under the attribute name ``net``, so the remapping is:

    patchify.scorer.scorer.X.weight  →  net.X.weight
    patchify.scorer.scorer.X.bias    →  net.X.bias

The Conv2d weight *shapes* are identical regardless of padding, so weights
transfer directly.  Sigmoid is new (fused in) and has no learnable params.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class PatchSelector(nn.Module):
    """Score-map CNN for event-based patch selection.

    Parameters
    ----------
    bins : int
        Number of voxel time-bins (input channels).  Default: 5 (DEVO).
    """

    def __init__(self, bins: int = 5) -> None:
        super().__init__()
        self.bins = bins
        # Sequential indices mirror DEVO's Scorer so checkpoint weights
        # load without shape conflicts (Conv2d at 0,2,4,6; ReLU at 1,3,5;
        # MaxPool at 7; Sigmoid at 8 is new / has no parameters).
        self.net = nn.Sequential(
            nn.Conv2d(bins, 8,  kernel_size=3, padding=1),  # 0
            nn.ReLU(inplace=True),                           # 1
            nn.Conv2d(8,   16, kernel_size=3, padding=1),   # 2
            nn.ReLU(inplace=True),                           # 3
            nn.Conv2d(16,  32, kernel_size=3, padding=1),   # 4
            nn.ReLU(inplace=True),                           # 5
            nn.Conv2d(32,  1,  kernel_size=3, padding=1),   # 6
            nn.MaxPool2d(kernel_size=4, stride=4),           # 7
            nn.Sigmoid(),                                    # 8  (fused)
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the score-map CNN.

        Parameters
        ----------
        x : torch.Tensor, shape (B, bins, H, W)

        Returns
        -------
        torch.Tensor, shape (B, 1, H//4, W//4), values in [0, 1]
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

# Keys that belong to the scorer inside the full eVONet checkpoint.
# Derived from DEVO/devo/selector.py::Scorer and DEVO/devo/enet.py::Patchifier.
DEVO_SCORER_KEYS: list[str] = [
    "patchify.scorer.scorer.0.weight",   # Conv2d(5→8)
    "patchify.scorer.scorer.0.bias",
    "patchify.scorer.scorer.2.weight",   # Conv2d(8→16)
    "patchify.scorer.scorer.2.bias",
    "patchify.scorer.scorer.4.weight",   # Conv2d(16→32)
    "patchify.scorer.scorer.4.bias",
    "patchify.scorer.scorer.6.weight",   # Conv2d(32→1)
    "patchify.scorer.scorer.6.bias",
]

# Prefix to strip when remapping to our ``net.*`` layout.
_DEVO_PREFIX = "patchify.scorer.scorer."


def _extract_scorer_state(raw: dict) -> OrderedDict:
    """Pull scorer weights out of a full DEVO checkpoint state-dict.

    Handles:
      - Full eVONet checkpoint  (key  'model_state_dict')
      - Legacy DataParallel     (keys prefixed with 'module.')
      - Bare scorer state-dict  (keys already 'net.*' or '0.*')
    """
    # Unwrap model_state_dict wrapper
    state: dict = raw.get("model_state_dict", raw)

    # Strip DataParallel prefix and drop deprecated lmbda key
    state = {
        k.replace("module.", ""): v
        for k, v in state.items()
        if "update.lmbda" not in k
    }

    # Extract scorer sub-keys and remap to our ``net.*`` layout
    scorer_state = OrderedDict()
    for k, v in state.items():
        if k.startswith(_DEVO_PREFIX):
            new_key = "net." + k[len(_DEVO_PREFIX):]   # e.g. net.0.weight
            scorer_state[new_key] = v

    if not scorer_state:
        # Already a bare state-dict (e.g. a previously exported scorer)
        scorer_state = OrderedDict(state)

    return scorer_state


def load_from_checkpoint(
    checkpoint_path: str | os.PathLike,
    device: str | torch.device = "cpu",
    bins: int = 5,
) -> PatchSelector:
    """Instantiate PatchSelector and load weights from a DEVO checkpoint.

    Parameters
    ----------
    checkpoint_path : path-like
        Path to ``DEVO.pth`` or any exported PatchSelector state-dict.
    device : str or torch.device
    bins : int

    Returns
    -------
    PatchSelector  (eval mode, on ``device``)
    """
    raw = torch.load(checkpoint_path, map_location=device)
    state = _extract_scorer_state(raw)

    model = PatchSelector(bins=bins)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_from_checkpoint] missing keys : {missing}")
    if unexpected:
        print(f"[load_from_checkpoint] unexpected keys: {unexpected}")

    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def export_jit(
    model: PatchSelector,
    save_path: str | os.PathLike,
    use_fp16: bool = True,
) -> torch.jit.ScriptModule:
    """Convert PatchSelector to fp16, TorchScript it, and save.

    Parameters
    ----------
    model : PatchSelector  (will be put in eval mode)
    save_path : destination ``.pt`` file
    use_fp16 : if True, calls .half() before scripting

    Returns
    -------
    torch.jit.ScriptModule
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if use_fp16:
        model = model.half()

    scripted = torch.jit.script(model)
    scripted.save(str(save_path))
    print(f"Saved TorchScript model → {save_path}")
    return scripted


# ---------------------------------------------------------------------------
# CLI: inspect checkpoint + export
# ---------------------------------------------------------------------------

def _inspect_checkpoint(path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Checkpoint: {path}")
    print(f"{'='*60}")
    raw = torch.load(path, map_location="cpu")
    state: dict = raw.get("model_state_dict", raw)
    state = {k.replace("module.", ""): v for k, v in state.items()
             if "update.lmbda" not in k}

    scorer_keys = [k for k in state if k.startswith("patchify.scorer.")]
    print(f"\nPatch-selection sub-network keys ({len(scorer_keys)} tensors):")
    for k in scorer_keys:
        print(f"  {k:55s}  {tuple(state[k].shape)}")

    other = [k for k in state if not k.startswith("patchify.scorer.")]
    print(f"\nOther keys in checkpoint: {len(other)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect + export DEVO PatchSelector")
    parser.add_argument("--checkpoint", default="DEVO.pth",
                        help="Path to DEVO.pth (or any eVONet checkpoint)")
    parser.add_argument("--out", default="models/patch_selector.pt",
                        help="Output TorchScript path")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Keep fp32 instead of converting to fp16")
    parser.add_argument("--random-init", action="store_true",
                        help="Skip checkpoint loading; export randomly initialised weights")
    args = parser.parse_args()

    # Step 1 – inspect
    if not args.random_init:
        if not Path(args.checkpoint).exists():
            print(f"[warn] Checkpoint not found at '{args.checkpoint}'.")
            print("[info] Expected scorer keys in DEVO.pth:")
            for k in DEVO_SCORER_KEYS:
                print(f"  {k}")
            print("\nFalling back to random initialisation for export.\n")
            args.random_init = True
        else:
            _inspect_checkpoint(args.checkpoint)

    # Step 2 – build model
    if args.random_init:
        model = PatchSelector(bins=5)
        print("[info] Using randomly initialised PatchSelector.")
    else:
        model = load_from_checkpoint(args.checkpoint, device="cpu")

    # Step 3 – export
    export_jit(model, args.out, use_fp16=not args.no_fp16)
    print("Done.")
