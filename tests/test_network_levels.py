"""Every `--levels` value reachable from the CLI must build and run.

`levels=5` shipped broken: `DoseUNet`'s default `strides` tuple had four
entries, so `build_network(levels=5)` raised `need 5 strides, got 4`. The
benchmark that motivated going deeper passed a five-element tuple explicitly and
therefore never exercised the path `scripts/train/train_doserad.py --levels 5` takes.
A six-GPU production run died on it seconds after initialising.

These tests use the *defaults*, because the defaults are what the CLI uses.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.geometry import BeamletGrid
from models.network import DoseUNet, build_network

GRID = BeamletGrid().shape  # (384, 64, 16)


@pytest.mark.parametrize("levels", [2, 3, 4, 5])
def test_default_strides_support_every_level(levels: int):
    """No explicit strides -- exactly how the training CLI constructs it."""
    net = build_network("unet", in_channels=2, base_features=8, levels=levels)
    assert net.levels == levels


@pytest.mark.parametrize("levels", [4, 5])
def test_forward_preserves_the_grid_shape(levels: int):
    """Anisotropic pooling must not collapse the 16-voxel v axis, and the
    decoder must return to full resolution -- render_to_volume assumes it."""
    net = build_network("unet", in_channels=2, base_features=4, levels=levels).eval()
    small = (64, 32, 16)  # same axis ratios, cheap enough for CPU
    with torch.inference_mode():
        out = net(torch.zeros(1, 2, *small))
    assert out.shape == (1, 1, *small)


def test_levels_beyond_available_strides_still_raises():
    """The guard itself must survive: asking for more levels than there are
    strides is a real error, not something to pad silently."""
    with pytest.raises(ValueError, match="strides"):
        DoseUNet(in_channels=2, base_features=4, levels=6)


def test_output_is_non_negative():
    """Softplus head: dose cannot be negative anywhere."""
    net = build_network("unet", in_channels=2, base_features=4, levels=5).eval()
    with torch.inference_mode():
        out = net(torch.randn(1, 2, 64, 32, 16))
    assert float(out.min()) >= 0.0
