"""`rendered_bias_loss` -- and the one property that makes it worth a retrain.

**THE TEST THAT MATTERS IS THE THIRD ONE.** The box-frame term already passes
its own tests (`tests/test_bias_loss.py`): it is selective for coherent error and
blind to scatter. It then failed in production because it was computed in the
wrong FRAME. So a test
suite that only re-checks selectivity would pass on the arm we already know
fails. What has to be pinned is that the rendered term is **not** the box term --
that `render . precompensate` really does move the number -- because if the two
agree, this retrain is the previous one with a new flag.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from models.geometry import BeamletGrid  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_doserad", REPO / "scripts" / "train" / "train_doserad.py")
TD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TD)


GRID = BeamletGrid(n_depth=64, n_lat_u=16, n_lat_v=8)


def _geometry(n: int, entry: float = 376.0) -> dict:
    """One straight ray per sample, aimed through the middle of a small volume.

    ``entry_depth`` is measured from ``ray_source``, not from the volume face:
    ``render_coordinates`` builds ``box_origin = ray_source + entry * direction``.
    The source sits 400 mm out along -y and the volume spans y in [-24, +24], so
    376 is what puts the box's first depth sample on the near face. Getting this
    wrong does not raise -- ``coords`` comes back ``None`` and every loss reads a
    clean 0.0, which is the shape of a test that passes while measuring nothing.
    """
    shape = (24, 48, 48)
    origin = np.array([-24.0, -24.0, -36.0])
    spacing = np.array([1.0, 1.0, 3.0])
    rays = []
    for _ in range(n):
        rays.append([0.0, -400.0, 0.0, 0.0, 0.0, 0.0, entry])
    return {
        "ray": torch.tensor(rays, dtype=torch.float64),
        "vol_origin": torch.tensor(np.tile(origin, (n, 1))),
        "vol_spacing": torch.tensor(np.tile(spacing, (n, 1))),
        "vol_shape": torch.tensor(np.tile(np.array(shape), (n, 1))),
    }


def _target(n: int) -> torch.Tensor:
    """A Bragg-ish box: rising plateau then a sharp peak, constant laterally."""
    depth = torch.linspace(0.0, 1.0, GRID.n_depth)
    profile = 0.25 + 0.15 * depth + torch.exp(-((depth - 0.75) ** 2) / 0.0015)
    t = profile[None, None, :, None, None].expand(
        n, 1, GRID.n_depth, GRID.n_lat_u, GRID.n_lat_v)
    return t.contiguous().to(torch.float32)


def test_zero_residual_is_zero_loss():
    t = _target(3)
    out = TD.rendered_bias_loss(t.clone(), t, _geometry(3), GRID)
    assert float(out) == pytest.approx(0.0, abs=1e-12)


def test_coherent_offset_scores_far_above_matched_scatter():
    """The property the whole term exists for, re-checked in the new frame.

    Pinned as a RATIO, not a threshold: the absolute value depends on the box
    and the ray, and a threshold would encode this fixture rather than the term.
    """
    t = _target(4)
    torch.manual_seed(0)
    hot = t * 1.01
    noisy = t + torch.randn_like(t) * (0.01 * t.amax())
    geo = _geometry(4)
    bias = float(TD.rendered_bias_loss(hot, t, geo, GRID))
    scatter = float(TD.rendered_bias_loss(noisy, t, geo, GRID))
    assert bias > 50 * scatter, f"bias {bias:.3e} vs scatter {scatter:.3e}"


def test_rendered_frame_is_not_the_box_frame():
    """THE LOAD-BEARING ONE. If these agree, the retrain is the old retrain.

    A residual concentrated in the Bragg peak is exactly where `precompensate`
    puts its amplification (`amp_max` 1.207, and it *concentrates* into the
    peak rather than spreading), so it is where the two frames must differ most.
    """
    t = _target(2)
    resid = torch.zeros_like(t)
    peak = int(0.75 * GRID.n_depth)
    resid[:, :, peak - 1: peak + 2] = 0.02 * float(t.amax())
    pred = t + resid
    box = float(TD.conditional_bias_loss(pred, t))
    rendered = float(TD.rendered_bias_loss(pred, t, _geometry(2), GRID))
    assert box > 0 and rendered > 0
    assert abs(rendered - box) / box > 0.05, (
        f"rendered {rendered:.6e} and box {box:.6e} agree to "
        f"{100 * abs(rendered - box) / box:.2f}% -- the frame change is a no-op "
        "on this fixture, so it would be a no-op in training too"
    )


def test_missing_geometry_raises_rather_than_falling_back():
    """A silent fallback would rerun the refuted arm under the new name."""
    t = _target(2)
    with pytest.raises(ValueError, match="rendered"):
        TD.total_loss(t.clone(), t, idd_weight=0.0, bias_weight=1.0,
                      bias_frame="rendered", geometry=None, grid=None)


def test_box_missing_the_volume_is_skipped_not_counted_as_unbiased():
    """A ray that never enters the volume must not dilute the mean toward zero."""
    t = _target(2)
    hot = t * 1.01
    near = _geometry(2)
    # Push the entry depth far past the far face so the box lands outside.
    far = _geometry(2, entry=10_000.0)
    on = float(TD.rendered_bias_loss(hot, t, near, GRID))
    off = float(TD.rendered_bias_loss(hot, t, far, GRID))
    assert on > 0
    assert off == pytest.approx(0.0, abs=1e-12), (
        "a fully-missing batch should return 0 by having counted nothing, not by "
        "averaging real bias against zeros"
    )


def test_flip_and_geometry_are_refused_together():
    """The mirrored box with an unmirrored ray renders into the wrong place."""
    from models.dataset import BeamletDataset
    with pytest.raises(ValueError, match="augment_flip"):
        BeamletDataset.__init__(
            object.__new__(BeamletDataset), root=".", patients=[],
            augment_flip=True, with_geometry=True)
