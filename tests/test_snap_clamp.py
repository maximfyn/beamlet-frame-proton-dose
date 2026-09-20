"""The three-way clamp: 0 below, the threshold in the band, untouched above.

**Only `0` or `>= c` is legal**, so for a voxel whose truth is `t`, predicting 0
costs `t` and predicting the threshold costs `|threshold - t|`. Clamping the whole
band down is cheap at the bottom and expensive at the top -- against a truth that
keeps its sub-cutoff dose, which is what the middle arm is for. The challenge's
labels are thresholded, so the submitted model disables it (`SNAP_ALPHA=0`); these
tests pin the mechanism for when the truth is not thresholded.
**The default α is 0.15, not 0.5.** `idd` integrates laterally, so sub-cutoff
errors cancel and the optimum is count-matching rather than pointwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.geometry import BeamletGrid, VolumeGeometry
from models.predictor import (SNAP_ALPHA, BeamletRequest, DosePredictor,
                              apply_cutoff_policy, clamp_threshold)

pytest.importorskip("torch")

# Cutoffs of the order the challenge serves (~1e-6).
CUTOFFS = [6.732e-07, 9.445281373494074e-07, 1.069e-06, 1.872e-06]


def _case():
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(22, 18, 14))
    ct = np.full(geom.shape, -1024.0, dtype=np.float32)
    ct[5:18, 3:15, 4:11] = 0.0
    return ct, geom


def _predict(cutoff, *, snap_alpha=SNAP_ALPHA):
    ct, geom = _case()
    p = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4),
                      snap_alpha=snap_alpha)
    reqs = [BeamletRequest(ray_source=(-50.0, 8.0, 8.0), ray_target=(50.0, 8.0, 8.0),
                           output_file_idx=0, idx_in_output=0, energy=100.0,
                           minimum_cutoff=cutoff)]
    return p.predict(ct, geom, reqs)[0]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_nothing_survives_between_zero_and_the_cutoff_in_float64(cutoff):
    """THE BAN-RISK INVARIANT, and it must be checked AFTER the float32 cast.

    Their `extract_beam` upcasts to float64 before testing `0 < v < c`, so a value
    that looks safe in float32 can violate it in float64 — the gap
    `clamp_threshold` exists to close. ⇒ read the emitted array back
    as float64 and assert the open interval is empty.
    """
    out = _predict(cutoff).astype(np.float64)
    bad = out[(out > 0.0) & (out < cutoff)]
    assert bad.size == 0, f"{bad.size} voxels in (0, c); e.g. {bad[:3]!r}"


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_the_three_arms(cutoff):
    """Unit-tested on the policy, because the predictor's stub emits a 1.4x
    dynamic range against a real beamlet's ~1e3 and so never lands in the band."""
    import torch

    t = clamp_threshold(cutoff)
    a = SNAP_ALPHA
    v = torch.tensor([0.0, 0.5 * a * t, 0.99 * a * t,      # -> zero
                      a * t, 0.5 * (a * t + t), 0.999 * t,  # -> t
                      t, 2 * t, 1e-3], dtype=torch.float32)
    out = apply_cutoff_policy(v, cutoff, a).numpy().astype(np.float64)
    assert (out[:3] == 0.0).all(), out[:3]
    assert np.allclose(out[3:6], np.float32(t)), out[3:6]
    assert np.array_equal(out[6:], v.numpy()[6:].astype(np.float64))
    # and the whole result obeys the platform's rule, read back in float64
    assert out[(out > 0.0) & (out < cutoff)].size == 0


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_alpha_zero_is_the_old_clamp_and_snaps_nothing(cutoff):
    """The switch to throw if a sub-cutoff mass head ever makes the band real."""
    import torch

    t = clamp_threshold(cutoff)
    v = torch.tensor([0.5 * t, 0.999 * t, t, 2 * t], dtype=torch.float32)
    out = apply_cutoff_policy(v, cutoff, 0.0).numpy().astype(np.float64)
    assert (out[:2] == 0.0).all()
    assert np.array_equal(out[2:], v.numpy()[2:].astype(np.float64))
    assert out[(out > 0.0) & (out < cutoff)].size == 0


def test_a_zero_cutoff_is_a_no_op():
    import torch

    v = torch.tensor([0.0, 1e-9, 1.0], dtype=torch.float32)
    assert torch.equal(apply_cutoff_policy(v, 0.0), v)


def test_alpha_is_the_measured_optimum_not_a_half():
    """`c/2` is the pointwise answer and `idd` is not pointwise; the measured
    plateau (against an unthresholded truth) is 0.10–0.20, and 0.5 is off it."""
    assert 0.10 <= SNAP_ALPHA <= 0.20


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_an_out_of_range_alpha_is_refused(bad):
    with pytest.raises(ValueError, match="snap_alpha"):
        DosePredictor(grid=BeamletGrid(), snap_alpha=bad)


def test_snap_alpha_is_read_per_call_not_captured_at_construction():
    """What licenses a CELL being a copy of one loaded predictor.

    Scoring several (body_mask, snap_alpha)
    cells of one checkpoint works by `copy.copy`-ing a single loaded `DosePredictor`
    and setting the two attributes on each copy -- so the network is uploaded
    once and a cell costs a forward pass rather than a second copy on the card.
    That is only valid while `predict` reads `self.snap_alpha` at CALL time.
    Capture it at construction and every cell would silently score the value the
    base predictor happened to be built with, with nothing to notice: the arms
    differ only inside `[alpha*t, t)`, which is a thin band of small numbers.
    """
    import copy

    import models.predictor as P

    seen = []
    real = P.apply_cutoff_policy

    def spy(block, cutoff, snap_alpha=P.SNAP_ALPHA):
        seen.append(float(snap_alpha))
        return real(block, cutoff, snap_alpha)

    ct, geom = _case()
    base = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    reqs = [BeamletRequest(ray_source=(-50.0, 8.0, 8.0), ray_target=(50.0, 8.0, 8.0),
                           output_file_idx=0, idx_in_output=0, energy=100.0,
                           minimum_cutoff=CUTOFFS[2])]
    P.apply_cutoff_policy = spy
    try:
        for alpha in (0.0, SNAP_ALPHA, 0.4):
            cell = copy.copy(base)
            cell.snap_alpha = alpha
            cell.predict(ct, geom, reqs)
    finally:
        P.apply_cutoff_policy = real

    assert seen == [0.0, SNAP_ALPHA, 0.4], seen


def test_the_two_cell_arms_are_a_real_difference_on_a_real_valued_block():
    """The predictor's stub CANNOT show this and the cells will look identical.

    Its output spans ~1.4x where a real beamlet spans ~1e3, so nothing it emits
    lands in `[alpha*t, t)` and both arms return byte-identical volumes. That is
    an artefact of the stub, not agreement between the arms -- so the difference
    is asserted here on a block with a real beamlet's range, and a cell run that
    shows no difference on real data is a finding rather than a null.
    """
    import torch

    cutoff = CUTOFFS[2]
    block = torch.linspace(0.0, 3.0 * cutoff, 512)
    three_way = apply_cutoff_policy(block, cutoff, SNAP_ALPHA)
    clamp_down = apply_cutoff_policy(block, cutoff, 0.0)
    assert not torch.equal(three_way, clamp_down)
    t = clamp_threshold(cutoff)
    # the band is exactly where they differ, and nowhere else
    differs = three_way != clamp_down
    assert torch.all(block[differs] >= SNAP_ALPHA * t)
    assert torch.all(block[differs] < t)
