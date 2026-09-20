"""The IDD loss term, and the checkpoint fields that keep old runs loadable.

``idd_distance_boxframe`` is a *proxy* -- the ranked number is computed after
rendering onto the CT grid. What these tests pin is that it is the same
arithmetic as the real metric applied to the beamlet frame, because a proxy that
quietly measures something else is worse than no proxy: it still moves, and
nobody rechecks a number that looks right.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "train_doserad", ROOT / "scripts" / "train" / "train_doserad.py")
trainer = importlib.util.module_from_spec(spec)
sys.modules["train_doserad"] = trainer
spec.loader.exec_module(trainer)

from evaluation.metrics import compute_idd_curve  # noqa: E402


def test_a_perfect_prediction_scores_zero():
    torch.manual_seed(0)
    target = torch.rand(3, 1, 16, 8, 4)
    assert float(trainer.idd_distance_boxframe(target, target)) == pytest.approx(0.0, abs=1e-7)


def test_it_is_blind_to_lateral_error_that_cancels_in_the_sum():
    """The property that makes it worth adding, stated as its own limitation.

    Moving dose sideways within a depth slice leaves the IDD untouched. That is
    exactly why it is an *additional* term and not a replacement: it controls the
    projection ``beam_mae`` cannot, and ``beam_mae`` controls the voxels it
    cannot.
    """
    target = torch.zeros(1, 1, 4, 4, 2)
    target[0, 0, 2, 1, 0] = 1.0
    shifted = torch.zeros_like(target)
    shifted[0, 0, 2, 3, 1] = 1.0

    assert float(trainer.idd_distance_boxframe(shifted, target)) == pytest.approx(0.0, abs=1e-7)
    assert float(trainer.beam_mae_proxy(shifted, target)) > 0.0


def test_it_matches_the_official_arithmetic_on_a_ray_aligned_box():
    """Against ``evaluation/metrics.py``, on the one geometry where they coincide.

    In the beamlet frame the box is already ray-aligned, so the official
    routine's rotate-and-resample reduces to a plain lateral sum. Laying the box
    on a CT-shaped grid along +x with unit spacing makes the two computable side
    by side -- which is the only way to show the proxy is the metric's arithmetic
    rather than something that merely resembles it.
    """
    torch.manual_seed(0)
    n_depth, n_u = 24, 12
    target = torch.rand(1, 1, n_depth, n_u, 1)

    # (z, y, x) with the beam along +x: the official code sums z out first, so
    # y is the lateral axis and x is depth.
    volume = target[0, 0, :, :, 0].permute(1, 0).reshape(1, n_u, n_depth).double().numpy()
    theirs = compute_idd_curve(volume, direction=np.array([1.0, 0.0, 0.0]),
                               spacing=(1.0, 1.0, 1.0))
    ours = target.sum(dim=(3, 4))[0, 0].double().numpy()

    # Their curve lives on a square grid sized to the in-plane diagonal, so it
    # is longer than the box and zero-padded either side. Line the two up on
    # their centres of mass and the *values* must agree: same lateral sum, one
    # resampling apart.
    start = int(round(np.argmax(theirs) - np.argmax(ours)))
    overlap = theirs[start:start + n_depth]
    assert overlap.shape == ours.shape
    assert np.abs(overlap - ours).max() < 1e-9 * max(ours.max(), 1.0) + 1e-6

    # And the padding is only zeros, so the sums agree too -- which is what
    # makes the RMS a rescaling of ours rather than a different quantity. The
    # tolerance is the resample's own float arithmetic (~1e-8 relative), not a
    # margin for disagreement: a real difference here would be a lateral axis
    # partly summed away, which is percent-scale.
    assert theirs.sum() == pytest.approx(ours.sum(), rel=1e-6)


def test_total_loss_reports_its_parts_and_respects_the_weight():
    torch.manual_seed(0)
    target = torch.rand(2, 1, 8, 4, 4)
    prediction = torch.rand(2, 1, 8, 4, 4)

    off, parts_off = trainer.total_loss(prediction, target, 0.0)
    on, parts_on = trainer.total_loss(prediction, target, 2.0)

    assert set(parts_off) == {"loss_dose"}
    assert set(parts_on) == {"loss_dose", "loss_idd_boxframe"}
    assert float(on) == pytest.approx(
        float(off) + 2.0 * parts_on["loss_idd_boxframe"], rel=1e-6)


def test_the_loss_term_is_differentiable_through_a_lateral_sum():
    torch.manual_seed(0)
    target = torch.rand(1, 1, 8, 4, 4)
    prediction = torch.rand(1, 1, 8, 4, 4, requires_grad=True)
    trainer.idd_distance_boxframe(prediction, target).backward()
    assert prediction.grad is not None
    assert float(prediction.grad.abs().sum()) > 0.0


def test_the_new_emissions_survive_the_naming_guard():
    """`idd_distance` is a leaderboard name; the beamlet-frame one must be qualified."""
    trainer.assert_boxframe(["val/idd_distance_boxframe", "train/loss_idd_boxframe"])
    with pytest.raises(ValueError, match="leaderboard metric name"):
        trainer.assert_boxframe(["val/idd_distance"])
