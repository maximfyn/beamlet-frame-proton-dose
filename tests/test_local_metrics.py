"""Tests for the two faithfully-portable DoseRAD metrics.

Two tiers, and the second is the one that matters:

1. **Analytic fixtures** -- cases whose answer is known without reference to
   any implementation (identical volumes score perfectly, an empty beam is
   ``nan``, a pure scaling of the IDD is invariant, and so on). These catch
   gross errors and run anywhere.
2. **Equivalence against the official implementation** -- exact equality, no
   tolerance, against ``github.com/DoseRAD2026/evaluation-setup``. It has no
   licence, so it is **imported from a clone and never vendored**.

Tier 1 alone would not catch a faithful-looking port that disagrees with the
scorer, which is the failure this whole track exists to prevent. Tier 2 runs
whenever a clone is found at ``external/evaluation-setup`` (or wherever
``DOSERAD_EVAL_REFERENCE`` points); without one, its 17 tests skip. They last
passed against commit ``fcb42a5f``::

    git clone https://github.com/DoseRAD2026/evaluation-setup external/evaluation-setup
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (  # noqa: E402
    aggregate_cohort,
    aggregate_plan,
    beam_direction,
    compute_idd_curve,
    idd_curve_distance,
    masked_beam_mae,
)

SPACING = (1.0, 1.0, 3.0)  # proton-CT, sitk (x, y, z) order


def _beam(shape=(12, 24, 28), seed=0):
    """A plausible beam: a bright core on a dim background, never negative."""
    rng = np.random.default_rng(seed)
    vol = rng.random(shape) * 0.05
    vol[shape[0] // 2 - 1 : shape[0] // 2 + 2, 6:14, 8:20] += 1.0
    return vol


# ---------------------------------------------------------------------------
# Tier 1 -- analytic
# ---------------------------------------------------------------------------


def test_identical_volumes_score_perfectly():
    gt = _beam()
    assert masked_beam_mae(gt, gt) == 0.0
    assert idd_curve_distance(gt, gt, np.array([1.0, 0.0, 0.0]), SPACING) == 0.0


def test_empty_ground_truth_is_nan_not_zero():
    """An all-zero GT beam must not silently score a perfect 0.0."""
    empty = np.zeros((8, 8, 8))
    assert math.isnan(masked_beam_mae(empty, empty))
    assert math.isnan(idd_curve_distance(_beam((8, 8, 8)), empty, np.array([1.0, 0.0, 0.0]), SPACING))


def test_mae_is_normalised_by_gt_peak():
    """A constant offset inside the mask gives that offset over the peak."""
    gt = np.zeros((4, 8, 8))
    gt[:, 2:6, 2:6] = 4.0  # peak 4.0; the mask is exactly this block
    pred = gt + 0.5
    assert masked_beam_mae(pred, gt) == pytest.approx(0.5 / 4.0)


def test_mae_ignores_voxels_below_the_ten_percent_mask():
    gt = np.zeros((4, 8, 8))
    gt[:, 2:6, 2:6] = 10.0
    pred = gt.copy()
    pred[:, 0, 0] = 999.0  # far outside the high-dose region
    assert masked_beam_mae(pred, gt) == 0.0


def test_mae_mask_boundary_is_inclusive():
    """`>= 0.1 * peak` -- a voxel sitting exactly on the boundary is scored."""
    gt = np.zeros((2, 4, 4))
    gt[0, 0, 0] = 10.0
    gt[0, 1, 1] = 1.0  # exactly 10 % of the peak
    pred = gt.copy()
    pred[0, 1, 1] = 2.0  # error of 1.0 at the boundary voxel
    # two masked voxels, errors 0.0 and 1.0, normalised by the peak of 10
    assert masked_beam_mae(pred, gt) == pytest.approx((1.0 / 2) / 10.0)


def test_idd_rejects_out_of_plane_beams():
    """A z component invalidates the resampling, so it must raise, not degrade."""
    with pytest.raises(ValueError, match="transverse"):
        compute_idd_curve(_beam(), np.array([0.0, 0.7, 0.7]), SPACING)


def test_idd_length_is_angle_independent():
    """The curve spans the in-plane diagonal, so every gantry angle agrees."""
    vol = _beam()
    lengths = {
        len(compute_idd_curve(vol, np.array([math.cos(a), math.sin(a), 0.0]), SPACING))
        for a in np.linspace(0.0, 2 * math.pi, 12, endpoint=False)
    }
    assert len(lengths) == 1


def test_idd_distance_is_invariant_to_a_shared_rescale():
    """Both curves are normalised by the GT peak, so a common factor cancels."""
    gt, pred = _beam(seed=1), _beam(seed=2)
    d = np.array([0.0, 1.0, 0.0])
    baseline = idd_curve_distance(pred, gt, d, SPACING)
    scaled = idd_curve_distance(pred * 7.5, gt * 7.5, d, SPACING)
    assert scaled == pytest.approx(baseline, rel=1e-12)


def test_idd_distance_grows_with_a_range_shift():
    """The metric exists to see distal falloff error; a shift must register."""
    gt = _beam()
    shifted = np.roll(gt, 3, axis=2)
    d = np.array([1.0, 0.0, 0.0])
    assert idd_curve_distance(shifted, gt, d, SPACING) > idd_curve_distance(gt, gt, d, SPACING)


def test_beam_direction_is_a_unit_vector_from_source_to_target():
    v = beam_direction((0.0, -1000.0, 34.0), (0.0, -62.0, 34.0))
    assert np.allclose(v, [0.0, 1.0, 0.0])
    assert np.linalg.norm(v) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tier 1 -- aggregation shape
# ---------------------------------------------------------------------------


def test_plan_aggregation_drops_nan_beams():
    plan = aggregate_plan([0.1, float("nan"), 0.3], [0.2, float("nan"), 0.4])
    assert plan["beam_mae_mean"] == pytest.approx(0.2)
    assert plan["idd_distance_mean"] == pytest.approx(0.3)
    # num_beams counts beams attempted, not beams that scored
    assert plan["num_beams"] == 3


def test_cohort_mean_is_over_plans_not_over_beams():
    """The distinction that would silently shift the number.

    A 1000-beam plan scoring 0.01 and a 10-beam plan scoring 0.05 average to
    0.03, not to the beam-count-weighted 0.0104.
    """
    cases = [
        {"beam_mae_mean": 0.01, "idd_distance_mean": 0.01, "num_beams": 1000},
        {"beam_mae_mean": 0.05, "idd_distance_mean": 0.05, "num_beams": 10},
    ]
    assert aggregate_cohort(cases)["beam_mae"]["mean"] == pytest.approx(0.03)


def test_cohort_skips_nan_cases_without_dropping_the_metric():
    cases = [
        {"beam_mae_mean": 0.02, "idd_distance_mean": float("nan")},
        {"beam_mae_mean": float("nan"), "idd_distance_mean": 0.06},
    ]
    agg = aggregate_cohort(cases)
    assert agg["beam_mae"]["mean"] == pytest.approx(0.02)
    assert agg["beam_mae"]["n_cases"] == 1
    assert agg["idd_distance"]["mean"] == pytest.approx(0.06)


# ---------------------------------------------------------------------------
# Tier 2 -- exact equivalence with the official implementation
# ---------------------------------------------------------------------------

# A clone at external/evaluation-setup is found without any setup; the env var
# overrides it, for a clone kept elsewhere.
_REFERENCE = os.environ.get("DOSERAD_EVAL_REFERENCE") or str(
    Path(__file__).resolve().parents[1] / "external" / "evaluation-setup"
)
_official = None

if _REFERENCE and Path(_REFERENCE).is_dir():
    sys.path.insert(0, str(Path(_REFERENCE).resolve()))
    try:
        from doserad2026_evaluator import metrics_beam as _official  # noqa: E402
    except ImportError:  # pragma: no cover - reported by the skip below
        _official = None

requires_reference = pytest.mark.skipif(
    _official is None,
    reason=(
        "clone github.com/DoseRAD2026/evaluation-setup to "
        "external/evaluation-setup, or point DOSERAD_EVAL_REFERENCE at a clone "
        "(it has no licence, so it is cited rather than vendored)"
    ),
)


@requires_reference
@pytest.mark.parametrize("seed", range(6))
def test_mae_matches_official_exactly(seed):
    """No tolerance: same arithmetic, same order, so equality is the bar."""
    gt, pred = _beam(seed=seed), _beam(seed=seed + 100)
    assert masked_beam_mae(pred, gt) == _official.masked_beam_mae(pred, gt)


@requires_reference
@pytest.mark.parametrize("angle_deg", [0.0, 10.0, 45.0, 70.0, 90.0, 137.0, 180.0, 285.0, 350.0])
def test_idd_matches_official_exactly(angle_deg):
    """Swept over gantry angles, including the 70-90 deg band local validation
    never covers (models/splits.py) -- the angles most likely to expose a
    resampling difference."""
    gt, pred = _beam(seed=3), _beam(seed=4)
    a = math.radians(angle_deg)
    d = np.array([-math.sin(a), math.cos(a), 0.0])

    assert np.array_equal(
        compute_idd_curve(gt, d, SPACING), _official.compute_idd_curve(gt, d, SPACING)
    )
    assert idd_curve_distance(pred, gt, d, SPACING) == _official.idd_curve_distance(
        pred, gt, d, SPACING
    )


@requires_reference
def test_plan_aggregation_matches_official_exactly():
    """`aggregate_plan` must reproduce `evaluate_beam_level`'s reduction."""
    gts = [_beam(seed=s) for s in range(4)]
    preds = [_beam(seed=s + 50) for s in range(4)]
    dirs = [np.array([math.cos(a), math.sin(a), 0.0]) for a in (0.0, 0.4, 1.1, 2.0)]

    official = _official.evaluate_beam_level(preds, gts, dirs, np.asarray(SPACING))
    ours = aggregate_plan(
        [masked_beam_mae(p, g) for p, g in zip(preds, gts)],
        [idd_curve_distance(p, g, d, SPACING) for p, g, d in zip(preds, gts, dirs)],
    )

    for key in ("beam_mae_mean", "beam_mae_std", "idd_distance_mean", "idd_distance_std"):
        assert ours[key] == official[key], key


@requires_reference
def test_official_directions_of_agrees_with_beam_direction():
    """Guards the keying assumption: our per-ray direction must equal theirs."""
    plan = {
        "beams": [
            {
                "beam_idx": 0,
                "gantry_angle": 0.0,
                "rays": [
                    {"ray_idx": 0, "ray_source": [-16.0, -1062.0, 34.0], "ray_target": [-16.0, -62.0, 34.0]},
                    {"ray_idx": 1, "ray_source": [500.0, -62.0, 34.0], "ray_target": [-16.0, -62.0, 34.0]},
                ],
            }
        ]
    }
    official = _official.directions_of(plan)
    for ray in plan["beams"][0]["rays"]:
        ours = beam_direction(ray["ray_source"], ray["ray_target"])
        assert np.array_equal(ours, official[0, ray["ray_idx"]])
