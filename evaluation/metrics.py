"""The two DoseRAD metrics that are faithfully computable off-platform.

Scope is deliberate. Of the six scored metrics, only these two are pure
functions of predicted vs ground-truth *beam* dose and therefore reproducible
from the data we hold:

======================  ===========================================
``beam_mae``            masked mean absolute error, per beam
``idd_distance``        normalised RMS integrated-depth-dose distance
======================  ===========================================

The other four are out of reach, and not by our choice:

* ``runtime`` is a fitted linear model rather than a measurement, so it is
  structurally out of scope here and always will be.
* ``plan_mae``, ``gamma_pass_rate`` and ``dvh_score`` are all properties of the
  *weight-composed* plan. The release ships ``image/`` + ``dose/`` +
  ``<PID>.json`` and nothing else -- no beamlet weights, no structure masks --
  confirmed against the dataset paper's §III.A, Appendix D Table 8 and Figure 4.
  Composing a plan needs
  weights, deriving weights needs inverse optimisation, and that needs a PTV.

**This is a property of the challenge, not of our setup: Level 2 is
platform-only for every participant.** So the two metrics here are not a partial
view of what others can see -- they are the whole of what anyone can measure
locally, which is why they are worth getting exactly right.

**This is a port, and ports drift.** The reference is
``github.com/DoseRAD2026/evaluation-setup`` at commit ``fcb42a5f``, files
``doserad2026_evaluator/metrics_beam.py`` and
``grand_challenge_evaluation/_common/evaluate.py``. That repository has no
licence, so it is cited rather than vendored -- but ``tests/test_local_metrics.py``
asserts exact equality against it whenever a clone sits at
``external/evaluation-setup`` (or ``DOSERAD_EVAL_REFERENCE`` points at one),
which is the only thing that makes the citations below checkable.
There is no tolerance margin anywhere in that test: these are the same
arithmetic in the same order, so any disagreement at all is a porting bug.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import SimpleITK as sitk

__all__ = [
    "masked_beam_mae",
    "compute_idd_curve",
    "idd_curve_distance",
    "aggregate_plan",
    "aggregate_cohort",
]


def masked_beam_mae(pred_beam: np.ndarray, gt_beam: np.ndarray) -> float:
    """Masked MAE for one beam, as a fraction of that beam's peak dose.

    Port of ``metrics_beam.py:20-40``. The mask is the beam's own high-dose
    region -- voxels at or above 10 % of the *ground-truth* beam maximum -- and
    the error is normalised by that same maximum, so the result is
    dimensionless and comparable across beams of very different weight.

    Returns ``nan`` for an empty beam, which is why every aggregation below
    uses ``nanmean``.
    """
    beam_max = float(np.max(gt_beam))
    if beam_max <= 0:
        return float("nan")

    mask = gt_beam >= 0.1 * beam_max
    if not mask.any():
        return float("nan")

    return float(np.mean(np.abs(pred_beam[mask] - gt_beam[mask])) / beam_max)


def compute_idd_curve(
    dose_3d: np.ndarray, direction: np.ndarray, spacing: Sequence[float]
) -> np.ndarray:
    """Integrated depth-dose curve along ``direction``.

    Port of ``metrics_beam.py:43-78``. Three details carry the fidelity and are
    the ones to check first if a ported number ever disagrees:

    * ``spacing`` is SimpleITK order ``(x, y, z)`` while ``dose_3d`` is numpy
      order ``(z, y, x)`` -- the caller passes ``image.GetSpacing()`` straight
      through, exactly as ``_common/evaluate.py:725`` does.
    * z is summed out *first*, reducing the resample to 2D. That is what makes
      the out-of-plane check below a hard error rather than an approximation:
      a beam with a z component would have its depth axis partly summed away.
    * the output grid is square and sized to the in-plane diagonal, so the
      curve length does not depend on gantry angle.
    """
    if abs(direction[2]) > 1e-9:
        raise ValueError("beam leaves the transverse plane; z cannot be summed")

    plane = dose_3d.sum(axis=0, dtype=np.float64)
    ny, nx = plane.shape
    sx, sy = float(spacing[0]), float(spacing[1])
    plane_centre = ((nx - 1) * sx / 2.0, (ny - 1) * sy / 2.0)

    step = max(sx, sy)
    n = int(math.ceil(math.hypot(nx * sx, ny * sy) / step)) + 1
    out_origin = (-(n - 1) * step / 2.0,) * 2

    source = sitk.GetImageFromArray(plane)
    source.SetSpacing((sx, sy))

    # Resample maps each *output* point back into the input, so rotating by the
    # beam angle about the plane centre lays the beam along the output's first
    # axis. Summing the other axis then collapses lateral dose against depth.
    to_beam = sitk.Euler2DTransform()
    to_beam.SetCenter((0.0, 0.0))
    to_beam.SetAngle(math.atan2(direction[1], direction[0]))
    to_beam.SetTranslation(plane_centre)
    aligned = sitk.Resample(
        source,
        (n, n),
        to_beam,
        sitk.sitkLinear,
        out_origin,
        (step, step),
        (1.0, 0.0, 0.0, 1.0),
        0.0,
        sitk.sitkFloat64,
    )

    return sitk.GetArrayFromImage(aligned).sum(axis=0)


def idd_curve_distance(
    pred_beam: np.ndarray,
    gt_beam: np.ndarray,
    direction: np.ndarray,
    spacing: Sequence[float],
) -> float:
    """Normalised RMS distance between predicted and ground-truth IDD curves.

    Port of ``metrics_beam.py:81-100``. Both curves are divided by the *ground
    truth* peak before the RMS, so the result is dimensionless. ``nan`` when
    the ground-truth curve is flat.
    """
    idd_pred = compute_idd_curve(pred_beam, direction, spacing)
    idd_gt = compute_idd_curve(gt_beam, direction, spacing)

    idd_max = float(np.max(idd_gt))
    if idd_max <= 0:
        return float("nan")

    return float(np.sqrt(np.mean((idd_pred / idd_max - idd_gt / idd_max) ** 2)))


def beam_direction(ray_source: Sequence[float], ray_target: Sequence[float]) -> np.ndarray:
    """Unit propagation direction of a proton ray.

    Port of ``metrics_beam.py:112-114`` (``directions_of``, proton branch). The
    official evaluator keys these by the ``beam_idx``/``ray_idx`` *fields* of
    the plan JSON; the caller here keys by the same fields rather than by
    positional order.
    """
    v = np.asarray(ray_target, dtype=float) - np.asarray(ray_source, dtype=float)
    return v / np.linalg.norm(v)


def aggregate_plan(maes: Iterable[float], idd_dists: Iterable[float]) -> Dict[str, float]:
    """Per-plan aggregation, matching ``_common/evaluate.py:745-748``.

    ``nanmean``/``nanstd``, so beams that were empty on the ground-truth side
    drop out rather than poisoning the plan.
    """
    maes = list(maes)
    idd_dists = list(idd_dists)
    return {
        "beam_mae_mean": float(np.nanmean(maes)),
        "beam_mae_std": float(np.nanstd(maes)),
        "idd_distance_mean": float(np.nanmean(idd_dists)),
        "idd_distance_std": float(np.nanstd(idd_dists)),
        "num_beams": len(maes),
    }


def aggregate_cohort(case_results: List[Dict]) -> Dict[str, Optional[Dict[str, float]]]:
    """Cohort aggregation, matching ``_common/evaluate.py:1028-1043``.

    Note the shape of this: it is the **unweighted mean over plans of each
    plan's own mean over beams**, not a flat mean over all beams. A plan with
    few beams counts exactly as much as a plan with many, and the challenge
    text says the same ("first averaged across all beams of each patient, and
    then averaged across all test patients"). Getting this wrong would shift
    the number without any individual metric being wrong.

    Leaderboard names are used for the keys, per ``AGGREGATE_NAME``
    (``_common/evaluate.py:264-268``): ``beam_mae_mean`` -> ``beam_mae``,
    ``idd_distance_mean`` -> ``idd_distance``.
    """
    out: Dict[str, Dict[str, float]] = {}
    for metric, name in (("beam_mae_mean", "beam_mae"), ("idd_distance_mean", "idd_distance")):
        values = [
            r[metric]
            for r in case_results
            if metric in r
            and r[metric] is not None
            and not (isinstance(r[metric], float) and np.isnan(r[metric]))
        ]
        out[name] = (
            {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_cases": len(values),
            }
            if values
            else None
        )
    return out
