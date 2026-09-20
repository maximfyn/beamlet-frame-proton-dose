"""The submission's clamp must survive being re-read in float64.

An early submission returned **5 implementation errors**, which the challenge
counts per dose map. The cause is a dtype asymmetry, not a rounding accident:

* our clamp is ``block < minimum_cutoff`` on a **float32** tensor, and torch
  casts the Python float to the tensor's dtype, so the effective threshold is
  ``float32(c)``;
* their check upcasts the frame to **float64** first
  (``evaluate.py:939``, ``extract_beam``) and compares against the float64 ``c``;
* for about **half of cutoffs of the size served (~1e-6)** ``float32(c) < c``, so ``[float32(c), c)``
  is a keep-window for us and a violation for them. Its only representable
  float32 value is ``float32(c)`` itself -- and exactly 5 frames of submission
  #1's output contained one, one voxel each.

**A test that compares in float32 cannot see any of this** -- numpy resolves
``float32_array < python_float`` in float32 (NEP 50), which silently re-applies
our own threshold instead of theirs. A float32 audit of our output
reported 0 violations against the platform's 5 for precisely that reason. Every
assertion below therefore compares the way `extract_beam` does, in float64.

The margin is pinned from **both** sides on purpose: too small re-opens the
window, too large deletes dose that ranks.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.predictor import CUTOFF_MARGIN, clamp_threshold

# Log-uniform over four decades around the ~1e-6 cutoffs the challenge
# serves, so the guarantee does not depend on any particular beamlets. Deterministic: a flaky margin test is worse than none.
CUTOFFS = np.exp(np.linspace(np.log(1e-8), np.log(1e-4), 20000))


def rounds_down(c: float) -> bool:
    """``float32(c) < c`` -- the half of the cutoffs that open the window."""
    return float(np.float32(c)) < c


def test_the_hazard_is_real_and_not_rare():
    """Guards the premise: if float32 stopped rounding cutoffs down, the rest of
    this file would pass vacuously and the clamp could quietly regress."""
    down = [c for c in CUTOFFS if rounds_down(c)]
    assert 0.3 < len(down) / len(CUTOFFS) < 0.7, (
        f"{len(down)}/{len(CUTOFFS)} cutoffs round down in float32"
    )


def test_nothing_survives_below_the_cutoff_in_the_evaluators_dtype():
    """The whole point: after clamping, no kept float32 value is < c in float64."""
    for c in CUTOFFS:
        thr = np.float32(clamp_threshold(c))
        # The largest float32 the clamp keeps out is one ULP below `thr`; the
        # smallest it keeps is `thr`. Only the second can reach their check.
        assert float(thr) >= c, f"cutoff {c:.9e}: clamp keeps {float(thr):.9e} < c"


def test_the_bare_cutoff_would_fail_this_same_test():
    """The regression this file exists to prevent, stated as a test.

    Without it a future simplification back to ``block < minimum_cutoff`` reads
    as harmless -- it is the obvious code, and it passes every metric.
    """
    leaks = [c for c in CUTOFFS if float(np.float32(c)) < c]
    assert leaks, "premise gone -- see test_the_hazard_is_real_and_not_rare"
    for c in leaks[:100]:
        # float32(c) is exactly the voxel value that survived in that submission.
        assert float(np.float32(c)) < c


def test_the_margin_is_not_free_to_grow():
    """A margin deletes dose in ``[c, c*(1+m))``. The cutoff is ~1e-3 of the
    beamlet peak, so 1e-5 erases a band ~1e-8 of peak -- five orders below the
    MC noise floor. At 1e-3 it would be 1e-6 of peak and still small, but the
    number would no longer be justified by anything measured, so it is pinned."""
    assert 1e-7 <= CUTOFF_MARGIN <= 1e-4
    for c in CUTOFFS[::97]:
        assert clamp_threshold(c) <= c * (1.0 + 2.0 * CUTOFF_MARGIN)


def test_margin_clears_the_float32_window_by_a_wide_factor():
    """0.5 ULP of float32 is 6e-8 relative and closes the *known* mechanism; the
    margin is there for the ones we cannot see (their cutoff is read from
    `run_manifest`, ours from the input JSON). Assert the headroom, so shrinking
    the margin to 'just enough' has to argue with this test."""
    for c in CUTOFFS[::97]:
        assert clamp_threshold(c) / c - 1.0 > 20 * np.finfo(np.float32).eps


def test_zero_and_negative_cutoffs_disable_the_clamp():
    """`minimum_cutoff` is 0.0 for a local scoring run and for the fixtures; a
    margin on zero is still zero, and a threshold of 0 must not zero real dose."""
    assert clamp_threshold(0.0) == 0.0
    assert clamp_threshold(-1.0) == 0.0


@pytest.mark.parametrize("c", [7.900642618e-07, 1.190604759e-06, 8.495790847e-07])
def test_the_shapes_of_cutoff_that_actually_failed(c: float):
    """Three cutoffs of the same magnitude and rounding behaviour as the five
    that failed. Chosen to *match* the hidden set's scale, not copied from it
    where it matters: the assertion is about float32 spacing near 1e-6, which is
    a property of the format, not of the challenge's data."""
    assert rounds_down(c), "this fixture is only meaningful if float32(c) < c"
    victim = np.float32(c)  # the value that survived our old clamp
    assert float(victim) < c, "premise: it is a violation in the evaluator's dtype"
    kept = np.where(
        np.array([victim], dtype=np.float32) < np.float32(clamp_threshold(c)),
        np.float32(0.0),
        np.array([victim], dtype=np.float32),
    )
    assert kept[0] == 0.0, "the margin must zero it"
    assert not np.any((kept.astype(np.float64) > 0) & (kept.astype(np.float64) < c))
