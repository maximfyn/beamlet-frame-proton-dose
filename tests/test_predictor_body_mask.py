"""The body mask must zero the same voxels a whole-volume mask would -- and only those.

**THE MASK IS `body_contour(ct)`, NOT `ct > AIR_HU`.** The bare threshold
deletes internal air stored at the reconstruction floor -- bowel gas, trachea --
where their contour-masked labels HAVE dose. An earlier submission shipped the
threshold and `idd` paid for it. ⇒ the pocket test below is the one that would have caught it, and it is why every
assertion here now compares against `body_contour` rather than the threshold.

`pred *= body_contour(ct)` is applied to the *block* rather than the volume for
runtime: a prediction is nonzero on ~1.6% of the grid, so masking 21M voxels per
beamlet would cost more than it buys.

That optimisation is what needs a test. The block carries its own ``(lo, hi)``
into the volume and **the window reverses** — ``lo``/``hi`` are ``(x, y, z)``
while the volume is indexed ``(z, y, x)``. A transposed window still runs, still
produces a block-shaped region of zeros, and still returns plausible dose; it
simply zeroes the wrong voxels. Nothing downstream raises and the platform would
score it.

So the invariant is not "some voxels got zeroed", it is **masking in the box is
indistinguishable from masking the volume**. Everything here uses a deliberately
**asymmetric** volume and body, because on the cubic fixture the rest of the
suite uses, a transpose is invisible.
"""

from __future__ import annotations

import numpy as np
import pytest

import models.geometry as G
from models.geometry import BeamletGrid, VolumeGeometry, body_contour
from models.predictor import BeamletRequest, DosePredictor, block_window

pytest.importorskip("torch")


def _asymmetric_case():
    """A volume whose three axes all differ, and a body offset in each."""
    geom = VolumeGeometry(
        origin=np.zeros(3), spacing=np.ones(3), shape=(22, 18, 14)
    )
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    # Different extents per axis: a transposed mask cannot coincide with this.
    ct[5:18, 3:15, 4:11] = 0.0
    requests = [
        BeamletRequest(
            ray_source=(-50.0, 8.0, float(v)),
            ray_target=(50.0, 8.0, float(v)),
            output_file_idx=0,
            idx_in_output=i,
            energy=100.0,
        )
        for i, v in enumerate((8.0, 10.0))
    ]
    return ct, geom, requests


def _predict(ct, geom, requests, *, body_mask):
    predictor = DosePredictor(
        grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4), body_mask=body_mask
    )
    return predictor.predict(ct, geom, requests)


def test_masking_the_block_equals_masking_the_volume():
    """The whole point, and the one a transposed window fails.

    Not `allclose` -- the two paths multiply by the same 0/1 field, so anything
    but bit equality means they disagreed about *which* voxels.
    """
    ct, geom, requests = _asymmetric_case()
    body = body_contour(ct)

    masked = _predict(ct, geom, requests, body_mask=True)
    unmasked = _predict(ct, geom, requests, body_mask=False)

    for got, raw in zip(masked, unmasked):
        assert np.array_equal(got, raw * body)


def test_the_mask_is_not_a_no_op():
    """Otherwise the test above passes vacuously.

    The predicted box extends past the body by construction (it is anchored on
    entry depth and runs the full beamlet depth), so an unmasked prediction MUST
    put dose in air. If this ever fails, the fixture stopped exercising the mask.
    """
    ct, geom, requests = _asymmetric_case()
    air = ct <= G.AIR_HU

    unmasked = _predict(ct, geom, requests, body_mask=False)
    masked = _predict(ct, geom, requests, body_mask=True)

    assert any(raw[air].any() for raw in unmasked), "fixture no longer tests anything"
    assert all(not got[air].any() for got in masked)


def test_the_mask_keeps_dose_inside_the_body_untouched():
    """It removes a floor outside; it must not scale anything inside."""
    ct, geom, requests = _asymmetric_case()
    body = body_contour(ct)

    masked = _predict(ct, geom, requests, body_mask=True)
    unmasked = _predict(ct, geom, requests, body_mask=False)

    for got, raw in zip(masked, unmasked):
        assert np.array_equal(got[body], raw[body])


def test_the_mask_is_on_by_default():
    """0 mm dilation, on, is the decided default."""
    predictor = DosePredictor(grid=BeamletGrid())
    assert predictor.body_mask is True


def test_block_window_reverses_lo_hi():
    """Pinned separately, because every other test would still pass if both
    the mask and `_to_host_volume` were transposed *together*."""
    lo, hi = np.array([1, 2, 3]), np.array([4, 6, 9])
    assert block_window(lo, hi) == (slice(3, 9), slice(2, 6), slice(1, 4))


# ---------------------------------------------------------------------------
# The pocket. This is the regression that cost an earlier submission on
# `idd`, and none of the tests above could see it: the asymmetric fixture is a
# solid slab, so `ct > AIR_HU` and `body_contour` agree on it exactly.
# ---------------------------------------------------------------------------


def _case_with_a_gas_pocket():
    """The same asymmetric body, with a pocket of reconstruction-floor air in it."""
    ct, geom, requests = _asymmetric_case()
    ct[8:15, 6:12, 6:9] = G.AIR_HU
    return ct, geom, requests


def test_the_pocket_fixture_actually_has_an_enclosed_pocket():
    """Otherwise the test below passes vacuously, the way the old suite did."""
    ct, _, _ = _case_with_a_gas_pocket()
    internal = body_contour(ct) & ~(ct > G.AIR_HU)
    assert internal.any(), "fixture no longer holds enclosed air"
    assert internal.sum() == 7 * 6 * 3


def test_the_mask_keeps_dose_in_enclosed_air_and_still_drops_the_padding():
    """Both halves, because either one alone is satisfied by doing nothing.

    Their labels carry dose in the pockets (`1ABB070`: 12.6% of pocket voxels,
    against 5.6e-09 of peak in the padding), so a mask that deletes the pocket
    scores a large fictional error and a mask that keeps the padding writes one.
    """
    ct, geom, requests = _case_with_a_gas_pocket()
    threshold = ct > G.AIR_HU
    internal = body_contour(ct) & ~threshold
    outside = ~body_contour(ct)

    unmasked = _predict(ct, geom, requests, body_mask=False)
    masked = _predict(ct, geom, requests, body_mask=True)

    assert any(raw[internal].any() for raw in unmasked), (
        "the beam does not reach the pocket -- fixture tests nothing"
    )
    for got, raw in zip(masked, unmasked):
        assert np.array_equal(got[internal], raw[internal]), (
            "dose was deleted from enclosed air -- the mask is `ct > AIR_HU` again"
        )
        assert not got[outside].any(), "padding dose survived the contour"


def test_body_contour_is_per_slice_binary_fill_holes():
    """One implementation, and it is the fast spelling of the obvious one.

    `body_contour` labels the volume once instead of calling
    `binary_fill_holes` per slice -- 2.8-3.2x faster for a bit-identical result,
    and runtime carries 2 of the 7 rank weights. Bit-identical is the claim, so
    `array_equal` is the assertion.
    """
    ndimage = pytest.importorskip("scipy.ndimage")
    ct, _, _ = _case_with_a_gas_pocket()
    body = ct > G.AIR_HU
    per_slice = np.stack(
        [ndimage.binary_fill_holes(body[z]) for z in range(body.shape[0])]
    )
    assert np.array_equal(body_contour(ct), per_slice)


def test_a_pocket_open_to_the_outside_on_its_own_slice_stays_open():
    """Per SLICE, not in 3-D. A 3-D fill would close a channel that is capped
    above and below but vents sideways on its own slice -- and the trachea is
    exactly that shape at the point it leaves the neck."""
    ct, _, _ = _asymmetric_case()
    # A channel through the body that reaches the volume edge on every slice.
    ct[8:15, 6:12, :9] = G.AIR_HU
    assert not (body_contour(ct) & ~(ct > G.AIR_HU)).any()
