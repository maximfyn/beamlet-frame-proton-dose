"""The sampled CT box is a function of the RAY, and reusing it must not leak.

`_cuboid` is the third memo built on the same observation — ``sample_ct`` and the
``grid_points_world`` feeding it take the ray and the entry depth, and **energy
is not an argument of either**, so a ray's beamlets share their box by
construction. It is worth more than the two before it (~2.3 ms/beamlet against
~0.8) because a clinical plan carries several energy layers per ray.

What that buys is speed; what it risks is the trap this project has already been
caught by once, in `tests/test_predictor_entry_cache.py`: a cache keyed on
something that does not identify the input hands the *next* patient the previous
one's anatomy, at the right shape, with plausible dose and nothing raised. So the
tests here are not "is it faster" — they are:

- **bit-exactness** against the same predictor with the memo off, on beamlets
  that share a ray, which is the case the memo actually changes;
- **a different ray**, and **a different image**, each of which must miss.

The memo hands the SAME tensor to several beamlets of a chunk. That is only
safe while every consumer reads it — `build_network_input` clamps out of place
and `models/physics.py` has no in-place operator — so the last test asserts the
box is unchanged after a predict rather than trusting that to stay true.
"""

from __future__ import annotations

import numpy as np
import pytest

import models.geometry as G
from models.geometry import BeamletGrid, VolumeGeometry
from models.predictor import BeamletRequest, DosePredictor

torch = pytest.importorskip("torch")

GRID = BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4)


def _volume():
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    ct[4:12, 4:12, 4:12] = 0.0
    return ct, geom


def _ray(energies, v=8.0, idx0=0):
    """Several beamlets on ONE ray — the case the memo exists for."""
    return [
        BeamletRequest(
            ray_source=(-50.0, 8.0, v),
            ray_target=(50.0, 8.0, v),
            output_file_idx=0,
            idx_in_output=idx0 + i,
            energy=float(e),
        )
        for i, e in enumerate(energies)
    ]


def test_a_rays_beamlets_predict_identically_with_the_memo_and_without():
    """Bit-exact, not close: the memo changes when the box is built, not what."""
    ct, geom = _volume()
    requests = _ray([90.0, 110.0, 130.0, 150.0])

    memoised = DosePredictor(grid=GRID, cuboid_memo=True).predict(ct, geom, requests)
    plain = DosePredictor(grid=GRID, cuboid_memo=False).predict(ct, geom, requests)

    for got, want in zip(memoised, plain):
        np.testing.assert_array_equal(got, want)


def test_a_second_ray_is_not_served_the_first_ones_box():
    """The miss that matters: same image, different ray."""
    ct, geom = _volume()
    requests = _ray([100.0], v=7.0) + _ray([100.0], v=9.0, idx0=1)

    memoised = DosePredictor(grid=GRID, cuboid_memo=True).predict(ct, geom, requests)
    plain = DosePredictor(grid=GRID, cuboid_memo=False).predict(ct, geom, requests)

    np.testing.assert_array_equal(memoised[0], plain[0])
    np.testing.assert_array_equal(memoised[1], plain[1])
    # And the two rays genuinely differ, or the test above proves nothing.
    assert not np.array_equal(memoised[0], memoised[1])


def test_a_second_image_does_not_inherit_the_first_ones_box():
    """The `test_predictor_entry_cache.py` trap, one stage further along.

    Identical ray geometry, different anatomy: a memo keyed on the ray alone
    would hand patient two patient one's CT box — right shape, plausible dose,
    nothing raised.
    """
    first, geom = _volume()
    second = first.copy()
    second[4:12, 4:12, 4:12] = 500.0        # denser patient, same grid
    request = _ray([100.0])[0]
    source = np.asarray(request.ray_source, dtype=float)
    target = np.asarray(request.ray_target, dtype=float)

    predictor = DosePredictor(grid=GRID, cuboid_memo=True)
    entry = predictor.resolve_entry(first, geom, source, target)
    assert entry is not None

    # Asserted on the BOX, not on the prediction: the stub's output does not
    # depend on the CT at all, so a leak would be invisible downstream of it --
    # and on the real network it would be dose from the wrong anatomy, which is
    # precisely the failure that has no symptom.
    one = predictor._cuboid(torch.as_tensor(first), geom, source, target, entry, "cpu")
    two = predictor._cuboid(torch.as_tensor(second), geom, source, target, entry, "cpu")
    assert not torch.equal(one, two), "the second image was served the first's box"

    fresh = DosePredictor(grid=GRID, cuboid_memo=False)._cuboid(
        torch.as_tensor(second), geom, source, target, entry, "cpu"
    )
    assert torch.equal(two, fresh)


def test_the_memo_actually_hits_within_a_ray():
    """Otherwise every test above passes on a memo that never fires."""
    ct, geom = _volume()
    predictor = DosePredictor(grid=GRID, cuboid_memo=True)
    calls = []
    original = predictor._cuboid.__func__

    def counting(self, ct_device, geom_, source, target, entry, device):
        before = self._cuboid_cache
        out = original(self, ct_device, geom_, source, target, entry, device)
        calls.append(before is not None and out is before[3])
        return out

    predictor._cuboid = counting.__get__(predictor, DosePredictor)
    predictor.predict(ct, geom, _ray([90.0, 110.0, 130.0, 150.0]))
    assert calls.count(True) >= 3, f"the memo missed on every beamlet: {calls}"


def test_the_shared_box_is_not_written_through():
    """The memo hands one tensor to several beamlets; a consumer that mutated it
    would corrupt the rest of the ray. Cheaper to assert than to keep verifying
    by reading every consumer."""
    ct, geom = _volume()
    predictor = DosePredictor(grid=GRID, cuboid_memo=True)
    predictor.predict(ct, geom, _ray([100.0]))
    assert predictor._cuboid_cache is not None
    box = predictor._cuboid_cache[3]
    snapshot = box.clone()
    predictor.predict(ct, geom, _ray([100.0, 140.0]))
    assert torch.equal(predictor._cuboid_cache[3], snapshot)


# ---------------------------------------------------------------------------
# The Bragg depth profile, memoised on the same cuboid
# ---------------------------------------------------------------------------

def test_the_bragg_channel_is_identical_across_the_memo():
    """The prior is per-energy; only the depth profile under it is per-cuboid.

    So the memo must not survive a change of box, and must not change the
    channel for a change of energy — the two ways a shared `wepl` goes wrong.
    """
    from models import physics

    physics.reset_bragg_memo()
    box = torch.linspace(-1000.0, 800.0, 16 * 8 * 4).reshape(16, 8, 4)
    other = box.flip(0).contiguous()

    first = physics.build_bragg_channel(box, 120.0, 1.0, strict_energy=False)
    second = physics.build_bragg_channel(box, 160.0, 1.0, strict_energy=False)
    physics.reset_bragg_memo()
    first_cold = physics.build_bragg_channel(box, 120.0, 1.0, strict_energy=False)
    physics.reset_bragg_memo()
    second_cold = physics.build_bragg_channel(box, 160.0, 1.0, strict_energy=False)

    assert torch.equal(first, first_cold)
    assert torch.equal(second, second_cold), "the second energy reused the first's channel"
    assert not torch.equal(first, second), "the two energies are indistinguishable"

    warm_other = physics.build_bragg_channel(other, 120.0, 1.0, strict_energy=False)
    physics.reset_bragg_memo()
    cold_other = physics.build_bragg_channel(other, 120.0, 1.0, strict_energy=False)
    assert torch.equal(warm_other, cold_other), "a new box was served the old profile"


def test_the_depth_spacing_is_part_of_the_key():
    """Same box, different grid: the profile is in millimetres, not voxels."""
    from models import physics

    physics.reset_bragg_memo()
    box = torch.linspace(-1000.0, 800.0, 16 * 8 * 4).reshape(16, 8, 4)
    one = physics.build_bragg_channel(box, 120.0, 1.0, strict_energy=False)
    three = physics.build_bragg_channel(box, 120.0, 3.0, strict_energy=False)
    assert not torch.equal(one, three)
