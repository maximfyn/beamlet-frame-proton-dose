"""The entry-depth memo must never outlive the image it was measured on.

`resolve_entry` is a function of the ray and the box -- **energy is not one of
its arguments** -- so the two beamlets sharing a ray have the same entry depth by
construction. That is half of every training plan and 11.7 ms/beamlet, the
largest single term left in `predict` once the host copy was pinned, so `predict` memoises it.

The failure a cache invites is not slowness, it is a **stale hit**: the same ray
against a *different patient* has a different entry depth, and the box is a
registration anchor, so a stale answer puts the whole beamlet's dose in the wrong
place. It raises nothing, the volume is the right shape, and the dose is
plausible everywhere it lands. These tests are that guard.

Cheap-looking alternatives that are wrong, so nobody re-introduces them:
`id(ct)` alone (CPython recycles ids, so a dead array's id can match a live one),
and hashing the CT's contents (85 MB per lookup, far more than the walk it saves).
The cache holds the array *reference* and compares identity, which cannot
false-positive because a held reference cannot be recycled.
"""

from __future__ import annotations

import numpy as np
import pytest

import models.geometry as G
from models.geometry import BeamletGrid, VolumeGeometry, find_entry_depth_box
from models.predictor import BeamletRequest, DosePredictor

GRID = BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4)


def _volume(body_at):
    """A CT whose body sits in a given slab, so entry depth is controllable."""
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    ct[:, :, body_at] = 0.0
    return ct, geom


def _request(idx=0, energy=100.0):
    return BeamletRequest(
        ray_source=(-50.0, 8.0, 8.0),
        ray_target=(50.0, 8.0, 8.0),
        energy=energy,
        output_file_idx=0,
        idx_in_output=idx,
    )


def test_a_second_patient_does_not_inherit_the_first_ones_entry_depth():
    """The stale hit, which is the only way this cache can be wrong.

    Same ray, two CTs whose bodies sit at different depths. The second must be
    re-walked; a cache keyed on the ray alone silently returns the first answer
    and anchors the box in the wrong place.
    """
    predictor = DosePredictor(grid=GRID)
    near_ct, geom = _volume(slice(4, 8))
    far_ct, _ = _volume(slice(10, 14))

    predictor.predict(near_ct, geom, [_request()])
    near = predictor._entry_cache[((-50.0, 8.0, 8.0), (50.0, 8.0, 8.0))]

    predictor.predict(far_ct, geom, [_request()])
    far = predictor._entry_cache[((-50.0, 8.0, 8.0), (50.0, 8.0, 8.0))]

    assert near != far, "the second image reused the first image's entry depth"
    assert far == find_entry_depth_box(far_ct, geom, np.array([-50.0, 8.0, 8.0]),
                                       np.array([50.0, 8.0, 8.0]), GRID)


def test_the_cache_is_dropped_when_the_image_changes():
    predictor = DosePredictor(grid=GRID)
    first, geom = _volume(slice(4, 8))
    second, _ = _volume(slice(10, 14))

    predictor.predict(first, geom, [_request()])
    predictor.predict(second, geom, [_request()])

    assert predictor._entry_cache_ct is second
    assert len(predictor._entry_cache) == 1


def test_a_second_patient_does_not_inherit_the_first_ones_UPLOADED_ct():
    """The same stale hit one level down, and it was live for one commit.

    The CT is uploaded once per image now rather than once per ``predict``
    call, which is worth an 85 MB copy per batch and is what lets the entry
    walk run on the device at all. The first version of that cache asked
    ``_entry_cache_ct is not ct`` -- but `predict` stamps that field *before*
    the upload is reached, so it answered "same image" for a CT it had never
    seen and every beamlet of patient two was sampled from patient one's
    anatomy. Shape right, dose plausible, nothing raised.
    """
    import torch

    predictor = DosePredictor(grid=GRID)
    first, geom = _volume(slice(4, 8))
    second, _ = _volume(slice(10, 14))

    predictor.predict(first, geom, [_request()])
    predictor.predict(second, geom, [_request()])

    assert predictor._ct_device_src is second
    assert torch.equal(predictor._ct_device.cpu(), torch.as_tensor(second))


def test_reset_drops_the_upload_as_well_as_the_memo():
    """A benchmark's second round must not measure a CT the container never keeps."""
    predictor = DosePredictor(grid=GRID)
    ct, geom = _volume(slice(4, 8))
    predictor.predict(ct, geom, [_request()])
    assert predictor._ct_device is not None

    predictor.reset_entry_cache()
    assert predictor._ct_device is None and predictor._ct_device_src is None


def test_two_beamlets_on_one_ray_walk_once():
    """The saving itself. Energy differs, the ray does not, so one walk serves both."""
    ct, geom = _volume(slice(4, 8))
    predictor = DosePredictor(grid=GRID)

    calls = []
    real = predictor.resolve_entry
    predictor.resolve_entry = lambda *a, **k: (calls.append(1), real(*a, **k))[1]

    predictor.predict(ct, geom, [_request(0, energy=80.0), _request(1, energy=140.0)])
    assert len(calls) == 1


def test_the_memo_does_not_change_the_answer():
    """Cached and uncached must agree beamlet for beamlet, not just on average."""
    ct, geom = _volume(slice(5, 9))
    requests = [_request(i, energy=e) for i, e in enumerate((80.0, 140.0, 200.0))]

    cached = DosePredictor(grid=GRID).predict(ct, geom, requests)
    fresh = [DosePredictor(grid=GRID).predict(ct, geom, [r])[0] for r in requests]

    for got, want in zip(cached, fresh):
        np.testing.assert_array_equal(got, want)


def test_a_ray_that_misses_is_cached_as_a_miss_and_still_counted_per_beamlet():
    """``None`` is a real answer worth caching -- but the zero-map counter must
    keep counting *requests*, or a full sweep's tally silently halves."""
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)  # no body: never resolves
    predictor = DosePredictor(grid=GRID)

    results = predictor.predict(ct, geom, [_request(0, 80.0), _request(1, 140.0)])

    assert predictor.n_zero_no_entry == 2, "counted rays instead of beamlets"
    assert all(np.count_nonzero(r) == 0 for r in results)
    assert len(predictor._entry_cache) == 1


def test_entry_depth_still_does_not_depend_on_energy():
    """The assumption the cache key encodes, guarded at the signature.

    ``predict`` looks the answer up under ``(ray_source, ray_target)`` alone,
    which is complete only while entry depth is a function of the ray and the
    box. An **energy-adaptive box is a natural accuracy idea** -- a 31.7 MeV
    beamlet does not need 384 mm of depth -- and the day someone adds it, this
    cache starts handing the first energy's answer to every other energy on the
    same ray. Nothing raises: the box is a registration anchor, so the dose
    simply lands in the wrong place.

    A signature check rather than a behavioural one, because the behaviour would
    still *look* right: the wrongness is in the key, not the walk.
    """
    import inspect

    taken = set(inspect.signature(DosePredictor.resolve_entry).parameters)
    assert "energy" not in taken, (
        "resolve_entry now depends on energy, so DosePredictor.predict's entry "
        "cache key -- (ray_source, ray_target) -- is incomplete and will return "
        "another beamlet's anchor. Add energy to the key."
    )
    assert taken == {"self", "ct", "geom", "source", "target"}, (
        f"resolve_entry's arguments changed to {sorted(taken)}; check that the "
        "cache key in DosePredictor.predict still covers all of them"
    )


def test_reset_entry_cache_makes_the_next_pass_cold():
    """What a benchmark needs, and why it exists.

    Timing one image twice with one predictor measures a hit rate the container
    cannot have -- it sees each image once. On 1ABB020 that was 21.7 ms/beamlet
    cold against 16.4 warm, and a harness that discards its first round as
    warmup reports the warm figure unless it calls this.
    """
    ct, geom = _volume(slice(4, 8))
    predictor = DosePredictor(grid=GRID)
    predictor.predict(ct, geom, [_request()])
    assert predictor._entry_cache

    predictor.reset_entry_cache()
    assert not predictor._entry_cache and predictor._entry_cache_ct is None

    calls = []
    real = predictor.resolve_entry
    predictor.resolve_entry = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    predictor.predict(ct, geom, [_request()])
    assert len(calls) == 1, "the reset did not force a fresh walk"


@pytest.mark.parametrize("chunk", [1, 8, 16, 64, 1000])
def test_the_chunk_knob_cannot_change_the_answer(chunk):
    """The default moved 64 -> 16 on a measurement; it must stay a pure knob."""
    ct, geom = _volume(slice(6, 10))
    source, target = np.array([-50.0, 8.0, 8.0]), np.array([50.0, 8.0, 8.0])
    reference = find_entry_depth_box(ct, geom, source, target, GRID, chunk=1)
    assert find_entry_depth_box(ct, geom, source, target, GRID, chunk=chunk) == reference


def test_two_beamlets_on_one_ray_build_render_coordinates_once():
    """The same argument as the entry memo, one stage later.

    Building them is 9.1 of `render_block`'s 10.2 ms/beamlet of device time and they are a function of the ray and the entry depth,
    not of energy -- so the pair sharing a ray must build them once.
    """
    import models.geometry_torch as GT

    ct, geom = _volume(slice(4, 8))
    predictor = DosePredictor(grid=GRID)
    calls = []
    original = GT.render_coordinates

    def counted(*args, **kwargs):
        calls.append(args[:3])
        return original(*args, **kwargs)

    GT.render_coordinates = counted
    try:
        predictor.predict(ct, geom, [_request(0, energy=90.0), _request(1, energy=140.0)])
    finally:
        GT.render_coordinates = original

    assert len(calls) == 1, f"built {len(calls)} times for one ray"


def test_render_coordinates_are_not_reused_across_geometries():
    """The stale hit, which is the only way this cache can be wrong.

    Same ray and the same entry depth against a *different* geometry maps to
    different voxels. The cache holds the geometry by reference and compares
    identity precisely so a recycled `id()` can never answer yes here.
    """
    import models.geometry_torch as GT

    ct, geom = _volume(slice(4, 8))
    other = VolumeGeometry(
        origin=np.array([10.0, 10.0, 10.0]), spacing=np.ones(3), shape=geom.shape
    )
    predictor = DosePredictor(grid=GRID)
    predictor.predict(ct, geom, [_request()])
    _, cached_geom, _ = predictor._render_cache
    assert cached_geom is geom

    built = GT.render_coordinates(
        np.array([-50.0, 8.0, 8.0]), np.array([50.0, 8.0, 8.0]),
        predictor._entry_cache[((-50.0, 8.0, 8.0), (50.0, 8.0, 8.0))],
        GRID, other, "cpu",
    )
    assert not np.array_equal(built[0], predictor._render_cache[2][0]) or \
        not np.array_equal(built[1], predictor._render_cache[2][1]), \
        "the two geometries must not produce the same bounds"


def test_the_same_array_on_a_moved_grid_is_re_walked():
    """One array, two geometries: the entry depth belongs to the pair."""
    predictor = DosePredictor(grid=GRID)
    ct, geom = _volume(slice(4, 8))
    moved = VolumeGeometry(origin=np.array([3.0, 0.0, 0.0]), spacing=np.ones(3),
                           shape=geom.shape)
    source, target = np.array([-50.0, 8.0, 8.0]), np.array([50.0, 8.0, 8.0])

    predictor.predict(ct, geom, [_request()])
    predictor.predict(ct, moved, [_request()])

    assert predictor._entry_cache[((-50.0, 8.0, 8.0), (50.0, 8.0, 8.0))] == \
        find_entry_depth_box(ct, moved, source, target, GRID)


def test_numpy_ray_endpoints_are_accepted():
    """Arrays are what most callers hold; they must not reach a dict key raw."""
    predictor = DosePredictor(grid=GRID)
    ct, geom = _volume(slice(4, 8))
    request = BeamletRequest(ray_source=np.array([-50.0, 8.0, 8.0]),
                             ray_target=np.array([50.0, 8.0, 8.0]), energy=100.0,
                             output_file_idx=0, idx_in_output=0)
    (as_array,) = predictor.predict(ct, geom, [request])
    (as_tuple,) = predictor.predict(ct, geom, [_request()])
    np.testing.assert_array_equal(as_array, as_tuple)
