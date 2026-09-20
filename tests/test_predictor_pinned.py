"""Each dose map the predictor returns must own its memory.

`models.predictor._to_host` exists because copying a dose map back from the GPU
was **58% of the whole predict half** -- 39.7 ms of 68.7 per beamlet, against
8.2 for the network. Pinned memory takes that to 9.6 ms.

The reason it needs a test rather than a benchmark is the shape of the faster
options. A single reused pinned buffer is faster still (6.3 ms), and `predict`
returns a **list** its callers hold whole -- `submission/inference.py` predicts a
window and only then writes it, and a scorer holds a chunk until it is
scored. Share one buffer across a call and every entry but the last is
quietly overwritten: right-shaped, right dtype, plausible dose, wrong beamlet.
Nothing raises, and the platform scores it.

So the invariant is not "the copy is fast", it is **two results from one call are
two arrays**. That was untested before the optimisation and is what makes the
optimisation safe to land.

The CUDA-only tests below are the ones that exercise the pinned path itself;
on a laptop they skip and only the contract is checked.
"""

from __future__ import annotations

import numpy as np
import pytest

import models.geometry as G
from models.geometry import BeamletGrid, VolumeGeometry
from models.predictor import BeamletRequest, DosePredictor

torch = pytest.importorskip("torch")

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="pinned memory needs a CUDA context"
)


def _two_beamlets():
    """A tiny volume with a body in it, and two rays that both resolve."""
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    ct[4:12, 4:12, 4:12] = 0.0  # water-ish block: entry resolves
    requests = [
        BeamletRequest(
            ray_source=(-50.0, 8.0, float(v)),
            ray_target=(50.0, 8.0, float(v)),
            output_file_idx=0,
            idx_in_output=i,
            energy=100.0,
        )
        for i, v in enumerate((7.0, 9.0))
    ]
    return ct, geom, requests


def test_two_results_from_one_call_are_independent_arrays():
    """The invariant a shared staging buffer would break, silently.

    Writing into the first result must not touch the second. A ring or a single
    reused buffer fails here; owning the memory passes.
    """
    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    first, second = predictor.predict(ct, geom, requests)

    assert first is not second
    assert not np.shares_memory(first, second)

    before = second.copy()
    first[:] = 12345.0
    np.testing.assert_array_equal(second, before)


def test_a_result_survives_the_next_call():
    """Callers may hold a chunk while the next one is predicted.

    Weaker than the test above but it fails differently: a buffer recycled
    *between* calls looks correct within one call and corrupts across two.
    """
    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))

    held = predictor.predict(ct, geom, requests[:1])[0]
    snapshot = held.copy()
    predictor.predict(ct, geom, requests)  # a second call, results discarded

    np.testing.assert_array_equal(held, snapshot)


def test_the_assembled_volume_is_float32_and_full_grid():
    """The output contract does not change because only the box now travels."""
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    ct, geom, requests = _two_beamlets()
    out = predictor.predict(ct, geom, requests)[0]
    assert out.shape == geom.shape and out.dtype == np.float32


def test_assembly_on_cpu_needs_no_cuda_context():
    """Pinning needs CUDA, and the stub and every laptop test run without one.

    `predict` runs torch on CPU deliberately rather than keeping a second
    resampling implementation, so the staging path must be skippable.
    """
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    block = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    out = predictor._to_host_volume(block, np.array([0, 0, 0]), np.array([2, 2, 2]),
                                    (4, 4, 4))
    assert out.dtype == np.float32 and predictor._staging is None
    np.testing.assert_array_equal(out[:2, :2, :2], np.arange(8).reshape(2, 2, 2))
    assert out[3, 3, 3] == 0.0


def test_a_box_that_misses_the_volume_yields_zeros_not_a_crash():
    """`render_block` returns ``None`` for a box entirely outside the grid."""
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    out = predictor._to_host_volume(None, np.array([0, 0, 0]), np.array([0, 0, 0]),
                                    (4, 4, 4))
    assert out.shape == (4, 4, 4) and not out.any()


@cuda_only
def test_the_boxed_transfer_equals_rendering_the_whole_volume():
    """The claim the optimisation rests on: outside the block is exactly zero.

    If `render_block` and `render_to_volume` ever disagree -- a bound off by one,
    a transposed axis -- the difference is dose in the wrong place, at the right
    shape, with nothing raising.
    """
    import models.geometry_torch as GT

    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4),
                              model=None)
    source = np.asarray(requests[0].ray_source, dtype=float)
    target = np.asarray(requests[0].ray_target, dtype=float)
    entry = predictor.resolve_entry(ct, geom, source, target)
    assert entry is not None

    prediction = torch.rand((16, 8, 4), device="cuda", dtype=torch.float32) * 1e-3
    whole = GT.render_to_volume(prediction, source, target, entry,
                                predictor.grid, geom).cpu().numpy()
    block, lo, hi = GT.render_block(prediction, source, target, entry,
                                    predictor.grid, geom)
    boxed = predictor._to_host_volume(block, lo, hi, geom.shape)
    np.testing.assert_array_equal(boxed, whole)


@cuda_only
def test_the_staging_buffer_is_reused_and_never_returned():
    """It is shared, so the results must not be views of it.

    This is the inverse of the earlier design, where the pinned buffer *was* the
    result and therefore could not be shared. Sharing is safe only because the
    value is copied into an array the caller owns -- if that copy ever became a
    view, every result in a chunk would alias.
    """
    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4))
    first, second = predictor.predict(ct, geom, requests)

    staging = predictor._staging
    assert staging is not None and staging.is_pinned()
    for result in (first, second):
        assert not np.shares_memory(result, staging.numpy())

    predictor.predict(ct, geom, requests)
    assert predictor._staging is staging, "a reused buffer was reallocated"
