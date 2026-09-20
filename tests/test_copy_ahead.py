"""A chunk's copy-out finishes while the next chunk's forward runs.

`predict` launches a chunk's forward pass and then has ~185 ms of A10G to wait
for, while the ~1.5 ms/beamlet of device-to-host and host copying it owes for the
*previous* chunk depends on nothing that forward produces. So the copies are
issued asynchronously and finished one iteration later.

Two things can go wrong and neither raises:

1. **A frame returned before its voxels arrive.** The window is written in the
   second half, so a missed flush hands the writer a frame that is zero exactly
   where the dose is — a plausible all-zero map, which is the failure this
   project has already shipped once (an early cutoff bug).
2. **A buffer reused while a copy is still reading it**, on the device side. That
   one needs CUDA to reproduce, so what is asserted here is the ordering — every
   test below runs more beamlets than one batch, which is the only case the
   pipeline engages at all.

The pinned path itself needs a GPU. On CPU the same two halves run with the
block standing in for the landing buffer, deliberately: a pipeline exercised only
on the box is a pipeline whose ordering is untested.
"""

from __future__ import annotations

import numpy as np
import pytest

import models.geometry as G
from models.geometry import BeamletGrid, VolumeGeometry
from models.predictor import BeamletRequest, DosePredictor

pytest.importorskip("torch")

GRID = BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4)


def _case(n: int):
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    ct[4:12, 4:12, 4:12] = 0.0
    requests = [
        BeamletRequest(
            ray_source=(-50.0, 8.0, 6.0 + (i % 4)),
            ray_target=(50.0, 8.0, 6.0 + (i % 4)),
            output_file_idx=0,
            idx_in_output=i,
            energy=100.0 + i,
        )
        for i in range(n)
    ]
    return ct, geom, requests


@pytest.mark.parametrize("n_beamlets", [5, 9, 17])
def test_the_pipelined_copy_out_returns_the_same_voxels(n_beamlets):
    """THE TEST. A missed flush shows up here as an all-zero frame."""
    ct, geom, requests = _case(n_beamlets)
    ahead = DosePredictor(grid=GRID, batch_size=2, copy_ahead=True)
    serial = DosePredictor(grid=GRID, batch_size=2, copy_ahead=False)

    got = ahead.predict(ct, geom, requests)
    want = serial.predict(ct, geom, requests)

    assert len(got) == len(want) == n_beamlets
    for i, (a, b) in enumerate(zip(got, want)):
        np.testing.assert_array_equal(a, b, err_msg=f"beamlet {i}")
    # And the frames are not all zero, or the comparison is vacuous.
    assert any(frame.any() for frame in got)


def test_every_frame_is_written_before_predict_returns():
    """The last chunk has no next forward to hide behind and must still flush."""
    ct, geom, requests = _case(7)          # 3 chunks of 2 and one of 1
    predictor = DosePredictor(grid=GRID, batch_size=2, copy_ahead=True)
    frames = predictor.predict(ct, geom, requests)
    assert frames[-1].any(), "the final chunk was never finished"


def test_two_results_from_one_call_are_still_two_arrays():
    """`test_predictor_pinned.py`'s invariant, under the pipeline.

    Several copies are in flight at once now, so a shared landing buffer would
    alias results within one call rather than across calls.
    """
    ct, geom, requests = _case(9)
    frames = DosePredictor(grid=GRID, batch_size=2, copy_ahead=True).predict(
        ct, geom, requests
    )
    for i in range(1, len(frames)):
        assert frames[i] is not frames[0]
        assert not np.shares_memory(frames[i], frames[0])
    before = frames[1].copy()
    frames[0][:] = 12345.0
    np.testing.assert_array_equal(frames[1], before)


def test_a_single_chunk_does_not_engage_the_pipeline():
    """Nothing to hide behind, so it takes the serial path and still works."""
    ct, geom, requests = _case(2)
    predictor = DosePredictor(grid=GRID, batch_size=8, copy_ahead=True)
    got = predictor.predict(ct, geom, requests)
    want = DosePredictor(grid=GRID, batch_size=8, copy_ahead=False).predict(
        ct, geom, requests
    )
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, b)
