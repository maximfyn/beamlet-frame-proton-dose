"""Recycling an output frame is worth ~0.6 s and corrupts output SILENTLY.

`models.predictor.FramePool` hands the same 85 MB array back to the next beamlet
instead of paying a calloc and ~2700 page faults for a fresh one — 1.47 ms
against 0.09. Everything about that is safe *except* the one thing this
file tests: a frame reused while somebody still holds it is the previous
beamlet's dose in this beamlet's map, at the right shape, with nothing raising
and the platform scoring it.

So the invariant is not "the pool is fast", it is **the bytes on disk are the
bytes the unpooled path writes**, and the only way to see that is to write a
whole slot both ways and compare the files. The unit tests below say why a
failure fails; the end-to-end one is what would actually catch a release moved
one line too early.

`tests/test_predictor_pinned.py` owns the neighbouring invariant — two
results from one call are two arrays — and it must keep passing *with a pool
enabled*, which is the last test here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

import models.geometry as G  # noqa: E402
from models.geometry import BeamletGrid, VolumeGeometry  # noqa: E402
from models.predictor import (  # noqa: E402
    BeamletRequest, DosePredictor, FramePool, block_window,
)
from submission import inference  # noqa: E402
from tests.test_submission_layout import SLOT_DIR, USED_SLOT, build_metadata, write_ct  # noqa: E402


# ---------------------------------------------------------------------------
# The pool on its own
# ---------------------------------------------------------------------------

def test_a_recycled_frame_comes_back_zero_where_the_last_one_wrote():
    """The whole correctness claim of `mark`: only the window is dirty."""
    pool = FramePool(capacity=2)
    frame = pool.take((4, 4, 4))
    window = block_window(np.array([1, 1, 1]), np.array([3, 3, 3]))
    frame[window] = 5.0
    pool.mark(frame, window)
    pool.release(frame)

    again = pool.take((4, 4, 4))
    assert again is frame, "the pool allocated instead of recycling"
    assert not again.any(), "a recycled frame carried the last beamlet's dose"


def test_a_frame_still_out_is_never_handed_out_twice():
    """Two live results are two arrays — the invariant a ring buffer breaks."""
    pool = FramePool(capacity=4)
    first = pool.take((4, 4, 4))
    second = pool.take((4, 4, 4))
    assert first is not second
    assert not np.shares_memory(first, second)


def test_releasing_something_the_pool_never_issued_is_a_no_op():
    """A caller mixing pooled and unpooled arrays cannot poison the pool."""
    pool = FramePool(capacity=2)
    stranger = np.zeros((4, 4, 4), dtype=np.float32)
    assert pool.release(stranger) is False
    assert pool.take((4, 4, 4)) is not stranger


def test_a_frame_of_a_different_grid_is_dropped_rather_than_reshaped():
    """Two images in one job have two grids; a wrong-shaped frame is not a frame."""
    pool = FramePool(capacity=4)
    small = pool.take((4, 4, 4))
    pool.release(small)
    large = pool.take((6, 5, 4))
    assert large.shape == (6, 5, 4)


def test_the_cap_bounds_what_is_held():
    """Each frame is 85 MB on the real path, so the pool cannot grow freely."""
    pool = FramePool(capacity=1)
    frames = [pool.take((4, 4, 4)) for _ in range(3)]
    for frame in frames:
        pool.release(frame)
    assert len(pool._free) == 1


# ---------------------------------------------------------------------------
# The predictor's own copy-out, pooled against not
# ---------------------------------------------------------------------------

def _two_beamlets():
    """A tiny volume with a body in it, and two rays that both resolve."""
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.ones(3), shape=(16, 16, 16))
    ct = np.full(geom.shape, G.AIR_HU, dtype=np.float32)
    ct[4:12, 4:12, 4:12] = 0.0
    requests = [
        BeamletRequest(
            ray_source=(-50.0, 8.0, float(v)),
            ray_target=(50.0, 8.0, float(v)),
            output_file_idx=0,
            idx_in_output=i,
            energy=100.0,
        )
        for i, v in enumerate((7.0, 9.0, 8.0, 6.0))
    ]
    return ct, geom, requests


def _predict_all(frame_pool: int) -> list:
    """Every beamlet through one predictor, releasing as a writer would."""
    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4),
                              frame_pool=frame_pool)
    kept = []
    for request in requests:
        volume = predictor.predict(ct, geom, [request])[0]
        kept.append(volume.copy())          # the copy is the "write"
        predictor.release_frame(volume)
    return kept, predictor


def test_the_pooled_path_predicts_the_same_voxels():
    """Bit-exact, not close: the pool changes allocation, never arithmetic."""
    pooled, predictor = _predict_all(frame_pool=4)
    plain, _ = _predict_all(frame_pool=0)
    assert predictor.frames.reused > 0, "nothing was recycled; the test proves nothing"
    for got, want in zip(pooled, plain):
        np.testing.assert_array_equal(got, want)


def test_a_result_still_held_survives_the_next_call_with_a_pool():
    """`test_predictor_pinned.py`'s invariant, restated where a pool exists.

    A caller that never releases keeps its array forever — which is exactly what
    the scorers do, and why the pool is off unless asked for.
    """
    ct, geom, requests = _two_beamlets()
    predictor = DosePredictor(grid=BeamletGrid(n_depth=16, n_lat_u=8, n_lat_v=4),
                              frame_pool=4)
    held = predictor.predict(ct, geom, requests[:1])[0]
    snapshot = held.copy()
    predictor.predict(ct, geom, requests)
    np.testing.assert_array_equal(held, snapshot)


# ---------------------------------------------------------------------------
# The whole container path, both ways, compared as bytes
# ---------------------------------------------------------------------------

class MarkingPredictor:
    """A predictor that takes frames from a pool exactly as the real one does.

    Each frame carries a marker unique to its beamlet, written into a *window*
    rather than the whole volume — so a frame recycled early keeps the previous
    beamlet's marker and the stack says which beamlet leaked into which.
    """

    batch_size = 2

    def __init__(self, capacity: int) -> None:
        self.frames = FramePool(capacity) if capacity else None
        self.calls = 0

    def predict(self, ct, geom, requests):
        out = []
        for request in requests:
            self.calls += 1
            volume = (self.frames.take(ct.shape) if self.frames is not None
                      else np.zeros(ct.shape, dtype=np.float32))
            window = (slice(0, 2), slice(0, 2), slice(0, 2))
            volume[window] = np.float32(request.idx_in_output + 1)
            if self.frames is not None:
                self.frames.mark(volume, window)
            out.append(volume)
        return out

    def release_frame(self, frame):
        return self.frames.release(frame) if self.frames is not None else False


def _invoke(tmp_path: Path, monkeypatch, predictor, n_beamlets: int,
            write_ahead: int) -> Path:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-1")
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(build_metadata(n_beamlets))
    )
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    monkeypatch.setattr(inference, "WRITE_AHEAD", write_ahead)
    inference.run(predictor)
    return output_root / SLOT_DIR.format(n=USED_SLOT + 1) / "output.mha"


# **Both schedules, and the serial one is the deterministic half.** With the
# writer on its own thread, whether a frame has come back by the time the next
# `take` runs is a race — real, and what the container does, but not something to
# assert a count on. At `write_ahead=0` the release happens on the producer's own
# thread, so every beamlet after the first MUST recycle; that arm is what pins
# reuse, and the overlapped arm is what pins the bytes under the real schedule.
@pytest.mark.parametrize("n_beamlets,write_ahead", [(7, 0), (16, 8)])
def test_the_written_stack_is_byte_identical_with_and_without_the_pool(
    tmp_path, monkeypatch, n_beamlets, write_ahead
):
    """THE TEST. A release moved before `writer.add` fails only here.

    More beamlets than the pool's capacity, so frames genuinely come back round
    and the writer is a round behind the producer — the schedule the container
    actually runs.
    """
    pooled_predictor = MarkingPredictor(capacity=4)
    pooled = _invoke(tmp_path / "pooled", monkeypatch, pooled_predictor, n_beamlets,
                     write_ahead)
    plain = _invoke(tmp_path / "plain", monkeypatch, MarkingPredictor(capacity=0),
                    n_beamlets, write_ahead)

    if write_ahead == 0:
        # A batch is taken before any of it is written, so the first batch is
        # the only one that allocates.
        assert pooled_predictor.frames.reused >= n_beamlets - MarkingPredictor.batch_size, (
            "the serial path releases before the next batch is taken, so every "
            "frame after the first batch must be a recycled one")
    assert pooled.read_bytes() == plain.read_bytes()

    # And the markers themselves, so a failure says which beamlet leaked rather
    # than only that the bytes differ.
    stack = sitk.GetArrayFromImage(sitk.ReadImage(str(pooled)))
    for idx in range(n_beamlets):
        assert stack[idx][0, 0, 0] == pytest.approx(idx + 1)
