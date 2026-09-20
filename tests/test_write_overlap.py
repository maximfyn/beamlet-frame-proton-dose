"""Overlapping the write with `predict` must change the clock and nothing else.

The write is CPU work that releases the GIL, and `predict` spends most of its
wall blocked on the GPU, so running the writer
on its own thread hides most of the write behind work already happening. What it
must not do is change a byte: the 4-D MetaImage is one zlib stream over frames
concatenated in stack order (`StreamingStackWriter`), so the queue *is* the
ordering, and a frame that overtakes another puts one beamlet's dose at another
beamlet's index. The volume is the right shape, the dose is plausible, and the
score is wrong for both.

The hang, not the crash, is the failure mode to fear here. A writer thread
that dies leaves the producer blocked on a full queue with nothing to drain it,
and a container that hangs does not fail loudly -- it runs to the platform's
500 s cap and returns no result at all.
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

from submission import inference  # noqa: E402
from tests.test_submission_layout import (  # noqa: E402
    SLOT_DIR,
    USED_SLOT,
    build_metadata,
    write_ct,
)

FRAMES = 7  # not a multiple of the batch size below, so the last window is short


class CountingPredictor:
    """Each frame carries its own index, so a swap is visible in the bytes."""

    batch_size = 2

    def predict(self, ct, geom, requests):
        out = []
        for r in requests:
            v = np.zeros(ct.shape, dtype=np.float32)
            # idx_in_output, written into the frame itself. A queue that
            # reorders puts frame k's marker at position j.
            v.reshape(-1)[0] = np.float32(r.idx_in_output + 1)
            out.append(v)
        return out


def invoke(tmp_path: Path, monkeypatch, predictor, write_ahead: int) -> bytes:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-1")
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(build_metadata(FRAMES))
    )
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    monkeypatch.setattr(inference, "WRITE_AHEAD", write_ahead)
    inference.run(predictor)
    return (output_root / SLOT_DIR.format(n=USED_SLOT + 1) / "output.mha").read_bytes()


@pytest.mark.parametrize("write_ahead", [1, 2, 3, 8])
def test_overlapped_output_is_byte_identical_to_serial(tmp_path, monkeypatch, write_ahead):
    """The claim the knob rests on: it is a schedule, not a format."""
    serial = invoke(tmp_path / "a", monkeypatch, CountingPredictor(), 0)
    overlapped = invoke(tmp_path / "b", monkeypatch, CountingPredictor(), write_ahead)
    assert overlapped == serial


def test_frames_land_in_stack_order(tmp_path, monkeypatch):
    """Read the stack back and check each frame carries its own index.

    Byte-equality above already implies this, but only against a serial path
    that could in principle share the bug. This reads the marker.
    """
    invoke(tmp_path, monkeypatch, CountingPredictor(), 3)
    path = tmp_path / "output" / SLOT_DIR.format(n=USED_SLOT + 1) / "output.mha"
    stack = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    assert stack.shape[0] == FRAMES
    markers = [float(frame.reshape(-1)[0]) for frame in stack]
    assert markers == [float(i + 1) for i in range(FRAMES)]


def test_a_failing_writer_raises_instead_of_hanging(tmp_path, monkeypatch):
    """The 500 s cap, not an exception, is what a deadlock here would cost.

    The consumer keeps draining after it fails precisely so the producer never
    blocks on a full queue with nobody reading it. This asserts the failure
    reaches the caller; pytest's own timeout is what would catch the hang.
    """
    original = inference.StreamingStackWriter.add

    def explode(self, frame):
        if self.frames_written >= 2:
            raise RuntimeError("disk went away")
        return original(self, frame)

    monkeypatch.setattr(inference.StreamingStackWriter, "add", explode)
    with pytest.raises(RuntimeError, match="disk went away"):
        invoke(tmp_path, monkeypatch, CountingPredictor(), 1)


def test_zero_write_ahead_starts_no_thread_at_all(tmp_path, monkeypatch):
    """The escape hatch must remove the machinery, not configure it to depth 1.

    It exists for a box that cannot spare a core for the writer, so "off" has to
    mean the writer runs inline on the calling thread.
    """
    seen: list[str] = []
    original = inference.threading.Thread

    def spy(*args, **kwargs):
        seen.append(kwargs.get("name", ""))
        return original(*args, **kwargs)

    monkeypatch.setattr(inference.threading, "Thread", spy)
    out = invoke(tmp_path, monkeypatch, CountingPredictor(), 0)
    assert out.startswith(b"ObjectType = Image")
    assert "stack-writer" not in seen


def test_a_writer_that_fails_on_the_first_frame_still_drains(tmp_path, monkeypatch):
    """The backpressure path, with the queue as shallow as it can be.

    Failing on frame one means the remaining six are pushed into a queue of one
    while the consumer is in its failed state. If that state stopped it
    consuming, the producer would block on a full queue and the container would
    hang rather than fail -- so this asserts the error arrives, and it is the
    depth-1 case that would deadlock first.
    """
    def explode(self, frame):
        raise RuntimeError("first frame already gone")

    monkeypatch.setattr(inference.StreamingStackWriter, "add", explode)
    with pytest.raises(RuntimeError, match="first frame already gone"):
        invoke(tmp_path, monkeypatch, CountingPredictor(), 1)
