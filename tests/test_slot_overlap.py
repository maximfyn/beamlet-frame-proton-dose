"""The writer may cross a slot boundary, and the bytes must not notice.

`run` used to create a writer and a consumer thread **per output slot** and join
at the end of each one, so the producer stood still while the queue emptied —
measured in the runtime fit as **10% of the wall**, ~140 ms a slot, ten
slots a job, with the GPU idle throughout. One consumer for
the whole invoke removes that: a slot ends by *queueing* its close and the next
slot's first batch starts immediately.

Two things have to hold, and they fail differently:

1. **The bytes.** One queue now carries frames for several files, so a frame
   reaching the wrong writer — or a close overtaking a frame — writes plausible
   dose into the wrong slot, at the right shape, with nothing raising. The first
   test writes three slots both ways and compares them byte for byte, then reads
   the markers back out of the stacks so a failure says *which* slot got which
   frame rather than only that they differ.
2. **The schedule.** Byte-identical output is exactly what the old, slow
   arrangement also produced, so a correctness test alone cannot tell whether the
   overlap happened at all. The second test makes both halves slow and asserts
   the wall clock is well under the serial sum — the only place the change is
   visible.

`SLOT_OVERLAP=0` is the A/B arm and the kill switch, so it is exercised here
rather than merely existing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from submission import inference  # noqa: E402
from tests.test_submission_layout import SLOT_DIR, write_ct  # noqa: E402

SLOTS = 3
PER_SLOT = 4


def metadata_over_slots(n_slots: int = SLOTS, per_slot: int = PER_SLOT) -> list:
    """One image, several output slots — the common shape of a scored job."""
    beamlets = []
    for slot in range(n_slots):
        for idx in range(per_slot):
            beamlets.append({
                "energy": 100.0 + slot * 10 + idx,
                "output_info": {
                    "output_file_idx": slot,
                    "idx_in_output": idx,
                    "minimum_cutoff": 0.0,
                },
            })
    return [{
        "image_file_idx": 0,
        "beams": [{"rays": [{
            "ray_source": [0.0, -500.0, 0.0],
            "ray_target": [0.0, 0.0, 0.0],
            "beamlets": beamlets,
        }]}],
    }]


class MarkingPredictor:
    """Each frame carries its own energy, which names its slot and position."""

    batch_size = 2

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def predict(self, ct, geom, requests):
        out = []
        for request in requests:
            if self.delay:
                time.sleep(self.delay)
            volume = np.zeros(ct.shape, dtype=np.float32)
            volume.reshape(-1)[0] = np.float32(request.energy)
            out.append(volume)
        return out


def invoke(tmp_path: Path, monkeypatch, predictor, overlap: bool) -> Path:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-1")
    (input_root / inference.INPUT_JSON_NAME).write_text(json.dumps(metadata_over_slots()))
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    monkeypatch.setattr(inference, "SLOT_OVERLAP", overlap)
    inference.run(predictor)
    return output_root


def test_the_slots_are_byte_identical_with_and_without_the_overlap(tmp_path, monkeypatch):
    """THE TEST. A close that overtakes a frame fails only here."""
    over = invoke(tmp_path / "over", monkeypatch, MarkingPredictor(), overlap=True)
    serial = invoke(tmp_path / "serial", monkeypatch, MarkingPredictor(), overlap=False)

    for slot in range(SLOTS):
        name = SLOT_DIR.format(n=slot + 1) + "/output.mha"
        assert (over / name).read_bytes() == (serial / name).read_bytes(), f"slot {slot}"


def test_every_frame_lands_in_its_own_slot(tmp_path, monkeypatch):
    """One queue, several files: the frame must reach the writer it was made for."""
    out = invoke(tmp_path, monkeypatch, MarkingPredictor(), overlap=True)
    for slot in range(SLOTS):
        stack = sitk.GetArrayFromImage(
            sitk.ReadImage(str(out / SLOT_DIR.format(n=slot + 1) / "output.mha"))
        )
        assert stack.shape[0] == PER_SLOT
        for idx in range(PER_SLOT):
            assert stack[idx][0, 0, 0] == pytest.approx(100.0 + slot * 10 + idx), (
                f"slot {slot} position {idx} holds another beamlet's dose"
            )


def test_every_slot_is_closed_and_readable_when_invoke_returns(tmp_path, monkeypatch):
    """The close is queued, not awaited — so the FINAL join is what guarantees it.

    A file still open when `/invoke` returns has an unpatched `CompressedDataSize`
    of zero, which reads as an empty image rather than as an error.
    """
    out = invoke(tmp_path, monkeypatch, MarkingPredictor(), overlap=True)
    for slot in range(SLOTS):
        path = out / SLOT_DIR.format(n=slot + 1) / "output.mha"
        head = path.read_bytes()[:4096]
        assert b"CompressedData = True" in head
        size = int(head.split(b"CompressedDataSize = ")[1].split(b"\n")[0])
        assert size > 0, f"slot {slot} was never closed: the size field is still 0"


def test_the_producer_does_not_wait_for_the_writer_at_a_slot_boundary(tmp_path, monkeypatch):
    """The schedule, which byte-equality cannot see.

    Both halves are made slow, so the serial arm pays predict *and* the drain for
    every slot while the overlapped arm hides all but the last drain behind the
    next slot's work. A wall-clock assertion, so the margin is wide: the
    measured gap is ~35% and the bar is 15%.
    """
    delay = 0.02
    original_add = inference.StreamingStackWriter.add

    def slow_add(self, volume):
        time.sleep(delay)
        return original_add(self, volume)

    monkeypatch.setattr(inference.StreamingStackWriter, "add", slow_add)

    # **Asserted on `drain_s`, not on the wall clock.** Both arms do the same
    # work, so a wall-clock ratio moves with whatever else the machine is doing —
    # it read 0.80x on an idle laptop and failed its own 0.85 bar on a busy one.
    # `drain_s` is the *same quantity* in both arms: the time the producer spent
    # waiting for the writer. Serial pays it at every slot; overlapped pays it
    # once, after the last.
    started = time.perf_counter()
    invoke(tmp_path / "serial", monkeypatch, MarkingPredictor(delay), overlap=False)
    serial_wall = time.perf_counter() - started
    serial_drain = inference.TIMING["drain_s"]

    started = time.perf_counter()
    invoke(tmp_path / "over", monkeypatch, MarkingPredictor(delay), overlap=True)
    over_wall = time.perf_counter() - started
    over_drain = inference.TIMING["drain_s"]

    assert over_drain < serial_drain / 2, (
        f"drain {over_drain:.3f}s overlapped against {serial_drain:.3f}s serial "
        f"over {SLOTS} slots — the producer is still waiting at each boundary "
        f"(wall {over_wall:.3f} vs {serial_wall:.3f})"
    )
