"""Where the container writes is a contract, and reading the code cannot check it.

`inference.run` wrote to ``/output/<slug>-N`` until 2026-08-14, one path level
short of the ``/output/images/<slug>-N`` both authorities require: the reference
algorithm writes there (the organisers' `example-submission`, `inference.py`)
and the evaluator reads there (`evaluation-setup`, `evaluate.py`,
``resolve_output_dir``). Every one of the ten files would have been written
successfully, to a place the platform never looks, with nothing raising — and
the cost of noticing is a permanently spent submission slot.

So the layout is asserted end-to-end against a synthetic ``/input`` rather than
described in prose. submission/inference.py carried the same omission, which is how the
container came to implement it.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from models.predictor import clamp_threshold  # noqa: E402
from submission import inference  # noqa: E402

# The literal the platform uses. Kept as a string rather than built from the
# code under test, so a change to that code fails here instead of agreeing
# with itself.
SLOT_DIR = "images/stacked-radiation-dose-map-{n}"
N_SLOTS = 10
USED_SLOT = 0
FRAMES_IN_USED_SLOT = 3


class ZeroPredictor:
    """Stands in for the network: the layout is what is under test, not dose."""

    batch_size = 2

    def predict(self, ct, geom, requests):
        return [np.zeros(ct.shape, dtype=np.float32) for _ in requests]


def write_ct(directory: Path) -> sitk.Image:
    directory.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.zeros((6, 5, 4), dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 3.0))
    image.SetOrigin((-237.0, -231.0, -156.0))
    sitk.WriteImage(image, str(directory / "ct.mha"), useCompression=True)
    return image


# A cutoff whose float32 rounds DOWN, the shape 51% of the platform's real ones
# have. Chosen for that property near 1e-6, not copied from the hidden set.
CUTOFF = 9.754714970e-07
assert float(np.float32(CUTOFF)) < CUTOFF


class ClampedPredictor:
    """Emits what a *correctly clamped* predictor emits: nothing positive below
    `clamp_threshold(cutoff)`, with one voxel sitting exactly on it.

    This does **not** re-test the clamp -- `tests/test_clamp_margin.py` owns
    that. It tests everything *after* it: float32 storage, compression, and the
    `JoinSeries` stacking. A write path that rescaled, re-quantised or promoted
    would put a value back inside `(0, cutoff)` with the clamp still perfect,
    and the platform would count it exactly the same way.
    """

    batch_size = 2

    def predict(self, ct, geom, requests):
        out = []
        for r in requests:
            v = np.zeros(ct.shape, dtype=np.float32)
            thr = np.float32(clamp_threshold(r.minimum_cutoff or CUTOFF))
            v.reshape(-1)[0] = thr            # the closest survivor to the cutoff
            v.reshape(-1)[1] = thr * np.float32(100.0)
            out.append(v)
        return out


def build_metadata(n_beamlets: int, minimum_cutoff: float = 0.0) -> list:
    """Exactly the nesting the platform ships: image -> beams -> rays ->
    beamlets -> output_info."""
    return [
        {
            "image_file_idx": 0,
            "beams": [
                {
                    "rays": [
                        {
                            "ray_source": [0.0, -500.0, 0.0],
                            "ray_target": [0.0, 0.0, 0.0],
                            "beamlets": [
                                {
                                    "energy": 100.0 + i,
                                    "output_info": {
                                        "output_file_idx": USED_SLOT,
                                        "idx_in_output": i,
                                        "minimum_cutoff": minimum_cutoff,
                                    },
                                }
                                for i in range(n_beamlets)
                            ],
                        }
                    ]
                }
            ],
        }
    ]


@pytest.fixture
def invoked(tmp_path, monkeypatch):
    """Run one full `invoke` against a synthetic mount, return the output root."""
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-1")
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(build_metadata(FRAMES_IN_USED_SLOT))
    )

    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    inference.run(ZeroPredictor())
    return output_root


def test_every_slot_is_written_under_images(invoked):
    """All ten slots every run, and all ten under `images/`."""
    for n in range(1, N_SLOTS + 1):
        path = invoked / SLOT_DIR.format(n=n) / "output.mha"
        assert path.exists(), f"slot {n} missing at the platform's path {path}"


def test_nothing_is_written_outside_images(invoked):
    """The exact regression: files that exist, in a place nothing reads.

    A slot directory at the output root means the `images/` level was dropped
    again, and every other assertion in this file could still pass.
    """
    stray = [p.name for p in invoked.iterdir() if p.name != "images"]
    assert not stray, f"wrote outside images/: {stray}"


def test_used_slot_holds_its_frames_on_the_input_grid(invoked):
    reference = sitk.ReadImage(str(
        invoked.parent / "input" / "images"
        / f"{inference.INPUT_DIR_BASE}-1" / "ct.mha"
    ))
    stack = sitk.ReadImage(str(invoked / SLOT_DIR.format(n=USED_SLOT + 1) / "output.mha"))

    assert sitk.GetArrayFromImage(stack).shape == (FRAMES_IN_USED_SLOT, 6, 5, 4)
    # "Each dose map sits on exactly its input image's grid. Do not resample."
    assert stack.GetSize()[:3] == reference.GetSize()
    assert stack.GetSpacing()[:3] == pytest.approx(reference.GetSpacing())
    assert stack.GetOrigin()[:3] == pytest.approx(reference.GetOrigin())


def test_every_written_stack_declares_compression(invoked):
    """An uncompressed output is an implementation error against every beam in
    the stack (`evaluate.py:is_compressed_mha`), so the placeholders are held to
    it too — the reference algorithm writes *its* placeholder uncompressed, and
    we deliberately do not follow it there.
    """
    for n in range(1, N_SLOTS + 1):
        path = invoked / SLOT_DIR.format(n=n) / "output.mha"
        assert b"CompressedData = True" in path.read_bytes()[:4096], f"slot {n}"


def test_the_written_bytes_hold_nothing_below_the_cutoff(tmp_path, monkeypatch):
    """The end of the chain the evaluator reads in float64.

    Read back exactly the way `extract_beam` does — ``sitk`` then **float64** —
    because the whole defect was that a float32 comparison cannot see it. Asserting in float32 here would pass on broken bytes.
    """
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-1")
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(build_metadata(FRAMES_IN_USED_SLOT, minimum_cutoff=CUTOFF))
    )
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    inference.run(ClampedPredictor())

    slot = output_root / SLOT_DIR.format(n=USED_SLOT + 1)
    stack = next(iter(slot.glob("*.mha")))
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(stack))).astype(np.float64)
    assert arr.max() > 0, "fixture wrote nothing; the assertion would be vacuous"
    violations = np.logical_and(arr > 0, arr < CUTOFF)
    assert not violations.any(), (
        f"{int(violations.sum())} written voxel(s) are in (0, cutoff) — the "
        f"write path reintroduced what the clamp removed"
    )
