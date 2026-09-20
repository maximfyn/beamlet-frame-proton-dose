"""The fallback path for a slot spanning several images, which nothing covered.

`run` streams a slot straight to disk when all its beamlets belong to one image
— the evaluator's own invariant says they do. When they do not, it falls back to
buffering every frame and stacking them with `JoinSeries`, because predicting
out of stack order cannot be streamed.

Every other test in this suite uses `image_file_idx: 0`, so that branch has
never been executed by anything but the platform. It is also the branch where
today's caches interact most: the predictor memoises the entry depths, the
uploaded CT and the render coordinates **per image**, and this is the only path
that alternates between images inside one invoke. A stale hit there would put
one patient's anatomy under another's beamlet with the shape still right.

The second test documents a limit rather than a feature. `JoinSeries`
requires identical geometry across frames, and the output contract requires each
dose map to sit on *its own input's* grid — so a slot spanning images with
different grids cannot be satisfied at all. What matters is that it fails
loudly: a silent stack on the wrong grid scores as noise with nothing in the log.
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
from tests.test_submission_layout import SLOT_DIR, USED_SLOT  # noqa: E402


class PerImagePredictor:
    """Marks each frame with its beamlet index and the CT it was given.

    The second marker is the point: it is read back off the *stack*, so a frame
    predicted from the wrong image is visible rather than merely suspected.
    """

    batch_size = 2

    def predict(self, ct, geom, requests):
        out = []
        for r in requests:
            v = np.zeros(ct.shape, dtype=np.float32)
            v.reshape(-1)[0] = np.float32(r.idx_in_output + 1)
            # Every CT below is filled with a constant that identifies it.
            v.reshape(-1)[1] = np.float32(ct.reshape(-1)[0])
            out.append(v)
        return out


def write_ct(directory: Path, fill: float, origin=(-237.0, -231.0, -156.0)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.full((6, 5, 4), fill, dtype=np.float32))
    image.SetSpacing((1.0, 1.0, 3.0))
    image.SetOrigin(origin)
    sitk.WriteImage(image, str(directory / "ct.mha"), useCompression=True)


def metadata_two_images(n_per_image: int) -> list:
    """One slot whose frames alternate between image 0 and image 1."""
    entries = []
    for image_idx in (0, 1):
        entries.append({
            "image_file_idx": image_idx,
            "beams": [{
                "rays": [{
                    "ray_source": [0.0, -500.0, 0.0],
                    "ray_target": [0.0, 0.0, 0.0],
                    "beamlets": [
                        {
                            "energy": 100.0 + i,
                            "output_info": {
                                "output_file_idx": USED_SLOT,
                                # Interleaved on purpose: image 0 takes the even
                                # stack positions, image 1 the odd ones, so a
                                # path that grouped by image and forgot to place
                                # by position would be visible immediately.
                                "idx_in_output": 2 * i + image_idx,
                                "minimum_cutoff": 0.0,
                            },
                        }
                        for i in range(n_per_image)
                    ],
                }]
            }],
        })
    return entries


def invoke(tmp_path: Path, monkeypatch, origins) -> Path:
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    for idx, (fill, origin) in enumerate(origins):
        write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-{idx + 1}",
                 fill, origin)
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(metadata_two_images(3))
    )
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)
    inference.run(PerImagePredictor())
    return output_root / SLOT_DIR.format(n=USED_SLOT + 1) / "output.mha"


SAME_GRID = [(11.0, (-237.0, -231.0, -156.0)), (22.0, (-237.0, -231.0, -156.0))]


def test_each_frame_lands_at_its_own_index(tmp_path, monkeypatch):
    stack = sitk.GetArrayFromImage(sitk.ReadImage(str(invoke(tmp_path, monkeypatch, SAME_GRID))))
    assert stack.shape[0] == 6
    assert [float(f.reshape(-1)[0]) for f in stack] == [float(i + 1) for i in range(6)]


def test_each_frame_was_predicted_from_its_own_image(tmp_path, monkeypatch):
    """The stale-cache failure, read off the output rather than inferred.

    Positions alternate between the two CTs, so a predictor handed a cached
    array from the previous image would stamp the wrong constant here — with the
    volume the right shape and the dose entirely plausible.
    """
    stack = sitk.GetArrayFromImage(sitk.ReadImage(str(invoke(tmp_path, monkeypatch, SAME_GRID))))
    seen = [float(f.reshape(-1)[1]) for f in stack]
    assert seen == [11.0, 22.0, 11.0, 22.0, 11.0, 22.0], seen


def test_frames_keep_the_input_grid(tmp_path, monkeypatch):
    written = sitk.ReadImage(str(invoke(tmp_path, monkeypatch, SAME_GRID)))
    assert written.GetSpacing()[:3] == (1.0, 1.0, 3.0)
    assert written.GetOrigin()[:3] == (-237.0, -231.0, -156.0)


def test_a_slot_spanning_two_grids_fails_loudly(tmp_path, monkeypatch):
    """It cannot be satisfied, so the only acceptable outcome is a refusal.

    One stacked file has one geometry; the contract wants each map on its own
    input's grid. Writing *something* anyway would score as noise silently.
    """
    mixed = [(11.0, (-237.0, -231.0, -156.0)), (22.0, (-100.0, -100.0, -50.0))]
    with pytest.raises(Exception):
        invoke(tmp_path, monkeypatch, mixed)
