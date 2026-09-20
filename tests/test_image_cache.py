"""One decompress per image, not per slot — and never two volumes resident.

`run` reads each CT with SimpleITK and holds it while a slot needs it. It used
to `clear()` after **every slot**, which bounded memory at one volume and, on
the shape the platform actually scores, re-decompressed the same 85 MB CT ten
times: **2.1 s of a 19.4 s job** at 1 image / 500 beams, against ~0.2 s for the
single read that was needed.

Both properties are in tension and the fix is only correct if it keeps both:
read each image once, and never hold two. The second is asserted
*behaviourally* rather than by inspecting the cache — an A, B, A slot order
costing **three** reads is precisely the evidence that A was evicted when B
arrived, which is what "only one resident" means. A test that reached into the
closure would assert the implementation instead of the property.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))

from submission import inference  # noqa: E402
from tests.test_submission_layout import write_ct  # noqa: E402


class Zeros:
    batch_size = 2

    def predict(self, ct, geom, requests):
        return [np.zeros(ct.shape, dtype=np.float32) for _ in requests]


def metadata(slots_to_image: dict[int, int]) -> list:
    """One beamlet per slot, each slot pointing at the given image index."""
    by_image: dict[int, list] = {}
    for slot, image_idx in slots_to_image.items():
        by_image.setdefault(image_idx, []).append(slot)
    return [
        {
            "image_file_idx": image_idx,
            "beams": [{"rays": [{
                "ray_source": [0.0, -500.0, 0.0],
                "ray_target": [0.0, 0.0, 0.0],
                "beamlets": [{
                    "energy": 100.0,
                    "output_info": {"output_file_idx": slot,
                                    "idx_in_output": 0,
                                    "minimum_cutoff": 0.0},
                } for slot in slots],
            }]}],
        }
        for image_idx, slots in by_image.items()
    ]


def run_with(tmp_path, monkeypatch, slots_to_image, n_images):
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    for idx in range(n_images):
        write_ct(input_root / "images" / f"{inference.INPUT_DIR_BASE}-{idx + 1}")
    (input_root / inference.INPUT_JSON_NAME).write_text(
        json.dumps(metadata(slots_to_image))
    )
    monkeypatch.setattr(inference, "INPUT_PATH", input_root)
    monkeypatch.setattr(inference, "OUTPUT_PATH", output_root)

    reads: list[str] = []
    original = inference.load_sitk_image
    monkeypatch.setattr(inference, "load_sitk_image",
                        lambda d: (reads.append(str(d)), original(d))[1])
    inference.run(Zeros())
    return reads


def test_one_image_across_many_slots_is_read_once(tmp_path, monkeypatch):
    """The scored shape: one CT, ten slots. It used to be read ten times."""
    reads = run_with(tmp_path, monkeypatch, {slot: 0 for slot in range(4)}, 1)
    assert len(reads) == 1, reads


def test_each_image_is_still_read_when_slots_alternate(tmp_path, monkeypatch):
    """Correctness first: a different image must still be fetched."""
    reads = run_with(tmp_path, monkeypatch, {0: 0, 1: 1, 2: 0}, 2)
    assert len(set(reads)) == 2
    # Slots run in order, so image 0 is evicted for 1 and fetched again for 2 --
    # the price of holding only one, and the reason this is a *cache* and not a
    # store.
    assert len(reads) == 3
