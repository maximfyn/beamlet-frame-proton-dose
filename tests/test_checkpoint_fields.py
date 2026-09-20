"""Every checkpoint the trainer writes must be loadable for inference.

`scripts/train/train_doserad.py` writes two kinds of file: `<run>_best.pt` at
each new best epoch, and `<run>_ep<NNN>.pt` under `--save-every-epoch`. The
submitted model is one of the latter, so a field that only the first one carries
breaks the path from training to a container.

That is exactly what happened: the per-epoch save wrote the channel list under
`channel_names`, while `models/predictor.py` reads `channels`. A file written
that way falls back to the legacy "3 channels means WEPL" rule and raises
`checkpoint is inconsistent` at load. Nothing caught it, because the only
checkpoints ever loaded here were written before the drift.

This reads the trainer's source rather than running it: a training run needs
shards and a GPU, and the property under test is which keys the payload has.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_doserad.py"

# What `DosePredictor.from_checkpoint` reads by name. `config` carries the
# widths; these are the fields that decide what the network *is* and what its
# numbers *mean*, and none of them can be recovered once absent -- a missing
# `dose_scale` silently falls back to this repository's own constant.
REQUIRED = {"model", "config", "in_channels", "channels", "grid", "arch", "rsp",
            "dose_scale", "condition_depth"}


def _saved_payloads() -> list[tuple[int, set[str]]]:
    """`(line, keys)` for every dict literal handed to ``torch.save``."""
    tree = ast.parse(TRAINER.read_text())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save" and node.args):
            continue
        payload = node.args[0]
        # `torch.save(emit({...}), path)` -- the guard wraps the dict in place.
        if isinstance(payload, ast.Call) and payload.args:
            payload = payload.args[0]
        if isinstance(payload, ast.Dict):
            keys = {k.value for k in payload.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            out.append((node.lineno, keys))
    return out


def test_the_trainer_has_both_save_paths():
    """A payload that stopped being a literal would make the checks below vacuous."""
    payloads = [p for p in _saved_payloads() if "model" in p[1]]
    assert len(payloads) >= 2, (
        f"expected the _best.pt and _ep<NNN>.pt saves, found {len(payloads)}")


@pytest.mark.parametrize("which", range(2))
def test_every_saved_checkpoint_carries_what_inference_reads(which: int):
    payloads = [p for p in _saved_payloads() if "model" in p[1]]
    line, keys = payloads[which]
    missing = REQUIRED - keys
    assert not missing, f"torch.save at line {line} omits {sorted(missing)}"
    assert "channel_names" not in keys, (
        f"torch.save at line {line} writes `channel_names`; predictor.py reads "
        "`channels`, so this file would not load for inference")
