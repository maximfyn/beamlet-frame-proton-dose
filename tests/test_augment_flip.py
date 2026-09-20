"""The flip must move the LABEL exactly as it moves the CT, on the same axes.

**THE FAILURE THIS EXISTS FOR IS SILENT.** Flipping the cuboid on `(1, 2)` and
the label on `(0, 1)` -- an off-by-one that reads as correct, because both are
three-dimensional and both flips "work" -- trains the network to predict a
depth-reversed dose from a laterally-mirrored CT. Nothing raises, the loss still
falls, and the arm is simply worse for a reason no metric names. So the invariant
is asserted on the returned sample rather than on the code that produces it.

The second half of the property is that the CHANNELS follow. They are built
from the cuboid after the flip, so equivariance is structural -- but
`build_lateral_prior` reads the lateral axes explicitly, and a builder that
consulted an axis rather than the cuboid would break it without touching the
label. Both optional channels are therefore exercised here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from models.dataset import BeamletDataset
from models.geometry import BeamletGrid


@pytest.fixture
def shard(tmp_path):
    """One two-row shard on a small box, with asymmetric content.

    The energies must be EXACT machine energies -- the Bragg prior is a
    table lookup and `physics` refuses an off-table value rather than
    interpolating, which is the same refusal inference relies on.
    """
    # A full-size box, not a reduced one: the lateral channel is the one whose
    # equivariance is not free, and a small box does not exercise it.
    grid = BeamletGrid()
    d, u, v = grid.shape
    rng = np.random.default_rng(0)
    pid = "1THB999"
    root = tmp_path / "proton"
    (root / pid).mkdir(parents=True)
    ct = (rng.normal(-200, 400, (2, d, u, v))).astype(np.int16)
    label = rng.random((2, d, u, v)).astype(np.float16)
    np.save(root / pid / "ct_i16.npy", ct)
    np.save(root / pid / "label_f16.npy", label)
    (root / pid / "meta.json").write_text(json.dumps(
        {"count": 2, "beamlets": [{"energy": 135.14582670578662}, {"energy": 181.9227472051713}]}))
    (root / pid / "_COMPLETE").write_text(json.dumps({"grid": grid.as_dict()}))
    return root, [pid], grid


def _load(shard, augment, seed=None):
    root, pids, grid = shard
    ds = BeamletDataset(root, pids, grid=grid, with_bragg=True,
                        with_lateral=True, rsp="hlut", augment_flip=augment)
    if seed is not None:
        torch.manual_seed(seed)
    return ds[0]


def test_no_augmentation_is_the_default_and_is_deterministic(shard):
    a, b = _load(shard, False), _load(shard, False)
    assert torch.equal(a["inputs"], b["inputs"])
    assert torch.equal(a["target"], b["target"])


def test_every_flip_moves_inputs_and_target_on_the_same_axes(shard):
    """For each of the four group elements, the sample IS the flipped sample."""
    plain = _load(shard, False)
    seen = set()
    for seed in range(40):
        got = _load(shard, True, seed=seed)
        # inputs are (C, d, u, v) and target (1, d, u, v): the lateral axes are
        # the LAST TWO of both, whatever the channel count.
        for axes in ([], [-2], [-1], [-2, -1]):
            want_in = torch.flip(plain["inputs"], dims=axes) if axes else plain["inputs"]
            if torch.allclose(got["inputs"], want_in, atol=1e-5):
                want_tg = torch.flip(plain["target"], dims=axes) if axes else plain["target"]
                assert torch.equal(got["target"], want_tg), (
                    f"inputs flipped on {axes} but the target did not follow")
                seen.add(tuple(axes))
                break
        else:
            raise AssertionError("augmented sample is not any flip of the plain one")
    assert len(seen) == 4, f"only saw {sorted(seen)} of the four group elements"


def test_depth_is_never_flipped(shard):
    """Axis 0 is the beam direction; reversing it is not a symmetry."""
    plain = _load(shard, False)
    for seed in range(40):
        got = _load(shard, True, seed=seed)
        for axes in ([-3], [-3, -2], [-3, -1], [-3, -2, -1]):
            assert not torch.allclose(
                got["inputs"], torch.flip(plain["inputs"], dims=axes), atol=1e-5), (
                f"a depth flip ({axes}) was produced")
