"""The torch backend must reproduce the scipy one, on real beamlets.

`models/geometry_torch.py` is only a speed change if it is numerically the same
operation, and "it's the same interpolation" is a claim, not a measurement --
`models/geometry_torch.py` lists three ways the two silently disagree, two of
which look like working code. This is where that claim is checked.

The `precompensate` comparison is the one that matters most: 8 chained Landweber
iterations, so a per-step difference has seven chances to compound, and its
output is a training *label* that would be baked into every shard.

Run against CUDA as well as CPU:

    DOSERAD_TORCH_DEVICE=cuda venv/bin/python -m pytest tests/test_geometry_torch.py -q
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
import torch

import reference_scipy as G
from models import geometry_torch as GT
from models.geometry import BeamletGrid, VolumeGeometry, find_entry_depth_box

PATIENT = "1ABB020"
DEVICE = os.environ.get("DOSERAD_TORCH_DEVICE", "cpu")


def _resolve_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    for candidate in (
        root / "data" / "dataset_raw" / "proton" / "training",
        Path("data/dataset_raw/proton/training"),
    ):
        for plan in (
            candidate / PATIENT / "plan_json" / f"{PATIENT}.json",
            candidate / PATIENT / f"{PATIENT}.json",
        ):
            if plan.exists():
                return candidate
    return None


DATA = _resolve_root()

pytestmark = pytest.mark.skipif(DATA is None, reason="no sample data available")


@pytest.fixture(scope="module")
def beamlet():
    root = DATA / PATIENT
    plan_path = next(
        p
        for p in (
            root / "plan_json" / f"{PATIENT}.json",
            root / f"{PATIENT}.json",
        )
        if p.exists()
    )
    plan = json.loads(plan_path.read_text())
    image = sitk.ReadImage(str(root / "image" / "ct.mha"))
    ct = sitk.GetArrayFromImage(image).astype(np.float32)
    geom = VolumeGeometry.from_sitk(image)
    grid = BeamletGrid()

    beam = plan["beams"][0]
    ray = beam["rays"][0]
    source = np.asarray(ray["ray_source"], dtype=float)
    target = np.asarray(ray["ray_target"], dtype=float)
    entry = find_entry_depth_box(ct, geom, source, target, grid)
    assert entry is not None

    dose_path = (
        root / "dose" / f"Dose_B{beam['beam_idx']}_R{ray['ray_idx']}_L0.mha"
    )
    dose = (
        sitk.GetArrayFromImage(sitk.ReadImage(str(dose_path))).astype(np.float32)
        if dose_path.exists()
        else None
    )
    return {
        "ct": ct,
        "dose": dose,
        "geom": geom,
        "grid": grid,
        "source": source,
        "target": target,
        "entry": entry,
        "device": torch.device(DEVICE),
    }


def test_grid_points_match(beamlet):
    """The coordinate build, before any interpolation touches it."""
    reference = G.grid_points_world(
        beamlet["source"], beamlet["target"], beamlet["entry"], beamlet["grid"]
    )
    got = GT.grid_points_world(
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        beamlet["device"],
    )
    assert np.abs(got.cpu().numpy() - reference).max() < 1e-9


def test_sample_ct_matches_scipy(beamlet):
    """CT gather: the network's input channel."""
    points = G.grid_points_world(
        beamlet["source"], beamlet["target"], beamlet["entry"], beamlet["grid"]
    )
    reference = G.sample_ct(beamlet["ct"], beamlet["geom"], points)
    got = GT.sample_ct(
        torch.as_tensor(beamlet["ct"], device=beamlet["device"]),
        beamlet["geom"],
        torch.as_tensor(points, device=beamlet["device"]),
    )
    diff = np.abs(got.cpu().numpy() - reference)
    # float32 rounding against HU_SCALE 1000. A half-voxel shift or a mishandled
    # edge would be hundreds of HU, not hundredths.
    assert diff.max() < 0.5, f"max {diff.max():.4g} HU"
    # And what actually reaches disk, the shard being int16:
    differing = np.count_nonzero(
        np.rint(reference).astype(np.int16) != np.rint(got.cpu().numpy()).astype(np.int16)
    )
    assert differing / reference.size < 1e-3


def test_render_to_volume_matches_scipy(beamlet):
    """The scatter back to the CT grid -- 55% of scored per-beamlet geometry."""
    rng = np.random.default_rng(0)
    prediction = rng.random(beamlet["grid"].shape).astype(np.float32)

    reference = G.render_to_volume(
        prediction,
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        beamlet["geom"],
    )
    got = GT.render_to_volume(
        torch.as_tensor(prediction, device=beamlet["device"]),
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        beamlet["geom"],
    )
    diff = np.abs(got.cpu().numpy() - reference)
    assert diff.max() < 1e-4, f"max {diff.max():.4g}"


def test_render_is_zero_outside_the_prediction_extent(beamlet):
    """The edge-shell trap, on the render side.

    `map_coordinates(cval=0)` returns a flat 0 outside the cuboid; `grid_sample`
    would blend the border voxel toward 0 across the outermost half voxel. With
    an all-ones prediction the two disagree by up to 0.5 on that shell, which no
    all-zeros fixture would expose.
    """
    prediction = np.ones(beamlet["grid"].shape, dtype=np.float32)
    reference = G.render_to_volume(
        prediction,
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        beamlet["geom"],
    )
    got = GT.render_to_volume(
        torch.as_tensor(prediction, device=beamlet["device"]),
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        beamlet["geom"],
    )
    assert np.abs(got.cpu().numpy() - reference).max() < 1e-4


def test_precompensate_matches_scipy(beamlet):
    """The label solve: 8 chained iterations, so differences can compound.

    This is the one that gates a GPU shard rebuild -- its output *is* the stored
    label. Compared relative to the label's own peak, because absolute dose
    values are ~1e-3 and an absolute tolerance would pass trivially.
    """
    if beamlet["dose"] is None:
        pytest.skip("no dose volume for this beamlet")

    reference = G.precompensate(
        beamlet["dose"],
        beamlet["geom"],
        beamlet["source"],
        beamlet["target"],
        beamlet["entry"],
        beamlet["grid"],
        iterations=8,
    )
    got = (
        GT.precompensate(
            torch.as_tensor(beamlet["dose"], device=beamlet["device"]),
            beamlet["geom"],
            beamlet["source"],
            beamlet["target"],
            beamlet["entry"],
            beamlet["grid"],
            iterations=8,
        )
        .cpu()
        .numpy()
    )

    peak = float(reference.max())
    assert peak > 0
    relative = np.abs(got - reference).max() / peak
    assert relative < 1e-4, f"max relative {relative:.4g} of peak {peak:.4g}"

    # float16 is what the shard stores, so this is the difference that survives.
    differing = np.count_nonzero(
        reference.astype(np.float16) != got.astype(np.float16)
    )
    assert differing / reference.size < 1e-2, f"{differing} float16 voxels differ"


def test_network_input_matches_scipy(beamlet):
    """Two implementations of what the network sees, kept honest by test."""
    np_input = G.build_network_input

    points = G.grid_points_world(
        beamlet["source"], beamlet["target"], beamlet["entry"], beamlet["grid"]
    )
    cuboid = G.sample_ct(beamlet["ct"], beamlet["geom"], points)
    reference = np_input(cuboid, 120.0)
    got = GT.build_network_input(
        torch.as_tensor(cuboid, device=beamlet["device"]), 120.0
    )
    assert np.abs(got.cpu().numpy() - reference).max() < 1e-6


def test_wepl_channel_matches_scipy(beamlet):
    np_wepl = G.build_wepl_channel

    points = G.grid_points_world(
        beamlet["source"], beamlet["target"], beamlet["entry"], beamlet["grid"]
    )
    cuboid = G.sample_ct(beamlet["ct"], beamlet["geom"], points)
    reference = np_wepl(cuboid, beamlet["grid"].depth_spacing)
    got = GT.build_wepl_channel(
        torch.as_tensor(cuboid, device=beamlet["device"]),
        beamlet["grid"].depth_spacing,
    )
    # cumsum over 384 depth steps, so compare relative to the channel's scale.
    assert np.abs(got.cpu().numpy() - reference).max() / reference.max() < 1e-5


def test_entry_depth_matches_the_numpy_walk_on_real_rays(beamlet):
    """The box anchor, on real geometry and on whatever device this run has.

    `tests/test_entry_walk.py` makes this comparison on a synthetic phantom so
    it runs on a laptop; this is the one that can actually see a device
    contracting a multiply-add the host computes in two rounded steps, because
    it needs real CUDA and a real CT to do it. Run it there:

        DOSERAD_TORCH_DEVICE=cuda venv/bin/python -m pytest \
            tests/test_geometry_torch.py -q -k entry_depth

    ``==``, not ``approx``. The depth is the box's anchor, so a difference of
    one step is a different set of voxels for that beamlet -- and it would be
    invisible downstream, since the prediction stays plausible wherever the box
    lands (`models/predictor.py`).
    """
    root = DATA / PATIENT
    plan_path = next(
        p for p in (root / "plan_json" / f"{PATIENT}.json", root / f"{PATIENT}.json")
        if p.exists()
    )
    plan = json.loads(plan_path.read_text())
    ct, geom, grid = beamlet["ct"], beamlet["geom"], beamlet["grid"]
    ct_device = torch.as_tensor(ct, device=beamlet["device"])

    rays = [
        (np.asarray(ray["ray_source"], dtype=float),
         np.asarray(ray["ray_target"], dtype=float))
        for beam in plan["beams"]
        for ray in beam["rays"]
    ]
    # Every 17th, so the sample spans beams and gantry angles rather than
    # sitting inside one beam's neighbourhood.
    picked = rays[::17][:48]
    assert len(picked) >= 8, "plan too small to say anything"

    mismatches = []
    for source, target in picked:
        expected = find_entry_depth_box(ct, geom, source, target, grid)
        got = GT.find_entry_depth_box(ct_device, geom, source, target, grid)
        if got != expected:
            mismatches.append((source.tolist(), target.tolist(), expected, got))
    assert not mismatches, f"{len(mismatches)}/{len(picked)} disagree: {mismatches[:3]}"
