"""The device entry walk must return the numpy walk's answer, exactly.

`models.geometry_torch.find_entry_depth_box` moves the box-anchoring walk off
the host, where it costs ~14 ms per call on every card measured. The depth it returns **is the box anchor**, so this is not an
interpolation tolerance question: two different depths are two different boxes,
and every voxel the network sees moves with them. Equality or nothing.

**Exactness here is measured, not argued.** Both walks build coordinates in
float64, but a device may contract a multiply-add the host computes in two
rounded steps, which can move a coordinate by an ulp -- and `round` decides an
integer index from it. That can only matter within an ulp of a half-integer, so
what these tests do is put real-shaped geometry through it and check.

Synthetic CT rather than a patient: this file's job is to run on a laptop in a
second, so the oracle is exercised on every commit rather than only where the
data lives. `tests/test_geometry_torch.py` carries the same comparison on real
beamlets and real CUDA, which is the one that can see an FMA difference.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import geometry_torch as GT
from models.geometry import BeamletGrid, VolumeGeometry, find_entry_depth_box

SHAPE = (48, 96, 96)  # (z, y, x), small enough to stay fast


@pytest.fixture(scope="module")
def phantom():
    """Air with a tissue ellipsoid, on a geometry shaped like a real CT's.

    Anisotropic spacing on purpose: `world_to_index` divides per axis, so equal
    spacing would hide an axis-order mistake in either implementation.
    """
    # Origin chosen so the ellipsoid's centre lands on world (0, 0, 0): the
    # rays below aim there, and semi-axes in millimetres are then readable
    # straight off the voxel counts -- 102 x, 90 y, 35 z.
    geom = VolumeGeometry(
        origin=np.array([-144.0, -144.0, -60.0]),
        spacing=np.array([3.0, 3.0, 2.5]),
        shape=SHAPE,
    )
    z, y, x = np.indices(SHAPE).astype(np.float64)
    centre = np.array([SHAPE[0] / 2, SHAPE[1] / 2, SHAPE[2] / 2])
    r = (
        ((z - centre[0]) / 14.0) ** 2
        + ((y - centre[1]) / 30.0) ** 2
        + ((x - centre[2]) / 34.0) ** 2
    )
    ct = np.where(r <= 1.0, 40.0, -1024.0).astype(np.float32)
    return ct, geom, BeamletGrid()


def rays(n: int = 12):
    """Sources on a circle in the z-plane, aimed at the isocentre.

    ``beam_frame`` refuses a ray with a z component, so these stay in plane --
    which is also the only geometry the dataset has (`models/geometry.py`).
    """
    out = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        # Slightly off-centre targets so the walk sees grazing rays too, which
        # are the ones whose first hit sits near a voxel boundary.
        offset = 18.0 * ((i % 5) - 2)
        source = np.array([1060.0 * np.cos(angle), 1060.0 * np.sin(angle), 0.0])
        target = np.array([offset * np.sin(angle), -offset * np.cos(angle), 0.0])
        out.append((source, target))
    return out


@pytest.mark.parametrize("index", range(12))
def test_device_walk_returns_the_numpy_answer(phantom, index):
    ct, geom, grid = phantom
    source, target = rays()[index]
    expected = find_entry_depth_box(ct, geom, source, target, grid)
    got = GT.find_entry_depth_box(
        torch.as_tensor(ct), geom, source, target, grid
    )
    assert got == expected, f"ray {index}: {got} vs numpy {expected}"


def test_a_ray_that_misses_returns_none_on_both(phantom):
    """The caller emits an all-zero map for these, so disagreeing is a wrong map."""
    ct, geom, grid = phantom
    source = np.array([1060.0, 900.0, 0.0])
    target = np.array([-1060.0, 900.0, 0.0])
    assert find_entry_depth_box(ct, geom, source, target, grid) is None
    assert GT.find_entry_depth_box(torch.as_tensor(ct), geom, source, target, grid) is None


@pytest.mark.parametrize("chunk", [1, 16, 137, 512, 4096])
def test_chunking_is_a_performance_knob_only(phantom, chunk):
    """Same claim the numpy walk makes for its own chunk (`models/geometry.py`)."""
    ct, geom, grid = phantom
    source, target = rays()[3]
    expected = find_entry_depth_box(ct, geom, source, target, grid)
    got = GT.find_entry_depth_box(
        torch.as_tensor(ct), geom, source, target, grid, chunk=chunk
    )
    assert got == expected


def test_the_walk_finds_the_box_edge_not_the_central_axis(phantom):
    """The rule under test, not just the port.

    This ray's central axis passes at y = 95 mm, outside the phantom's 90 mm
    semi-axis, so an axis-only walk resolves nothing. The box is +-32 mm wide,
    so its near edge reaches y = 63 mm and enters -- which is the continuity
    that makes the anchor a box corner rather than the beam core
    (`DosePredictor.resolve_entry`). Without this, a port that quietly
    sampled the axis would pass every equality test above, because both walks
    would then agree on all the rays that hit head-on.
    """
    ct, geom, grid = phantom
    source = np.array([1060.0, 95.0, 0.0])
    target = np.array([0.0, 95.0, 0.0])
    expected = find_entry_depth_box(ct, geom, source, target, grid)
    got = GT.find_entry_depth_box(torch.as_tensor(ct), geom, source, target, grid)
    assert expected is not None, "the box edge should reach the phantom"
    assert got == expected
