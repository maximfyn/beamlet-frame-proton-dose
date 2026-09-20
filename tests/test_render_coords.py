"""The separable coordinate build against the matmul it replaces.

`render_coordinates` used to materialise an ``(N, 3)`` float64 field of world
coordinates, subtract the box origin from it, and take three matmuls. It now
sums three one-dimensional terms per axis, because the map is affine in the
voxel index. That is 9.12 of `render_block`'s 10.20 ms/beamlet of device time and float64 runs at 1/32 rate on the cards that score us.

**It reassociates float64 additions**, so bit-identity is not something the
construction gives -- it is something this file measures. The property that
actually matters is the one at the end of the pipeline: `_sample_trilinear`
casts these coordinates to the sampler's dtype (float32) before touching them,
and it compares them against the volume's bounds in float64. So the two
questions are whether the float32 the sampler sees is the same, and whether any
coordinate lands on the other side of an in/out-of-volume boundary.

The reference here is the *old code*, transcribed. That is deliberate: the scipy
oracle in `tests/test_geometry_torch.py` already says both are the right
geometry, and what this file is for is the narrower question of what changed.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import geometry_torch as GT
from models.geometry import BeamletGrid, VolumeGeometry, beam_frame

# Real CT proportions: anisotropic spacing, and an origin ~250 mm out so the
# coordinates carry the magnitude that decides how much precision is left.
GEOM = VolumeGeometry(
    origin=np.array([-249.5, -230.0, -60.0]),
    spacing=np.array([3.0, 3.0, 2.5]),
    shape=(48, 160, 160),
)
GRID = BeamletGrid()
# CPU proves the algebra; only a GPU can show a contracted multiply-add, and
# that is the machine this runs on in production. On a CUDA box:
#     DOSERAD_TORCH_DEVICE=cuda venv/bin/python -m pytest tests/test_render_coords.py
DEVICE = os.environ.get("DOSERAD_TORCH_DEVICE", "cpu")


def reference_coords(source, target, entry, grid, geom, device):
    """`render_coordinates` as it was before the separable form."""
    from models.geometry import _box_index_bounds

    lo, hi = _box_index_bounds(source, target, entry, grid, geom)
    if np.any(hi <= lo):
        return lo, hi, None
    direction, lat_u, lat_v = beam_frame(source, target)
    depth, u, v = grid.axis_coordinates()
    to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    box_origin = to(source) + entry * to(direction)
    axes = [
        torch.arange(int(lo[a]), int(hi[a]), device=device, dtype=torch.float64)
        for a in range(3)
    ]
    iz, iy, ix = torch.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    voxels_world = torch.stack([ix, iy, iz], dim=-1) * to(geom.spacing) + to(geom.origin)
    relative = voxels_world - box_origin
    return lo, hi, torch.stack(
        [
            (relative @ to(direction) - float(depth[0])) / grid.depth_spacing,
            (relative @ to(lat_u) - float(u[0])) / grid.lat_u_spacing,
            (relative @ to(lat_v) - float(v[0])) / grid.lat_v_spacing,
        ],
        dim=-1,
    )


def rays(n: int = 8):
    out = []
    for i in range(n):
        angle = 2.0 * np.pi * i / n
        offset = 20.0 * ((i % 4) - 1.5)
        out.append((
            np.array([1060.0 * np.cos(angle), 1060.0 * np.sin(angle), 0.0]),
            np.array([offset * np.sin(angle), -offset * np.cos(angle), 0.0]),
        ))
    return out


@pytest.mark.parametrize("index", range(8))
def test_the_sampler_sees_identical_coordinates(index):
    """float32 is what `_sample_trilinear` casts to, so float32 is the contract."""
    source, target = rays()[index]
    entry = 830.0
    _, _, new = GT.render_coordinates(source, target, entry, GRID, GEOM, DEVICE)
    _, _, old = reference_coords(source, target, entry, GRID, GEOM, DEVICE)
    assert new is not None and old is not None
    assert torch.equal(new.to(torch.float32), old.to(torch.float32))


@pytest.mark.parametrize("index", range(8))
def test_float64_agreement_is_at_the_last_bits(index):
    """A bound, so a future change that degrades this fails rather than drifts."""
    source, target = rays()[index]
    _, _, new = GT.render_coordinates(source, target, 830.0, GRID, GEOM, DEVICE)
    _, _, old = reference_coords(source, target, 830.0, GRID, GEOM, DEVICE)
    scale = old.abs().max().item()
    assert (new - old).abs().max().item() <= 1e-12 * max(scale, 1.0)


@pytest.mark.parametrize("index", range(8))
def test_no_voxel_changes_side_of_the_in_volume_test(index):
    """The one difference that would not be a rounding difference.

    `_sample_trilinear` decides in float64 whether a coordinate is outside the
    prediction cuboid and substitutes a fill value if it is. A coordinate that
    crossed that boundary would not shift a dose slightly -- it would replace an
    interpolated value with the fill, or the reverse.
    """
    source, target = rays()[index]
    _, _, new = GT.render_coordinates(source, target, 830.0, GRID, GEOM, DEVICE)
    _, _, old = reference_coords(source, target, 830.0, GRID, GEOM, DEVICE)
    sizes = torch.tensor([s - 1 for s in GRID.shape], dtype=torch.float64,
                         device=new.device)
    outside_new = ((new < 0) | (new > sizes)).any(dim=-1)
    outside_old = ((old < 0) | (old > sizes)).any(dim=-1)
    assert torch.equal(outside_new, outside_old)
