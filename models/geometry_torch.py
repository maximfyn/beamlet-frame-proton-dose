"""Torch backend for the beamlet-box resampling, on CPU or CUDA.

Same mathematics as the ``scipy`` reference in ``tests/reference_scipy.py`` --
``map_coordinates(order=1)`` and ``grid_sample(mode="bilinear")`` are both plain
trilinear interpolation -- so this is a speed change, not a semantic one. That
claim is only worth anything because it is tested: ``tests/test_geometry_torch.py``
compares every function here against that reference on real beamlets, and
the three ways they can silently disagree are listed at the end of this
docstring.

Why this exists
---------------
Both halves of the pipeline are dominated by this arithmetic:

* **preprocessing** -- ``precompensate`` is 70.6% of it, being one render +
  sample per Landweber iteration (the released shards use 32);
* **inference** -- ``render_to_volume`` is 55% of per-beamlet CPU geometry, and
  runtime is a scored metric.

One port serves both. The functions take the same arguments as their scipy
namesakes plus a device, and keep every intermediate on it.

**This module is the only resampling on the production path** — preprocessing,
:mod:`models.dataset` and ``predictor.predict`` all call it. There is
deliberately **no scipy branch**: torch runs on CPU too, so a fallback buys
nothing, and two implementations of what the network sees fail silently —
train and inference disagreeing about where a beamlet starts, both looking
correct in isolation. ``tests/reference_scipy.py``
is an **oracle, not a fallback**, and lives outside ``models/`` so it cannot be
imported by accident and quietly become the second one.

The three traps
---------------
1. **``align_corners=True``.** ``-1`` maps to index 0 and ``+1`` to ``n-1``,
   which is exactly ``map_coordinates``' index convention. ``False`` shifts by
   half a voxel, is worth 440 HU of error, and looks like it works.
2. **Padding.** ``grid_sample`` pads with 0 whatever ``cval`` says, so sample
   ``volume - fill`` and add ``fill`` back.
3. **The edge shell.** ``map_coordinates(mode="constant")`` does *no*
   interpolation beyond the edge and returns ``cval`` flat; ``grid_sample``
   blends the border voxel toward the padding across the outermost half voxel.
   Unguarded that is 1884 HU of error. :func:`_sample_trilinear` forces the
   scipy semantics rather than relying on the border happening to be air.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import (
    AIR_HU,
    BODY_HU_THRESHOLD,
    BeamletGrid,
    VolumeGeometry,
    _box_index_bounds,
    _first_depth_reaching_volume,
    beam_frame,
)


def _sample_trilinear(
    volume: torch.Tensor, coords: torch.Tensor, fill_value: float
) -> torch.Tensor:
    """Interpolate ``volume`` at continuous ``coords``, matching scipy exactly.

    ``volume`` is 3-D and ``coords`` is ``(..., 3)`` holding indices in
    **array-axis order** -- ``(axis0, axis1, axis2)`` against ``volume.shape`` --
    which is the one convention both callers convert into, so the axis flip
    ``grid_sample`` needs lives here and only here.
    """
    sizes = torch.tensor(
        [s - 1 for s in volume.shape], dtype=coords.dtype, device=coords.device
    )
    # grid_sample's last dim is (x, y, z), addressing (W, H, D) -- the reverse of
    # array-axis order.
    normalized = (2.0 * coords / sizes - 1.0).flip(-1)

    src = (volume - fill_value).reshape(1, 1, *volume.shape)
    out = F.grid_sample(
        src,
        normalized.reshape(1, *coords.shape[:-1], 3).to(src.dtype),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(coords.shape[:-1])
    out = out + fill_value

    outside = ((coords < 0) | (coords > sizes)).any(dim=-1)
    return torch.where(outside, out.new_tensor(fill_value), out)


def _world_to_index(
    geom: VolumeGeometry, points_world: torch.Tensor, device
) -> torch.Tensor:
    """World ``(..., 3)`` in (x, y, z) -> continuous index in **array-axis** (z, y, x)."""
    origin = torch.as_tensor(geom.origin, dtype=torch.float64, device=device)
    spacing = torch.as_tensor(geom.spacing, dtype=torch.float64, device=device)
    return ((points_world - origin) / spacing).flip(-1)


def grid_points_world(
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    device,
) -> torch.Tensor:
    """World coordinates of every box sample, built on ``device``.

    ``(*grid.shape, 3)`` ordered ``(depth, u, v)``, in float64 -- the coordinate
    build stays double precision because a half-voxel error here is the
    ``align_corners`` trap by another route; only the interpolation drops to the
    sampler's dtype.
    """
    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    depth, u, v = grid.axis_coordinates()

    to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    origin = to(ray_source) + entry_depth * to(direction)
    return (
        origin
        + to(depth)[:, None, None, None] * to(direction)
        + to(u)[None, :, None, None] * to(lat_u)
        + to(v)[None, None, :, None] * to(lat_v)
    )


def sample_volume(
    volume: torch.Tensor,
    geom: VolumeGeometry,
    points_world: torch.Tensor,
    fill_value: float,
) -> torch.Tensor:
    """Trilinear sampling of ``volume`` at world points, on a device.

    ``volume`` is ``(z, y, x)``; the scipy reference is
    ``tests/reference_scipy.py``.
    """
    return _sample_trilinear(
        volume, _world_to_index(geom, points_world, volume.device), fill_value
    )


def sample_ct(
    ct: torch.Tensor, geom: VolumeGeometry, points_world: torch.Tensor
) -> torch.Tensor:
    """CT Hounsfield values on the beamlet box; air outside the volume."""
    return sample_volume(ct, geom, points_world, AIR_HU)


def sample_dose(
    dose: torch.Tensor, geom: VolumeGeometry, points_world: torch.Tensor
) -> torch.Tensor:
    """Training-only: the regression label on the box grid."""
    return sample_volume(dose, geom, points_world, 0.0)


def render_coordinates(
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    geom: VolumeGeometry,
    device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor | None]:
    """``(lo, hi, coords)`` for rendering the box back onto ``geom``'s voxels.

    ``coords`` indexes the *prediction* cuboid in array-axis order
    ``(depth, u, v)``, one entry per CT voxel in the box's bounding box.

    Split out from :func:`render_to_volume` because it depends only on geometry,
    never on values -- so :func:`precompensate` builds it **once** and reuses it
    across every iteration instead of rebuilding a meshgrid and three dot
    products every time, which the scipy version does.
    """
    lo, hi = _box_index_bounds(ray_source, ray_target, entry_depth, grid, geom)
    if np.any(hi <= lo):
        return lo, hi, None

    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    depth, u, v = grid.axis_coordinates()
    to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    box_origin = to(ray_source) + entry_depth * to(direction)

    axes = [
        torch.arange(int(lo[a]), int(hi[a]), device=device, dtype=torch.float64)
        for a in range(3)
    ]
    # **Separable, because the map is affine in the voxel index.** Projecting a
    # voxel onto a beam axis is
    #
    #     Σ_a (i_a · spacing_a + origin_a − box_origin_a) · w_a
    #
    # which splits into three one-dimensional terms plus a constant, so the
    # whole field is three broadcast sums instead of a meshgrid, an (N, 3)
    # float64 tensor, a second one for the difference, and three matmuls over
    # every CT voxel of the bounding box. That build was **9.12 of
    # `render_block`'s 10.20 ms/beamlet of device time** and
    # float64 runs at 1/32 rate on the cards that score us, so the arithmetic it
    # avoids is the expensive kind.
    #
    # **This reassociates float64 additions** -- x+y+z here against whatever
    # order a matmul chooses -- so it is not bit-identical to the form it
    # replaces by construction. What makes it safe is measured, not argued:
    # `tests/test_render_coords.py` compares the two on real geometry.
    spacing, origin_rel = to(geom.spacing), to(geom.origin) - box_origin

    def project(axis: torch.Tensor, first: float, step: float) -> torch.Tensor:
        scaled = spacing * axis
        return (
            (axes[2] * scaled[2])[:, None, None]
            + (axes[1] * scaled[1])[None, :, None]
            + (axes[0] * scaled[0])[None, None, :]
            + (torch.dot(origin_rel, axis) - first)
        ) / step

    return (
        lo,
        hi,
        torch.stack(
            [
                project(to(direction), float(depth[0]), grid.depth_spacing),
                project(to(lat_u), float(u[0]), grid.lat_u_spacing),
                project(to(lat_v), float(v[0]), grid.lat_v_spacing),
            ],
            dim=-1,
        ),
    )


def render_to_volume(
    prediction: torch.Tensor,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    geom: VolumeGeometry,
    out: torch.Tensor | None = None,
    coords: torch.Tensor | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> torch.Tensor:
    """Map a predicted beamlet cuboid back onto the CT voxel grid, on device.

    Pass ``coords``/``bounds`` from :func:`render_coordinates` to skip rebuilding
    them; they depend on geometry alone.
    """
    device = prediction.device
    values, lo, hi = render_block(
        prediction, ray_source, ray_target, entry_depth, grid, geom, coords, bounds
    )
    if out is None:
        out = torch.zeros(geom.shape, dtype=torch.float32, device=device)
    if values is None:
        return out
    out[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] = values.to(out.dtype)
    return out


def render_block(
    prediction: torch.Tensor,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    geom: VolumeGeometry,
    coords: torch.Tensor | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[torch.Tensor | None, np.ndarray, np.ndarray]:
    """``(values, lo, hi)`` -- the box's own voxels, without the volume around them.

    :func:`render_to_volume` is this plus a zero-filled CT grid to drop it into,
    and is written in terms of it so there is one implementation of the mapping.

    Worth having separately because **the volume is almost entirely zero**: the
    block is a median **5.5%** of the CT grid and at worst 7.9% (measured over 54
    beamlets on two patients), and everything outside it is exactly zero by
    construction -- this is the only write. Anything that has to *move* a dose
    map should move the block: at 85 MB a frame that is an 18x difference in
    bytes, which is what the device-to-host copy costs (`models/predictor.py`).

    ``values`` is ``None`` when the box misses the volume entirely, and ``lo``/
    ``hi`` are still returned so the caller can size an empty result.
    """
    if coords is None or bounds is None:
        lo, hi, coords = render_coordinates(
            ray_source, ray_target, entry_depth, grid, geom, prediction.device
        )
    else:
        lo, hi = bounds
    if coords is None:
        return None, lo, hi
    return _sample_trilinear(prediction, coords, 0.0), lo, hi


def precompensate(
    dose: torch.Tensor,
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    iterations: int = 32,        # what the released shards were built with
    relaxation: float = 0.8,
) -> torch.Tensor:
    """Pre-compensation with every iteration on device.

    70.6% of preprocessing, and the reason the rebuild is GPU work at all. Both
    coordinate sets are geometry-only, so they are built once here rather than
    once per iteration -- that alone removes all but one iteration's worth of the
    coordinate arithmetic a per-iteration rebuild would repeat.
    """
    device = dose.device
    lo, hi = _box_index_bounds(ray_source, ray_target, entry_depth, grid, geom)
    if np.any(hi <= lo):
        return torch.zeros(grid.shape, dtype=torch.float32, device=device)

    dose_sub = dose[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
    sub_geom = VolumeGeometry(
        origin=geom.origin + lo * geom.spacing,
        spacing=geom.spacing.copy(),
        shape=tuple(dose_sub.shape),
    )

    points = grid_points_world(ray_source, ray_target, entry_depth, grid, device)
    sample_coords = _world_to_index(sub_geom, points, device)
    _, _, render_coords = render_coordinates(
        ray_source, ray_target, entry_depth, grid, sub_geom, device
    )

    x = _sample_trilinear(dose_sub, sample_coords, 0.0)
    for _ in range(iterations):
        rendered = torch.zeros(dose_sub.shape, dtype=x.dtype, device=device)
        rendered[...] = _sample_trilinear(x, render_coords, 0.0).to(x.dtype)
        residual = dose_sub - rendered
        x = torch.clamp(
            x + relaxation * _sample_trilinear(residual, sample_coords, 0.0), min=0.0
        )
    return x


def find_entry_depth_box(
    ct: torch.Tensor,
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    grid: BeamletGrid,
    hu_threshold: float = BODY_HU_THRESHOLD,
    max_depth: float = 1400.0,
    step: float = 1.0,
    chunk: int = 512,
) -> float | None:
    """`geometry.find_entry_depth_box` with the walk on ``ct``'s device.

    The anchoring rule, unchanged: the first depth at which any sample of the
    beamlet *box* is inside the body, nearest-neighbour, out-of-volume counting
    as air. Semantics are the numpy function's and are asserted against it
    rather than argued -- ``tests/test_geometry_torch.py`` compares the two
    beamlet by beamlet, which is the same oracle discipline every other
    function here carries.

    **Why it exists.** The numpy walk costs **~14 ms per call on every card**
    measured -- it is host
    work, so the GPU stands idle for the whole of it, and on A10G-class silicon
    that is a quarter of ``predict``. Nothing about it needs to be on the host:
    the CT is already resident on the device for the sampling that follows.

    **This is a lower bound on exactness, not a bit-identity guarantee.** The
    coordinates are built in float64 as they are in numpy, but a device may fuse
    a multiply-add the host computes in two steps, which can move a coordinate
    by an ulp. That only changes an answer where a coordinate sits within an ulp
    of a half-integer -- ``round`` is half-to-even in both -- and the oracle test
    is what says it does not happen on real geometry rather than a probability
    argument. If it ever does, the answer is not to loosen the test: the entry
    depth **is** the box anchor, so a disagreement is two different boxes.

    ``chunk`` is depths per launch, and unlike the numpy walk's 16 it is large:
    the early exit is worth much less when the samples it skips are free and the
    launch it saves is not. The result is identical for any value, which the
    test also pins.
    """
    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    start = _first_depth_reaching_volume(
        geom, ray_source, direction, lat_u, lat_v, grid
    )
    if not np.isfinite(start):
        return None

    steps = np.arange(0.0, max_depth, step)
    begin = max(int(np.searchsorted(steps, start, side="right")) - 1, 0)
    if begin >= steps.size:
        return None

    device = ct.device
    to = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    _, u, v = grid.axis_coordinates()
    lateral = to(u)[:, None, None] * to(lat_u) + to(v)[None, :, None] * to(lat_v)
    source, direction_t = to(ray_source), to(direction)
    origin, spacing = to(geom.origin), to(geom.spacing)
    # (x, y, z), matching `world_to_index`'s output order -- the array is (z, y, x).
    upper = torch.tensor(
        [geom.shape[2], geom.shape[1], geom.shape[0]], device=device
    )

    for lo in range(begin, steps.size, chunk):
        block = to(steps[lo : lo + chunk])
        points = (
            source + block[:, None, None, None] * direction_t + lateral
        )
        index = torch.round((points - origin) / spacing).to(torch.long)
        inside = ((index >= 0) & (index < upper)).all(dim=-1)
        if not bool(inside.any()):
            continue
        # Clamp only to make the gather legal; `inside` is what decides. The
        # numpy walk writes AIR_HU into the outside entries instead, which is
        # the same test one line later.
        safe = torch.minimum(index.clamp_min(0), upper - 1)
        hu = ct[safe[..., 2], safe[..., 1], safe[..., 0]]
        hit = (inside & (hu > hu_threshold)).flatten(1).any(dim=1)
        found = torch.nonzero(hit)
        if found.numel():
            return float(steps[lo + int(found[0])])
    return None


def build_network_input(ct_cuboid: torch.Tensor, energy_mev: float) -> torch.Tensor:
    """The ``(2, depth, u, v)`` network input, on device.

    Mirrors the scipy reference in ``tests/reference_scipy.py`` exactly;
    ``tests/test_geometry_torch.py`` asserts they agree. Two implementations of
    "what the network sees" is a bug class that fails silently, so they are kept
    honest by test, not by convention.
    """
    from .geometry import HU_CLIP, HU_SCALE, MAX_ENERGY_MEV

    ct_norm = ct_cuboid.clamp(*HU_CLIP) / HU_SCALE
    energy = torch.full_like(ct_norm, energy_mev / MAX_ENERGY_MEV)
    return torch.stack([ct_norm, energy]).to(torch.float32)


def build_wepl_channel(
    ct_cuboid: torch.Tensor, depth_spacing: float, normalizer: float | None = None
) -> torch.Tensor:
    """Water-equivalent path length along the beam, on device."""
    from .geometry import HU_SCALE, WEPL_SCALE_MM

    if normalizer is None:
        normalizer = WEPL_SCALE_MM
    stopping = (1.0 + ct_cuboid / HU_SCALE).clamp(min=0.0)
    wepl = torch.cumsum(stopping, dim=0) * depth_spacing
    return (wepl / normalizer).to(torch.float32)
