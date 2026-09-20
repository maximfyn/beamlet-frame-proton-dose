"""The scipy reference implementation — a test oracle, not production code.

Every function here has a counterpart in :mod:`models.geometry_torch`, which is
what preprocessing, training and inference actually call. These exist for one
purpose: to be an **independent** implementation to check that one against.
That independence is not decorative — comparing the two is how the three ways
``map_coordinates`` and ``grid_sample`` silently disagree were found, two of
which look like working code.

They live under ``tests/`` rather than ``models/`` deliberately. While they sat
beside the production functions, "which implementation does inference use?" was
a question you had to read code to answer, and for a while the answer differed
between preprocessing and inference. An oracle in the test tree cannot be
imported by accident.

Do not call anything here from ``models/``, ``scripts/`` or ``submission/``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from models.geometry import (
    AIR_HU,
    HU_CLIP,
    HU_SCALE,
    MAX_ENERGY_MEV,
    WEPL_SCALE_MM,
    BeamletGrid,
    VolumeGeometry,
    _box_index_bounds,
    beam_frame,
)

def grid_points_world(
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
) -> np.ndarray:
    """World coordinates of every sample in the beamlet box.

    Returns an array of shape ``(*grid.shape, 3)`` ordered ``(depth, u, v)``.
    """
    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    depth, u, v = grid.axis_coordinates()

    origin = np.asarray(ray_source, dtype=np.float64) + entry_depth * direction
    return (
        origin
        + depth[:, None, None, None] * direction
        + u[None, :, None, None] * lat_u
        + v[None, None, :, None] * lat_v
    )



def _sample_volume(
    volume: np.ndarray,
    geom: VolumeGeometry,
    points_world: np.ndarray,
    fill_value: float,
    order: int = 1,
) -> np.ndarray:
    """Interpolate ``volume`` (numpy z, y, x) at world points ``(..., 3)``."""
    index = geom.world_to_index(points_world)  # (..., 3) as x, y, z
    coords = np.stack(
        [index[..., 2].ravel(), index[..., 1].ravel(), index[..., 0].ravel()]
    )
    sampled = map_coordinates(
        volume, coords, order=order, mode="constant", cval=fill_value
    )
    return sampled.reshape(points_world.shape[:-1])



def sample_ct(
    ct: np.ndarray, geom: VolumeGeometry, points_world: np.ndarray
) -> np.ndarray:
    """Sample CT Hounsfield values on the beamlet box (air outside the volume)."""
    return _sample_volume(ct, geom, points_world, fill_value=AIR_HU)



def sample_dose(
    dose: np.ndarray, geom: VolumeGeometry, points_world: np.ndarray
) -> np.ndarray:
    """Sample a dose volume on the beamlet box.

    Training-only: this is how the ground-truth *label* is put on the same grid
    as the input. Never call this on the inference path -- doing so would
    reintroduce exactly the dependence on ground truth that this module removes.
    """
    return _sample_volume(dose, geom, points_world, fill_value=0.0)



def crop_to_box(
    volume: np.ndarray,
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
) -> tuple[np.ndarray, VolumeGeometry]:
    """Crop a volume to the beamlet box's bounding box, with matching geometry.

    Every grid sample lies inside these bounds by construction, so working on the
    crop is equivalent to working on the full volume -- but the box covers well
    under 1% of a typical CT, so the per-iteration array arithmetic in
    :func:`precompensate` gets dramatically cheaper.
    """
    lo, hi = _box_index_bounds(ray_source, ray_target, entry_depth, grid, geom)
    if np.any(hi <= lo):
        empty = VolumeGeometry(geom.origin.copy(), geom.spacing.copy(), (0, 0, 0))
        return volume[:0, :0, :0], empty

    cropped = volume[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]]
    sub_geom = VolumeGeometry(
        origin=geom.origin + lo * geom.spacing,
        spacing=geom.spacing.copy(),
        shape=cropped.shape,
    )
    return cropped, sub_geom



def precompensate(
    dose: np.ndarray,
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    iterations: int = 8,
    relaxation: float = 0.8,
) -> np.ndarray:
    """Beam-grid label whose *rendered* form reproduces the CT-grid ground truth.

    Naively sampling the dose onto the beamlet grid and rendering it back loses
    accuracy: measured over beamlets spanning the full energy range, the round
    trip alone costs ~0.013 beam MAE, which is larger than the entire error of
    the submitted model (0.0065 on the final test). The loss is sub-voxel
    misalignment rather than rotation -- it shows up even for beams that run
    exactly along a CT axis -- so no choice of box orientation avoids it.

    Solving for the label the model *should* predict, instead of the one that
    falls out of naive sampling, removes almost all of it: the same measurement
    drops to ~0.0014 after 8 iterations. This runs in preprocessing so the
    training loop stays a plain loss in the beamlet frame, with fixed shapes and
    no differentiable resampling.

    Landweber iteration on ``render(x) ~= dose``; ``sample`` is the adjoint of
    ``render`` up to a constant, and the non-negativity clamp keeps the result a
    physically plausible dose field (measured: peak rises ~9%, total variation
    essentially unchanged, so the target stays as learnable as the naive one).

    Ground truth is used here purely as the regression *label*. Nothing about
    the box -- its position, orientation, or extent -- depends on it.

    Runs on the box's bounding-box crop rather than the whole CT: identical
    result, but the full-volume subtraction was over half the cost.
    """
    dose_sub, sub_geom = crop_to_box(
        dose, geom, ray_source, ray_target, entry_depth, grid
    )
    points = grid_points_world(ray_source, ray_target, entry_depth, grid)
    if dose_sub.size == 0:
        return np.zeros(grid.shape, dtype=np.float32)

    rendered = np.empty(sub_geom.shape, dtype=np.float32)
    x = sample_dose(dose_sub, sub_geom, points)
    for _ in range(iterations):
        rendered[...] = 0.0
        render_to_volume(
            x, ray_source, ray_target, entry_depth, grid, sub_geom, out=rendered
        )
        residual = dose_sub - rendered
        x = np.clip(
            x + relaxation * sample_dose(residual, sub_geom, points), 0.0, None
        )
    return x



def build_network_input(ct_cuboid: np.ndarray, energy_mev: float) -> np.ndarray:
    """Assemble the ``(2, depth, u, v)`` network input: normalized CT + energy."""
    ct_norm = np.clip(ct_cuboid, *HU_CLIP) / HU_SCALE
    energy = np.full_like(ct_norm, energy_mev / MAX_ENERGY_MEV)
    return np.stack([ct_norm, energy]).astype(np.float32)



def stopping_power_from_hu(hu: np.ndarray) -> np.ndarray:
    """Relative stopping power from Hounsfield units (water = 1.0 at 0 HU).

    The standard linear approximation. Good enough for a network input channel;
    it is not a calibrated clinical curve.
    """
    return np.clip(1.0 + hu / HU_SCALE, 0.0, None)



def build_wepl_channel(
    ct_cuboid: np.ndarray, depth_spacing: float, normalizer: float = WEPL_SCALE_MM
) -> np.ndarray:
    """Water-equivalent path length along the beam, as a ``(depth, u, v)`` channel.

    Axis 0 of the cuboid *is* the beam direction, so WEPL is one cumulative sum
    -- exact, and far easier for the network than inferring range from a limited
    receptive field. The Bragg peak sits where ``WEPL = R(energy)``, a 1-D
    lookup, and small range errors are punished hard by the IDD metric and 1 mm
    gamma.

    Lives here rather than in the dataset so training and inference build this
    channel with **the same code**, exactly as they already do for
    :func:`build_network_input`. Two implementations of "what the network sees"
    is a bug class that fails silently: both run, and only one matches what the
    model was trained on.
    """
    wepl = np.cumsum(stopping_power_from_hu(ct_cuboid), axis=0) * depth_spacing
    return (wepl / normalizer).astype(np.float32)



def render_to_volume(
    prediction: np.ndarray,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    geom: VolumeGeometry,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Map a predicted beamlet cuboid back onto the CT voxel grid.

    The submission contract requires each dose map to sit on exactly the input
    image's grid, so the prediction has to come back out of the beamlet frame.
    Only CT voxels inside the box's world bounding box are touched.
    """
    if out is None:
        out = np.zeros(geom.shape, dtype=np.float32)

    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    depth, u, v = grid.axis_coordinates()
    box_origin = np.asarray(ray_source, dtype=np.float64) + entry_depth * direction

    lo, hi = _box_index_bounds(ray_source, ray_target, entry_depth, grid, geom)
    if np.any(hi <= lo):
        return out

    iz, iy, ix = np.meshgrid(
        np.arange(lo[2], hi[2]),
        np.arange(lo[1], hi[1]),
        np.arange(lo[0], hi[0]),
        indexing="ij",
    )
    voxels_world = (
        np.stack([ix, iy, iz], axis=-1) * geom.spacing + geom.origin
    )

    # World -> beamlet-local: project onto the (orthonormal) frame axes.
    relative = voxels_world - box_origin
    local = np.stack(
        [
            (relative @ direction - depth[0]) / grid.depth_spacing,
            (relative @ lat_u - u[0]) / grid.lat_u_spacing,
            (relative @ lat_v - v[0]) / grid.lat_v_spacing,
        ]
    )
    values = map_coordinates(
        prediction,
        local.reshape(3, -1),
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(ix.shape)

    out[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]] = values.astype(np.float32)
    return out
