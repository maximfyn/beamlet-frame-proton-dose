"""Geometry-only beamlet localization for DoseRAD proton dose prediction.

Every quantity here is derived from the beam geometry that a submission
legitimately receives at inference time (``ray_source``, ``ray_target``,
``energy``) plus the patient CT. **Ground-truth dose is never read by any
function that builds model input.** The only function that touches GT dose is
``models.geometry_torch.sample_dose``, which produces the regression *label*
for training and is never called on the inference path.

The grid
--------
Each beamlet gets a box aligned to its own ray:

* ``depth``    -- along the ray, anchored at the raytraced patient surface
* ``lat_u``    -- in the transverse (x-y) plane, perpendicular to the ray
* ``lat_v``    -- world z

Beam direction always lies in the x-y plane for this dataset (gantry rotates
about z, couch angle is absent), verified over every ray of five patients.
That makes world z exactly perpendicular to every beam, so ``lat_v`` uses the
CT's native 3 mm z spacing, while ``depth``/``lat_u`` use the native 1 mm
in-plane spacing.

**Matching the spacing is not matching the phase.** Equal spacing does not mean
``lat_v`` escapes interpolation: the box's z samples sit at
``ray_source_z + (k - (n_lat_v - 1) / 2) * 3`` -- ``(k - 11.5) * 3`` for the
released 24-voxel box -- whose offset from the CT z planes is arbitrary and
measures a **median 0.40-0.63 voxels** over every ray of five
patients, the worst case for linear interpolation in both sampling and
rendering. Snapping that origin to the CT lattice would be a <=1.5 mm shift of a
registration anchor and remove ~25% of the resampling floor; it is **not
implemented here**, and pre-compensated labels address the same floor instead.

The frame is built from the ray endpoints and a cross product rather than from
``pyRadPlan.geometry.get_beam_rotation_matrix``: this dataset's ``gantry_angle``
sign convention is mirrored relative to pyRadPlan's (the x component of the
resulting axis comes out negated), and deriving the axis straight from
``ray_target - ray_source`` is both exact for divergent rays and free of that
ambiguity.

The default box is ``(384, 64, 16)`` at ``(1, 1, 3)`` mm (the released models use
``(384, 64, 24)``), **anchored where the box first
reaches the body**, not where the central axis does — one rule, every beamlet,
pinned by ``test_entry_is_box_anchored_for_every_beamlet`` and
``test_the_axis_rule_stays_deleted``.

**Entry depth is a registration anchor, not a physical surface.** Changing the
rule moves nearly every anchor at once, so the convention and the shards must
change together and any existing checkpoint is invalidated — which is why
:mod:`models.predictor` refuses a checkpoint whose recorded ``grid`` disagrees
with its own.

**Labels are pre-compensated.** Naive sample-and-render-back loses more accuracy
than a good model's entire error budget, and the loss is sub-voxel misalignment,
so no choice of box orientation avoids it. ``models.geometry_torch.precompensate``
solves for the label whose *rendered* form matches the CT-grid truth, in
preprocessing, so training stays a plain beamlet-frame loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# CT Hounsfield window used to normalize the density channel.
HU_CLIP = (-1000.0, 2000.0)
HU_SCALE = 1000.0

# Normalization constant for the energy channel. The DoseRAD proton cohort spans
# 31.7-200.8 MeV; the rounder cap leaves headroom without rescaling behaviour.
MAX_ENERGY_MEV = 250.0

# WEPL is normalized by roughly the deepest range in the cohort so the channel
# stays O(1); 200.8 MeV reaches a little under 300 mm of water.
WEPL_SCALE_MM = 300.0

# Hounsfield value above which a voxel counts as patient rather than air, used
# to find where the ray enters the body.
BODY_HU_THRESHOLD = -500.0

AIR_HU = -1024.0


def body_contour(ct: "np.ndarray", air_hu: float = AIR_HU) -> "np.ndarray":
    """``ct > air_hu`` with every per-slice interior pocket closed.

    **`ct > AIR_HU` IS AN HU THRESHOLD AND THE LABELS ARE MASKED BY A
    CONTOUR.** Internal air stored at the reconstruction floor -- bowel gas,
    trachea, rectum -- is *excluded* by a threshold and *enclosed* by a closed
    curve, so the bare comparison punches holes straight through the patient and
    deletes dose their ground truth keeps. Measured on their own released labels: `1ABB070` carries dose in **12.6%** of its pocket voxels
    at 1.3e-05 of peak, against **5.6e-09** in the FOV padding. Shipping the bare
    threshold costs measurably in `idd`.

    **Per axial SLICE, not in 3-D**, which is what a contour is: a closed
    curve drawn on one slice. A 3-D fill would also close a pocket that is open
    to the outside on its own slice but capped above and below.
    A pocket touching the slice border is connected to the background and
    stays open, which is what keeps the FOV padding out.

    **One pass over the volume, not one call per slice.** The obvious spelling
    -- `binary_fill_holes` per slice -- is **2.8-3.2x slower** for a bit-identical
    result (590 vs 211 ms on `1ABB070`, 1490 vs 473 ms on `1THB054`), and runtime
    carries **2 of the 7** rank weights. Label the complement once with in-plane
    connectivity, then keep the components that never reach a slice border.
    """
    from scipy import ndimage

    body = ct > air_hu
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    labels, n = ndimage.label(~body, structure=structure)
    if n == 0:
        return body
    edge = np.concatenate([labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                           labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])
    outside = np.zeros(n + 1, dtype=bool)
    outside[np.unique(edge)] = True
    # Label 0 is `body` itself, never a pocket; a body voxel on the border
    # would otherwise mark it "outside" and the OR below would be the only thing
    # saving the patient.
    outside[0] = False
    return ~outside[labels] | body


@dataclass(frozen=True)
class BeamletGrid:
    """Fixed-shape sampling box, in beamlet-local coordinates.

    The extents are a global constant of the pipeline -- chosen once from the
    dataset's physics, never per beamlet from that beamlet's own dose (deriving
    the box from the target is exactly the leak this module exists to remove).

    Defaults were picked by measuring captured dose mass over beamlets spanning
    the full 31.7-200.8 MeV range: a 384 mm depth window holds ~100% of dose
    mass, +-32 mm lateral holds ~98% (the remainder is the low-amplitude nuclear
    halo, which spreads much further than it is worth covering). Treat the
    lateral extents and ``entry_margin_mm`` as tunable hyperparameters.

    Re-measured 2026-08-13 under box anchoring over 144 beamlets from 12
    patients, thoracic and abdominal. The depth *extent* survived; its *placement* did not. See ``entry_margin_mm``.
    """

    n_depth: int = 384
    depth_spacing: float = 1.0
    n_lat_u: int = 64
    lat_u_spacing: float = 1.0
    n_lat_v: int = 16
    lat_v_spacing: float = 3.0
    # How far upstream of the entry the box starts -- 16 mm until 2026-08-13.
    # That figure was justified partly by "protons deposit a little dose in air
    # just before entry", which is false: ground truth is exactly zero outside
    # the body, verified empirically. What remains is raytrace accuracy, ~1-2
    # voxels, and under box anchoring the axis-vs-box gap the margin also
    # absorbed is zero by construction.
    #
    # The 12 mm this frees is not saved, it is moved to the distal end, where it
    # was needed: thoracic beamlets range further in mm through low-density lung,
    # and at [-16, 367] the worst beamlet lost 1.67% of its dose mass past the
    # back face. Same box, same cost, better placed -- [-4, 379] cuts that worst
    # case to 0.22% and lowers the beam-MAE floor slightly. Extending to
    # [-4, 411] instead cuts it to 0.005% but does not move the floor at all, so
    # it buys sub-10%-isodose tail for +8% box volume, which the
    # runtime does not pay for, so it was declined.
    entry_margin_mm: float = 4.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_depth, self.n_lat_u, self.n_lat_v)

    def as_dict(self) -> dict:
        """The box as a shard sidecar and a checkpoint record it -- one dict,
        both places.

        **The spacings are here because a box is not its shape.** Until
        2026-08-23 three places described a box by ``shape`` +
        ``entry_margin_mm`` alone, and the predictor's mismatch guard compared
        exactly those -- so ``(768, 128, 40)`` at 0.5 mm depth and the same
        shape at 1.0 mm read as the *same box*. That is the `_fd` family, i.e.
        precisely the fine-spacing variant a reshard would use, and the
        failure is the invisible one the guard exists to prevent: half the depth
        reach, every Bragg curve read at twice its true depth, nothing raised.
        Pinned by ``test_the_box_record_carries_its_spacings``.
        """
        return {
            "shape": list(self.shape),
            "depth_spacing": self.depth_spacing,
            "lat_u_spacing": self.lat_u_spacing,
            "lat_v_spacing": self.lat_v_spacing,
            "entry_margin_mm": self.entry_margin_mm,
            "anchor": "box",
        }

    @classmethod
    def from_dict(cls, record: dict) -> "BeamletGrid":
        """Rebuild a box from what a shard or a checkpoint stored.

        Missing spacings mean a record written before ``as_dict`` existed, and
        every box that predates it is the default lattice -- so the defaults are
        the right fill, not a guess. Do not extend that leniency to ``shape``:
        a record without one describes nothing.
        """
        n_depth, n_lat_u, n_lat_v = (int(v) for v in record["shape"])
        default = cls()
        return cls(
            n_depth=n_depth,
            n_lat_u=n_lat_u,
            n_lat_v=n_lat_v,
            depth_spacing=float(record.get("depth_spacing", default.depth_spacing)),
            lat_u_spacing=float(record.get("lat_u_spacing", default.lat_u_spacing)),
            lat_v_spacing=float(record.get("lat_v_spacing", default.lat_v_spacing)),
            entry_margin_mm=float(
                record.get("entry_margin_mm", default.entry_margin_mm)),
        )

    def axis_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Beamlet-local coordinates (mm) along each axis.

        Depth runs from ``-entry_margin_mm`` (upstream of the surface) forward;
        the lateral axes are centred on the ray.
        """
        depth = np.arange(self.n_depth) * self.depth_spacing - self.entry_margin_mm
        lat_u = (np.arange(self.n_lat_u) - (self.n_lat_u - 1) / 2.0) * self.lat_u_spacing
        lat_v = (np.arange(self.n_lat_v) - (self.n_lat_v - 1) / 2.0) * self.lat_v_spacing
        return depth, lat_u, lat_v


@dataclass(frozen=True)
class VolumeGeometry:
    """Geometry of a SimpleITK volume, in the identity-direction case."""

    origin: np.ndarray  # world (x, y, z) of voxel [0, 0, 0]
    spacing: np.ndarray  # (x, y, z) mm
    shape: tuple[int, int, int]  # numpy array shape, (z, y, x)

    @classmethod
    def from_sitk(cls, image) -> "VolumeGeometry":
        direction = np.asarray(image.GetDirection()).reshape(3, 3)
        if not np.allclose(direction, np.eye(3)):
            raise ValueError(
                "Non-identity direction cosines are not supported; "
                f"got {direction.tolist()}"
            )
        return cls(
            origin=np.asarray(image.GetOrigin(), dtype=np.float64),
            spacing=np.asarray(image.GetSpacing(), dtype=np.float64),
            shape=tuple(reversed(image.GetSize())),
        )

    def world_to_index(self, points_world: np.ndarray) -> np.ndarray:
        """World (..., 3) in (x, y, z) -> continuous index (..., 3) in (x, y, z)."""
        return (points_world - self.origin) / self.spacing


def beam_frame(
    ray_source: np.ndarray, ray_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal beamlet frame ``(depth, lat_u, lat_v)`` in world coordinates.

    ``lat_v`` is world z (always perpendicular to the beam for this dataset) and
    ``lat_u`` completes a right-handed frame. Raises if the ray turns out to
    have a z component, which would invalidate the native-spacing assumption
    rather than silently degrade sampling quality.
    """
    direction = np.asarray(ray_target, dtype=np.float64) - np.asarray(
        ray_source, dtype=np.float64
    )
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise ValueError("ray_source and ray_target coincide; beam direction undefined")
    direction = direction / norm

    if abs(direction[2]) > 1e-9:
        raise ValueError(
            "Beam direction has a z component "
            f"({direction[2]:.3e}); the lat_v=world-z assumption does not hold"
        )

    lat_v = np.array([0.0, 0.0, 1.0])
    lat_u = np.cross(direction, lat_v)
    lat_u /= np.linalg.norm(lat_u)
    return direction, lat_u, lat_v


def _first_depth_reaching_volume(
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    direction: np.ndarray,
    lat_u: np.ndarray,
    lat_v: np.ndarray,
    grid: BeamletGrid,
) -> float:
    """Smallest depth at which any box sample can lie inside the volume.

    A slab test against the volume's world bounds, widened by the box's lateral
    reach so it stays a lower bound on the true answer rather than an estimate of
    it. Purely a search-window narrowing: depths below this cannot produce a hit,
    because every sample there rounds to an index outside the array and counts as
    air. Returns 0.0 when the ray starts inside.

    This matters because the sources sit ~1060 mm out while entry lands at
    790-950 mm, so an unclipped walk spends ~85% of its samples in guaranteed
    air, and this walk pays 1024 lateral samples for every one of those depths.
    """
    _, u, v = grid.axis_coordinates()
    # world_to_index rounds, so a point counts as inside while its index is in
    # [-0.5, n - 0.5). Use those bounds, not the voxel centres.
    lo_world = geom.origin - 0.5 * geom.spacing
    hi_world = geom.origin + (np.array(geom.shape)[::-1] - 0.5) * geom.spacing
    reach = np.abs(u).max() * np.abs(lat_u) + np.abs(v).max() * np.abs(lat_v)
    lo_world = lo_world - reach
    hi_world = hi_world + reach

    source = np.asarray(ray_source, dtype=np.float64)
    start = 0.0
    for axis in range(3):
        d = direction[axis]
        if abs(d) < 1e-12:
            # Parallel to this slab: either always within it or never.
            if not (lo_world[axis] <= source[axis] <= hi_world[axis]):
                return np.inf
            continue
        t0 = (lo_world[axis] - source[axis]) / d
        t1 = (hi_world[axis] - source[axis]) / d
        start = max(start, min(t0, t1))
    return start


def find_entry_depth_box(
    ct: np.ndarray,
    geom: VolumeGeometry,
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    grid: BeamletGrid,
    hu_threshold: float = BODY_HU_THRESHOLD,
    max_depth: float = 1400.0,
    step: float = 1.0,
    chunk: int = 16,
) -> float | None:
    """Depth at which the beamlet *box* first reaches the patient.

    The anchoring rule for the whole pipeline, applied to every beamlet. Walks
    the depths sampling the full lateral extent the network is fed, and returns
    the first depth at which any sample is inside the body (nearest neighbour;
    out-of-volume counts as air).

    Sampling the whole extent rather than the central axis is what makes this one
    continuous function of geometry: a beamlet that grazes the patient still
    resolves, so there is no tangential class, no gate and no discontinuity in
    the anchor. The anchor is a box corner rather than the beam core -- less
    physically meaningful, but consistent, which is what a registration anchor
    needs.

    Returns ``None`` for a genuine miss, which the caller must still handle: the
    box is wide (+-32 mm), not infinite.

    Walks in chunks of ``chunk`` depths and stops at the first hit, starting from
    the first depth that can reach the volume at all. ``chunk`` is a pure
    performance knob -- the result is identical for any value, which
    ``tests/test_geometry.py`` asserts.

    **This function has a device twin that must not drift from it**:
    ``geometry_torch.find_entry_depth_box``, which inference uses on CUDA while
    preprocessing builds every shard with *this* one. **This is the definition**
    and that one is held to it by ``tests/test_entry_walk.py`` and
    ``tests/test_geometry_torch.py``, with ``==``. A change here that is not
    mirrored there anchors training and inference on different boxes -- the
    silent failure `models/geometry_torch.py`'s docstring describes, where
    neither side looks wrong on its own.

    **A larger chunk is not a faster one.** 64 overshoots: past the slab bound the walk needs a median of 120 mm on an
    abdominal case and 186 on a thoracic one, so a 64-deep block computes ~2
    blocks' worth of lateral samples beyond the hit and throws them away.
    Measured over 24 beamlets on each of `1ABB020` / `1THB054`::

        chunk      8      16      32      64     128     256
        1ABB020 11.77   11.87   12.59   13.21   16.62   23.70   ms/beamlet
        1THB054 17.85   17.70   18.32   19.12   23.28   25.41
    """
    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    _, u, v = grid.axis_coordinates()

    steps = np.arange(0.0, max_depth, step)
    start = _first_depth_reaching_volume(
        geom, ray_source, direction, lat_u, lat_v, grid
    )
    if not np.isfinite(start):
        # The ray never reaches the volume's slab: no entry, as in the torch twin.
        return None
    begin = int(np.searchsorted(steps, start, side="right")) - 1
    begin = max(begin, 0)

    source = np.asarray(ray_source, dtype=np.float64)
    lateral = u[:, None, None] * lat_u + v[None, :, None] * lat_v
    upper = np.array([geom.shape[2], geom.shape[1], geom.shape[0]])

    for lo in range(begin, steps.size, chunk):
        block = steps[lo : lo + chunk]
        points = source + block[:, None, None, None] * direction + lateral

        index = np.round(geom.world_to_index(points)).astype(int)
        inside = np.all((index >= 0) & (index < upper), axis=-1)
        if not inside.any():
            continue

        hu = np.full(inside.shape, AIR_HU)
        valid = index[inside]
        hu[inside] = ct[valid[:, 2], valid[:, 1], valid[:, 0]]

        hits = np.flatnonzero(np.any(inside & (hu > hu_threshold), axis=(1, 2)))
        if hits.size:
            return float(block[hits[0]])
    return None


def _box_index_bounds(
    ray_source: np.ndarray,
    ray_target: np.ndarray,
    entry_depth: float,
    grid: BeamletGrid,
    geom: VolumeGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Index-space ``(lo, hi)`` in (x, y, z) covering the beamlet box, clamped."""
    direction, lat_u, lat_v = beam_frame(ray_source, ray_target)
    depth, u, v = grid.axis_coordinates()
    box_origin = np.asarray(ray_source, dtype=np.float64) + entry_depth * direction

    corners = np.array(
        [
            box_origin + d * direction + uu * lat_u + vv * lat_v
            for d in (depth[0], depth[-1])
            for uu in (u[0], u[-1])
            for vv in (v[0], v[-1])
        ]
    )
    lo = np.floor(geom.world_to_index(corners.min(axis=0))).astype(int)
    hi = np.ceil(geom.world_to_index(corners.max(axis=0))).astype(int) + 1
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [geom.shape[2], geom.shape[1], geom.shape[0]])
    return lo, hi
