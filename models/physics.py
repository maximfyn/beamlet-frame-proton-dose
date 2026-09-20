"""The proton beam model, as a *prior* on the beamlet's depth-dose curve.

What this is for
----------------
Without it the network would receive ``energy`` only as a scalar channel and
have to reverse-engineer the entire Bragg curve family from data. The curves
themselves are public: pyRadPlan distributes matRad's ``Generic`` proton machine under the
BSD-3-Clause license, exported here as ``models/bragg_generic.npz`` (see NOTICE),
and the dataset paper §II.C.2 states the DoseRAD beam
parameters were *"derived from matRad's native 'Generic' proton machine"*.

The link is exact, not approximate: the release's **85 energies are a subset of
the machine's 114**, matching to 4.9e-5 MeV (measured over all 85 entries of
the dataset's energy table, ``beam_parameters.json``). So a beamlet's Bragg curve is a table lookup with no
interpolation in energy.

**A prior, not a target.** DoseRAD simulated with Geant4; the Generic machine
supplied the *source* parameters, not the ground truth its own fitted kernels
would predict. Everything here is a feature handed to the network, never a
quantity anything is regressed onto or scored against.

The geometry it assumes
-----------------------
**Parallel**, and this was checked rather than inherited. The release contradicts
itself -- ``beam_parameters.json`` declares ``"source_model": "point_source"``
while the paper §II.C.2 says spots were *"initialized as parallel pencil beams on
a source plane located 100 cm upstream"*. Two independent measurements settle it
for the data we are actually given:

* every ray of every beam in the five plans is **parallel to within 9e-7
  degrees**, each ``ray_source`` exactly 1000.0000 mm upstream of its own
  ``ray_target``, with the 15 sources spread over ~85 mm. A point source would
  put all 15 at one point;
* the dose itself does not diverge. A point source 1000 mm upstream predicts a
  lateral centroid tilt of **1.00 mrad per mm** of the ray's off-axis offset;
  measured over 445 beamlets the regression slope is **+0.024 mrad/mm in u and
  +0.056 in v (r = +0.003, +0.027)** -- zero. The residual drift that remains
  (median 6.4 mrad in u, and 12.5 mrad on *central* rays, where a point source
  predicts exactly zero) is heterogeneity deflection, uncorrelated with offset.

So ``models/geometry.py``'s constant-width, ray-aligned box is the correct
assumption, and ``sigma_spot`` from the release is the spot size **at the patient**
rather than at an isocentre to be projected. ``"point_source"`` describes the
machine the parameters came from, not the geometry that was simulated.

Cost
----
No preprocessing and no rebuild. The machine table is public and long predates
the challenge, so the rules' cut-off for additional public data is met. At inference the prior is a ``cumsum`` plus a
gather along depth, both already on the device the CT sits on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

# Where pyRadPlan's machine file sat when the table was exported, relative to the
# repo root; recorded in the export as its provenance. `TABLE_RELPATH` is the
# copy that travels; `scripts/data/export_bragg_table.py` re-derives or checks it
# from a pyRadPlan clone and the dataset.
MACHINE_RELPATH = "pyRadPlan/data/machines/protons_Generic.mat"

# The tracked export, and the default source. Loading prefers it precisely
# because it is the one that exists everywhere the code runs -- a fallback to
# the `.mat` that worked locally and failed where it was not vendored would be the same
# class of bug in the other direction.
TABLE_RELPATH = "models/bragg_generic.npz"

# The common water-equivalent-depth grid every Bragg curve is resampled onto.
# 400 mm covers the deepest curve in the machine (361.9 mm at 236.1 MeV) with
# room to spare; 0.5 mm is half the box's depth spacing, so the resample never
# limits resolution along the axis the curve is read out on.
WEPL_STEP_MM = 0.5
WEPL_MAX_MM = 400.0

# Curves are stored normalised to unit peak: the network learns the scale, and a
# prior carrying matRad's absolute normalisation would invite reading it as a
# dose prediction. See the module docstring -- prior, not target.


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@lru_cache(maxsize=4)
def load_bragg_table(path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(energies, curves)`` -- ``(n_e,)`` MeV and ``(n_e, n_wepl)`` unit-peak.

    With no ``path``, reads the tracked ``.npz`` export, which is the only copy
    that exists in a container or a fresh checkout. Given a ``.mat`` it re-derives
    the table from the machine itself, which is what the exporter does and what
    ``--check`` compares against.

    Each machine energy's ``(depths, Z)`` pair is resampled onto the common
    grid; past a curve's last tabulated depth the value is **exactly zero**
    rather than held, which is what makes the prior vanish past the range. The
    truncation discards a tail worth ~0.15% of peak.

    Cached because parsing costs ~0.4 s and every dataloader worker and every
    predictor wants the same table.
    """
    if path is None:
        table = repo_root() / TABLE_RELPATH
        if not table.exists():
            raise FileNotFoundError(
                f"{table} is missing. It is a tracked artifact -- regenerate it "
                "with scripts/data/export_bragg_table.py from a pyRadPlan clone "
                f"(machine vendored as {MACHINE_RELPATH}) and the dataset's "
                "beam_parameters.json."
            )
        stored = np.load(table, allow_pickle=False)
        if (float(stored["wepl_step_mm"]) != WEPL_STEP_MM
                or float(stored["wepl_max_mm"]) != WEPL_MAX_MM):
            raise ValueError(
                f"{table} was exported on a different depth grid "
                f"({float(stored['wepl_step_mm'])} mm to "
                f"{float(stored['wepl_max_mm'])} mm) than this module reads it "
                f"on ({WEPL_STEP_MM} mm to {WEPL_MAX_MM} mm); re-export"
            )
        return stored["energies_mev"], stored["curves"]

    import scipy.io as sio

    mat = sio.loadmat(str(path), simplify_cells=True)
    data = mat["machine"]["data"]

    grid = wepl_grid()
    energies = np.array([float(entry["energy"]) for entry in data], dtype=np.float64)
    curves = np.zeros((len(data), grid.size), dtype=np.float32)
    for i, entry in enumerate(data):
        depths = np.asarray(entry["depths"], dtype=np.float64)
        z = np.asarray(entry["Z"], dtype=np.float64)
        # `offset` is 0 for every energy in this machine, but it is part of
        # matRad's depth convention, so apply it rather than assume it.
        depths = depths + float(entry.get("offset", 0.0) or 0.0)
        peak = float(z.max())
        if peak <= 0:
            continue
        resampled = np.interp(grid, depths, z / peak, left=0.0, right=0.0)
        curves[i] = resampled

    order = np.argsort(energies)
    return energies[order], curves[order]


LATERAL_KEYS = ("sigma1", "sigma2", "weight")


@lru_cache(maxsize=4)
def load_lateral_table(path: str | None = None) -> dict[str, np.ndarray]:
    """``{sigma1, sigma2, weight}``, each ``(n_e, n_wepl)``, on the WEPL grid.

    matRad's Generic machine is ``dataType: 'doubleGauss'`` -- its lateral kernel
    is a narrow Gaussian plus a wide one carrying the nuclear halo, tabulated per
    energy **and per depth** on the same axis as ``Z``.

    ``sigma1`` is **exactly 0 at zero depth**, which is the tell that these are
    the *scattering* kernels with the initial spot excluded. They therefore
    compose with DoseRAD's own spot rather than replacing it::

        sigma_core(d) = sqrt(sigma_spot**2 + sigma1(d)**2)

    with ``sigma_spot`` from the release's ``beam_parameters.json`` (4.00-7.99 mm)
    -- which is itself matRad's ``initFocus.emittance.sigmaX``, to 5 decimals. And
    variances simply add because the beam is **parallel**, measured: there is no
    geometric divergence term (`models/physics.py` module docstring).

    Still a prior. DoseRAD simulated with Geant4 under a *single*-Gaussian
    **source**; the halo in the patient is physics their simulation produces
    regardless, and this is matRad's fitted description of that shape, not theirs.
    """
    stored = _load_npz(path)
    return {key: stored[key] for key in LATERAL_KEYS}


def _load_npz(path: str | None):
    if path is not None:
        raise ValueError("lateral kernels are read from the export, not the .mat")
    table = repo_root() / TABLE_RELPATH
    stored = np.load(table, allow_pickle=False)
    missing = [k for k in LATERAL_KEYS if k not in stored]
    if missing:
        raise KeyError(
            f"{table} predates the lateral kernels ({missing} absent); "
            "re-run scripts/data/export_bragg_table.py"
        )
    return stored


def wepl_grid() -> np.ndarray:
    """The common water-equivalent depth axis, in mm."""
    return np.arange(0.0, WEPL_MAX_MM + 0.5 * WEPL_STEP_MM, WEPL_STEP_MM)


def nearest_energy_index(energy_mev: float, energies: np.ndarray) -> int:
    """Index of the machine energy for this beamlet.

    A nearest-neighbour lookup rather than an interpolation, because the
    dataset's 85 energies *are* machine energies -- verified to 4.9e-5 MeV over
    every energy present in the local plans. :func:`assert_energy_is_tabulated`
    is what stops that silently becoming an approximation on a test patient
    carrying an energy this machine does not have.
    """
    return int(np.abs(energies - float(energy_mev)).argmin())


ENERGY_TOLERANCE_MEV = 1e-2

# Inference-path near-misses, so a sweep can report them once instead of per
# beamlet. Not a counter of *errors*: a near-miss is a slightly worse input
# feature, and the submission still has to produce dose for that beamlet.
untabulated_energies: dict[float, int] = {}


def assert_energy_is_tabulated(
    energy_mev: float, tolerance: float = ENERGY_TOLERANCE_MEV, strict: bool = True
) -> None:
    """Check a beamlet's energy against the machine's table.

    The prior's justification is that the lookup is *exact*: the release's 85
    energies are machine energies to 4.9e-5 MeV. An energy off the table would
    quietly degrade to "the nearest curve", which is a different and unvalidated
    thing -- and the hidden test set is *"much larger and more varied"* than what
    we hold, so it is not hypothetical.

    **Strict while training, lenient while inferring, and the asymmetry is
    deliberate.** During training an off-table energy means the cohort is not
    what we think it is and we want to hear about it immediately. On the
    inference path the same raise would abort the beamlet, and a run that
    throws returns nothing at all. The nearest curve is a ~1-2 MeV interpolation
    error in one *input channel*, which costs a little accuracy on that beamlet
    and nothing else. Degrading beats failing; the near-misses are counted so the choice is
    visible rather than silent.
    """
    energies, _ = load_bragg_table()
    i = nearest_energy_index(energy_mev, energies)
    delta = abs(float(energies[i]) - float(energy_mev))
    if delta <= tolerance:
        return
    if strict:
        raise ValueError(
            f"energy {energy_mev:.4f} MeV is {delta:.4f} MeV from the nearest "
            f"machine energy {energies[i]:.4f}; the Bragg prior is a table "
            "lookup and this checkpoint was trained on exact hits"
        )
    key = round(float(energy_mev), 4)
    if len(untabulated_energies) >= 1024:
        # A counter, not a log: a long-lived process fed many off-table
        # energies must not grow a dict without bound.
        untabulated_energies.clear()
    if key not in untabulated_energies:
        print(f"[physics] energy {energy_mev:.4f} MeV is not tabulated "
              f"({delta:.4f} MeV from {energies[i]:.4f}); using the nearest "
              "Bragg curve for the prior channel", flush=True)
    untabulated_energies[key] = untabulated_energies.get(key, 0) + 1


_TABLE_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def bragg_curves_tensor(device, dtype=torch.float32) -> torch.Tensor:
    """``(n_e, n_wepl)`` curve table, resident on ``device``."""
    key = (str(device), str(dtype))
    if key not in _TABLE_CACHE:
        _, curves = load_bragg_table()
        _TABLE_CACHE[key] = torch.as_tensor(curves, dtype=dtype, device=device)
    return _TABLE_CACHE[key]


def relative_stopping_power(ct: torch.Tensor, mode: str = "linear") -> torch.Tensor:
    """HU → the per-voxel factor accumulated as depth, clamped at zero.

    Named for stopping power, but under ``hlut`` the values are **mass density**
    (g/cm³) used in its place, so the accumulated depth tracks water-equivalent
    depth without being it (paper §2.1).

    ``linear`` is ``1 + HU/1000``, what `models/geometry.py` has always used.
    ``hlut`` is a piecewise-linear lookup, and it is **the dataset's table exactly**: the
    dataset's own ``proton/training/beam_parameters.json`` gives ``hu_to_density``
    as 10 linearly interpolated anchors "used by G4DCM", and they match this table
    at **all ten** (2026-08-27). The container's ``Data1.dat`` ``:CT2D`` differs
    at the second knot (-600 against -999) and is a left-over from another study,
    not the DoseRAD conversion.

    Measured on the Bragg prior, 8 patients per region: RMS median
    **0.0412 → 0.0322 abdominal** and 0.0636 → 0.0620 thoracic, peak within 3 mm
    **71% → 79%** and 57% → 62%. The linear form puts fat at 0.900 against
    0.917 and bone at 1.700 against 1.484, so a body with real subcutaneous fat
    accumulates depth too slowly and ranges too long.
    """
    from .geometry import HU_SCALE

    if mode == "linear":
        return (1.0 + ct / HU_SCALE).clamp(min=0.0)
    if mode != "hlut":
        raise ValueError(f"unknown rsp mode {mode!r}; have 'linear', 'hlut'")

    table = torch.as_tensor(_load_npz(None)["hlut"], dtype=ct.dtype, device=ct.device)
    # `.contiguous()`: a column of a (10, 2) table is strided, and
    # `torch.searchsorted` warns and copies when its boundary tensor is.
    hu, rsp = table[:, 0].contiguous(), table[:, 1]
    index = torch.clamp(torch.searchsorted(hu, ct.contiguous()) - 1, 0, len(hu) - 2)
    left, right = hu[index], hu[index + 1]
    frac = ((ct - left) / (right - left)).clamp(0.0, 1.0)
    out = rsp[index] * (1.0 - frac) + rsp[index + 1] * frac
    # Outside the table, hold the ends -- and air stays air.
    out = torch.where(ct <= hu[0], torch.zeros_like(out), out)
    return torch.where(ct >= hu[-1], rsp[-1].expand_as(out), out).clamp(min=0.0)


def build_lateral_prior(
    ct_cuboid: torch.Tensor,
    energy_mev: float,
    depth_spacing: float,
    rsp: str = "linear",
    strict_energy: bool = True,
) -> torch.Tensor:
    """The analytic double-Gaussian lateral kernel on the box, ``(depth, u, v)``.

    Normalised to sum 1 over ``(u, v)`` at every depth, so it is directly
    comparable with -- and composable with -- ``FactorisedDoseNet``'s softmax
    kernel.

    ``sigma_core(d) = sqrt(sigma_spot(E)**2 + sigma1(wepl(d))**2)``, because
    ``sigma1`` is exactly 0 at zero depth and is therefore the *scattering* part
    with the spot excluded. Variances add because the beam is **parallel**
    (measured), so there is no divergence term.

    Each component carries ``1/(2*pi*sigma**2)``. Dropping it looks harmless
    because the kernel is renormalised anyway, but it sets the **core-to-halo
    ratio**, which is the entire content of a double Gaussian -- omitted, the
    halo gets ``(sigma2/sigma_core)**2`` (~22x) too much mass.

    **A prior, and a partial one**: measured L1 residual **0.102 abdominal /
    0.141 thoracic** against real beamlets, so it explains most of the shape and
    not all of it. It is a starting shape for a free kernel, never
    a replacement for one.
    """
    from .geometry import BeamletGrid

    assert_energy_is_tabulated(energy_mev, strict=strict_energy)
    energies, _ = load_bragg_table()
    index = nearest_energy_index(energy_mev, energies)
    stored = _load_npz(None)
    lateral = load_lateral_table()

    # The box handed in, not the default one: a 24-voxel v axis with the
    # default grid's 16 coordinates produced a (D, 64, 16) kernel that the
    # caller's `torch.cat` then refused.
    grid = BeamletGrid(n_depth=ct_cuboid.shape[0], n_lat_u=ct_cuboid.shape[1],
                       n_lat_v=ct_cuboid.shape[2], depth_spacing=depth_spacing)
    _, u_axis, v_axis = grid.axis_coordinates()
    u = torch.as_tensor(u_axis, dtype=torch.float32, device=ct_cuboid.device)
    v = torch.as_tensor(v_axis, dtype=torch.float32, device=ct_cuboid.device)

    n_u, n_v = ct_cuboid.shape[1], ct_cuboid.shape[2]
    u0, v0 = (n_u - 8) // 2, (n_v - 4) // 2
    core = ct_cuboid[:, u0:u0 + 8, v0:v0 + 4]
    wepl = (torch.cumsum(relative_stopping_power(core, rsp).mean(dim=(1, 2)), dim=0)
            * depth_spacing)

    to = lambda a: torch.as_tensor(  # noqa: E731
        a, dtype=torch.float32, device=ct_cuboid.device)
    position = (wepl / WEPL_STEP_MM).long().clamp(0, lateral["sigma1"].shape[1] - 1)
    sigma1 = to(lateral["sigma1"][index])[position]
    sigma2 = to(lateral["sigma2"][index])[position]
    weight = to(lateral["weight"][index])[position]
    spot = float(stored["sigma_spot_mm"][index])

    core_sigma = torch.sqrt(spot ** 2 + sigma1 ** 2).clamp(min=1e-3)[:, None, None]
    halo_sigma = torch.sqrt(spot ** 2 + sigma2 ** 2).clamp(min=1e-3)[:, None, None]
    w = weight[:, None, None]

    radius = u[None, :, None] ** 2 + v[None, None, :] ** 2
    unit = lambda s: torch.exp(-0.5 * radius / s ** 2) / (2.0 * np.pi * s ** 2)  # noqa: E731
    kernel = (1.0 - w) * unit(core_sigma) + w * unit(halo_sigma)
    return kernel / kernel.sum(dim=(1, 2), keepdim=True).clamp(min=1e-30)


#: The depth profile and the body mask of the LAST cuboid, and the cuboid itself.
#: **Keyed on the tensor's IDENTITY, and it holds the tensor** — CPython
#: recycles ids, so an `id()` key could match a dead cuboid and hand one
#: beamlet another's anatomy; a held reference cannot be recycled. The same rule
#: `models/predictor.py`'s three memos use, for the same reason.
_WEPL_MEMO: tuple | None = None


def reset_bragg_memo() -> None:
    """Forget the cached depth profile. **A benchmark needs this**, nothing else
    does: timing one image twice through one process would otherwise measure a
    hit rate the container — one process per job — can never have."""
    global _WEPL_MEMO
    _WEPL_MEMO = None


def _wepl_and_body(ct_cuboid, depth_spacing: float, rsp: str):
    """``(wepl, ct > AIR_HU)`` for this box, reused across the ray's beamlets.

    **ENERGY IS NOT AN ARGUMENT OF EITHER**, which is the whole point: the
    converted HU, its cumulative sum along depth, and the body mask are
    functions of the *cuboid*, and a ray's beamlets share one cuboid by
    construction (`models/predictor.py`, `_cuboid`). Only the curve lookup below
    depends on the energy. Measured on the A10G at the released 384×64×24 box:
    ``build_bragg_channel`` **1.06 ms/beamlet**, of which
    ``relative_stopping_power`` alone is **0.71**.

    Safe to hand the same tensors to several beamlets because both callers
    only read them — ``torch.cumsum`` is out of place and the mask is multiplied,
    not written — and `models/physics.py` contains no in-place operator at all.

    Depth one, and that is not a limitation here: the misses are once per ray
    and the hits are the further energy layers that follow it.
    """
    from .geometry import AIR_HU

    global _WEPL_MEMO
    if _WEPL_MEMO is not None:
        src, spacing, mode, wepl, in_body = _WEPL_MEMO
        if src is ct_cuboid and spacing == depth_spacing and mode == rsp:
            return wepl, in_body
    stopping = relative_stopping_power(ct_cuboid, rsp)
    wepl = torch.cumsum(stopping, dim=0) * depth_spacing
    in_body = ct_cuboid > AIR_HU
    _WEPL_MEMO = (ct_cuboid, depth_spacing, rsp, wepl, in_body)
    return wepl, in_body


def build_bragg_channel(
    ct_cuboid: torch.Tensor,
    energy_mev: float,
    depth_spacing: float,
    strict_energy: bool = True,
    rsp: str = "linear",
) -> torch.Tensor:
    """The Bragg prior on the beamlet box: ``(depth, u, v)``, unit peak.

    A depth coordinate is accumulated along axis 0 -- the box's first axis
    *is* the beam direction, so this is a plain ``cumsum`` of
    ``relative_stopping_power`` (a density for the released models, read as a
    stand-in for water-equivalent depth), the same quantity ``build_wepl_channel`` produces and by the same
    approximation as ``build_wepl_channel`` (``rsp`` selects which: the released
    models use the dataset's HU-to-density table, ``linear`` is 1 + HU/1000).

    The curve is then read out **per voxel** at that voxel's own WEPL, so
    heterogeneity moves the Bragg peak laterally as well as in depth -- a
    beamlet clipping the edge of a rib does not get one flat range. Voxels past
    the curve's end read exactly 0.

    **The prior is zero outside the body, and that is load-bearing rather than
    tidy.** WEPL stops accumulating in air, so for a beamlet whose Bragg peak
    lies beyond the patient the lookup freezes part-way up the curve and stays
    there for the rest of the box -- while the truth falls to zero at the exit
    surface. Unmasked, those beamlets came out *anti*-correlated with the ground
    truth (r ~ -0.7, normalised RMS 0.72) and dragged the p90 to 0.64 against a
    median of 0.067; masking is what makes the tail behave.

    **``ct > -1024`` is the HU contour, which is NOT the dose contour.** The
    release sets HU outside the CT-**and**-MRI intersection to -1024 (paper
    §II.B.1) while masking dose by the CT-**only** contour (§II.C.3), so a shell
    of real tissue reads as air, and -1024 voxels can carry real beam. Measured
    over **both** regions, 8 patients and 320 beamlets each:

    * the shell is real and **thoracic** -- dose inside it is **13.0% of a
      beamlet peak** (median), 29.8% max, against **0.5%** median abdominal;
    * **refilling it fails identically in both.** A 3 mm dilation nearly zeroes
      the *mean* peak offset (thoracic **+4.17 → +0.28 mm**, abdominal
      **+2.97 → +0.21**) while making RMS ~60% **worse** (**0.0636 → 0.1021**,
      **0.0412 → 0.0674**); 6 and 12 mm are worse again. Right on average, wrong
      per beamlet.

    **Do not add a shell correction.** ``idd_distance`` is an **RMS**, so
    trading bias for variance is the wrong direction -- and a constant offset is
    exactly what a network removes for free, which is the whole reason this is an
    input channel and not a target. Note the rejection is the *measurement*,
    not an inability to apply the fix: ``anatomical_region`` ships with every
    image and site-specific routing is allowed, so a thoracic-only refill was
    available and was priced with that assumed.

    **The residual deep bias is not the shell either.** The abdominal shell
    carries 26x less dose yet the bias is only modestly smaller, and a *uniform*
    3 mm refill centres both -- the signature of a roughly constant ~3 mm WEPL
    under-count. ``1 + HU/1000`` is the suspect: it puts fat (-100 HU) at
    0.900 against the table's 0.917 and bone (+700) at 1.700 against 1.484.

    This is the analytic prior, **not** a dose estimate: it is normalised to
    unit peak, carries no lateral kernel and no absolute scale, and costs a
    cumsum rather than the pencil-beam dose calculation that was priced and
    parked.
    """

    assert_energy_is_tabulated(energy_mev, strict=strict_energy)
    energies, _ = load_bragg_table()
    index = nearest_energy_index(energy_mev, energies)

    wepl, in_body = _wepl_and_body(ct_cuboid, depth_spacing, rsp)

    curve = bragg_curves_tensor(ct_cuboid.device)[index]
    prior = _lookup(curve, wepl)
    return (prior * in_body).to(torch.float32)


def _lookup(curve: torch.Tensor, wepl: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``curve`` at ``wepl`` mm; 0 past the end.

    Written out rather than deferred to ``grid_sample`` because the out-of-range
    convention is the whole point: ``grid_sample`` clamps to the border value,
    which would hold the curve's last sample forever and put a floor of dose
    everywhere past the range. That is the opposite of what the prior is for.
    """
    position = wepl / WEPL_STEP_MM
    last = curve.numel() - 1
    lower = position.floor().clamp(0, last)
    frac = (position - lower).clamp(0.0, 1.0)
    lo = lower.to(torch.long)
    hi = (lo + 1).clamp(max=last)
    value = curve[lo] * (1.0 - frac) + curve[hi] * frac
    return torch.where(position > last, torch.zeros_like(value), value)


def build_bragg_idd_prior(
    ct_cuboid: torch.Tensor,
    energy_mev: float,
    depth_spacing: float,
    central_u: int = 8,
    central_v: int = 4,
    rsp: str = "linear",
    strict_energy: bool = True,
) -> torch.Tensor:
    """A ``(depth,)`` prior for the *integrated* depth dose, unit peak.

    The 3-D channel above answers "what does the curve say here"; this answers
    "what does the curve say this beamlet's IDD is", which is the quantity the
    factorised head predicts and ``idd_distance`` scores. It is read on the
    central ``central_u x central_v`` voxels rather than the single axis voxel,
    which costs nothing and stops one noisy voxel of CT setting the range.

    Deliberately *not* the lateral mean of the 3-D channel: out at +-32 mm the
    box is mostly tissue the beamlet never reaches, and averaging its WEPL in
    drags the prior's range around by anatomy that carries no dose.
    """
    n_u, n_v = ct_cuboid.shape[1], ct_cuboid.shape[2]
    u0, v0 = (n_u - central_u) // 2, (n_v - central_v) // 2
    core = ct_cuboid[:, u0:u0 + central_u, v0:v0 + central_v]
    return build_bragg_channel(
        core, energy_mev, depth_spacing, rsp=rsp, strict_energy=strict_energy
    ).mean(dim=(1, 2))
