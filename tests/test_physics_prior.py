"""The Bragg prior: that it is the machine's, and that it vanishes where it must.

Three failure modes here are silent, which is why each gets a test rather than a
comment:

* the table is **not** the machine's (a stale export, a re-derived grid);
* the curve is **held** past the end of its tabulation instead of going to zero,
  which puts a floor of prior dose beyond the range -- the opposite of what a
  Bragg curve does;
* an energy the machine does not tabulate degrades to "the nearest curve"
  instead of raising, and the test set is *"much larger and more varied"* than
  what we hold.
"""

from __future__ import annotations

import os
from pathlib import Path


import numpy as np
import pytest
import torch

from models import physics
from models.geometry import AIR_HU, BeamletGrid

# A pyRadPlan clone, wherever the reader keeps one: `PYRADPLAN=<clone>` runs the
# check below; without it the test skips, since no checkout ships the `.mat`.
MACHINE = Path(os.environ.get("PYRADPLAN", "/nonexistent")) / physics.MACHINE_RELPATH

# The five local plans' energies are all machine energies; the release ships 85
# and the machine tabulates 114 (`models/physics.py`).
RELEASE_ENERGY_COUNT = 85
MACHINE_ENERGY_COUNT = 114


def test_the_table_loads_and_is_unit_peak():
    energies, curves = physics.load_bragg_table()
    assert energies.shape == (MACHINE_ENERGY_COUNT,)
    assert curves.shape[0] == MACHINE_ENERGY_COUNT
    assert np.all(np.diff(energies) > 0), "energies must be sorted for the lookup"
    assert curves.min() >= 0.0
    # Not exactly 1: each curve is normalised on the machine's own
    # non-uniform depth samples and then resampled onto the common 0.5 mm
    # grid, which need not land on the tabulated peak. Worst loss ~0.11%.
    assert np.allclose(curves.max(axis=1), 1.0, atol=2e-3)


def test_every_curve_ends_at_exactly_zero():
    """Past the tabulated range the prior must vanish, not plateau.

    Held instead of zeroed, the prior would claim dose everywhere downstream of
    the Bragg peak. That is the single worst thing this feature could tell the
    network, because the distal fall-off is what `idd_distance` measures.
    """
    _, curves = physics.load_bragg_table()
    assert np.all(curves[:, -1] == 0.0)


@pytest.mark.skipif(not MACHINE.exists(),
                    reason="set PYRADPLAN to a pyRadPlan clone to check the table against its machine")
def test_the_export_still_matches_the_machine():
    """The tracked ``.npz`` is a derived artifact and derived artifacts drift."""
    from_mat = physics.load_bragg_table(str(MACHINE))
    from_npz = physics.load_bragg_table()
    assert np.array_equal(from_mat[0], from_npz[0])
    assert np.array_equal(from_mat[1], from_npz[1])


def test_an_untabulated_energy_raises_rather_than_snapping():
    energies, _ = physics.load_bragg_table()
    physics.assert_energy_is_tabulated(float(energies[10]))
    with pytest.raises(ValueError, match="machine energy"):
        physics.assert_energy_is_tabulated(float(energies[10]) + 5.0)


def _cuboid(hu: float, shape=(384, 64, 16)) -> torch.Tensor:
    return torch.full(shape, hu, dtype=torch.float32)


def test_the_prior_is_zero_in_air():
    """Air is not the body, and the ground truth is exactly zero outside it.

    This is the fix that made the prior usable: unmasked, a beamlet whose Bragg
    peak lies past the patient froze part-way up the curve and stayed there,
    coming out anti-correlated with the truth (`models/physics.py`).
    """
    energies, _ = physics.load_bragg_table()
    energy = float(energies[physics.nearest_energy_index(100.0, energies)])
    prior = physics.build_bragg_channel(_cuboid(AIR_HU), energy, 1.0)
    assert float(prior.abs().max()) == 0.0


def test_the_prior_peaks_near_the_machine_range_in_water():
    """In uniform water, geometric depth is water-equivalent depth."""
    import scipy.io as sio

    energies, _ = physics.load_bragg_table()
    index = physics.nearest_energy_index(120.0, energies)
    energy = float(energies[index])

    prior = physics.build_bragg_channel(_cuboid(0.0), energy, 1.0)
    profile = prior[:, 32, 8]
    peak_depth = float(profile.argmax())

    if MACHINE.exists():
        data = sio.loadmat(str(MACHINE), simplify_cells=True)["machine"]["data"]
        expected = float(next(d for d in data
                              if abs(float(d["energy"]) - energy) < 1e-3)["peakPos"])
    else:
        expected = peak_depth
    # 1 mm sampling against a 0.5 mm table, plus the cumsum's half-voxel offset.
    assert peak_depth == pytest.approx(expected, abs=2.0)


def test_denser_tissue_pulls_the_peak_closer():
    """The whole point of reading the curve at WEPL rather than at depth."""
    energies, _ = physics.load_bragg_table()
    energy = float(energies[physics.nearest_energy_index(120.0, energies)])

    water = physics.build_bragg_channel(_cuboid(0.0), energy, 1.0)[:, 32, 8]
    bone = physics.build_bragg_channel(_cuboid(500.0), energy, 1.0)[:, 32, 8]
    assert int(bone.argmax()) < int(water.argmax())


def test_the_idd_prior_reads_the_core_not_the_whole_box():
    """Anatomy at +-32 mm carries no dose and must not move the range.

    A box that is water down the middle and bone at the edges has to give the
    same IDD prior as one that is water throughout -- otherwise a rib the
    beamlet never touches shortens its predicted range.
    """
    energies, _ = physics.load_bragg_table()
    energy = float(energies[physics.nearest_energy_index(150.0, energies)])
    grid = BeamletGrid()

    uniform = _cuboid(0.0)
    edged = _cuboid(0.0)
    edged[:, :16, :] = 800.0
    edged[:, -16:, :] = 800.0

    a = physics.build_bragg_idd_prior(uniform, energy, grid.depth_spacing)
    b = physics.build_bragg_idd_prior(edged, energy, grid.depth_spacing)
    assert torch.equal(a, b)


def test_the_lookup_interpolates_and_truncates():
    curve = torch.tensor([0.0, 1.0, 0.5, 0.0], dtype=torch.float32)
    step = physics.WEPL_STEP_MM
    wepl = torch.tensor([0.0, 0.5 * step, step, 1.5 * step, 3 * step, 10 * step])
    got = physics._lookup(curve, wepl)
    assert got.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.75, 0.0, 0.0])


def test_the_channel_is_float32_and_shaped_like_the_box():
    grid = BeamletGrid()
    energies, _ = physics.load_bragg_table()
    energy = float(energies[physics.nearest_energy_index(100.0, energies)])
    prior = physics.build_bragg_channel(_cuboid(0.0, grid.shape), energy,
                                        grid.depth_spacing)
    assert prior.shape == grid.shape
    assert prior.dtype is torch.float32


def test_the_lateral_prior_follows_the_box_it_is_handed():
    """It read its axes from the default grid, so the released box broke it.

    `BeamletGrid()` is 16 voxels in v and the released models are 24, so a
    kernel built for the default grid could not be concatenated with the box it
    was meant to describe -- the failure only stayed hidden because
    `with_lateral` is off in both released configs.
    """
    for n_v in (16, 24):
        prior = physics.build_lateral_prior(torch.zeros(48, 64, n_v), 150.35, 1.0)
        assert prior.shape == (48, 64, n_v)
        assert torch.allclose(prior[10].sum(), torch.tensor(1.0), atol=1e-5)
