#!/usr/bin/env python3
"""Predict one proton beamlet in a water phantom, on CPU, with no dataset.

    python examples/water_phantom.py --energy 150
    python examples/water_phantom.py --checkpoint checkpoints/e240_FINAL.pt --out base.png

Builds a 300 mm water box in air, fires a single beamlet along +x through its
centre, and compares the network's Bragg peak with the depth-dose curve of the
Generic proton machine in water (``models/bragg_generic.npz``) -- a consistency
check in a geometry the network was never trained on (every training beamlet was
in a patient), not an independent accuracy test.

This is also the smallest example of using the model as a library:
``DosePredictor.from_checkpoint`` + ``predict(ct, geometry, [BeamletRequest])``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.geometry import VolumeGeometry  # noqa: E402
from models.physics import (  # noqa: E402
    load_bragg_table, nearest_energy_index, relative_stopping_power, wepl_grid)
from models.predictor import BeamletRequest, DosePredictor  # noqa: E402

WATER_HU, AIR_HU = 0.0, -1000.0


def water_phantom(nx=400, ny=120, nz=40, spacing=(1.0, 1.0, 3.0), x0=50.0, x1=350.0):
    """Air volume with a water box from x0 to x1 mm; numpy order (z, y, x), HU."""
    ct = np.full((nz, ny, nx), AIR_HU, dtype=np.float32)
    ix0, ix1 = int(x0 / spacing[0]), int(x1 / spacing[0])
    ct[3:nz - 3, 10:ny - 10, ix0:ix1] = WATER_HU
    geom = VolumeGeometry(origin=np.zeros(3), spacing=np.array(spacing), shape=ct.shape)
    return ct, geom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/tall24_it32_rbias_w32_ep002.pt"))
    ap.add_argument("--energy", type=float, default=150.0, help="MeV; snapped to the nearest tabulated energy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="water_phantom.png")
    args = ap.parse_args()

    energies, curves = load_bragg_table()
    k = nearest_energy_index(args.energy, energies)
    energy = float(energies[k])

    ct, geom = water_phantom()
    nz, ny, nx = ct.shape
    yc, zc = (ny - 1) / 2 * geom.spacing[1], (nz - 1) / 2 * geom.spacing[2]
    request = BeamletRequest(ray_source=(-1000.0, yc, zc), ray_target=(1000.0, yc, zc),
                             energy=energy, output_file_idx=0, idx_in_output=0)

    predictor = DosePredictor.from_checkpoint(args.checkpoint, device=args.device)
    t0 = time.perf_counter()
    dose = predictor.predict(ct, geom, [request])[0]
    print(f"predicted one beamlet at {energy:.1f} MeV in {time.perf_counter() - t0:.2f} s "
          f"on {args.device}")

    # Integrated depth dose along the beam axis, against depth in water.
    idd = dose.sum(axis=(0, 1))
    depth = np.arange(nx) * geom.spacing[0] + geom.spacing[0] / 2 - 50.0
    in_water = (depth >= 0) & (depth < 300)
    peak_pred = float(depth[in_water][np.argmax(idd[in_water])])

    # The machine's curve is tabulated over WATER-EQUIVALENT depth. The dataset's
    # HU-to-density table, which the prior reads in place of a stopping-power
    # curve, puts 0 HU slightly above 1, so the fair comparison converts that
    # curve to physical depth in this phantom.
    rsp = float(relative_stopping_power(torch.tensor([WATER_HU]), mode="hlut")[0])
    depth_ref = wepl_grid() / rsp
    peak_ref = float(depth_ref[np.argmax(curves[k])])
    print(f"Bragg peak depth in water: network {peak_pred:.1f} mm, "
          f"Generic machine {peak_ref:.1f} mm (table density at 0 HU = {rsp:.4f}), "
          f"difference {peak_pred - peak_ref:+.1f} mm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figure")
        return 0

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    sl = dose[nz // 2]
    a.imshow(sl / sl.max(), origin="lower", cmap="magma", aspect="auto",
             extent=[depth[0], depth[-1], 0, ny * geom.spacing[1]])
    a.axvline(0, color="w", lw=0.8, ls=":")
    a.set(xlabel="depth in water (mm)", ylabel="lateral (mm)", xlim=(-20, 300),
          title=f"predicted dose, {energy:.0f} MeV beamlet, central slice")
    b.plot(depth_ref, curves[k] / curves[k].max(), "k--", lw=1, label="Generic machine (water)")
    b.plot(depth[in_water], idd[in_water] / idd[in_water].max(), color="C3", lw=1.5,
           label="network (integrated)")
    b.set(xlabel="depth in water (mm)", ylabel="relative dose", xlim=(0, 300),
          title="integrated depth dose")
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
