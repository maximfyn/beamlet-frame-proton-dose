#!/usr/bin/env python3
"""Rebuild ``models/bragg_generic.npz`` from its two public sources, or check it.

The table is tracked so that inference needs neither source. This script is
how to regenerate it, or confirm the tracked copy is what it claims to be:

* **matRad's Generic proton machine**, as shipped by
  [pyRadPlan](https://github.com/e0404/pyRadPlan) (BSD-3-Clause) at
  ``pyRadPlan/data/machines/protons_Generic.mat``: the Bragg curves, the
  double-Gaussian lateral kernels and the initial spot widths. The export
  records the file's SHA-256; the copy it was built from is
  ``24d1a5f2...``, byte-identical to upstream pyRadPlan's.
* **The DoseRAD2026 dataset's HU-to-density table**, ``hu_to_density`` in
  ``proton/training/beam_parameters.json``: the ten anchors the Monte Carlo
  ground truth was simulated with, read as the ``hlut`` conversion in place of a
  stopping-power curve.

    git clone https://github.com/e0404/pyRadPlan
    python scripts/data/export_bragg_table.py --pyradplan pyRadPlan --check \
        --beam-parameters <dataset>/proton/training/beam_parameters.json

The Bragg curves, kernels and spot widths come from the machine alone, so the
`--check` above verifies them from a clone plus the dataset's one JSON file; the
864 GB image download is not needed for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models import physics  # noqa: E402

ATTRIBUTION = (
    "Bragg curves resampled from matRad's 'Generic' proton machine, vendored by "
    "pyRadPlan (BSD-3-Clause), created by matRad Developer Team @ DKFZ. "
    "Double-Gaussian dose kernels fitted to MC simulations using a generic "
    "machine model. Used as a PRIOR only: DoseRAD ground truth is Geant4."
)
MACHINE_IN_PYRADPLAN = "pyRadPlan/data/machines/protons_Generic.mat"
DEFAULT_BEAM_PARAMETERS = "data/dataset_raw/proton/training/beam_parameters.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def machine_entries(mat_path: Path) -> list:
    """The machine's per-energy entries, sorted by energy."""
    import scipy.io as sio

    data = sio.loadmat(str(mat_path), simplify_cells=True)["machine"]["data"]
    order = np.argsort([float(d["energy"]) for d in data])
    return [data[i] for i in order]


def lateral_kernels(entries: list) -> dict:
    """``sigma1``/``sigma2``/``weight`` resampled onto the shared WEPL grid.

    Held (``left``/``right`` edge values), not zeroed, past the tabulation --
    the opposite of ``Z``. A width of zero is not "no kernel", it is a delta
    function. Past the range ``Z`` is zero anyway, so the held value is never
    read for real dose.
    """
    grid = physics.wepl_grid()
    out = {}
    for key in physics.LATERAL_KEYS:
        table = np.zeros((len(entries), grid.size), dtype=np.float32)
        for row, entry in enumerate(entries):
            depths = np.asarray(entry["depths"], dtype=np.float64)
            depths = depths + float(entry.get("offset", 0.0) or 0.0)
            values = np.asarray(entry[key], dtype=np.float64)
            table[row] = np.interp(grid, depths, values,
                                   left=values[0], right=values[-1])
        out[key] = table
    return out


def dataset_hlut(beam_parameters: Path) -> np.ndarray:
    """``(10, 2)`` HU and density anchors, linearly interpolated by the reader."""
    entries = json.loads(beam_parameters.read_text())["hu_to_density"]["entries"]
    return np.array([[e["hu"], e["density_g_cm3"]] for e in entries], dtype=np.float64)


def build(mat_path: Path, beam_parameters: Path) -> dict:
    energies, curves = physics.load_bragg_table(str(mat_path))
    entries = machine_entries(mat_path)
    return {
        "energies_mev": energies.astype(np.float64),
        "curves": curves.astype(np.float32),
        "sigma_spot_mm": np.array([float(e["initFocus"]["emittance"]["sigmaX"])
                                   for e in entries], dtype=np.float64),
        "hlut": dataset_hlut(beam_parameters),
        **lateral_kernels(entries),
        "wepl_step_mm": np.float64(physics.WEPL_STEP_MM),
        "wepl_max_mm": np.float64(physics.WEPL_MAX_MM),
        "source_sha256": np.str_(sha256(mat_path)),
        "source_path": np.str_(physics.MACHINE_RELPATH),
        "attribution": np.str_(ATTRIBUTION),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pyradplan", required=True,
                        help="a clone of github.com/e0404/pyRadPlan")
    parser.add_argument("--beam-parameters", default=DEFAULT_BEAM_PARAMETERS,
                        help="the dataset's beam_parameters.json "
                             f"(default {DEFAULT_BEAM_PARAMETERS})")
    parser.add_argument("--out", default=None,
                        help=f"output .npz; defaults to {physics.TABLE_RELPATH}")
    parser.add_argument("--check", action="store_true",
                        help="compare the existing export against the sources "
                             "instead of writing")
    args = parser.parse_args()

    root = physics.repo_root()
    mat_path = Path(args.pyradplan) / MACHINE_IN_PYRADPLAN
    beam_parameters = Path(args.beam_parameters)
    if not beam_parameters.is_absolute() and not beam_parameters.exists():
        beam_parameters = root / beam_parameters
    out_path = Path(args.out) if args.out else root / physics.TABLE_RELPATH
    for path in (mat_path, beam_parameters):
        if not path.exists():
            raise SystemExit(f"{path} not found")

    fresh = build(mat_path, beam_parameters)
    if args.check:
        if not out_path.exists():
            raise SystemExit(f"{out_path} does not exist; nothing to check")
        stored = np.load(out_path, allow_pickle=False)
        problems = []
        if str(stored["source_sha256"]) != str(fresh["source_sha256"]):
            problems.append(
                f"source hash differs: export {str(stored['source_sha256'])[:16]} "
                f"vs .mat {str(fresh['source_sha256'])[:16]}")
        if not np.array_equal(stored["energies_mev"], fresh["energies_mev"]):
            problems.append("energy grid differs")
        for key in ("curves", "sigma_spot_mm", "hlut") + physics.LATERAL_KEYS:
            if key not in stored:
                problems.append(f"{key} absent from the export")
            elif not np.array_equal(stored[key], fresh[key]):
                worst = float(np.abs(stored[key] - fresh[key]).max())
                problems.append(f"{key} differs, max |delta| = {worst:.3e}")
        if problems:
            print("\n".join("MISMATCH: " + p for p in problems))
            return 1
        print(f"{out_path.name} matches {mat_path.name} and {beam_parameters.name} "
              f"({len(fresh['energies_mev'])} energies, "
              f"sha256 {str(fresh['source_sha256'])[:16]}...)")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **fresh)
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB, "
          f"{len(fresh['energies_mev'])} energies x {fresh['curves'].shape[1]} depths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
