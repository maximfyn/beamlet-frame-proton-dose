#!/usr/bin/env python3
"""Build the beamlet training set from raw CT + Monte Carlo dose.

Cropping each beamlet around its own ground-truth dose would produce data no
blind model could ever reproduce. Here the box comes only from `models.geometry`, i.e. from the ray
geometry a submission actually receives; ground truth is used solely as the
regression label.

Per patient, writes memory-mappable arrays plus a metadata sidecar::

    <out>/<PID>/ct_i16.npy     (N, 384, 64, V) int16    sampled CT, HU
    <out>/<PID>/label_f16.npy  (N, 384, 64, V) float16  pre-compensated dose
    <out>/<PID>/meta.json      per-beamlet geometry and statistics
    <out>/<PID>/_COMPLETE      written last; resumption skips these

with ``V = --n-lat-v`` (default 16; the released models use 24).

`_COMPLETE` is only touched after both arrays are flushed, so a crashed patient
is redone rather than silently half-read.

Storage note: int16 CT costs at most 0.5 HU against a normalization scale of
1000, and float16 labels measured 0.0% degradation of the beam-MAE floor, so
neither is worth storing wider.

This script writes; `models/dataset.py` reads. Accept or reject a run by
executing the contract, not by reading the script that produced it::

    DOSERAD_BEAMLET_ROOT=<out-root> \
      python -m pytest tests/test_dataset_contract.py -q

Four things there fail silently rather than loudly:

- **Labels are raw, never peak-normalized.** Level 2 metrics sum beamlets with
  clinical weights, so relative magnitude *between* beamlets is load-bearing.
  Scaling happens at load time (`models.dataset.DOSE_SCALE = 1e-3`).
- **`_COMPLETE` records the beamlet cap and the box**, so a marker from a
  truncated or differently-boxed run cannot satisfy a full one.
- **In development, only train + val patients were preprocessed**, so the 8
  held-out test patients were only ever seen through the inference path. The
  released base model trains on all 75, which is why the README builds all three
  splits and deselects the contract check that guards this.
- **A complete training plan holds 1,080 beamlets**, so anything iterating them
  must count and report skips. A bare `if not path.exists(): continue` once hid
  a truncated download that looked like a coherent gantry band rather than loss.

Raw `.npy` (not `.npz`/HDF5) because compression defeats the memory-mapping the
set is designed around.

Usage
-----
    # short test
    python scripts/data/preprocess_beamlets.py --patients 1ABB006 --max-beamlets 24

    # full train+val run
    python scripts/data/preprocess_beamlets.py --splits train val --workers 32

    # rebuild on GPUs, several workers each (zlib on the dose files is the floor)
    python scripts/data/preprocess_beamlets.py --workers 30 --gpus 0,3,4,5,6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.splits import get_splits  # noqa: E402
import torch  # noqa: E402

from models import geometry_torch as GT  # noqa: E402
from models.geometry import (  # noqa: E402
    BeamletGrid,
    VolumeGeometry,
    beam_frame,
    find_entry_depth_box,
)

DEFAULT_RAW_ROOT = Path("data/dataset_raw/proton/training")
LOCAL_RAW = Path(__file__).resolve().parents[2] / "data" / "dataset_raw" / "proton" / "training"
# The released set's name. Pass --out-root explicitly for any set that is not
# the canonical one.
DEFAULT_OUT_ROOT = Path("data/dataset_beamlet_tall24_it32/proton")


def resolve_raw_root(explicit: str | None) -> Path:
    for candidate in ([Path(explicit)] if explicit else []) + [DEFAULT_RAW_ROOT, LOCAL_RAW]:
        if candidate.exists():
            return candidate
    raise SystemExit("no raw dataset root found; pass --raw-root")


def find_ct(patient_dir: Path) -> Path | None:
    for name in ("ct_def_masked.mha", "ct.mha"):
        path = patient_dir / "image" / name
        if path.exists():
            return path
    return None


def find_plan(patient_dir: Path) -> Path | None:
    pid = patient_dir.name
    for path in (patient_dir / "plan_json" / f"{pid}.json", patient_dir / f"{pid}.json"):
        if path.exists():
            return path
    return None


def enumerate_beamlets(plan: dict) -> list[tuple[dict, dict, dict]]:
    return [
        (beam, ray, beamlet)
        for beam in plan["beams"]
        for ray in beam["rays"]
        for beamlet in ray["beamlets"]
    ]


def dose_path(patient_dir: Path, beam: dict, ray: dict, beamlet: dict) -> Path:
    return patient_dir / "dose" / (
        f"Dose_B{beam['beam_idx']}_R{ray['ray_idx']}_L{beamlet['beamlet_idx']}.mha"
    )


def build_one(
    ct: np.ndarray,
    geom: VolumeGeometry,
    grid: BeamletGrid,
    patient_dir: Path,
    beam: dict,
    ray: dict,
    beamlet: dict,
    iterations: int,
    device: "torch.device",
    ct_device: "torch.Tensor",
    label_cutoff: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """CT cuboid, pre-compensated label, and metadata for a single beamlet.

    Sampling and ``precompensate`` always run through ``models.geometry_torch``;
    ``device`` selects CPU or CUDA, and there is no scipy branch. One
    implementation on the production path is the point -- the shards and the
    predictor resample with the same code, which is the train/inference symmetry
    of `models/geometry.py`. scipy survives only as the independent oracle that
    ``tests/test_geometry_torch.py`` checks this against.

    ``ct_device`` is the CT already resident on that device, uploaded once per
    patient rather than once per beamlet.
    """
    path = dose_path(patient_dir, beam, ray, beamlet)
    if not path.exists():
        return None

    source = np.asarray(ray["ray_source"], dtype=float)
    target = np.asarray(ray["ray_target"], dtype=float)
    try:
        beam_frame(source, target)
    except ValueError:
        return None

    # Box-anchored: the same single rule the predictor applies, so the shards and
    # inference cannot disagree about where a beamlet starts (`models/geometry.py`).
    entry = find_entry_depth_box(ct, geom, source, target, grid)
    if entry is None:  # box misses the patient entirely
        return None

    dose = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
    if not np.isfinite(dose).all() or dose.max() <= 0.0:
        return None

    # Thresholded HERE, on the CT grid, and only then pre-compensated. The order
    # is load-bearing and not interchangeable: inference renders the prediction
    # back to the CT grid and *then* clamps, so the consistent training target is
    # `precompensate(threshold(dose))`. Thresholding the pre-compensated label
    # instead would be the wrong composition and would look identical on disk.
    # `<` matches the evaluator and `predictor.py`'s clamp.
    if label_cutoff > 0:
        dose[dose < label_cutoff] = 0.0
        if dose.max() <= 0.0:
            return None

    points = GT.grid_points_world(source, target, entry, grid, device)
    ct_cuboid = GT.sample_ct(ct_device, geom, points).cpu().numpy()
    label = (
        GT.precompensate(
            torch.as_tensor(dose, device=device),
            geom,
            source,
            target,
            entry,
            grid,
            iterations=iterations,
        )
        .cpu()
        .numpy()
    )

    meta = {
        "beam_idx": beam["beam_idx"],
        "ray_idx": ray["ray_idx"],
        "beamlet_idx": beamlet["beamlet_idx"],
        "energy": float(beamlet["energy"]),
        "gantry_angle": float(beam.get("gantry_angle", float("nan"))),
        "ray_source": source.tolist(),
        "ray_target": target.tolist(),
        "entry_depth": float(entry),
        "label_peak": float(label.max()),
        "label_sum": float(label.sum()),
        "dose_peak": float(dose.max()),
        "dose_sum": float(dose.sum()),
    }
    return ct_cuboid, label, meta


def process_patient(
    pid: str,
    raw_root: Path,
    out_root: Path,
    grid: BeamletGrid,
    iterations: int,
    max_beamlets: int | None,
    overwrite: bool,
    gpus: tuple[int, ...] = (),
    label_cutoff: float = 0.0,
) -> dict:
    started = time.time()
    out_dir = out_root / pid
    marker = out_dir / "_COMPLETE"
    if marker.exists() and not overwrite:
        # A marker left by a capped test run must not satisfy a full run, or the
        # dataset silently ends up truncated for those patients.
        try:
            previous = json.loads(marker.read_text())
        except (ValueError, OSError):
            previous = None
        if not isinstance(previous, dict):
            previous = {}  # marker from an older format: treat as unknown provenance
        wanted = grid.as_dict()
        if (
            previous.get("max_beamlets", "missing") == max_beamlets
            and previous.get("grid", "missing") == wanted
            # A raw and a thresholded set differ in no other visible way: same
            # shapes, same dtypes, same file sizes. Without this a stale marker
            # silently satisfies a run that wanted the other one.
            and previous.get("label_cutoff", 0.0) == label_cutoff
            # Labels pre-compensated with different Landweber counts look
            # identical on disk too, and a resume that mixed them would train
            # on two label definitions at once. A marker from before this field
            # existed cannot say, so it counts as stale rather than as a match.
            and previous.get("iterations", "missing") == iterations
        ):
            return {"patient": pid, "status": "skipped", "count": previous.get("count", 0)}
        return {
            "patient": pid,
            "status": "stale",
            "detail": f"built with max_beamlets={previous.get('max_beamlets', '?')} "
                      f"grid={previous.get('grid', '?')} "
                      f"label_cutoff={previous.get('label_cutoff', 0.0)} "
                      f"iterations={previous.get('iterations', '?')}, now "
                      f"{max_beamlets} {wanted} {label_cutoff} {iterations}; "
                      "rerun with --overwrite",
        }

    patient_dir = raw_root / pid
    ct_path, plan_path = find_ct(patient_dir), find_plan(patient_dir)
    if ct_path is None or plan_path is None:
        return {"patient": pid, "status": "error", "detail": "missing ct or plan json"}

    plan = json.loads(plan_path.read_text())
    image = sitk.ReadImage(str(ct_path))
    ct = sitk.GetArrayFromImage(image).astype(np.float32)
    geom = VolumeGeometry.from_sitk(image)

    # Workers are spread over the allowed GPUs by patient id, so a pool of N
    # processes shares the devices without a coordinator. The CT goes up once
    # per patient and is reused by all 1,080 beamlets; only the dose volume is
    # uploaded per beamlet (~20 ms against the ~424 ms its zlib decompression
    # costs, so the transfer is not the thing to optimise).
    #
    # blake2b, not hash(): Python randomises string hashing per process, so
    # hash(pid) would assign patients to different GPUs on every run. That is
    # harmless while every device is the same model and forward grid_sample is
    # deterministic, but it makes a rebuild unreproducible by construction, and
    # it would stop being cosmetic the moment the GPUs are not identical.
    if gpus:
        digest = hashlib.blake2b(pid.encode(), digest_size=8).digest()
        device = torch.device(
            "cuda", gpus[int.from_bytes(digest, "big") % len(gpus)]
        )
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    ct_device = torch.as_tensor(ct, device=device)

    beamlets = enumerate_beamlets(plan)
    if max_beamlets is not None:
        beamlets = beamlets[:max_beamlets]
    if not beamlets:
        return {"patient": pid, "status": "error", "detail": "no beamlets in plan"}

    out_dir.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)  # stale marker must not survive a redo

    # Stream straight to disk; a patient's arrays are ~1.6 GB together.
    ct_out = np.lib.format.open_memmap(
        out_dir / "ct_i16.npy", mode="w+", dtype=np.int16,
        shape=(len(beamlets), *grid.shape),
    )
    label_out = np.lib.format.open_memmap(
        out_dir / "label_f16.npy", mode="w+", dtype=np.float16,
        shape=(len(beamlets), *grid.shape),
    )

    records, written, failed = [], 0, []
    try:
        for beam, ray, beamlet in beamlets:
            built = build_one(
                ct, geom, grid, patient_dir, beam, ray, beamlet, iterations,
                device, ct_device, label_cutoff,
            )
            if built is None:
                failed.append(
                    f"B{beam['beam_idx']}_R{ray['ray_idx']}_L{beamlet['beamlet_idx']}"
                )
                continue
            ct_cuboid, label, meta = built
            ct_out[written] = np.rint(ct_cuboid).astype(np.int16)
            label_out[written] = label.astype(np.float16)
            meta["index"] = written
            records.append(meta)
            written += 1
        ct_out.flush()
        label_out.flush()
    finally:
        del ct_out, label_out

    if written == 0:
        return {"patient": pid, "status": "error", "detail": "no usable beamlets"}

    # Trim the unused tail left by skipped beamlets.
    if written < len(beamlets):
        for name, dtype in (("ct_i16.npy", np.int16), ("label_f16.npy", np.float16)):
            full = np.load(out_dir / name, mmap_mode="r")
            trimmed = np.array(full[:written], dtype=dtype)
            del full
            np.save(out_dir / name, trimmed)

    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "patient": pid,
                "count": written,
                "attempted": len(beamlets),
                "failed": failed,
                "ct_path": str(ct_path),
                # Per patient, not just in the run manifest: a shard directory
                # has to be self-describing, because the manifest is one file and
                # patient directories get copied, resumed and re-run separately.
                "provenance": provenance(),
                "grid": grid.as_dict(),
                # `label_cutoff`, not `args.label_cutoff` -- this runs in a
                # worker process, which has the parameter and not the parsed
                # arguments. The `args.` version raised NameError only after a
                # patient's arrays were fully written, so a 40-minute run looked
                # healthy the whole way and produced no `_COMPLETE` at all.
                "label_cutoff": label_cutoff,
                "volume": {
                    "origin": geom.origin.tolist(),
                    "spacing": geom.spacing.tolist(),
                    "shape": list(geom.shape),
                },
                "beamlets": records,
            },
            indent=1,
        )
    )
    marker.write_text(
        json.dumps(
            {"count": written, "max_beamlets": max_beamlets,
             "grid": grid.as_dict(), "label_cutoff": label_cutoff,
             "iterations": iterations}
        )
    )
    elapsed = time.time() - started
    return {
        "patient": pid,
        "status": "ok",
        "count": written,
        "failed": len(failed),
        "seconds": round(elapsed, 1),
        "per_beamlet": round(elapsed / max(written, 1), 3),
    }


def provenance() -> dict:
    """Identify the code that produced a run.

    The git commit alone is not trustworthy here: code can reach a worker by
    rsync, so the checkout's HEAD can lag the files actually being executed.
    The hash of ``models/geometry.py`` pins what really ran.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except Exception:
        dirty = None
    def sha(*parts: str) -> str:
        return hashlib.sha256(root.joinpath(*parts).read_bytes()).hexdigest()[:16]

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        # Both halves of "what built this": the box geometry and the driver that
        # applies it. v1 recorded neither, which is why nobody can say which code
        # produced it.
        # NOT the checkpoint's `geometry_sha256`, which covers three files at
        # full length: this is one file, truncated, and the two never compare.
        "geometry_py_sha16": sha("models", "geometry.py"),
        # Hash THIS file by its own path, not by a spelled-out one: a spelled
        # path silently stops describing the run once the file moves.
        "preprocess_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest()[:16],
    }




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--splits", nargs="*", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--patients", nargs="*", default=None,
                        help="explicit patient ids, overrides --splits")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32,
                        help="Landweber iterations per label (the released shards use 32)")
    parser.add_argument("--label-cutoff", type=float, default=0.0, metavar="GY",
                        help="zero ground-truth dose below this, ON THE CT GRID, before "
                             "pre-compensation. Default 0.0 = raw labels, which is every "
                             "shard set built before 2026-08-15. The platform thresholds "
                             "both sides at 0.985e-3 x each beamlet's GT peak, which "
                             "is per-beamlet and cannot be matched by one flat number, "
                             "so this must ERR LOW: below costs nothing (the inference "
                             "clamp zeroes that band on both sides anyway), above "
                             "deletes signal that is still scored and no clamp recovers "
                             "it. Served cutoffs are of order 1e-6, so any "
                             "value here must sit well below that and 0.0 needs no "
                             "argument. Recorded in "
                             "_COMPLETE, because a thresholded shard set is otherwise "
                             "indistinguishable from a raw one.")
    parser.add_argument("--max-beamlets", type=int, default=None,
                        help="cap per patient, for short test runs")
    parser.add_argument("--overwrite", action="store_true")
    # Box dimensions are parameters, not contract (`models/geometry_torch.py`). They are
    # recorded into _COMPLETE, so a shard built with one box can never satisfy a
    # run that wants another.
    _defaults = BeamletGrid()
    parser.add_argument("--n-depth", type=int, default=_defaults.n_depth)
    parser.add_argument("--n-lat-u", type=int, default=_defaults.n_lat_u)
    parser.add_argument("--n-lat-v", type=int, default=_defaults.n_lat_v)
    # Counts alone cannot express a fine-depth box: one of
    # (768, 128, 40) at (0.5, 1, 3) mm, and without these the same counts at
    # 1 mm depth would build a box of twice the reach under the right name.
    parser.add_argument("--depth-spacing", type=float,
                        default=_defaults.depth_spacing)
    parser.add_argument("--lat-u-spacing", type=float,
                        default=_defaults.lat_u_spacing)
    parser.add_argument("--lat-v-spacing", type=float,
                        default=_defaults.lat_v_spacing)
    parser.add_argument("--entry-margin-mm", type=float,
                        default=_defaults.entry_margin_mm)
    parser.add_argument(
        "--gpus",
        default="",
        help="comma-separated CUDA indices to spread workers over, e.g. 3,5,6. "
        "Empty runs the same code on CPU. precompensate is 70%% of this script "
        "and 13.4x faster on a GPU; what remains is zlib decompression of the "
        "dose volumes, which is CPU-bound, so use several workers per GPU.",
    )
    args = parser.parse_args()

    raw_root = resolve_raw_root(args.raw_root)
    # `out_root` is the directory that HOLDS the `<PID>/` shards, so it ends
    # in `proton/` -- the modality level `models/dataset.py` and this file's own
    # usage line both read from. The local default omitted it, so a laptop build
    # landed one level above every reader; the default root has always had
    # it. A `--out-root` without `proton/` builds shards nothing finds, and
    # the failure surfaces as `FileNotFoundError` in the *trainer*, an hour and
    # a reshard later.
    out_root = Path(args.out_root) if args.out_root else (
        DEFAULT_OUT_ROOT if DEFAULT_RAW_ROOT.exists()
        else Path(__file__).resolve().parents[2] / "data" / "dataset_beamlet_tall24_it32" / "proton"
    )
    grid = BeamletGrid(
        n_depth=args.n_depth,
        n_lat_u=args.n_lat_u,
        n_lat_v=args.n_lat_v,
        depth_spacing=args.depth_spacing,
        lat_u_spacing=args.lat_u_spacing,
        lat_v_spacing=args.lat_v_spacing,
        entry_margin_mm=args.entry_margin_mm,
    )

    gpus = tuple(int(g) for g in args.gpus.split(",") if g.strip() != "")
    if gpus and not torch.cuda.is_available():
        raise SystemExit("--gpus given but torch reports no CUDA device")

    if args.patients:
        patients = list(args.patients)
    else:
        splits = get_splits()
        patients = [pid for split in args.splits for pid in splits[split]]

    missing = [p for p in patients if not (raw_root / p).exists()]
    if missing:
        raise SystemExit(f"patients absent from {raw_root}: {missing}")

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"raw     : {raw_root}")
    print(f"out     : {out_root}")
    print(f"patients: {len(patients)}  workers: {args.workers}  "
          f"iterations: {args.iterations}"
          + (f"  max-beamlets: {args.max_beamlets}" if args.max_beamlets else ""))
    print(f"grid    : {grid.shape} @ "
          f"({grid.depth_spacing}, {grid.lat_u_spacing}, {grid.lat_v_spacing}) mm")
    print(f"device  : {'cuda ' + str(list(gpus)) if gpus else 'cpu (torch)'}")
    prov = provenance()
    print(f"code    : geometry.py.sha16={prov['geometry_py_sha16']} git={prov['git_commit'][:8]}"
          f"{' (dirty)' if prov['git_dirty'] else ''}\n", flush=True)

    started = time.time()
    results = []
    # CUDA cannot be re-initialized in a forked child, and fork is the default
    # start method on Linux -- so a GPU run must spawn. Spawn re-imports this
    # module per worker and pays ~2-3 s of CUDA init each, which is noise against
    # a run measured in hours.
    context = multiprocessing.get_context("spawn" if gpus else "fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {
            pool.submit(
                process_patient, pid, raw_root, out_root, grid,
                args.iterations, args.max_beamlets, args.overwrite, gpus,
                args.label_cutoff,
            ): pid
            for pid in patients
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            elapsed = time.time() - started
            eta = elapsed / done * (len(patients) - done)
            detail = (
                f"n={result.get('count', 0):4d} failed={result.get('failed', 0):3d} "
                f"{result.get('per_beamlet', 0):.2f}s/beamlet"
                if result["status"] == "ok" else result.get("detail", result["status"])
            )
            print(f"[{done:3d}/{len(patients)}] {result['patient']:>9} "
                  f"{result['status']:>7}  {detail}   eta={eta / 60:.1f}m", flush=True)

    # A patient finished by an earlier run counts: a correct resume must exit 0.
    ok = [r for r in results if r["status"] in ("ok", "skipped")]
    total = sum(r["count"] for r in ok)
    elapsed = time.time() - started
    print(f"\n{'=' * 60}")
    print(f"patients ok : {len(ok)}/{len(patients)}")
    print(f"beamlets    : {total}")
    print(f"wall        : {elapsed / 60:.1f} min")
    for r in results:
        if r["status"] == "error":
            print(f"  ERROR {r['patient']}: {r.get('detail')}")

    # Per host: several nodes can build disjoint patient subsets into one output
    # root -- each patient owns its own directory, so the manifest is the only
    # shared write, and a single name would leave only the last node's record.
    (out_root / f"_manifest_{socket.gethostname().split('.')[0]}.json").write_text(
        json.dumps(
            {
                "provenance": provenance(),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "raw_root": str(raw_root),
                "splits": args.splits if not args.patients else None,
                "patients": patients,
                "iterations": args.iterations,
                "backend": "torch-cuda" if gpus else "torch-cpu",
                "gpus": list(gpus),
                "max_beamlets": args.max_beamlets,
                "grid": grid.as_dict(),
                "label_cutoff": args.label_cutoff,
                "beamlets": total,
                "wall_minutes": round(elapsed / 60, 2),
                "results": results,
            },
            indent=1,
        )
    )

    return 0 if len(ok) == len(patients) else 1


if __name__ == "__main__":
    raise SystemExit(main())
