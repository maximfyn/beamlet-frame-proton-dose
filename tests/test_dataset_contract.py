"""The preprocessing -> training shard contract.

`scripts/data/preprocess_beamlets.py` writes shards; `models/dataset.py` reads them.
Nothing else couples the two, so this file *is* the interface. It exists so that
a preprocessing run -- including one produced by a delegated agent on another
machine -- can be accepted or rejected by executing one command rather than by
reading the script that produced it.

Every assertion here guards a decision that was made by measurement
(`scripts/data/preprocess_beamlets.py`) and would fail silently if reverted:

* `int16` CT + `float16` labels. The 105 GB storage choice rests on the measured
  fact that this costs 0.0% of the beam-MAE floor. Widening or narrowing the
  dtype changes the size of the dataset on disk by ~2x either way.
* Raw `.npy`, so `np.load(mmap_mode="r")` works. `.npz`/HDF5 were rejected
  *because* compression defeats memory-mapping, and the loader depends on the
  page cache holding the set.
* `_COMPLETE` written last, carrying the parameters the shard was built with. A
  marker from a truncated test run must never satisfy a full run.
* Labels stored raw, never peak-normalized -- Level 2 metrics sum beamlets with
  clinical weights, so relative magnitude between beamlets is load-bearing.

The synthetic tests run anywhere. The real-data test skips unless shards are
present, so this suite is meaningful on a laptop and stricter where shards exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from models.splits import get_splits
from models.dataset import DOSE_SCALE, BeamletDataset
from models.geometry import BeamletGrid

GRID = BeamletGrid()
SHAPE = GRID.shape  # (384, 64, 16)

CT_DTYPE = np.int16
LABEL_DTYPE = np.float16

# Measured pre-compensated peak distribution over 31,968 beamlets.
PEAK_P1, PEAK_MEDIAN, PEAK_P99 = 8.1e-4, 1.26e-3, 1.74e-3

# Every patient plans -- and the release ships -- 1,080 beamlets. Shards built
# from the truncated HuggingFace download hold ~1,000. See scripts/data/preprocess_beamlets.py.
EXPECTED_PER_PATIENT = 1080
TRUNCATED_PER_PATIENT = 1000


def write_shard(root: Path, pid: str, n: int = 3, complete: bool = True,
                grid: BeamletGrid | None = None) -> Path:
    """Write a synthetic shard in exactly the layout preprocessing produces."""
    shard = root / pid
    shard.mkdir(parents=True, exist_ok=True)
    grid = grid or GRID
    shape = grid.shape

    rng = np.random.default_rng(abs(hash(pid)) % (2**32))
    ct = rng.integers(-1000, 2000, size=(n, *shape), dtype=CT_DTYPE)
    label = (rng.random((n, *shape)) * PEAK_MEDIAN).astype(LABEL_DTYPE)

    np.save(shard / "ct_i16.npy", ct)
    np.save(shard / "label_f16.npy", label)

    (shard / "meta.json").write_text(
        json.dumps(
            {
                "patient": pid,
                "count": n,
                "attempted": n,
                "failed": [],
                "ct_path": f"/fake/{pid}.mha",
                "volume": {
                    "origin": [0.0, 0.0, 0.0],
                    "spacing": [1.0, 1.0, 3.0],
                    "shape": [512, 512, 200],
                },
                "beamlets": [
                    {
                        "energy": 31.7 + i * 10.0,
                        "label_peak": float(label[i].max()),
                        "dose_sum": float(label[i].sum()),
                    }
                    for i in range(n)
                ],
            }
        )
    )
    if complete:
        (shard / "_COMPLETE").write_text(json.dumps(
            {"count": n, "max_beamlets": None, "grid": grid.as_dict()}))
    return shard


@pytest.fixture
def shard_root(tmp_path: Path) -> Path:
    root = tmp_path / "proton"
    write_shard(root, "1ABB006", n=3)
    write_shard(root, "1THB002", n=2)
    return root


# --------------------------------------------------------------------------
# Layout and dtypes
# --------------------------------------------------------------------------


def test_loader_consumes_what_preprocessing_writes(shard_root: Path):
    ds = BeamletDataset(shard_root, ["1ABB006", "1THB002"])
    assert len(ds) == 5

    sample = ds[0]
    assert sample["inputs"].shape == (2, *SHAPE)
    assert sample["target"].shape == (1, *SHAPE)
    assert sample["inputs"].dtype is torch.float32
    assert sample["target"].dtype is torch.float32


def test_storage_dtypes_are_the_measured_decision(shard_root: Path):
    """int16 CT + float16 labels -- the basis of the 105 GB sizing."""
    for pid in ("1ABB006", "1THB002"):
        ct = np.load(shard_root / pid / "ct_i16.npy", mmap_mode="r")
        label = np.load(shard_root / pid / "label_f16.npy", mmap_mode="r")
        assert ct.dtype == CT_DTYPE, f"{pid} CT dtype drifted -- storage size assumption broken"
        assert label.dtype == LABEL_DTYPE, f"{pid} label dtype drifted"


def test_arrays_are_memory_mappable(shard_root: Path):
    """Guards the .npy-not-.npz decision: compression would defeat mmap."""
    ct = np.load(shard_root / "1ABB006" / "ct_i16.npy", mmap_mode="r")
    assert isinstance(ct, np.memmap), "shard is no longer memory-mappable"


def test_cuboid_shape_matches_the_grid(shard_root: Path):
    ct = np.load(shard_root / "1ABB006" / "ct_i16.npy", mmap_mode="r")
    assert ct.shape[1:] == SHAPE == (384, 64, 16)


# --------------------------------------------------------------------------
# Count consistency -- the failure mode that silently truncates training
# --------------------------------------------------------------------------


def test_counts_agree_across_marker_meta_and_arrays(shard_root: Path):
    for pid in ("1ABB006", "1THB002"):
        shard = shard_root / pid
        marker = json.loads((shard / "_COMPLETE").read_text())
        meta = json.loads((shard / "meta.json").read_text())
        ct = np.load(shard / "ct_i16.npy", mmap_mode="r")
        label = np.load(shard / "label_f16.npy", mmap_mode="r")

        assert marker["count"] == meta["count"] == len(ct) == len(label)
        assert len(meta["beamlets"]) == meta["count"], (
            "per-beamlet records out of step with the array -- energies would "
            "be misaligned with their cuboids"
        )


def test_marker_records_build_parameters(shard_root: Path):
    """A marker from a truncated run must not satisfy a full run."""
    marker = json.loads((shard_root / "1ABB006" / "_COMPLETE").read_text())
    assert "max_beamlets" in marker, "marker cannot prove which run built it"


def test_incomplete_shard_is_skipped_not_silently_partial(tmp_path: Path):
    root = tmp_path / "proton"
    write_shard(root, "1ABB006", n=3, complete=True)
    write_shard(root, "1ABB021", n=2, complete=False)

    ds = BeamletDataset(root, ["1ABB006", "1ABB021"])
    assert len(ds) == 3, "a shard without _COMPLETE leaked into the dataset"


def test_no_complete_shards_raises_rather_than_training_on_nothing(tmp_path: Path):
    root = tmp_path / "proton"
    write_shard(root, "1ABB006", n=2, complete=False)
    with pytest.raises(RuntimeError):
        BeamletDataset(root, ["1ABB006"])


# --------------------------------------------------------------------------
# Label semantics
# --------------------------------------------------------------------------


def test_labels_are_raw_not_peak_normalized(shard_root: Path):
    """Per-beamlet peak normalization would destroy relative magnitude, which
    Level 2 plan metrics depend on."""
    label = np.load(shard_root / "1ABB006" / "label_f16.npy", mmap_mode="r")
    peaks = np.asarray(label, dtype=np.float32).reshape(len(label), -1).max(axis=1)
    assert not np.allclose(peaks, 1.0), "labels look peak-normalized"
    assert peaks.max() < 1e-1, "labels look rescaled away from physical dose"


def test_global_scale_puts_targets_near_unity(shard_root: Path):
    ds = BeamletDataset(shard_root, ["1ABB006"])
    target = ds[0]["target"]
    assert target.max() < 100.0, "target scale is far from O(1); DOSE_SCALE drifted"
    assert DOSE_SCALE == pytest.approx(1e-3)


def test_float16_preserves_label_precision():
    """The 105 GB decision assumed float16 costs 0.0% of the MAE floor.

    Checked against the measured peak range rather than in the abstract: at
    ~1e-3, float16 has ~1e-7 resolution, four orders below the 0.0025 floor.
    """
    values = np.linspace(PEAK_P1, PEAK_P99, 10_000, dtype=np.float64)
    round_tripped = values.astype(np.float16).astype(np.float64)
    rel_err = np.abs(round_tripped - values).max() / PEAK_MEDIAN
    assert rel_err < 1e-3, f"float16 relative error {rel_err:.2e} is no longer negligible"


def test_wepl_adds_exactly_one_channel(shard_root: Path):
    plain = BeamletDataset(shard_root, ["1ABB006"], with_wepl=False)
    wepl = BeamletDataset(shard_root, ["1ABB006"], with_wepl=True)

    assert plain.n_channels == 2 and wepl.n_channels == 3
    assert wepl[0]["inputs"].shape == (3, *SHAPE)
    # WEPL is a cumulative sum along the beam axis -- must be non-decreasing.
    channel = wepl[0]["inputs"][2].numpy()
    assert np.all(np.diff(channel, axis=0) >= -1e-6), "WEPL channel is not cumulative"


# --------------------------------------------------------------------------
# Leak guard -- test patients must not exist as tensors at all
# --------------------------------------------------------------------------


def _real_root() -> Path | None:
    override = os.environ.get("DOSERAD_BEAMLET_ROOT")
    candidates = [Path(override)] if override else []
    candidates.append(Path(__file__).resolve().parents[1] / "data" / "dataset_beamlet_tall24_it32" / "proton")
    for c in candidates:
        if c.is_dir():
            return c
    return None


def test_test_patients_are_never_preprocessed():
    """Preprocessed tensors for the 8 held-out test patients should not exist
    at all -- evaluation must go through the real
    inference path."""
    root = _real_root()
    if root is None:
        pytest.skip("no real shard root present")

    present = {p.name for p in root.iterdir() if p.is_dir()}
    leaked = present & set(get_splits()["test"])
    assert not leaked, f"test-split patients were preprocessed: {sorted(leaked)}"


def test_real_shards_satisfy_the_contract():
    """Runs wherever shards exist; skips on a checkout without them."""
    root = _real_root()
    if root is None:
        pytest.skip("no real shard root present")

    complete = sorted(p for p in root.iterdir() if (p / "_COMPLETE").exists())
    if not complete:
        pytest.skip(f"no completed shards under {root} yet")

    for shard in complete:
        marker = json.loads((shard / "_COMPLETE").read_text())
        meta = json.loads((shard / "meta.json").read_text())
        ct = np.load(shard / "ct_i16.npy", mmap_mode="r")
        label = np.load(shard / "label_f16.npy", mmap_mode="r")

        assert ct.dtype == CT_DTYPE and label.dtype == LABEL_DTYPE, shard.name
        # Against the shard's OWN marker, not the default box. This is the
        # command `scripts/data/preprocess_beamlets.py` tells you to run to
        # accept or reject a build, and from 2026-08-23 there is more than one
        # box to build -- pinned to `BeamletGrid()` it would have
        # rejected every correct fine-depth (0.5 mm) run. Reading the marker is also
        # the stricter check: it catches an array disagreeing with the box its
        # own marker claims, which the fixed constant never could.
        built = BeamletGrid.from_dict(marker["grid"]).shape if marker.get("grid") else SHAPE
        assert ct.shape[1:] == built and label.shape[1:] == built, shard.name
        assert marker["count"] == meta["count"] == len(ct) == len(label), shard.name
        assert len(meta["beamlets"]) == meta["count"], shard.name


def test_real_shards_account_for_skipped_beamlets():
    """Skips must be counted and reported, never silently dropped.

    A bare `if not path.exists(): continue` hid a *local* download fault for
    weeks: 80 files/patient were missing because our HuggingFace fetch read only
    the first page of a limit=1000 listing, and the last 80 filenames in
    lexicographic order are beams 7-9 (`Dose_B10...` sorts before `Dose_B8...`).
    The release itself ships all 1,080 (`scripts/data/preprocess_beamlets.py`).
    """
    root = _real_root()
    if root is None:
        pytest.skip("no real shard root present")

    complete = sorted(p for p in root.iterdir() if (p / "_COMPLETE").exists())
    full_runs = [
        s for s in complete
        if json.loads((s / "_COMPLETE").read_text()).get("max_beamlets") is None
    ]
    if not full_runs:
        pytest.skip("only truncated (max_beamlets) runs present")

    for shard in full_runs:
        meta = json.loads((shard / "meta.json").read_text())
        assert "failed" in meta, f"{shard.name} does not account for skipped beamlets"
        assert len(meta["failed"]) == meta["attempted"] - meta["count"], shard.name


def test_real_shards_were_not_built_from_the_truncated_download():
    """Shards holding <=1000 beamlets came from an incomplete raw copy.

    A complete plan holds 1,080 beamlets. A download that enumerates the remote
    directory instead of the plan JSON silently stops at 1,000, which looks like
    a coherent gantry band rather than like loss -- so the count is asserted
    rather than trusted.
    """
    root = _real_root()
    if root is None:
        pytest.skip("no real shard root present")

    complete = sorted(p for p in root.iterdir() if (p / "_COMPLETE").exists())
    full_runs = [
        s for s in complete
        if json.loads((s / "_COMPLETE").read_text()).get("max_beamlets") is None
    ]
    if not full_runs:
        pytest.skip("only truncated (max_beamlets) runs present")

    short = {
        s.name: json.loads((s / "_COMPLETE").read_text())["count"]
        for s in full_runs
        if json.loads((s / "_COMPLETE").read_text())["count"] <= TRUNCATED_PER_PATIENT
    }
    if short:
        pytest.fail(
            f"{len(short)} shard(s) built from a truncated download "
            f"(e.g. {dict(list(short.items())[:3])}); expected "
            f"{EXPECTED_PER_PATIENT}/patient. Re-fetch and rebuild."
        )


def test_every_shard_agrees_on_its_label_cutoff(shard_root):
    """A cohort mixing raw and thresholded labels trains a model matching neither.

    Nothing else distinguishes the two: same shapes, same dtypes, same file
    sizes, same beamlet counts. The marker is the only record, which is why
    `train_doserad.shard_label_cutoff` reads it rather than trusting a flag --
    and why a run that half-completed under a different cutoff must fail here
    rather than at the end of a training run.
    """
    root = shard_root
    markers = sorted(root.glob("*/_COMPLETE"))
    if not markers:
        pytest.skip("no completed shards")

    seen = {}
    for m in markers:
        seen.setdefault(
            float(json.loads(m.read_text()).get("label_cutoff", 0.0) or 0.0), []
        ).append(m.parent.name)

    assert len(seen) == 1, (
        "shards were built with different label cutoffs: "
        + "; ".join(f"{c:g} -> {len(v)} patients (e.g. {v[:2]})" for c, v in sorted(seen.items()))
    )


def test_a_shard_with_data_but_no_marker_stops_training(tmp_path: Path):
    """The failure the 2026-08-15 reshard actually produced.

    Every patient wrote its arrays and then died before `_COMPLETE`. The loader
    skips such shards with a printed line and trains on what is left, so a
    half-finished rebuild silently changes the cohort. `shard_label_cutoff` runs
    first and refuses instead.
    """
    from scripts.train.train_doserad import shard_label_cutoff

    (tmp_path / "1ABB006").mkdir()
    (tmp_path / "1ABB006" / "_COMPLETE").write_text(json.dumps({"label_cutoff": 1e-6}))
    assert shard_label_cutoff(tmp_path) == 1e-6

    (tmp_path / "1THB002").mkdir()
    (tmp_path / "1THB002" / "meta.json").write_text("{}")      # arrays written, died before the marker
    with pytest.raises(SystemExit, match="no _COMPLETE"):
        shard_label_cutoff(tmp_path)


def test_an_unreadable_marker_is_not_treated_as_raw_labels(tmp_path: Path):
    """Unknown is not 0.0. A *missing* label_cutoff key does mean raw labels --
    no build before 2026-08-15 wrote one -- but a corrupt marker means we do not
    know, and guessing stamps the checkpoint with a cutoff it was not trained on."""
    from scripts.train.train_doserad import shard_label_cutoff

    (tmp_path / "1ABB006").mkdir()
    (tmp_path / "1ABB006" / "_COMPLETE").write_text("{not json")
    with pytest.raises(SystemExit, match="unreadable"):
        shard_label_cutoff(tmp_path)

    (tmp_path / "1ABB006" / "_COMPLETE").write_text(json.dumps({"count": 1080}))
    assert shard_label_cutoff(tmp_path) == 0.0                 # missing key = pre-2026-08-15


# ---------------------------------------------------------------------------
# The writer, end to end
# ---------------------------------------------------------------------------
#
# `scripts/data/preprocess_beamlets.py` had NO test that executed it -- it appeared in
# this file only in comments. On 2026-08-15 a worker referenced `args.label_cutoff`
# where only the `label_cutoff` parameter is in scope; `ast.parse` passed, the
# whole suite passed, and the crash landed after each patient's arrays were
# written. Two nodes ran 40 minutes, wrote 49 GB, and produced no `_COMPLETE`.
# A NameError in a worker is exactly what "the tests pass" cannot tell you.


def _synthetic_patient(root: Path, pid: str = "1ABB001", n_beamlets: int = 2) -> Path:
    """A patient with the three things the script reads: CT, plan, dose maps."""
    import SimpleITK as sitk

    d = root / pid
    (d / "image").mkdir(parents=True)
    (d / "dose").mkdir()

    shape, spacing = (32, 140, 140), (1.0, 1.0, 3.0)     # sitk (x, y, z)
    ct = np.full(shape, -1024.0, dtype=np.float32)
    ct[8:24, 40:100, 40:100] = 40.0                       # a body the ray must cross
    img = sitk.GetImageFromArray(ct)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(d / "image" / "ct.mha"))

    beamlets = []
    for i in range(n_beamlets):
        dose = np.zeros(shape, dtype=np.float32)
        dose[12:20, 60:80, 50:90] = 1.0e-3
        dose[10:22, 50:90, 45:95] += 1.0e-9               # sub-cutoff dust to threshold
        dm = sitk.GetImageFromArray(dose)
        dm.SetSpacing(spacing)
        sitk.WriteImage(dm, str(d / "dose" / f"Dose_B0_R0_L{i}.mha"))
        beamlets.append({"beamlet_idx": i, "energy": 120.0 + 10 * i})

    (d / f"{pid}.json").write_text(json.dumps({
        "beams": [{"beam_idx": 0, "gantry_angle": 90.0, "rays": [{
            "ray_idx": 0,
            "ray_source": [-300.0, 70.0, 48.0],
            "ray_target": [300.0, 70.0, 48.0],
            "beamlets": beamlets,
        }]}]
    }))
    return d


def test_the_writer_runs_and_marks_what_it_finished(tmp_path: Path):
    """Run the real script on a real (tiny) patient and check its three outputs.

    Via the CLI rather than by importing `process_patient`, because the defect
    that motivated this lived in a *worker process* -- the scope a direct call
    would not reproduce.
    """
    import subprocess
    import sys

    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    _synthetic_patient(raw)

    proc = subprocess.run(
        [sys.executable, "scripts/data/preprocess_beamlets.py",
         "--patients", "1ABB001", "--raw-root", str(raw), "--out-root", str(out),
         "--label-cutoff", "1e-6", "--workers", "1", "--iterations", "1"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "patients ok : 1/1" in proc.stdout, proc.stdout[-2000:]

    shard = out / "1ABB001"
    for name in ("ct_i16.npy", "label_f16.npy", "meta.json", "_COMPLETE"):
        assert (shard / name).exists(), f"{name} missing -- {proc.stdout[-2000:]}"

    # The cutoff must reach BOTH records: the marker is what the staleness check
    # and `shard_label_cutoff` read, meta.json is what makes a copied shard
    # directory self-describing.
    assert json.loads((shard / "_COMPLETE").read_text())["label_cutoff"] == 1e-6
    assert json.loads((shard / "meta.json").read_text())["label_cutoff"] == 1e-6
    # And the Landweber count, which is what lets a resume refuse to mix labels
    # built with a different one.
    assert json.loads((shard / "_COMPLETE").read_text())["iterations"] == 1

    # A resume with the same settings finds the shard done and succeeds.
    same = subprocess.run(
        [sys.executable, "scripts/data/preprocess_beamlets.py",
         "--patients", "1ABB001", "--raw-root", str(raw), "--out-root", str(out),
         "--label-cutoff", "1e-6", "--workers", "1", "--iterations", "1"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=600,
    )
    assert same.returncode == 0, same.stdout[-2000:]
    assert "patients ok : 1/1" in same.stdout, same.stdout[-2000:]

    # A resume asking for another count must not reuse the shard as done.
    again = subprocess.run(
        [sys.executable, "scripts/data/preprocess_beamlets.py",
         "--patients", "1ABB001", "--raw-root", str(raw), "--out-root", str(out),
         "--label-cutoff", "1e-6", "--workers", "1", "--iterations", "2"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=600,
    )
    assert again.returncode != 0, again.stdout[-2000:]
    assert "stale" in again.stdout and "iterations=1" in again.stdout, again.stdout[-2000:]
    assert "patients ok : 0/1" in again.stdout, again.stdout[-2000:]


# --------------------------------------------------------------------------
# The box the shards were built with
# --------------------------------------------------------------------------
#
# Resharding to a second box makes the box a variable for the first time. Everything below
# guards the same failure: the arrays come off disk at their real shape, so a
# box mismatch never shows up as a shape error -- it shows up as dose in the
# wrong place, or as a Bragg curve read at the wrong depth, with nothing raised.

FINE = BeamletGrid(n_depth=768, depth_spacing=0.5, n_lat_u=128, n_lat_v=40)


def test_the_marker_records_the_spacings_not_only_the_shape():
    """A 0.5 mm-depth box and a 1 mm box of the same counts must not look alike."""
    coarse = BeamletGrid(n_depth=768, n_lat_u=128, n_lat_v=40)
    assert FINE.shape == coarse.shape
    assert FINE.as_dict() != coarse.as_dict()
    assert BeamletGrid.from_dict(FINE.as_dict()) == FINE


def test_a_shard_built_with_another_box_is_refused_not_reshaped(tmp_path: Path):
    """Reading 0.5 mm shards under the 1 mm default is the silent half.

    The cuboid would load fine -- it is just an array -- while `with_bragg`
    sampled the depth-dose curve at `grid.depth_spacing`, putting every Bragg
    peak at twice its true depth. That prior is the single input
    channel doing the most work, so the run would train and quietly under-
    perform its own floor, which is exactly the reading the arms exist to make.
    """
    root = tmp_path / "proton"
    write_shard(root, "1ABB006", n=2, grid=FINE)
    with pytest.raises(ValueError, match="different sampling box"):
        BeamletDataset(root, ["1ABB006"])
    # Named correctly, it loads, and the cuboid is the new box's.
    ds = BeamletDataset(root, ["1ABB006"], grid=FINE)
    assert ds[0]["inputs"].shape == (2, *FINE.shape)


def test_shards_disagreeing_on_the_box_stop_training(tmp_path: Path):
    """Same shape as the label-cutoff rule above, for the parameter beside it.

    A half-finished reshard leaves the old box's patients next to the new box's.
    `train_doserad.shard_grid` has to refuse rather than pick one -- a cohort
    spanning two boxes trains a model that matches neither, and the checkpoint
    would then record whichever box happened to win.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "train_doserad",
        Path(__file__).resolve().parents[1] / "scripts" / "train" / "train_doserad.py",
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    root = tmp_path / "proton"
    write_shard(root, "1ABB006", n=2, grid=FINE)
    assert train.shard_grid(root) == FINE

    write_shard(root, "1THB002", n=2)  # the old box
    with pytest.raises(SystemExit, match="disagree on the sampling box"):
        train.shard_grid(root)
