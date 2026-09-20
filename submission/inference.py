"""Container I/O: read the mounted run, predict, write the ten output slots.

Everything model- or geometry-related lives behind `models.predictor`; this file
only moves data.

What bites here, none of it loudly
----------------------------------
Authority is <https://doserad2026.grand-challenge.org/>; where its instructions page
and the organisers' example code differ, the code is what the platform runs.
Everything below either fails silently or gets a submission rejected; the rest of the API
is visible the moment it is wrong and is not written down.

* **A submission is TWO uploads.** An algorithm is a container image *plus* a
  separately uploaded **Model**, extracted read-only at `/opt/ml/model`; the
  instructions page never says so, their `do_save.sh` does. We ship weights that
  way, so an image uploaded without its Model **starts, serves `/health`, and
  scores as noise**. `init_model` refuses rather than allow that, unless
  `DOSERAD_ALLOW_STUB=1`.
* **Build 4-D stacks with `sitk.JoinSeries`**, never `GetImageFromArray` on a
  stacked 4-D array — SimpleITK reads the extra axis as vector components and
  writes a file that looks valid and is wrong.
* **Each dose map sits on exactly its input image's grid.** The evaluator reads
  the *plan's* spacing, not ours, so a grid mistake scores identically to a
  clean run and is invisible to every metric.
* **Zero everything below that beamlet's `minimum_cutoff`.** The public
  `Implementation Errors Count` counts **dose maps holding any non-zero value
  below the cutoff, not violating voxels**, so one stray voxel costs a whole map
  and repeat offenders can be rejected.
* `LABEL org.grand-challenge.api-method="invoke"`, or the submission is rejected.
* `GET /health` → 200 once loaded; never load inside `/invoke`. Whether time
  before this is free is not documented.
* **The platform-side importer has OOM-killed on oversized stacks** — a
  540-map single slot decompressed to 24.1 GiB and failed. Fixed by Grand
  Challenge *"as long as you adhere to the expected stack sizes"*. Spreading
  frames across the ten slots is the endorsed remedy and is what we do. Nothing
  here can detect that failure; it happens after execution.

Memory
------
Each dose map is a full CT volume (~85 MB as float32 on the measured
447x455x105 grid) and a run can carry 500 beamlets, so this file is written
around two rules, both still load-bearing:

* **one output slot at a time** -- predict that slot's beamlets, write the
  stack, release it. Peak is one slot, not the whole run.
* **one prediction window at a time within a slot** -- convert each numpy volume
  to its sitk frame as it arrives, rather than holding the slot's volumes and
  the slot's frames side by side.

Measured peak container memory is **2.6 GB** (2026-08-15, twice, at the scored
50-frame layout). An older estimate here said ~8.5 GB per slot; it assumed
`sitk.JoinSeries` held one resident, which `StreamingStackWriter` never does.
⇒ **Host RAM does not bind.** Take the platform's 32 GB for vCPU -- the write is
CPU-bound -- not for headroom.
"""

import glob
import json
import os
import time

# Import time, not init time: what the platform waits for is the whole gap
# between starting our process and `/health` answering, and the model load sits
# inside it. The `[ready]` line at the end of `init_model` reports that gap, so
# a preliminary log gives a **lower bound on their health timeout** — if the job
# ran, they waited at least this long.
_IMPORTED_AT = time.perf_counter()

import queue
import threading
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from models.geometry import VolumeGeometry
from models.predictor import BeamletRequest, DosePredictor

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
NUM_OUTPUT_FILES = 10
INPUT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
INPUT_JSON_NAME = "stacked-proton-beam-level-metadata.json"

# Output compression level. 1 is what we ship (see StreamingStackWriter); the
# variable exists so the level-6 arm of the A/B can be run from the *same*
# image, rather than rebuilding 11.9 GB to change one constant and thereby
# comparing two builds instead of two settings.
#
# No default, deliberately: the Dockerfile is the only place a setting's value
# is written, so `docker inspect` reports the real configuration and a build
# that drops an ENV line fails at import rather than silently falling back to a
# number written somewhere else. Same rule CHECKPOINT_PATH already follows.
ZLIB_LEVEL = int(os.environ["ZLIB_LEVEL"])

# Whether the body mask ships. Same no-default rule, and strictly parsed: a
# typo'd "true" silently reading as False would change a scored metric without
# changing anything visible, which is the failure shape this file exists to
# avoid. Why it ships on is in `DosePredictor.body_mask`.
_BODY_MASK_RAW = os.environ["BODY_MASK"]
if _BODY_MASK_RAW not in ("0", "1"):
    raise RuntimeError(
        f"BODY_MASK={_BODY_MASK_RAW!r} is not '0' or '1'. It decides whether a "
        "scored metric is computed on masked output; there is no safe guess."
    )
BODY_MASK = _BODY_MASK_RAW == "1"

# Parsed and range-checked here rather than deep in the predictor: a container
# built with a typo must fail at import, not emit 2,329 silently wrong dose maps.
try:
    SNAP_ALPHA = float(os.environ["SNAP_ALPHA"])
except (KeyError, ValueError) as exc:
    raise SystemExit(
        f"SNAP_ALPHA must be a float in [0, 1); got "
        f"{os.environ.get('SNAP_ALPHA')!r}. It is the fraction of the cutoff below "
        "which sub-cutoff dose is zeroed rather than snapped up; 0 disables the "
        "snap entirely."
    ) from exc
if not 0.0 <= SNAP_ALPHA < 1.0:
    raise SystemExit(f"SNAP_ALPHA={SNAP_ALPHA!r} is outside [0, 1).")

def _usable_cpus() -> int:
    """Cores this *process* may run on, not cores the host has.

    `os.cpu_count()` reports the machine, so inside a container pinned to a
    cpuset it over-reports and we would size the pool to hardware we cannot
    touch. The platform does not publish what it pins us to, and oversubscribing
    a compressor is how a parallel write gets slower than a serial one.
    `sched_getaffinity` is Linux-only, which is where the container runs; the
    fallback is for the Mac, where this file is only ever tested.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# How the write is parallelised. These two DO carry defaults, unlike ZLIB_LEVEL:
# they change no output byte-for-byte -- the voxels and the stream are identical
# at any setting -- so a build that omits them is not misconfigured, only
# slower. `ZLIB_WORKERS = 0` means "ask the machine", which is the right default
# for a platform that does not publish its instance type.
# **HALF THE CORES, NOT ALL OF THEM — THE WRITER WAS STEALING THE PRODUCER'S
# CPU** (2026-08-29, A10G, the challenge's runtime fit, one env flag apart):
#
#     ZLIB_WORKERS      2       3       4       6       8
#     fitted (s)    18.63   16.82   16.88   16.95   17.83
#
# ⇒ **8 costs 5.7% against 4**, and the container's own log says why: at 8 the
# write is 19 ms/beamlet and `predict` **37**; at 4 the write slows to 22 and
# `predict` falls to **35**, with queue-wait still 0.0 — so the write hides
# either way and the only thing the extra threads changed was how much of an
# 8-vCPU box the *producer* could have. And 2 is worse than 8: there the write
# stops hiding (32 ms/beamlet, queue-wait 0.1, drain 0.2) — ⇒ **this is a
# plateau between 3 and 6 with a cliff on the low side**, not a monotone knob.
# Expressed as a FRACTION of what the process may use, not as the literal 4:
# the platform does not publish what it pins us to, and the measured fact is
# "half", on the 8 vCPU our own `[env]` line reports there.
ZLIB_WORKERS = int(os.environ.get("ZLIB_WORKERS", "0")) or max(1, _usable_cpus() // 2)
# Blocks per frame. More than workers, deliberately: frames finish unevenly and
# a queue of small blocks keeps every worker fed to the end. Measured 6.05x at
# 32/8 against 5.78x at 16/8.
# Blocks per frame. **This, not ZLIB_LEVEL, is the write's real lever**: on
# real predicted frames, level 1→9 costs 2.0x while blocks 32→1 costs a further
# 12x, because the block count drives both the thread parallelism and how much
# of the frame the all-zero cache can skip. Measured 2026-08-17 through this
# writer on predicted output, not ground truth. No default, same rule as
# ZLIB_LEVEL: the Dockerfile is the only place a value is written.
ZLIB_BLOCKS = int(os.environ["ZLIB_BLOCKS"])
# Frames `predict` may run ahead of the writer. The write is CPU work with the
# GIL released, and `predict` spends most of its wall BLOCKED on the GPU
# (the host waits inside `render_block` for
# four times that stage's own kernels), so the two halves want the same wall
# clock and different resources. At 0 this is the old strictly-serial loop,
# which is also the escape hatch if a box ever has fewer cores than the writer
# wants. Output is byte-identical at any value: same frames, same order, one
# consumer.
# Each queued frame is a whole 85 MB volume, and `predict` already returns a
# batch of them, so this is bounded rather than unbounded on purpose.
#
# **The default is BATCH_SIZE, and the batch is why.** `predict` computes a
# whole batch and then hands over all of it at once, so a queue shallower than
# one batch fills on the handover and the producer blocks *instead of starting
# the next batch* -- the writer then runs while predict waits, which is the
# serial schedule with extra steps. Measured 2026-08-24 on 8 pinned cores
# 64 beamlets, alternating arms:
#
#     write-ahead      0       3       8      16
#     ms/beamlet   62.77   59.30   54.44   54.45
#     vs serial    1.000   0.945   0.867   0.867
#
# ⇒ depth 3 buys a third of what depth 8 buys, and depth 16 buys nothing more:
# **one batch is the whole answer**, and queue-wait falls from 0.4 s to 0.0
# exactly there. Do not "tune" this above the batch size; it costs 85 MB a
# frame and buys nothing.
WRITE_AHEAD = int(os.environ.get("WRITE_AHEAD") or os.environ["BATCH_SIZE"])
# Output frames recycled instead of re-allocated per beamlet. **The saving is
# the page faults, not the copy**: a frame is ~85 MB, `np.zeros` is a calloc, and
# writing a 1.6 MB block over an 11 MB window first-touches ~2700 pages -- 1.47
# ms fresh against 0.09 ms on a frame that is already hot, i.e. **1.15
# ms/beamlet, ~0.6 s of the scored number** (measured on an A10G
# 2026-08-26). Voxels are identical either way, so this touches no accuracy
# metric; what it touches is the one place a frame can be recycled while
# somebody is still reading it, which is why the release below sits *after*
# `writer.add` and nowhere else.
#
# **The capacity is the live set, not a tuning knob.** `predict` returns a
# whole batch and the queue holds `WRITE_AHEAD` more, so that many frames are
# alive at the peak with or without this -- the pool holds the same ones instead
# of freeing and re-faulting them. Deeper buys nothing and costs 85 MB a frame.
# 0 switches it off, which is the arm to run if a recycle is ever suspected.
FRAME_POOL = os.environ.get("FRAME_POOL", "1") != "0"
# Whether the writer may keep working across a SLOT boundary. **This is where
# 10% of the wall was** (measured 2026-08-29): a writer and a consumer thread
# per slot, joined at the end of each one, leaves the
# producer standing still while the queue empties — ~140 ms a slot, ten slots a job,
# **6.2 ms/beamlet on `(1,225)`**, which is one of the four shapes the platform
# actually fits. The GPU is idle for every millisecond of it, and the next slot's
# first batch has no dependency whatever on the previous slot's bytes: they are
# different files. ⇒ one consumer for the whole invoke, and a slot ends by
# *queueing* its close rather than waiting for it. 0 restores the per-slot join.
SLOT_OVERLAP = os.environ.get("SLOT_OVERLAP", "1") != "0"
# Whether the sampled CT box is reused across a ray's beamlets
# (`models/predictor.py`, `_cuboid`). The third per-ray memo and the largest;
# 0 is the kill switch and the A/B arm, so one image prices it.
CUBOID_MEMO = os.environ.get("CUBOID_MEMO", "1") != "0"
# **OFF, AND IT IS OFF BECAUSE IT WAS MEASURED** (2026-08-29). A chunk's
# copy-out can finish while the NEXT chunk's forward runs
# (`models/predictor.py`, `_issue_host_volume`), and the slack it aims at is
# real — the forward call returns in **1.8 ms** and completes in **185.7**, so
# the host is free for 184 ms of every chunk. It still buys **nothing**:
# **15.94 · 15.95 against 15.96** in the runtime fit, a 0.02 s difference on a
# 0.07 s repeat spread. ⇒ the copy-out was already off the critical path, and
# what remains beside the forward is GPU work rather than host work.
# Kept rather than deleted, and default OFF rather than removed: it is
# byte-identical in its output (verified) and costs nothing where it sits,
# but an asynchronous D2H whose buffer lifetime is load-bearing is not shipped
# for a 0.1% that the noise cannot even resolve.
COPY_AHEAD = os.environ.get("COPY_AHEAD", "0") != "0"


def _zlib_header(level: int) -> bytes:
    """The 2-byte zlib header, computed rather than hardcoded to `78 01`.

    `FLEVEL` is advisory and no decoder reads it, but `FCHECK` is not: the
    16-bit header must be a multiple of 31 or a strict inflater rejects the
    stream before reading a byte of data. Deriving it means a changed
    `ZLIB_LEVEL` cannot silently emit a header describing a different one.
    """
    cmf = 0x78                       # deflate, 32 KiB window
    flevel = 0 if level <= 1 else 1 if level < 6 else 2 if level == 6 else 3
    flg = flevel << 6
    flg |= 31 - ((cmf << 8 | flg) % 31)
    return bytes((cmf, flg))


def _adler_combine(adler1: int, adler2: int, len2: int) -> int:
    """zlib's `adler32_combine`, which CPython does not expose.

    Needed because the stack's checksum must cover every byte in order, while
    the blocks that produce those bytes are compressed out of order across
    threads. Recomputing it sequentially instead costs 31.7 ms per 85 MB frame
    against ~14 ms for the whole parallel compress -- it would *be* the
    bottleneck. Pinned against `zlib.adler32` over random splits in
    `tests/test_streaming_stack_writer.py`.
    """
    base = 65521
    remainder = len2 % base
    sum1 = adler1 & 0xffff
    sum2 = (remainder * sum1) % base
    sum1 += (adler2 & 0xffff) + base - 1
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + base - remainder
    if sum1 >= base:
        sum1 -= base
    if sum1 >= base:
        sum1 -= base
    if sum2 >= (base << 1):
        sum2 -= (base << 1)
    if sum2 >= base:
        sum2 -= base
    return sum1 | (sum2 << 16)

# Wall-clock split for one `invoke`, printed at the end of `run`. The whole
# runtime reversal came from having a per-beamlet predictor figure and no
# figure at all for the write the platform obliges us to perform, so the
# container reports both itself rather than leaving it to be inferred.
# `other` in the invoke line is a REMAINDER, and on the A10G it is 2.7-3.9 s
# of a 19 s job -- the second largest term now that the write is hidden
#. These three name the candidates so the next run does not
# have to guess: reading the CTs (an 85 MB decompress each), the placeholder
# files every unused slot still needs, and the drain of whatever the writer
# still owes after the last batch.
TIMING = {"predict_s": 0.0, "write_s": 0.0, "queue_wait_s": 0.0,
          "image_load_s": 0.0, "placeholder_s": 0.0, "drain_s": 0.0}


def report_environment() -> None:
    """Print what this container was given: cores, GPU, torch, free output space.

    Written to the container's own log, before `/health` returns, so it costs
    nothing that is timed. It reports the resources this process can see about
    itself -- enough to explain a runtime number afterwards -- and nothing about
    the machine or the job around it.

    Every branch is defensive on purpose. A diagnostic that can fail the run is
    worse than no diagnostic: this must never be the reason a submission errors.
    """
    try:
        facts = {"cpu_count": os.cpu_count()}
        try:
            facts["sched_affinity"] = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass
        try:
            import torch
            facts["torch"] = torch.__version__
            facts["cuda"] = torch.version.cuda
            if torch.cuda.is_available():
                facts["gpu"] = torch.cuda.get_device_name(0)
                major, minor = torch.cuda.get_device_capability(0)
                # sm_86 or not decides whether a compiled kernel package applies.
                facts["gpu_arch"] = f"sm_{major}{minor}"
        except Exception:                                         # noqa: BLE001
            pass
        try:
            usage = os.statvfs("/output")
            facts["output_free_mib"] = round(usage.f_bavail * usage.f_frsize / 2**20)
        except Exception:                                         # noqa: BLE001
            pass
        print(f"[env] {json.dumps(facts, sort_keys=True)}", flush=True)
    except Exception as exc:                                      # noqa: BLE001
        print(f"[env] diagnostics failed, ignoring: {exc}", flush=True)


def report_run_shape(metadata: list) -> None:
    """Print the shape of this job: how many images, beamlets and output slots.

    One line in the container's own log, so a runtime number afterwards can be
    read against the work it covered. Summarised rather than listed: a job's
    per-beamlet cutoffs are all distinct, and printing them would be thousands
    of characters.
    """
    try:
        summary = {"n_images": len(metadata), "n_beams": 0, "slots": {}, "cutoffs": set()}
        for entry in metadata:
            for beam in entry.get("beams", []):
                for ray in beam.get("rays", []):
                    for beamlet in ray.get("beamlets", []):
                        info = beamlet.get("output_info", {})
                        slot = info.get("output_file_idx")
                        summary["n_beams"] += 1
                        summary["slots"][slot] = summary["slots"].get(slot, 0) + 1
                        summary["cutoffs"].add(info.get("minimum_cutoff"))
        depths = sorted(summary["slots"].values())
        # Summarised, not listed. The cutoffs are **per beamlet** and all
        # distinct, so a 225-beamlet job printed 225 floats — thousands of
        # characters in the one log the platform returns. n/min/max keeps every
        # question they answer (are they per-beamlet? what range? one or many?).
        cutoffs = sorted(c for c in summary["cutoffs"] if c is not None)
        span = (f"n={len(cutoffs)} min={cutoffs[0]:.4g} max={cutoffs[-1]:.4g}"
                if cutoffs else "none")
        print(f"[shape] images={summary['n_images']} beams={summary['n_beams']} "
              f"slots_used={len(summary['slots'])} stack_depths={depths} "
              f"minimum_cutoffs({span})", flush=True)
    except Exception as exc:                                      # noqa: BLE001
        print(f"[shape] diagnostics failed, ignoring: {exc}", flush=True)



def warm_up(predictor) -> None:
    """Run one synthetic batch through the real path, before /health returns.

    **Everything before /health is untimed and every job is a fresh container**,
    so a cost paid here is paid out of the platform's pocket and a cost paid on
    the first `/invoke` is paid out of ours -- on *every job*, not once per
    submission. The first submission's own log shows the size of it: its
    1-beamlet jobs spent **0.5 s in `predict`** against 0.047 s/beamlet on its
    309-beamlet job, and the first job of a run cost 1.5 s/beamlet against a
    0.046 steady state.

    What is actually cold: the CUDA context, cuDNN's algorithm choice for these
    convolution shapes, the caching allocator's first arena, and the pinned
    staging buffer (`models/predictor.py`). All of them are shape-dependent and
    none of them care about the *values*, so a synthetic CT warms exactly what a
    real one would -- the network input shape comes from the beamlet grid, not
    from the image.

    **A full batch, not one beamlet.** cuDNN tunes per shape, and the shape
    that matters is the batched one the loop actually runs.

    **It may never take the container down.** A warmup is an optimisation;
    failing it should cost the seconds it would have saved and nothing else,
    which is why the whole thing is inside one `except`.
    """
    started = time.perf_counter()
    try:
        # Small enough to build instantly, big enough for the box to intersect
        # it; out-of-volume samples are air by construction (`models/geometry.py`).
        ct = np.full((24, 48, 48), -1000.0, dtype=np.float32)
        ct[8:16, 16:32, 16:32] = 40.0
        geom = VolumeGeometry(
            origin=np.array([-72.0, -72.0, -30.0]),
            spacing=np.array([3.0, 3.0, 2.5]),
            shape=ct.shape,
        )
        batch = max(getattr(predictor, "batch_size", 1), 1)

        def make(count):
            return [
                BeamletRequest(
                    ray_source=(600.0, 0.0, 0.0),
                    ray_target=(0.0, 0.0, 0.0),
                    energy=100.0 + i,
                    output_file_idx=0,
                    idx_in_output=i,
                    minimum_cutoff=1e-6,
                )
                for i in range(count)
            ]

        # **TWO shapes, and the second is the point.** A 500-beamlet job is
        # 62 batches of `batch` and one short batch, and everything that
        # specialises per shape -- cuDNN's autotuner, `torch.compile`'s graphs --
        # meets that short one for the first time *inside the timed window*.
        # Measured cost of such a first encounter: **100.3 ms/beamlet against
        # 27.8**. Warming a full batch and a short one puts
        # both in the untimed window, and for `torch.compile` the second shape
        # is also what marks the batch dimension dynamic, so no third size can
        # trigger a recompile later.
        volumes = predictor.predict(ct, geom, make(batch))
        if batch > 1:
            volumes = volumes + predictor.predict(ct, geom, make(max(batch // 2, 1)))
        # **AND A BATCH OF ONE, WHICH THE OTHER TWO DO NOT COVER.** Marking the
        # batch dimension dynamic buys every size *except* 1: PyTorch specialises
        # 0 and 1 unconditionally, so a slot holding a single beamlet recompiles
        # — **~8 s of `predict`, inside the timed window** (2026-08-24, A10G).
        # It is not a corner case: a stack of depth 1 is the
        # normal shape of a small job, and it lands on the tiny jobs that
        # anchor the fit's intercept.
        if batch > 2:
            volumes = volumes + predictor.predict(ct, geom, make(1))
        # **A warmup that renders nothing warms nothing.** If the synthetic
        # ray misses, or the box lands outside this little volume, `predict`
        # returns all-zero maps having skipped the render and the copy out --
        # the two stages after the forward pass -- and the first real job pays
        # for them anyway. Silence would look identical to success.
        touched = sum(int(np.any(v)) for v in volumes)
        if not touched:
            print("[warmup] produced only zeros: the synthetic geometry no "
                  "longer intersects, so render and copy-out were NOT warmed",
                  flush=True)
    except Exception as exc:                                      # noqa: BLE001
        # **A COMPILED CONTAINER MUST NOT CARRY ITS FAILURE PAST /health.**
        # Swallowing this and returning is right for an eager warmup -- the
        # warmup is an optimisation and may cost only itself. It is wrong when
        # the module is compiled: the *reason* the warmup failed is then almost
        # certainly the compilation, the compiled module is still installed, and
        # the first real job meets the same failure inside the timed window,
        # where the platform sees a job returning 0 instead of 201. That is how
        # the first `torch.compile` run died.
        #
        # The final phase returns **no logs** and allows **two** submissions,
        # so nothing that depends on an unobservable property of the platform
        # may decide whether we score at all. Drop the fast path and serve.
        if getattr(predictor, "disable_compilation", lambda: False)():
            print(f"[warmup] COMPILATION ABANDONED, serving eager: {exc}",
                  flush=True)
            try:
                predictor.predict(ct, geom, make(batch))
            except Exception as second:                           # noqa: BLE001
                print(f"[warmup] eager warmup also failed, ignoring: {second}",
                      flush=True)
            return
        print(f"[warmup] skipped, ignoring: {exc}", flush=True)
        return
    finally:
        # Nothing synthetic may survive into the first real image: the memo, the
        # uploaded CT and the render coordinates all key on this fake geometry.
        try:
            predictor.reset_entry_cache()
        except Exception:                                         # noqa: BLE001
            pass
    print(f"[warmup] {time.perf_counter() - started:.1f} s, untimed "
          f"(before /health)", flush=True)


def init_model():
    # 8, not 32: batch is not a lever. On an A100,
    # b16 and b32 move `predict` by 0.993x and 1.013x (both CIs straddle 1)
    # while the forward alone drops 8-11% -- the forward is not what the wall
    # is made of. ⇒ leave it at 8; the dtype
    # below is the knob that does move it.
    # No default -- see ZLIB_LEVEL above.
    batch_size = int(os.environ["BATCH_SIZE"])
    checkpoint = os.environ.get("CHECKPOINT_PATH") or None

    # The dtype the FORWARD PASS runs in; everything around it stays float32
    # (`models/predictor.py`). Defaulted rather than required, unlike
    # BATCH_SIZE, because an unset value here is the behaviour every container
    # before this one had -- and because this is the first runtime knob on the
    # path that changes voxel values, so it has to be asked for.
    # An unsupported value raises here rather than falling back: init runs
    # before /health, so a refusal costs a container that never starts instead
    # of a submission slot that scores a silent fp32 run as a precision win.
    infer_dtype = os.environ.get("INFER_DTYPE", "fp32")

    # Whether to run one batch through the path before /health. Defaulted on and
    # switchable, so a single image can measure both arms with `--env` rather
    # than costing a build apiece — the warmup's whole benefit is per fresh
    # process, which is exactly what a long-lived proxy cannot show.
    warmup_enabled = os.environ.get("WARMUP", "1") != "0"

    # torch.compile mode, "" for eager. Off by default: compilation fuses
    # operations, so bit-identity is not guaranteed and owes an accuracy answer
    # the way fp16 did (the ahead-of-time package in the scored image did write
    # volumes byte-identical to eager on the A10G) -- but unlike fp16 it is a measured *win* on the A10G (forward
    # 0.855x, `predict` 0.899x), and its cost lands in the warmup's untimed
    # window rather than in a scored job.
    compile_mode = os.environ.get("COMPILE_MODE", "") or None
    if compile_mode:
        # **The rootfs is READ-ONLY on the platform**, and Triton writes its
        # compiled kernels to `$HOME/.triton` — so compilation dies at first use
        # with `OSError: [Errno 30] Read-only file system`, *after* a healthy
        # start, and the job returns 0 instead of 201. `/tmp` is the tmpfs the
        # harness mounts, so it is the one writable place a compiled container
        # has. Set before torch is asked to compile anything, and only when
        # compiling, so a normal run's environment is untouched.
        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/inductor")

    # Weights arrive on the platform's read-only /opt/ml/model mount, not baked
    # into the image, so "the weights are missing" is now a *runtime* condition
    # rather than a build failure. `from_checkpoint` raises on a missing path,
    # and only an unset CHECKPOINT_PATH selects StubModel -- which would start,
    # pass /health, answer /invoke and score as noise.
    #
    # So reaching the stub must always be a choice. Set DOSERAD_ALLOW_STUB=1 to
    # ask for it; otherwise absent weights stop the container before /health.
    allow_stub = os.environ.get("DOSERAD_ALLOW_STUB") == "1"
    if checkpoint is None and not allow_stub:
        raise RuntimeError(
            "CHECKPOINT_PATH is unset and DOSERAD_ALLOW_STUB is not 1. This "
            "container would run as an untrained stub and score as noise. Mount "
            "the model at /opt/ml/model, or set DOSERAD_ALLOW_STUB=1 to say you "
            "meant it."
        )
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise RuntimeError(
            f"CHECKPOINT_PATH={checkpoint} does not exist. On the platform the "
            f"weights arrive on the read-only /opt/ml/model mount; if that mount "
            f"is absent or empty the Model was never uploaded alongside the "
            f"container image. Failing here rather than scoring as noise."
        )

    print(f"Initializing DosePredictor (batch_size={batch_size}, "
          f"body_mask={BODY_MASK}, snap_alpha={SNAP_ALPHA}, "
          f"infer_dtype={infer_dtype}, frame_pool={FRAME_POOL}, "
          f"cuboid_memo={CUBOID_MEMO}, slot_overlap={SLOT_OVERLAP}, "
          f"copy_ahead={COPY_AHEAD}, "
          f"warmup={warmup_enabled}, compile={compile_mode or 'off'}, "
          f"checkpoint={checkpoint or 'STUB (explicitly allowed)'})", flush=True)
    # Here, not in `run`: everything before /health returns 200 is untimed, so
    # the diagnostics cost nothing that is scored.
    report_environment()
    # No `grid=` here on purpose: the box comes off the checkpoint
    # (`models/predictor.py`), so a resharded arm ships without this file
    # knowing anything about it. Naming a box here would be a second place to
    # keep in step, and the failure is silent -- misplaced dose, no error.
    predictor = DosePredictor.from_checkpoint(
        checkpoint,
        batch_size=batch_size,
        body_mask=BODY_MASK,
        snap_alpha=SNAP_ALPHA,
        infer_dtype=infer_dtype,
        compile_mode=compile_mode,
        # The live set, not a knob: a batch in flight plus the write-ahead
        # queue, which is what this path already peaks at (`FRAME_POOL`).
        frame_pool=(batch_size + WRITE_AHEAD + 2) if FRAME_POOL else 0,
        cuboid_memo=CUBOID_MEMO,
        copy_ahead=COPY_AHEAD,
    )
    print(f"Sampling box {predictor.grid.shape} @ "
          f"({predictor.grid.depth_spacing}, {predictor.grid.lat_u_spacing}, "
          f"{predictor.grid.lat_v_spacing}) mm", flush=True)
    # Resolved, not requested: a header echoing the argument
    # prints "bf16" for a run that fell back, and this path has no fallback to
    # print -- but it does have a stub, which has no dtype at all.
    print(f"[precision] forward pass in "
          f"{getattr(predictor.model, 'autocast', 'fp32 (stub)')}", flush=True)
    if warmup_enabled:
        warm_up(predictor)
    else:
        print("[warmup] disabled (WARMUP=0)", flush=True)
    # **Resolved, not requested**. The init line above echoes
    # the mode we asked for; this one says what survived the warmup, which is the
    # only place the difference is visible — the preliminary phase returns these
    # logs, the final phase returns none, and both run the same algorithm.
    # **AND IT MUST COUNT AOTI, WHICH IT DID NOT.** A loaded package sets
    # `compile_mode = None` (it replaces compilation rather than joining it), so
    # this line read "eager" through the whole 2026-08-26 run while a compiled
    # `.so` was serving every forward. That is the same class of error as the
    # three fits measured eager: the one line written to say what is running,
    # saying the wrong thing, in the only phase that returns logs at all.
    _model = getattr(predictor, "model", None)
    print(f"[compile] resolved: "
          f"{'aoti' if getattr(_model, 'aoti', None) is not None else getattr(_model, 'compile_mode', None) or 'eager'}",
          flush=True)
    print(f"[ready] {time.perf_counter() - _IMPORTED_AT:.1f} s from import to "
          f"/health-capable", flush=True)
    return predictor


def load_json_file(location):
    with open(location) as handle:
        return json.load(handle)


def load_sitk_image(location: Path):
    matches = glob.glob(str(location / "*.mha"))
    if not matches:
        raise FileNotFoundError(f"No .mha file found in {location}")
    return sitk.ReadImage(matches[0])


def collect_requests(metadata) -> dict:
    """slot -> {idx_in_output: (image_file_idx, BeamletRequest)}."""
    per_slot: dict[int, dict[int, tuple[int, BeamletRequest]]] = defaultdict(dict)
    for image_data in metadata:
        image_idx = image_data["image_file_idx"]
        for beam in image_data.get("beams", []):
            for ray in beam.get("rays", []):
                for beamlet in ray.get("beamlets", []):
                    request = BeamletRequest.from_metadata(ray, beamlet)
                    per_slot[request.output_file_idx][request.idx_in_output] = (
                        image_idx,
                        request,
                    )
    return per_slot


class StreamingStackWriter:
    """Write a compressed 4-D MetaImage one frame at a time.

    `sitk.JoinSeries` is the documented way to build the stack, and it is what
    kills the container at the scored layout: it needs every frame of the slot
    resident, then allocates a second copy of all of them, and then the writer
    compresses from that. At 50 frames of 85 MB that is ~13 GB of the box's 16
    GiB before anything else -- measured, as an OOM kill at 14.97 GiB anon RSS.

    A MetaImage with ``CompressedData = True`` is a text header followed by a
    single zlib stream over the raw voxel buffer, and the buffer's order --
    x fastest, then y, z, and finally frame -- is exactly the order the frames
    are produced in. So the stack can be compressed incrementally and never
    exist in memory at all. Peak becomes one frame.

    ``CompressedDataSize`` is only known at the end, so it is written as a
    fixed-width zero-padded placeholder and patched in place on close; the field
    is parsed as an integer, so the padding is harmless.
    """

    def __init__(self, path: Path, reference: sitk.Image, n_frames: int,
                 level: int | None = None) -> None:
        size = reference.GetSize()          # (x, y, z)
        spacing = reference.GetSpacing()
        origin = reference.GetOrigin()
        d = reference.GetDirection()        # 3x3, row-major
        # MetaIO stores TransformMatrix column-major, so ITK's row-major
        # direction goes in transposed -- as SimpleITK's own writer does.
        matrix = (d[0], d[3], d[6], 0, d[1], d[4], d[7], 0, d[2], d[5], d[8], 0, 0, 0, 0, 1)

        header = (
            "ObjectType = Image\n"
            "NDims = 4\n"
            "BinaryData = True\n"
            "BinaryDataByteOrderMSB = False\n"
            "CompressedData = True\n"
            f"CompressedDataSize = {0:020d}\n"
            f"TransformMatrix = {' '.join(f'{v:.17g}' for v in matrix)}\n"
            f"Offset = {origin[0]:.17g} {origin[1]:.17g} {origin[2]:.17g} 0\n"
            "CenterOfRotation = 0 0 0 0\n"
            f"ElementSpacing = {spacing[0]:.17g} {spacing[1]:.17g} {spacing[2]:.17g} 1\n"
            f"DimSize = {size[0]} {size[1]} {size[2]} {n_frames}\n"
            "ElementType = MET_FLOAT\n"
            "ElementDataFile = LOCAL\n"
        ).encode("ascii")

        self.path = path
        self.expected_frames = n_frames
        self.frames_written = 0
        self._shape = (size[2], size[1], size[0])   # numpy order
        self._size_offset = header.index(b"CompressedDataSize = ") + len("CompressedDataSize = ")
        self._handle = open(path, "wb")
        self._handle.write(header)
        self._level = ZLIB_LEVEL if level is None else level
        # Level 1, not zlib's default 6. The platform only requires that the
        # output *be* compressed -- it never scores the file's size -- while
        # runtime carries double weight against a hard 500 s cap. On a real
        # 85 MB *predicted* frame (1.6% non-zero; `render_to_volume` zero-fills
        # and only touches the beamlet box) level 1 takes 0.09 s for 1.7 MB
        # against level 6's 0.31 s for 1.4 MB. Across 500 frames that trade is
        # minutes. The earlier 0.17/0.50 s figures were measured on
        # ground-truth frames, which are ~4.6x denser than what we write.
        # One zlib stream, built from independently-deflated blocks -- the pigz
        # construction. The write is 65% of invoke and ran on one core of eight
        #, so this is the largest runtime lever we have: measured 6.05x at
        # 32 blocks / 8 workers on a real predicted frame.
        #
        # It works because `Z_SYNC_FLUSH` ends a block on a byte boundary, so raw
        # deflate blocks concatenate into one valid stream. Three details are
        # load-bearing and each was verified rather than assumed:
        #   * `-15` window == raw deflate, no per-block header or checksum;
        #   * the Adler-32 covers the *whole* uncompressed stack, so it is
        #     combined from per-block sums (`_adler_combine`) rather than
        #     recomputed -- a sequential pass over 85 MB costs 31.7 ms against
        #     14 ms for the parallel compress, i.e. it would have become the
        #     bottleneck and capped the win at ~1.8x;
        #   * `zlib` releases the GIL, so threads give real parallelism; a pool
        #     of processes would have to copy 85 MB per frame.
        #
        # **Compression ratio is free here only because the data is ~98%
        # zeros** (`render_to_volume` touches the beamlet box and nothing else):
        # measured -0.1% at 32 blocks. On dense data independent dictionaries
        # cost real size, so do not carry this number over to anything else.
        # Blocking is also ~1.47x faster than one stream at a *single* worker, so
        # this never loses if the platform gives us fewer cores than we expect.
        self._handle.write(_zlib_header(self._level))
        self._compressed_bytes = 2
        self._blocks_per_frame = max(1, ZLIB_BLOCKS)
        self._adler = 1
        # Compressed form of an all-zero block, by length -- see `_deflate_cached`.
        self._zero_blocks: dict[int, tuple] = {}
        self._pool = (ThreadPoolExecutor(max_workers=ZLIB_WORKERS)
                      if ZLIB_WORKERS > 1 else None)
        # Set by whoever finishes this writer -- the consumer thread, once it
        # reaches this slot's close item. The writer owns it because the thing
        # being waited for is *this file being finished*, and a caller that
        # tracked it separately would be tracking a second copy of that fact.
        self.done = threading.Event()

    def _deflate(self, chunk: memoryview):
        compressor = zlib.compressobj(self._level, zlib.DEFLATED, -15)
        return (compressor.compress(chunk) + compressor.flush(zlib.Z_SYNC_FLUSH),
                zlib.adler32(chunk), len(chunk))

    def _deflate_cached(self, chunk: memoryview):
        """:meth:`_deflate`, but an all-zero block is compressed once per run.

        **A predicted frame is ~98% zeros** and the box that holds the rest is
        contiguous in z, so most of the 32 blocks contain no dose at all and
        deflate them to the same handful of bytes every time -- 500 frames a job,
        the same answer each time. Deflate is deterministic for a given level and
        input, so caching that answer by block length is exact, not approximate:
        the file this writes is **byte-identical** to the one the plain path
        writes, which is asserted rather than reasoned about
        (`tests/test_streaming_stack_writer.py`).

        The scan that decides is much cheaper than the compression it skips --
        one linear pass over the block against deflate's match-finding over the
        same bytes -- and it is only wasted on the minority of blocks that do
        carry dose.

        Keyed on **length**, so it can only ever be consulted for a block that
        was just proven all-zero; two lengths occur per frame, the last block
        being short. Races between the pool's threads are benign: two threads
        may compress the same zero block once each and store identical bytes.
        """
        if np.any(np.frombuffer(chunk, dtype=np.uint8)):
            return self._deflate(chunk)
        cached = self._zero_blocks.get(len(chunk))
        if cached is None:
            cached = self._deflate(chunk)
            self._zero_blocks[len(chunk)] = cached
        return cached

    def add(self, volume: np.ndarray) -> None:
        if volume.shape != self._shape:
            raise ValueError(
                f"frame shape {volume.shape} does not match the input image grid "
                f"{self._shape}; outputs must sit on exactly the input's grid"
            )
        if self.frames_written >= self.expected_frames:
            raise ValueError(f"more than the declared {self.expected_frames} frame(s)")
        started = time.perf_counter()
        raw = memoryview(
            np.ascontiguousarray(volume, dtype="<f4").data
        ).cast("B")
        step = (len(raw) + self._blocks_per_frame - 1) // self._blocks_per_frame
        chunks = [raw[i:i + step] for i in range(0, len(raw), step)]
        mapper = self._pool.map if self._pool is not None else map
        for block, adler, length in mapper(self._deflate_cached, chunks):
            self._handle.write(block)
            self._compressed_bytes += len(block)
            self._adler = _adler_combine(self._adler, adler, length)
        TIMING["write_s"] += time.perf_counter() - started
        self.frames_written += 1

    def close(self) -> None:
        if self.frames_written != self.expected_frames:
            raise ValueError(
                f"{self.path}: wrote {self.frames_written} of "
                f"{self.expected_frames} declared frame(s)"
            )
        # A final, empty, fixed-Huffman block sets BFINAL and ends the deflate
        # data; then the Adler-32 of the whole stack, big-endian, which is what
        # makes the concatenation a *zlib* stream rather than raw deflate.
        tail = b"\x03\x00" + self._adler.to_bytes(4, "big")
        self._handle.write(tail)
        self._compressed_bytes += len(tail)
        self._handle.seek(self._size_offset)
        self._handle.write(f"{self._compressed_bytes:020d}".encode("ascii"))
        self._handle.close()
        if self._pool is not None:
            self._pool.shutdown(wait=True)


def write_placeholder(out_path: Path) -> None:
    """Unused slots still need a file; a single trivial frame satisfies it."""
    frame = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.float32))
    sitk.WriteImage(sitk.JoinSeries([frame]), str(out_path), useCompression=True)


def run(predictor):
    metadata_path = INPUT_PATH / INPUT_JSON_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found")

    for key in TIMING:
        TIMING[key] = 0.0
    invoke_started = time.perf_counter()

    metadata = load_json_file(metadata_path)
    if isinstance(metadata, dict):
        metadata = [metadata]

    report_run_shape(metadata)

    # Resolved once, and tolerant of a predictor that has no pool: `run` is
    # driven by test doubles and by anything satisfying the predict contract,
    # and a frame that is never handed back simply costs the allocation it would
    # have saved (`models/predictor.py`, `FramePool`).
    release = getattr(predictor, "release_frame", None) or (lambda frame: False)

    per_slot = collect_requests(metadata)
    print(f"Parsed {sum(len(v) for v in per_slot.values())} beamlets across "
          f"{len(per_slot)} output slot(s); zlib level {ZLIB_LEVEL}", flush=True)

    # **At most one CT, evicted when a DIFFERENT one is asked for** — not
    # after every slot. Clearing per slot was a memory decision and it bounded
    # memory correctly, but the platform's jobs are typically **one image across
    # many slots** (the evaluator's own invariant is that a stacked output file
    # belongs to exactly one image), so it re-decompressed the same 85 MB volume
    # once per slot: measured **2.1 s of a 19.4 s job** at 1 image / 500 beams,
    # against ~0.2 s for the one read that was needed. Lazy eviction holds the
    # same single volume and reads each image once.
    image_cache: dict[int, tuple[sitk.Image, np.ndarray, VolumeGeometry]] = {}

    def get_image(image_idx: int):
        if image_idx not in image_cache:
            # Before the load, so the peak is still one volume and never two.
            image_cache.clear()
            started = time.perf_counter()
            image = load_sitk_image(INPUT_PATH / f"images/{INPUT_DIR_BASE}-{image_idx + 1}")
            # `copy=False`: `GetArrayFromImage` has already produced a fresh
            # array and nothing else holds it, so when the CT is stored float32
            # this is an 85 MB copy for nothing. It still converts when it must.
            array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
            image_cache[image_idx] = (image, array, VolumeGeometry.from_sitk(image))
            TIMING["image_load_s"] += time.perf_counter() - started
        return image_cache[image_idx]

    # **ONE CONSUMER FOR THE WHOLE INVOKE, NOT ONE PER SLOT.** A queue item is
    # `(writer, frame)`, and `(writer, None)` means "you have all of that slot's
    # frames now, close it". So a slot ENDS BY QUEUEING ITS CLOSE and the
    # producer starts the next one immediately, where it used to stand still
    # until the writer had caught up (`SLOT_OVERLAP`). Ordering is unchanged and
    # unchanged for the same reason as before: one consumer, one queue, frames
    # reaching the stream in the order they were produced -- which is the order
    # the 4-D buffer requires. Two writers can now be alive at once, on
    # different files, and only one is ever being written to.
    overlapped = WRITE_AHEAD > 0
    frames: "queue.Queue" = queue.Queue(maxsize=max(WRITE_AHEAD, 1))
    failure: list[BaseException] = []

    def drain() -> None:
        # **The sentinel is the ONLY way out of this loop**, and the try
        # covers the whole body for that reason -- including the `get`. A
        # consumer that dies leaves the producer blocked on a full queue with
        # nobody reading it, and a container that hangs does not fail: it runs to
        # the platform's 500 s cap and is removed from the ranking, which costs
        # a submission slot and produces no result to show for
        # it. So a failure here is *recorded* and draining continues; the
        # producer raises it on its own thread.
        while True:
            try:
                item = frames.get()
                if item is None:
                    return
                writer, frame = item
                if failure:
                    continue
                if frame is None:
                    # The same guard the per-slot `finally` carried: a slot
                    # that fell short of its declared frames is left UNCLOSED
                    # rather than closed with a header that lies about its
                    # length. And `done` is set whether or not it closed --
                    # it says "this consumer is finished with that slot", which
                    # is what a waiter needs; a `done` that only fires on the
                    # happy path is a hang wearing a flag.
                    try:
                        if writer.frames_written == writer.expected_frames:
                            writer.close()
                        print(f"Wrote {writer.frames_written} frame(s) to "
                              f"{writer.path}", flush=True)
                    finally:
                        writer.done.set()
                    continue
                writer.add(frame)
                # HERE AND NOWHERE EARLIER. `add` returns with the frame
                # compressed and written, and this thread drops its reference
                # next -- so this is the first instant nobody is reading it. One
                # line earlier (before the add, or on the producer side at
                # handover) hands the next beamlet a frame this one is still
                # deflating: right shape, plausible dose, wrong beamlet, nothing
                # raised (`models/predictor.py`, `FramePool`).
                release(frame)
            except BaseException as exc:            # noqa: BLE001
                failure.append(exc)

    # At 0 the thread is never created, rather than created with a queue of
    # one. An escape hatch that still runs the machinery it is meant to switch
    # off is not one -- and this is the switch a box with fewer cores than the
    # writer wants would be turned off with.
    consumer = None
    if overlapped:
        consumer = threading.Thread(target=drain, name="stack-writer", daemon=True)
        consumer.start()

    try:
        for slot_idx in range(NUM_OUTPUT_FILES):
            # `images/`, matching the input side above and both of the org's own
            # artifacts: their example algorithm writes
            # /output/images/stacked-radiation-dose-map-N/, and their evaluator's
            # resolve_output_dir reads .../output/images/<slug>-N. The submission
            # instructions draw the tree without this level, which is where the
            # earlier path came from -- but the docs are a simplification and the
            # code is what scores you. Writing one level up produces no readable
            # output for any beam, with nothing to notice until the score is zero.
            # Pinned by tests/test_submission_layout.py.
            output_dir = OUTPUT_PATH / "images" / f"stacked-radiation-dose-map-{slot_idx + 1}"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / "output.mha"

            slot = per_slot.get(slot_idx)
            if not slot:
                started = time.perf_counter()
                write_placeholder(out_path)
                TIMING["placeholder_s"] += time.perf_counter() - started
                continue

            # idx_in_output is contiguous and 0-based per the contract; fail loudly
            # rather than silently shifting every later beam in the stack.
            expected = list(range(max(slot) + 1))
            missing = [i for i in expected if i not in slot]
            if missing:
                raise ValueError(
                    f"slot {slot_idx} missing idx_in_output {missing[:5]} "
                    f"(have {len(slot)}, need {len(expected)})"
                )

            # Predict grouped by image so each CT is used once, then place results
            # back at their stack position.
            by_image: dict[int, list[int]] = defaultdict(list)
            for position in expected:
                by_image[slot[position][0]].append(position)

            window_size = max(predictor.batch_size, 1)

            if len(by_image) == 1:
                # The evaluator's own invariant: a stacked output file belongs to
                # exactly one image within a run (evaluate.py:validate_run_beams).
                # So the slot's beams can be produced in idx_in_output order and
                # compressed straight to disk, and no frame outlives its window.
                image, array, geom = get_image(next(iter(by_image)))
                writer = StreamingStackWriter(out_path, image, len(expected))
                for start in range(0, len(expected), window_size):
                    window = expected[start : start + window_size]
                    started = time.perf_counter()
                    volumes = predictor.predict(array, geom, [slot[p][1] for p in window])
                    TIMING["predict_s"] += time.perf_counter() - started
                    for offset in range(len(window)):
                        volume = volumes[offset]
                        volumes[offset] = None      # drop as we go, not at the end
                        if consumer is None:
                            writer.add(volume)
                            release(volume)
                        else:
                            waited = time.perf_counter()
                            frames.put((writer, volume))
                            TIMING["queue_wait_s"] += time.perf_counter() - waited
                        del volume
                        if failure:
                            raise failure[0]
                    del volumes
                if consumer is None:
                    if writer.frames_written == writer.expected_frames:
                        writer.close()
                    print(f"Wrote {writer.frames_written} frame(s) to {out_path}",
                          flush=True)
                else:
                    # **QUEUED, NOT AWAITED — this line is the 10%.** The next
                    # slot's first batch depends on nothing this writer still
                    # owes: a different file, a different stack, and the frames
                    # already handed over. Waiting here (which is what a
                    # per-slot `join` did) stops the GPU for ~140 ms, ten times
                    # a job. It is also what makes the close ORDERED rather
                    # than raced -- it travels the same queue as the frames, so
                    # it cannot overtake one.
                    frames.put((writer, None))
                    if not SLOT_OVERLAP:
                        # The old schedule, kept only as the A/B arm that prices
                        # this: stand still until the slot is bytes on disk.
                        # **Bounded by the consumer being alive**, never by
                        # the event alone -- a consumer that died holding the
                        # last close would otherwise park the producer here
                        # until the platform's 500 s cap, which scores as a
                        # removed submission rather than a slow one.
                        started = time.perf_counter()
                        while not writer.done.wait(0.5):
                            if failure or not consumer.is_alive():
                                break
                        TIMING["drain_s"] += time.perf_counter() - started
            else:
                # A slot spanning several images should not happen, but predicting
                # out of stack order cannot be streamed, so fall back rather than
                # write frames to the wrong positions. Memory here is the old
                # profile and only safe because this case carries few frames.
                # NOT `frames`: that name is the invoke-long queue now, and
                # rebinding it here would hand every later slot a list where the
                # producer expects a Queue -- after this branch had already run
                # once, on a job nobody re-tests because it "should not happen".
                stack: list = [None] * len(expected)
                for image_idx, positions in by_image.items():
                    image, array, geom = get_image(image_idx)
                    for start in range(0, len(positions), window_size):
                        window = positions[start : start + window_size]
                        volumes = predictor.predict(array, geom, [slot[p][1] for p in window])
                        for offset, position in enumerate(window):
                            volume = volumes[offset]
                            volumes[offset] = None
                            frame = sitk.GetImageFromArray(volume)
                            # JoinSeries requires identical origin/spacing/direction,
                            # and the contract requires each dose map to sit on its
                            # input's grid.
                            frame.CopyInformation(image)
                            stack[position] = frame
                            del volume, frame
                        del volumes
                # Never GetImageFromArray on a stacked 4D array: SimpleITK reads the
                # extra axis as vector components and silently misplaces the data.
                # Compression level is set explicitly so this path is on the same
                # arm as the streaming one when the A/B varies ZLIB_LEVEL.
                writer = sitk.ImageFileWriter()
                writer.SetFileName(str(out_path))
                writer.UseCompressionOn()
                writer.SetCompressionLevel(ZLIB_LEVEL)
                writer.Execute(sitk.JoinSeries(stack))
                print(f"Wrote {len(stack)} frame(s) to {out_path} (multi-image slot)", flush=True)
                del stack
    finally:
        # **The one place the invoke waits for the writer, and it is after the
        # LAST slot** -- which is the whole change: the other nine drains now
        # happen while the GPU is busy. Reached on the error path too, so no
        # thread outlives the invoke and no output file is left half-written
        # with the container reporting success.
        if consumer is not None:
            started = time.perf_counter()
            frames.put(None)
            consumer.join()
            TIMING["drain_s"] += time.perf_counter() - started
        if failure:
            raise failure[0]

    total = time.perf_counter() - invoke_started
    n = sum(len(v) for v in per_slot.values()) or 1
    # `other` is the MAIN THREAD's remainder, not total - predict - write.
    # With the writer on its own thread those two overlap and can sum past the
    # total, which as a subtraction would print a negative -- and a log
    # parser reading this line with `[\d.]+` would not fail on a minus sign, it
    # would simply stop matching and silently return nothing.
    other = max(total - TIMING["predict_s"] - TIMING["queue_wait_s"], 0.0)
    overlap = "on" if WRITE_AHEAD > 0 else "off"
    print(
        f"invoke: total {total:.1f} s for {n} beamlet(s) ({total / n:.3f} s/beamlet)"
        f" | predict {TIMING['predict_s']:.1f} s ({TIMING['predict_s'] / n:.3f})"
        f" | write {TIMING['write_s']:.1f} s ({TIMING['write_s'] / n:.3f})"
        f" | other {other:.1f} s | zlib level {ZLIB_LEVEL}"
        # Appended, never inserted: log parsers match everything above.
        f" | write-overlap {overlap} (ahead {WRITE_AHEAD}, "
        f"queue-wait {TIMING['queue_wait_s']:.1f} s)"
        # What `other` is made of, so the remainder stops being a mystery.
        f" | images {TIMING['image_load_s']:.1f} s"
        f" | placeholders {TIMING['placeholder_s']:.1f} s"
        f" | drain {TIMING['drain_s']:.1f} s"
        # **The verdict, not colour.** A pool whose frames are never handed
        # back degrades silently to the old allocate-per-beamlet path and reads
        # exactly like a win that did not arrive; "0% recycled" in the log is the
        # only place that is visible, and the preliminary phase is the only phase
        # that returns logs at all (`models/predictor.py`, `FramePool`).
        f" | frames {predictor.frames.summary() if getattr(predictor, 'frames', None) else 'pool off'}",
        flush=True,
    )
