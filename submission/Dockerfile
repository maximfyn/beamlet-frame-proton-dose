FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
# The platform provides a scratch directory and forbids writing anywhere else.
# Python's default is to drop __pycache__ next to every module it imports, which
# is a write into /opt/app -- harmless in itself, but it is the only thing
# standing between this container and "writes nothing outside /output", and it
# breaks outright under a read-only root filesystem. The harness runs with one.
ENV PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# **gcc, because Inductor builds kernels with a C compiler** -- for the AOTI
# export (README step 6, what the scored image ran) and for `torch.compile` --
# and the -runtime base ships none: without it Inductor fails at first use with
# `Failed to find C compiler`. AOTI measured on an A10G: forward 0.855x,
# `predict` 0.899x. Installed before the USER switch, since apt
# needs root.
USER root
RUN apt-get update -qq && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user:user submission/requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user submission/app.py /opt/app/
COPY --chown=user:user submission/inference.py /opt/app/
COPY --chown=user:user models /opt/app/models/

# Weights arrive on the platform's read-only model mount, NOT baked into the
# image. On Grand Challenge an algorithm is a container image *plus* a separately
# uploaded Model, extracted at /opt/ml/model -- their own do_save.sh builds both
# and ends by insisting the model be uploaded separately.
#
# Why this way round: the image is ~11.9 GB, almost all CUDA torch, and baking
# the checkpoint in means re-building and re-uploading every byte of it for each
# new checkpoint. Split, the image changes only when code or dependencies do and
# a checkpoint is ~50-200 MB on its own. Load time is unaffected either way --
# it happens before /health returns 200, which the platform does not measure
# (measured: 3 s -> 29 s of weight load moved invoke by 0.2 s).
#
# The cost is that "weights missing" becomes a runtime condition instead of a
# build failure. `init_model` therefore refuses to start unless
# DOSERAD_ALLOW_STUB=1, so an unmounted model cannot quietly score as noise.
#
# Side effect worth having: this image now builds from a clean checkout with no
# checkpoint present, which the old COPY made impossible.
ENV CHECKPOINT_PATH=/opt/ml/model/checkpoint.pt

# 8 costs the same as 32 at a quarter the peak VRAM: batch size moves `predict`
# by less than its own noise on the current path (see `init_model`).
# **A BUILD ARG, because the platform gives no way to set env at submission
# time.** A `--env` reaches a local `docker run` and nothing else, so any
# value the submitted container must have is baked here or it is not there.
ARG BATCH_SIZE=8
ENV BATCH_SIZE=${BATCH_SIZE}

# torch.compile mode, "" for eager. Off by default (`submission/inference.py`).
ARG COMPILE_MODE=""
ENV COMPILE_MODE=${COMPILE_MODE}

# Writer queue depth; empty means "one full batch", which is the measured
# optimum (`submission/inference.py`'s WRITE_AHEAD comment). 0 serialises it.
ARG WRITE_AHEAD=""
ENV WRITE_AHEAD=${WRITE_AHEAD}

# Output compression level. The platform only requires that output *be*
# compressed and never scores its size, while runtime carries double weight
# against a hard 500 s cap. Levels are lossless with respect to each other --
# 1 and 6 write bit-identical voxels -- so this is size traded for CPU and
# nothing else. Set here rather than only in code so `docker inspect` shows
# which arm an image is on, and so the A/B can override it per run.
ARG ZLIB_LEVEL=1
ENV ZLIB_LEVEL=${ZLIB_LEVEL}

# Blocks per frame -- the write's dominant lever (see `inference.py`). A build
# arg for the same reason as ZLIB_LEVEL: the image declares its own speed.
ARG ZLIB_BLOCKS=32
ENV ZLIB_BLOCKS=${ZLIB_BLOCKS}

# Body mask (a slice-wise filled body contour, `models/geometry.py`). A BUILD ARG, so one clean checkout produces
# either variant and `docker inspect` says which -- rather than a code edit per
# image, which would make the two indistinguishable after the fact.
# On by default, as submitted; `DosePredictor.body_mask` gives the reason.
ARG BODY_MASK=1
ENV BODY_MASK=${BODY_MASK}

# The cutoff clamp. Only `0` or `>= c` is legal below a beamlet's served cutoff.
# SNAP_ALPHA=0 zeroes the whole sub-cutoff band -- what the submitted model ran
# with, and the right choice because the challenge's labels are themselves
# thresholded at the cutoff. A positive alpha instead snaps the top of the band
# up to the threshold, which only pays against an UNthresholded truth (see
# `SNAP_ALPHA` in `models/predictor.py`). Default 0 so a plain build reproduces
# the submission.
ARG SNAP_ALPHA=0
ENV SNAP_ALPHA=${SNAP_ALPHA}

LABEL org.grand-challenge.api-method="invoke"

# Provenance. A checkpoint records the network it needs but not the *geometry*
# it assumes, and a checkpoint trained under one geometry loads fine under
# another while predicting dose in the wrong place. These make an image
# self-identifying, so
# `docker inspect` answers "what is this" without opening anything.
#
# No ARG defaults: a hand-run `docker build` leaves them empty, which reads as
# unknown provenance and is true. The build command in README.md supplies
# all four.
ARG GIT_SHA
ARG CHECKPOINT_STEP
ARG CHECKPOINT_EPOCH
ARG CHECKPOINT_GEOMETRY_SHA
LABEL org.doserad.commit="$GIT_SHA"
LABEL org.doserad.checkpoint-step="$CHECKPOINT_STEP"
LABEL org.doserad.checkpoint-epoch="$CHECKPOINT_EPOCH"
LABEL org.doserad.checkpoint-geometry-sha="$CHECKPOINT_GEOMETRY_SHA"
# The runtime knobs, so a deliberately SLOW image cannot be mistaken for the
# fast one by `docker inspect` -- the two differ in nothing else visible.
# THIS SET USED TO OMIT THE BIGGEST ONE. `zlib-blocks` is the write's REAL
# lever -- 32 -> 1 is **12x**, against 2x for the whole level range -- so two
# images could differ in the one knob no label recorded (2026-08-29). `snap-alpha`
# and `body-mask` change a SCORED METRIC and were likewise invisible.
# ⇒ every build-arg that changes runtime or a metric is a label. Add the label in
# the same commit as the arg.
LABEL org.doserad.batch-size="$BATCH_SIZE"
LABEL org.doserad.compile-mode="$COMPILE_MODE"
LABEL org.doserad.write-ahead="$WRITE_AHEAD"
LABEL org.doserad.zlib-blocks="$ZLIB_BLOCKS"
LABEL org.doserad.zlib-level="$ZLIB_LEVEL"
LABEL org.doserad.body-mask="$BODY_MASK"
LABEL org.doserad.snap-alpha="$SNAP_ALPHA"

ENTRYPOINT ["python", "app.py"]
