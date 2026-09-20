"""Compile the forward pass at BUILD time into a file the container just loads.

    # on a GPU box, at build time
    python3 -c "from models.aoti import export_package; export_package(...)"

    # in the container, at run time
    net = load_package(path, identity)   # or None, and the caller stays eager

Why this exists
---------------
`torch.compile` is worth ~10% of the forward pass (0.855x) and costs three things that
have nothing to do with the speed:

* **compilation at every job start** — every job is a fresh container, so the
  cost is paid per job, before `/health`, against a timeout nobody has published;
* **a write-then-execute on the scratch mount** — Inductor writes a `.so` and
  `dlopen`s it, which a `noexec` scratch refuses *after* a healthy start;
* **a batch of exactly 1** — dynamic shapes cover every size except 0 and 1, so
  a singleton stack recompiles inside the timed window, and jobs too small to
  fill their slots cannot avoid that shape (`submission/inference.py`).

AOTInductor removes the first two at once by moving the compile to build time.
The artefact is a bare `.so` that lives in the image, is `dlopen`ed from a
read-only layer exactly as every `import torch` already is, and consults no
cache. **The third it does NOT remove**: `torch.export` specialises 0 and 1
exactly as `torch.compile` does, so the exported range starts at 2 whatever
`Dim(min=…)` is told, and batch 1 is served eagerly rather than compiled
(`MIN_BATCH`). What AOTI buys there is that a singleton costs eager instead of a
*recompile* inside the timed window — the failure mode, not the speed.

**A BARE `.so` MUST CARRY ITS OWN CUDA KERNELS, AND BY DEFAULT IT DOES NOT.**
This is what "loads and does not run" was (2026-08-26): with
`embed_kernel_binary` off, Inductor emits `loadKernel("<build-box path>.cubin",
…)` into the wrapper and hands the `.cubin` files to the *packager* — which is
how a `.pt2` gets them to the runtime, by unpacking them into a scratch
directory and passing it as `cubin_dir`. A bare object has no packager, so the
paths point at a machine that is not there. `dlopen` still succeeds, because a
kernel is loaded lazily on first use, and the first forward then dies inside
`cuModuleLoad` with an error the C++ shim prints to **stderr** and converts to a
bare `run_func_(...) API call failed`. ⇒ `export_package` sets
**`aot_inductor.embed_kernel_binary`**, which links each cubin into the object's
own `.rodata` and switches the call to the `cuModuleLoadData` overload. The
kernels then travel in the same file as the weights, and nothing is read from
any path at run time.

**A stale package computes the wrong dose without failing.** The `.so`
contains weights, so a package built for one checkpoint and served beside
another is the same class of error as a geometry mismatch: it loads, it runs, it
is wrong. Every package therefore carries the identity of the checkpoint that
produced it and `load_package` refuses on any mismatch.

**AND A PACKAGE IS NOT BELIEVED UNTIL IT HAS RUN.** Both ends smoke-test:
`export_package` runs the object it just wrote and compares it against eager on
the build box, and `load_package` runs it at the recorded shape *and* at batch 1
before returning it. Three fits were measured eager because nothing checked, and
"it loaded" was twice mistaken for "it works" — a load proves `dlopen`, which is
the half of this that was never the problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

# Batch 64 is well past anything the container runs (BATCH_SIZE is 8) and
# costs nothing to allow: the range is a guard, not a tuning knob.
MAX_BATCH = 64
# **AND THE FLOOR IS 2, WHICH IS NOT A CHOICE.** `torch.export` specialises
# sizes 0 and 1 unconditionally, so `Dim("batch", min=1)` is accepted and then
# compiled with a range of [2, 64] anyway -- measured 2026-08-26 on an A10G,
# where the batch-1 probe came back "dim value is too small at 0, expected it to
# be >= 2, but got: 1". ⇒ **a package does NOT cover batch 1**, and the standing
# earlier claim that it did was reading the argument rather than the artefact.
# `models/predictor.py` serves anything below this eagerly: padding 1 to 2 would
# cost ~2x a batch-1 forward to save the 0.855x AOTI is worth, which is slower
# on exactly the singleton jobs it would be for.
MIN_BATCH = 2


def package_identity(checkpoint_identity: dict, extra: Optional[dict] = None) -> dict:
    """What a package must agree with before it may be used."""
    import torch

    facts = {
        "checkpoint_step": str(checkpoint_identity.get("step", "")),
        "checkpoint_epoch": str(checkpoint_identity.get("epoch", "")),
        "geometry_sha": str(checkpoint_identity.get("geometry_sha", "")),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        # Kernels are per architecture. A package built on sm_86 is not valid
        # on anything else, and the failure would be a wrong answer or a crash
        # deep in a loader, neither of which names the cause.
        facts["gpu_arch"] = f"sm_{major}{minor}"
    facts.update(extra or {})
    return facts


def input_spec(example) -> dict:
    """Shape, **stride** and dtype of the tensor the package was built for.

    **Stride, not just shape.** AOTInductor bakes the input's strides into the
    graph and checks them when `AOTI_RUNTIME_CHECK_INPUTS=1`; without the env var
    it does not check and simply computes with the layout it assumed. So a
    package exported from a contiguous `randn` and served a channels-last tensor
    is the silent-wrong class again, and recording the stride is what lets both
    ends refuse instead.
    """
    return {
        "shape": list(example.shape),
        "stride": list(example.stride()),
        "dtype": str(example.dtype),
        "device": str(example.device),
        # What the artefact will actually accept, so the caller can route the
        # sizes it will not (`MIN_BATCH`).
        "min_batch": MIN_BATCH,
    }


def _probe(spec: dict, batch: int, device: str):
    """A zero tensor with the package's exact layout, at a chosen batch size.

    The stride of a contiguous (or channels-last) layout does not depend on the
    batch dimension, so the recorded strides serve every size in the exported
    range -- which is what makes the batch-1 half of the smoke test free.
    """
    import torch

    shape = (batch, *tuple(spec["shape"])[1:])
    dtype = getattr(torch, str(spec["dtype"]).split(".")[-1])
    return torch.empty_strided(shape, tuple(spec["stride"]),
                               dtype=dtype, device=device).zero_()


def export_package(net, example_batch, path: str | Path, identity: dict) -> Path:
    """Export ``net`` for any batch in [MIN_BATCH, MAX_BATCH] and write the package.

    Build time only: this needs the GPU it is compiling for, and takes minutes.
    """
    import torch
    import torch._inductor as inductor

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.eval()
    batch = torch.export.Dim("batch", min=MIN_BATCH, max=MAX_BATCH)
    exported = torch.export.export(net, (example_batch,),
                                   dynamic_shapes={"x": {0: batch}})
    # **A BARE `.so`, NOT A `.pt2` PACKAGE.** `aoti_load_package` unpacks the
    # archive into a temp directory and `dlopen`s from *there* -- so a `noexec`
    # scratch defeats it exactly as it defeats Triton, which is the whole thing
    # AOTI was supposed to escape (measured 2026-08-25: "Error in dlopen:
    # /tmp/…/aotinductor/model/…"). A bare object is `dlopen`ed in place, and
    # in place is the image layer: read-only, and executable like every other
    # `.so` the container already loads.
    # **AND THEREFORE `embed_kernel_binary`.** Without it the wrapper keeps
    # `loadKernel("<this box>/….cubin", …)` and the cubins ride in the archive
    # this deliberately does not build -- see the module docstring; that is the
    # whole of "loads in place and dies on the first forward".
    produced = inductor.aot_compile(
        exported.module(), (example_batch,),
        options={
            "aot_inductor.output_path": str(path),
            "aot_inductor.embed_kernel_binary": True,
        },
    )
    produced = Path(produced)
    Path(str(path) + ".json").write_text(json.dumps(
        dict(identity, input=input_spec(example_batch)), sort_keys=True, indent=1))

    # **RUN IT HERE, WHERE FAILING IS CHEAP.** The alternative costs a
    # container build, an upload and a timing run to discover that the object
    # does not execute -- which is exactly what 2026-08-26 spent. On the build
    # box the loader's own smoke test is one call away and the GPU is already hot.
    loaded = load_package(path, identity, device=str(example_batch.device))
    if loaded is None:
        raise RuntimeError(
            f"{path} was written but will not load and run on the box that "
            f"built it -- the reason is in the '[aoti]' line above, and the "
            f"C++ shim prints its own cause on stderr. Shipping it would put "
            f"that failure inside the scored window."
        )
    with torch.no_grad():
        reference = net(example_batch)
        diff = (loaded(example_batch).float() - reference.float()).abs().max().item()
    # Not a tolerance: this path has been bit-identical on CPU, and a nonzero
    # value on CUDA is a fact worth having in the build log rather than a gate.
    print(f"[aoti] max abs diff vs eager on this box: {diff:g}", flush=True)
    return produced


def load_package(path: str | Path, identity: dict,
                 device: str = "cuda") -> Optional[Callable]:
    """The compiled forward, or ``None`` — never an exception, never a mismatch.

    Returns None rather than raising for the same reason compilation falls
    back to eager (`submission/inference.py`): the fast path may not decide
    whether the container scores at all. A refusal here costs the speedup; a
    wrong package would cost the submission.
    """
    path = Path(path)
    try:
        if not path.is_file():
            return None
        recorded = json.loads(Path(str(path) + ".json").read_text())
    except Exception as exc:                                      # noqa: BLE001
        print(f"[aoti] unreadable package beside {path}: {exc}; staying eager",
              flush=True)
        return None

    mismatch = {k: (recorded.get(k), v) for k, v in identity.items()
                if recorded.get(k) != v}
    if mismatch:
        # Loud, because this is the case that would otherwise be silent and wrong.
        print(f"[aoti] REFUSING a package built for something else: {mismatch}; "
              f"staying eager", flush=True)
        return None
    spec = recorded.get("input")
    if not spec:
        # An object from before the input spec existed cannot be smoke-tested,
        # and an untested package is the thing that cost three measurements.
        print(f"[aoti] REFUSING {path.name}: it records no input shape, so it "
              f"cannot be checked before the timed window; staying eager",
              flush=True)
        return None
    try:
        import torch
        import torch._export as export_api

        loaded = export_api.aot_load(str(path), device)
    except Exception as exc:                                      # noqa: BLE001
        print(f"[aoti] load failed ({type(exc).__name__}: {exc}); staying eager",
              flush=True)
        return None

    # **A LOAD IS NOT A RUN.** `dlopen` succeeds while the kernels are still
    # unresolved -- they are fetched on first use -- so everything that can be
    # wrong about a package is wrong *after* this point unless something calls
    # it. Both ends of the exported range, under the context the real forward
    # uses (`models/predictor.py` runs inside `inference_mode`), and before
    # `/health`: a failure here is a slower container, the same failure inside
    # `/invoke` is a zero.
    floor = int(spec.get("min_batch", 1))
    batches = sorted({int(spec["shape"][0]), floor}, reverse=True)
    failed = None
    try:
        with torch.inference_mode():
            for size in batches:
                failed = size
                loaded(_probe(spec, size, device))
        failed = None
    except Exception as exc:                                      # noqa: BLE001
        # **Separate the one hypothesis that costs a round trip to test.**
        # `inference_mode` is the only thing about the call site that is not
        # obviously innocent, and a box that has already failed can answer it
        # for free -- otherwise the answer is another image, another bake and
        # another timing run.
        # **AT THE SIZE THAT FAILED.** Retrying at a different one answers a
        # different question and answers it wrongly: on 2026-08-26 this retried
        # at the batch that worked, reported "it does run under `no_grad`", and
        # pointed at `inference_mode` when the actual cause was the batch.
        try:
            with torch.no_grad():
                loaded(_probe(spec, failed, device))
        except Exception:                                         # noqa: BLE001
            note = ""
        else:
            note = (f" AT BATCH {failed} IT DOES RUN UNDER `no_grad`, so the "
                    f"difference is `inference_mode` and the fix belongs in "
                    f"`forward_device`, not here.")
        print(f"[aoti] loaded {path.name} and it will not RUN at batch "
              f"{failed} ({type(exc).__name__}: {exc}); staying eager. The "
              f"cause is on stderr, printed by the object itself as 'Error: …' "
              f"-- read the container's stderr, not only its stdout.{note}",
              flush=True)
        return None
    loaded.min_batch = floor
    print(f"[aoti] loaded {path.name} in place and ran it at batch "
          f"{'/'.join(str(b) for b in batches)} of {floor}-{MAX_BATCH} "
          f"(no extraction, no scratch); anything under {floor} stays eager",
          flush=True)
    return loaded


def export_from_predictor(predictor, path: str | Path, identity: dict):
    """Export the net a predictor actually runs, at the shape it actually sees.

    **The example shape is captured, never written down.** It follows from the
    box geometry, the channel list the checkpoint asks for and the batch size,
    and a hardcoded guess that drifts from any of them exports a package that
    loads and computes the wrong thing. So this hooks the module, runs one real
    forward through `warm_up`'s synthetic geometry, and exports the tensor that
    arrived.
    """
    import numpy as np
    import torch

    from models.geometry import VolumeGeometry
    from models.predictor import BeamletRequest

    model = predictor.model
    seen = {}

    def capture(_module, args):
        # **Shape and STRIDE only, never the tensor.** `forward_device` runs
        # under `torch.inference_mode()`, so what arrives here is an *inference
        # tensor*, and `torch.export` traces with autograd:
        # "Inference tensors cannot be saved for backward". Cloning does not
        # launder it -- the clone is an inference tensor too. Values are safe
        # to discard because this network has **no data-dependent control
        # flow**: it is convolutions and elementwise ops, so the exported graph
        # is a function of the shape, layout and dtype alone.
        # The layout is *not* implied by the shape: `channels_last` makes
        # `forward_device` hand over an NDHWC-strided view, and an example built
        # with `torch.randn(shape)` would be contiguous and describe a different
        # tensor (`input_spec`).
        x = args[0]
        seen.setdefault("spec", (tuple(x.shape), tuple(x.stride()), x.dtype, x.device))

    handle = model.net.register_forward_pre_hook(capture)
    try:
        # The same synthetic phantom `submission/inference.py` warms with: small
        # enough to build instantly, and the box intersects it.
        ct = np.full((24, 48, 48), -1000.0, dtype=np.float32)
        ct[8:16, 16:32, 16:32] = 40.0
        geom = VolumeGeometry(origin=np.array([-72.0, -72.0, -30.0]),
                              spacing=np.array([3.0, 3.0, 2.5]), shape=ct.shape)
        batch = max(getattr(predictor, "batch_size", 1), 1)
        predictor.predict(ct, geom, [
            BeamletRequest(ray_source=(600.0, 0.0, 0.0), ray_target=(0.0, 0.0, 0.0),
                           energy=100.0 + i, output_file_idx=0, idx_in_output=i,
                           minimum_cutoff=1e-6)
            for i in range(batch)
        ])
    finally:
        handle.remove()
        predictor.reset_entry_cache()

    if "spec" not in seen:
        raise RuntimeError(
            "the forward never ran, so no input shape was captured -- the "
            "synthetic geometry missed the box (`submission/inference.py` warns "
            "about exactly this) and an exported package would be a guess."
        )
    shape, stride, dtype, device = seen["spec"]
    # Built outside inference mode, which is the point.
    example = torch.randn(shape, dtype=dtype, device=device)
    if tuple(example.stride()) != stride:
        example = torch.empty_strided(shape, stride, dtype=dtype,
                                      device=device).copy_(example)
    print(f"[aoti] exporting at {shape} stride {stride} {dtype}", flush=True)
    return export_package(model.net, example, path, identity)
