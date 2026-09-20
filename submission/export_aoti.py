"""Compile the network ahead of time (AOTInductor) for the GPU it will run on.

The scored container carried this package: the forward runs at 0.855x its eager
time, and `predict` at 0.899x. Without it the container serves the same network eagerly -- same dose,
slower. Kernels are specific to the GPU architecture that compiles them, so run
this on the card you will infer on (the challenge scored on an NVIDIA A10G).

Runs INSIDE the built inference image, which already holds `models/` at
/opt/app. The runtime image has no CUDA toolkit, so the host's is mounted in.
See README.md, "Ahead-of-time kernels", for the two commands.
"""
import json
import os
import sys

sys.path.insert(0, "/opt/app")

import torch  # noqa: E402

from models.aoti import export_from_predictor, package_identity  # noqa: E402
from models.predictor import DosePredictor  # noqa: E402

ck = os.environ.get("CHECKPOINT_PATH", "/opt/ml/model/checkpoint.pt")
out_dir = os.environ.get("AOTI_OUT", "/out")

# The package is keyed on the checkpoint it was exported from, and the loader
# refuses a package whose identity does not match the weights it is handed.
raw = torch.load(ck, map_location="cpu", weights_only=True)
identity = package_identity({
    "step": raw.get("step", ""),
    "epoch": raw.get("epoch", ""),
    "geometry_sha": raw.get("geometry_sha256", "")
                    or (raw.get("config") or {}).get("geometry_sha256", ""),
})
del raw

predictor = DosePredictor.from_checkpoint(ck, device="cuda", batch_size=8)
out = export_from_predictor(predictor, os.path.join(out_dir, "net.so"), identity)
print("[aoti] wrote", out, os.path.getsize(out) // 1024, "KiB")
print("[aoti] identity", json.dumps(identity, sort_keys=True))
