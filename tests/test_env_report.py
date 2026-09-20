"""The environment line explains a runtime number after the fact.

It reports what this container can see about itself -- cores, GPU, torch, free
space under `/output` -- and nothing about the machine or the job around it.

**A diagnostic may never be the reason a submission errors**, so the failure
paths are pinned here rather than discovered on a platform.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from submission import inference  # noqa: E402


def emitted() -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inference.report_environment()
    line = next(l for l in buf.getvalue().splitlines() if l.startswith("[env] "))
    return json.loads(line[len("[env] "):])


def test_it_reports_the_cores_this_process_may_use():
    facts = emitted()
    assert facts["cpu_count"] == __import__("os").cpu_count()


def test_it_reports_the_torch_build():
    """The wheel decides which kernels exist, so a runtime number needs it."""
    facts = emitted()
    assert facts["torch"] == __import__("torch").__version__


def test_it_reports_nothing_about_the_host_or_the_job():
    """Self-description only: no mounts, no environment names, no directory listings."""
    facts = emitted()
    forbidden = [k for k in facts
                 if k.startswith("mount") or k in {"env_keys", "model_dir",
                                                   "cgroup_cpu_max", "rootfs_writable",
                                                   "scratch_exec"}]
    assert not forbidden, f"the env line carries {forbidden}"


def test_a_failing_probe_cannot_fail_the_run(monkeypatch):
    """`os.cpu_count` raising is not a reason for a submission to error."""
    monkeypatch.setattr(inference.os, "cpu_count",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inference.report_environment()                 # must not raise
    assert "diagnostics failed" in buf.getvalue()


def test_free_space_is_reported_when_output_exists(tmp_path, monkeypatch):
    """`/output` is a mount only on the platform, so the call is faked here."""
    import os
    real = os.statvfs
    monkeypatch.setattr(inference.os, "statvfs", lambda path: real(tmp_path))
    assert emitted()["output_free_mib"] >= 0
