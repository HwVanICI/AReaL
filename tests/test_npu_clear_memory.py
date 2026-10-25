# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from areal.infra.platforms import npu


def test_clear_memory_releases_idle_host_cache(monkeypatch):
    """Offload boundaries clear idle host backups after device cleanup."""
    events = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            synchronize=lambda: events.append("synchronize"),
            empty_cache=lambda: events.append("device"),
            memory=SimpleNamespace(host_empty_cache=lambda: events.append("host")),
        ),
        raising=False,
    )
    monkeypatch.setattr(npu.gc, "collect", lambda: events.append("gc"))

    npu.NPUPlatform().clear_memory()

    assert events == ["synchronize", "gc", "device", "host"]
