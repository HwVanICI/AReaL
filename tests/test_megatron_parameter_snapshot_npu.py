# SPDX-License-Identifier: Apache-2.0

"""CPU backup memory and four-NPU regressions for the target NPU stack.

Run in the target NPU image with AREAL_TEST_QWEN3_PATH pointing to a local
Qwen3-0.6B checkpoint. The runner requires Megatron-Core 0.16.1 and validates
snapshot switching and subsequent updates with optimizer-step gather overlap
disabled for compatibility with the target MindSpeed stack.
"""

import os
import subprocess
import sys
import weakref
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
def test_qwen3_parameter_snapshot_restores_policy_and_optimizer_updates_npu(tmp_path):
    """Check old-policy logp, restored parameters, and subsequent updates."""
    pytest.importorskip("torch_npu", reason="requires the target Ascend NPU stack")
    if not torch.npu.is_available() or torch.npu.device_count() < 4:
        pytest.skip("distributed parameter snapshot regression requires 4 NPUs")

    checkpoint = os.environ.get("AREAL_TEST_QWEN3_PATH")
    if not checkpoint:
        pytest.fail("set AREAL_TEST_QWEN3_PATH to a local Qwen3-0.6B checkpoint")
    model_path = Path(checkpoint).expanduser().resolve()
    if not model_path.is_dir():
        pytest.fail(f"Qwen3-0.6B checkpoint directory does not exist: {model_path}")

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "tests/torchrun/run_megatron_parameter_snapshot_npu.py"
    output = tmp_path / "qwen3_parameter_snapshot_npu.out"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(repo_root), env.get("PYTHONPATH")) if path
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            "--max-restarts=0",
            str(runner),
            "--model-path",
            str(model_path),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        env=env,
        check=True,
        timeout=900,
    )
    assert output.read_text(encoding="utf-8").strip() == "Passed"


@pytest.mark.slow
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("forward_failure", [False, True])
def test_parameter_snapshot_cpu_backup_avoids_npu_parameter_copy(
    monkeypatch, dtype, forward_failure
):
    """Keep the current policy on pinned CPU during normal/failed forwards."""
    pytest.importorskip("torch_npu", reason="requires the target Ascend NPU stack")
    if not torch.npu.is_available():
        pytest.skip("CPU parameter backup memory check requires an NPU")

    import areal.utils.torch_npu_compat  # noqa: F401
    from areal.api.engine_api import _ParameterSnapshot
    from areal.infra.platforms import current_platform

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    weight = torch.nn.Parameter(torch.ones((4096, 4096), dtype=dtype, device="npu"))
    refs = [("weight", weight)]
    storage = weight.untyped_storage().data_ptr()
    snapshot = _ParameterSnapshot()
    snapshot.capture(iter(refs))
    with torch.no_grad():
        weight.fill_(2)
    # Warm up transfer buffers before measuring live allocator usage.
    with snapshot.activate(iter(refs)):
        pass
    current_platform.synchronize()

    backups = []
    empty_like = torch.empty_like

    def record_backup(*args, **kwargs):
        backup = empty_like(*args, **kwargs)
        backups.append(weakref.ref(backup))
        return backup

    monkeypatch.setattr(torch, "empty_like", record_backup)
    before = torch.npu.memory_allocated()
    parameter_bytes = weight.numel() * weight.element_size()
    context = (
        pytest.raises(RuntimeError, match="forward failed")
        if forward_failure
        else nullcontext()
    )
    with context:
        with snapshot.activate(iter(refs)):
            current_platform.synchronize()
            active = torch.npu.memory_allocated()
            assert active - before < parameter_bytes // 4
            assert len(backups) == 1
            assert backups[0]().device.type == "cpu"
            assert backups[0]().is_pinned()
            assert backups[0]().dtype == dtype
            assert backups[0]().shape == weight.shape
            torch.testing.assert_close(
                weight.detach().cpu(),
                snapshot._snapshot_tensors_cpu[0][1],
                rtol=0,
                atol=0,
            )
            if forward_failure:
                raise RuntimeError("forward failed")
    current_platform.synchronize()
    torch.testing.assert_close(
        weight.detach().cpu(),
        torch.full(weight.shape, 2.0, dtype=dtype, device="cpu"),
        rtol=0,
        atol=0,
    )
    assert weight.untyped_storage().data_ptr() == storage
    assert snapshot.state == "ready"
    if not forward_failure:
        # An exception traceback can retain the generator's local variables.
        assert all(backup() is None for backup in backups)
    snapshot.clear()
