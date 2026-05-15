import sys

from areal.api.cli_args import SchedulingSpec
from areal.infra.rpc.ray_rpc_server import _build_python_module_cmd
from areal.infra.scheduler.ray import RayScheduler


def test_ray_scheduler_detects_guard_module_commands():
    assert RayScheduler._uses_http_guard(
        SchedulingSpec(cmd="python3 -m areal.infra.rpc.guard")
    )
    assert RayScheduler._uses_http_guard(
        SchedulingSpec(cmd="areal.experimental.agent_service.guard")
    )


def test_ray_scheduler_does_not_treat_rpc_server_as_guard():
    assert not RayScheduler._uses_http_guard(
        SchedulingSpec(cmd="python -m areal.infra.rpc.rpc_server")
    )


def test_ray_http_launcher_accepts_python_module_command():
    assert _build_python_module_cmd("python3 -m areal.infra.rpc.guard") == [
        sys.executable,
        "-m",
        "areal.infra.rpc.guard",
    ]


def test_ray_http_launcher_accepts_bare_module_command():
    assert _build_python_module_cmd("areal.infra.rpc.guard") == [
        sys.executable,
        "-m",
        "areal.infra.rpc.guard",
    ]
