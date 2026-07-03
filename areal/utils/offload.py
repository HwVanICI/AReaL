# SPDX-License-Identifier: Apache-2.0

"""Utilities for torch_memory_saver (TMS) configuration and setup.

This module handles the environment variable setup required for TMS to work
properly with LD_PRELOAD hooks.
"""

import os
from collections.abc import Mapping
from contextlib import nullcontext

_TMS_ENV_VARS = (
    "TMS_INIT_ENABLE",
    "TMS_INIT_ENABLE_CPU_BACKUP",
    "TMS_HOOK_MODE",
)


def is_tms_enabled() -> bool:
    return os.environ.get("TMS_INIT_ENABLE", "0") == "1"


if is_tms_enabled():
    from torch_memory_saver import torch_memory_saver
else:

    class MockTorchMemorySaver:
        def disable(self):
            return nullcontext()

        def pause(self):
            pass

        def resume(self):
            pass

    torch_memory_saver = MockTorchMemorySaver()


def get_tms_env_vars() -> dict[str, str]:
    """Get environment variables for torch_memory_saver (TMS)."""
    try:
        import torch_memory_saver as tms_pkg
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "torch_memory_saver is required when train-engine offload is enabled."
        ) from e

    # Locate the LD_PRELOAD shared library
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(tms_pkg.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )

    if not os.path.exists(dynlib_path):
        raise RuntimeError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    env_vars = {
        "LD_PRELOAD": dynlib_path,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    return env_vars


def sanitize_tms_env_vars(env: Mapping[str, str] | None) -> dict[str, str]:
    """Return an env copy with TMS preload hooks disabled.

    Non-TMS LD_PRELOAD entries are preserved so callers keep unrelated preload
    libraries such as stdbuf's line-buffering hook.
    """
    sanitized = dict(env or {})
    if ld_preload := sanitized.get("LD_PRELOAD"):
        ld_preload_entries = [
            entry
            for entry in ld_preload.split(os.pathsep)
            if entry and "torch_memory_saver" not in os.path.basename(entry)
        ]
        if ld_preload_entries:
            sanitized["LD_PRELOAD"] = os.pathsep.join(ld_preload_entries)
        else:
            sanitized.pop("LD_PRELOAD", None)
    else:
        sanitized.pop("LD_PRELOAD", None)

    for key in _TMS_ENV_VARS:
        sanitized.pop(key, None)

    # Keep explicit off flags for subprocesses that still inspect TMS env vars.
    sanitized.setdefault("LD_PRELOAD", "")
    sanitized["TMS_INIT_ENABLE"] = "0"
    sanitized["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
    return sanitized
