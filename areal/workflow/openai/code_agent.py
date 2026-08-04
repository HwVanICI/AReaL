# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import re
import resource
import subprocess
import sys
import tempfile
from functools import lru_cache
from typing import Literal

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper
from areal.utils import logging

logger = logging.getLogger("CodeAgent")


SandboxBackend = Literal["auto", "daytona", "subprocess"]


def _daytona_available() -> bool:
    return importlib.util.find_spec("daytona") is not None and bool(
        os.getenv("DAYTONA_API_KEY")
    )


@lru_cache(maxsize=1)
def _warn_unsafe_subprocess() -> None:
    logger.warning(
        "Using subprocess execution for model-generated code because Daytona is "
        "unavailable or was not selected. Code will run with access to the host "
        "filesystem, network, and environment."
    )


def _resolve_sandbox_backend(
    backend: SandboxBackend,
) -> Literal["daytona", "subprocess"]:
    if backend == "auto":
        backend = "daytona" if _daytona_available() else "subprocess"
    elif backend not in ("daytona", "subprocess"):
        raise ValueError(
            "sandbox_backend must be one of: 'auto', 'daytona', 'subprocess'"
        )

    if backend == "daytona" and not _daytona_available():
        raise RuntimeError(
            "Daytona execution requires the optional 'daytona' dependency and "
            "DAYTONA_API_KEY. Install it with `uv sync --extra sandbox`."
        )
    if backend == "subprocess":
        _warn_unsafe_subprocess()
    return backend


def remove_thinking(text: str):
    # Remove XML-style thinking blocks: <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove "Thought:" or "Thinking:" style prefixes
    text = re.sub(
        r"^\s*(Thought|Thinking|Reasoning):.*?$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Remove OpenAI-style hidden reasoning markers if present
    text = re.sub(r"<<.*?>>", "", text, flags=re.DOTALL)

    return text.strip()


def _extract_code_block(text: str):
    text = remove_thinking(text)
    pattern = r"```(?:[pP]ython)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        text = max(matches, key=len).strip()

    return text.strip()


def _limit_resources():
    # CPU time limit (seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    # Memory limit (~512MB)
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))


def _build_test_script(code: str, tests: str | list[str], imports: str = "") -> str:
    test_code = "\n\n".join(tests) if isinstance(tests, list) else tests
    return imports + "\n\n" + code + "\n\n" + test_code + "\n\n" + "test_check()"


def _run_subprocess_test(
    code: str,
    tests: str | list[str],
    imports: str = "",
    timeout: int = 10,
) -> tuple[bool, str]:
    """
    Run full test (with test_check) in isolation.
    """
    script = _build_test_script(code, tests, imports)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit_resources if os.name == "posix" else None,
        )

        success = result.returncode == 0
        error = result.stderr.strip()

        return success, error

    except subprocess.TimeoutExpired:
        return False, "Timeout"

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def _run_daytona_test(
    code: str,
    tests: str | list[str],
    imports: str = "",
    timeout: int = 10,
) -> tuple[bool, str]:
    from areal.infra.sandbox import DaytonaRunner

    script = _build_test_script(code, tests, imports)
    with DaytonaRunner(network_block_all=True) as runner:
        result = runner.run(script, timeout=timeout)
    return result.exit_code == 0, result.stderr


def code_reward_fn(
    completions: str,
    tests: str | list[str],
    imports: str = "",
    sandbox_backend: Literal["daytona", "subprocess"] = "subprocess",
) -> float:
    """
    Binary reward: 1 if passes all tests, else 0
    """

    code = _extract_code_block(completions)
    if sandbox_backend == "daytona":
        ok, _ = _run_daytona_test(code, tests, imports)
    else:
        ok, _ = _run_subprocess_test(code, tests, imports)

    return 1.0 if ok else 0.0


class CodeAgent:
    def __init__(self, sandbox_backend: SandboxBackend = "auto", **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self.distill = bool(self.kwargs.pop("distill", False))
        self.prompt_logprobs_topk = self.kwargs.pop("prompt_logprobs_topk", 0)
        self.sandbox_backend = _resolve_sandbox_backend(sandbox_backend)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None) or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key", None) or os.getenv("OPENAI_API_KEY")
        client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, http_client=http_client, max_retries=0
        )
        comp: ChatCompletion = await client.chat.completions.create(
            messages=data["messages"],
            model="default",
            metadata={
                "distill": "true" if self.distill else "false",
                "prompt_logprobs_topk": str(self.prompt_logprobs_topk),
            },
            **self.kwargs,
        )

        reward_fn = AsyncRewardWrapper(code_reward_fn)
        reward = await reward_fn(
            completions=comp.choices[0].message.content,
            tests=data["tests"],
            imports=data.get("imports", ""),
            sandbox_backend=self.sandbox_backend,
        )
        return float(reward)
