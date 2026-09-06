"""Tests for E2B backend forwarding in the SWE rollout workflow."""

import sys
from types import ModuleType
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from examples.swe.agent import (
    SWEAgentWorkflow,
    _configured_agent_config,
    _configured_sandbox_backend,
)
from examples.swe.utils import SWEEnvConfig


def test_swe_env_config_supports_omegaconf_structured_config():
    config = OmegaConf.structured(SWEEnvConfig)

    assert config.sandbox_backend == "aenv"
    assert config.opencode_config == {}
    assert config.harness_timeout is None
    assert config.eval_timeout == 900.0
    assert config.session_idle_timeout == 1200.0
    assert config.session_poll_interval == 30.0
    assert config.terminal_exit_grace == 15.0
    config.session_idle_timeout = None
    config.terminal_exit_grace = None
    assert config.session_idle_timeout is None
    assert config.terminal_exit_grace is None
    config.sandbox_backend = "e2b"
    assert config.sandbox_backend == "e2b"


def test_configured_sandbox_backend_validates_value():
    assert _configured_sandbox_backend({}) == "aenv"
    assert _configured_sandbox_backend({"sandbox_backend": " E2B "}) == "e2b"
    with pytest.raises(ValueError, match="sandbox_backend"):
        _configured_sandbox_backend({"sandbox_backend": "local"})


def test_configured_agent_config_selects_codex_config():
    assert (
        _configured_agent_config(
            {"codex_agent_config": "train_codex_time3600"}, "codex"
        )
        == "train_codex_time3600"
    )


def test_configured_agent_config_selects_opencode_config():
    assert (
        _configured_agent_config(
            {"opencode_agent_config": "train_opencode_time3600"}, "opencode"
        )
        == "train_opencode_time3600"
    )


@pytest.mark.asyncio
async def test_run_episode_forwards_e2b_and_proxy_session(monkeypatch, tmp_path):
    run_agent = AsyncMock(return_value=(1.0, {"env": "success"}))
    lifecycle = ModuleType("aweagent.lifecycle")
    lifecycle.run_agent_with_reward = run_agent
    package = ModuleType("aweagent")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "aweagent", package)
    monkeypatch.setitem(sys.modules, "aweagent.lifecycle", lifecycle)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    workflow = object.__new__(SWEAgentWorkflow)

    reward = await workflow._run_episode(
        data={"instance_id": "task-1", "problem_statement": "fix", "image": "img"},
        agent_type="cc",
        config_name="train_cc_time3600",
        base_url="http://proxy:9000",
        api_key="trajectory-key",
        llm_model=None,
        opencode_provider=None,
        codex_provider=None,
        sandbox_backend="e2b",
        harness_timeout=222.0,
        eval_timeout=111.0,
        session_idle_timeout=321.0,
        session_poll_interval=7.0,
        terminal_exit_grace=11.0,
    )

    assert reward == 1.0
    kwargs = run_agent.await_args.kwargs
    assert kwargs["override_base_url"] == "http://proxy:9000"
    assert kwargs["override_api_key"] == "trajectory-key"
    assert kwargs["sandbox_backend"] == "e2b"
    assert kwargs["override_harness_timeout"] == 222.0
    assert kwargs["override_eval_timeout"] == 111.0
    assert kwargs["override_session_idle_timeout"] == 321.0
    assert kwargs["override_session_poll_interval"] == 7.0
    assert kwargs["override_terminal_exit_grace"] == 11.0


@pytest.mark.asyncio
async def test_run_episode_forwards_codex_model_and_provider(monkeypatch, tmp_path):
    run_agent = AsyncMock(return_value=(1.0, {"env": "success"}))
    lifecycle = ModuleType("aweagent.lifecycle")
    lifecycle.run_agent_with_reward = run_agent
    package = ModuleType("aweagent")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "aweagent", package)
    monkeypatch.setitem(sys.modules, "aweagent.lifecycle", lifecycle)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    workflow = object.__new__(SWEAgentWorkflow)

    reward = await workflow._run_episode(
        data={"instance_id": "task-2", "problem_statement": "fix", "image": "img"},
        agent_type="codex",
        config_name="train_codex_time3600",
        base_url="http://proxy:9000",
        api_key="trajectory-key",
        llm_model="qwen3-coder",
        opencode_provider=None,
        codex_provider="areal",
        sandbox_backend="e2b",
    )

    assert reward == 1.0
    kwargs = run_agent.await_args.kwargs
    assert kwargs["override_llm_model"] == "qwen3-coder"
    assert kwargs["override_codex_provider"] == "areal"
    assert kwargs["sandbox_backend"] == "e2b"


@pytest.mark.asyncio
async def test_run_episode_forwards_opencode_model_and_provider(monkeypatch, tmp_path):
    run_agent = AsyncMock(return_value=(1.0, {"env": "success"}))
    lifecycle = ModuleType("aweagent.lifecycle")
    lifecycle.run_agent_with_reward = run_agent
    package = ModuleType("aweagent")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "aweagent", package)
    monkeypatch.setitem(sys.modules, "aweagent.lifecycle", lifecycle)
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    workflow = object.__new__(SWEAgentWorkflow)

    reward = await workflow._run_episode(
        data={"instance_id": "task-3", "problem_statement": "fix", "image": "img"},
        agent_type="opencode",
        config_name="train_opencode_time3600",
        base_url="http://proxy:9000",
        api_key="trajectory-key",
        llm_model="qwen3-coder",
        opencode_provider="areal",
        codex_provider=None,
        sandbox_backend="e2b",
        opencode_config={"tools": {"task": False}},
        max_tokens=32768,
        max_completion_tokens=4096,
        harness_timeout=1800.0,
        eval_timeout=900.0,
        session_idle_timeout=1200.0,
        session_poll_interval=30.0,
        terminal_exit_grace=15.0,
    )

    assert reward == 1.0
    kwargs = run_agent.await_args.kwargs
    assert kwargs["override_llm_model"] == "qwen3-coder"
    assert kwargs["override_opencode_provider"] == "areal"
    assert kwargs["override_opencode_config"] == {"tools": {"task": False}}
    assert kwargs["override_max_tokens"] == 32768
    assert kwargs["override_max_completion_tokens"] == 4096
    assert kwargs["override_harness_timeout"] == 1800.0
    assert kwargs["override_eval_timeout"] == 900.0
    assert kwargs["override_session_idle_timeout"] == 1200.0
    assert kwargs["override_session_poll_interval"] == 30.0
    assert kwargs["override_terminal_exit_grace"] == 15.0
    assert kwargs["sandbox_backend"] == "e2b"


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", ["cc", "codex", "opencode"])
async def test_delegated_e2b_run_keeps_whole_episode_timeout(monkeypatch, agent_type):
    workflow = object.__new__(SWEAgentWorkflow)
    workflow.econfig = {
        "agent_type": agent_type,
        "sandbox_backend": "e2b",
        "harness_timeout": 1800.0,
        "eval_timeout": 900.0,
        "session_idle_timeout": None,
        "terminal_exit_grace": None,
    }
    workflow.gen_args = {}
    workflow.timeout = 1.0
    workflow._run_episode = AsyncMock(return_value=1.0)
    captured = {}

    async def wait_for(awaitable, timeout):
        captured["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr("examples.swe.agent.asyncio.wait_for", wait_for)

    reward = await workflow.run(
        {"instance_id": "task", "problem_statement": "fix", "image": "img"},
        base_url="http://proxy:9000",
        api_key="trajectory-key",
    )

    assert reward == 1.0
    assert captured["timeout"] == 1.0
    kwargs = workflow._run_episode.await_args.kwargs
    assert kwargs["session_idle_timeout"] is None
    assert kwargs["terminal_exit_grace"] is None
