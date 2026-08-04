from __future__ import annotations

import pytest

import areal.workflow.openai.code_agent as code_agent


@pytest.fixture(autouse=True)
def clear_subprocess_warning_cache():
    code_agent._warn_unsafe_subprocess.cache_clear()
    yield
    code_agent._warn_unsafe_subprocess.cache_clear()


def test_resolve_sandbox_backend_auto_with_daytona_selects_daytona(monkeypatch):
    """Auto mode selects Daytona when its SDK and credentials are available."""
    monkeypatch.setattr(code_agent, "_daytona_available", lambda: True)

    backend = code_agent._resolve_sandbox_backend("auto")

    assert backend == "daytona"


def test_resolve_sandbox_backend_auto_without_daytona_warns_once(monkeypatch):
    """Auto mode warns only once when it falls back to unsafe host execution."""
    warnings = []
    monkeypatch.setattr(code_agent, "_daytona_available", lambda: False)
    monkeypatch.setattr(code_agent.logger, "warning", warnings.append)

    first_backend = code_agent._resolve_sandbox_backend("auto")
    second_backend = code_agent._resolve_sandbox_backend("auto")

    assert first_backend == "subprocess"
    assert second_backend == "subprocess"
    assert len(warnings) == 1
    assert "model-generated code" in warnings[0].lower()


def test_resolve_sandbox_backend_explicit_unavailable_daytona_raises(monkeypatch):
    """Explicit Daytona mode fails instead of silently selecting subprocess."""
    monkeypatch.setattr(code_agent, "_daytona_available", lambda: False)

    with pytest.raises(RuntimeError, match="DAYTONA_API_KEY"):
        code_agent._resolve_sandbox_backend("daytona")


def test_build_test_script_combines_list_of_tests():
    """List-valued dataset tests become one executable Python script."""
    script = code_agent._build_test_script(
        "class Solution: pass",
        ["def helper(): pass", "def test_check(): helper()"],
        "from typing import List",
    )

    assert "def helper(): pass\n\ndef test_check(): helper()" in script
    assert script.endswith("test_check()")


def test_code_reward_fn_uses_selected_daytona_backend(monkeypatch):
    """Reward evaluation delegates to the selected Daytona runner."""
    calls = []

    def fake_run(code, tests, imports):
        calls.append((code, tests, imports))
        return True, ""

    monkeypatch.setattr(code_agent, "_run_daytona_test", fake_run)

    reward = code_agent.code_reward_fn(
        "```python\nclass Solution: pass\n```",
        "def test_check(): pass",
        sandbox_backend="daytona",
    )

    assert reward == 1.0
    assert calls == [
        ("class Solution: pass", "def test_check(): pass", ""),
    ]
