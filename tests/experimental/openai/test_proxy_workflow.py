import pytest

from areal.experimental.openai.proxy import workflow as proxy_workflow_module
from areal.infra.workflow_context import WorkflowContext, set as set_workflow_context


class _DummyAgent:
    async def run(self, data, **extra_kwargs):
        del data, extra_kwargs
        return 0.0


class _DummyProxyClient:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.session_id = None
        self._session_api_key = "dummy-session-key"

    @property
    def session_api_key(self):
        return self._session_api_key

    async def __aenter__(self):
        self.session_id = "dummy-session"
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb

    async def set_last_reward(self, reward):
        del reward

    async def export_interactions(self, discount=1.0, style="individual"):
        del discount, style
        return {}


@pytest.mark.asyncio
async def test_proxy_workflow_returns_none_when_export_is_empty(monkeypatch):
    async def _fake_get_aiohttp_session():
        return object()

    async def _fake_grant_capacity(self, session):
        del self, session

    monkeypatch.setattr(
        proxy_workflow_module.workflow_context,
        "get_aiohttp_session",
        _fake_get_aiohttp_session,
    )
    monkeypatch.setattr(
        proxy_workflow_module.OpenAIProxyWorkflow,
        "_grant_capacity",
        _fake_grant_capacity,
    )
    monkeypatch.setattr(
        proxy_workflow_module,
        "OpenAIProxyClient",
        _DummyProxyClient,
    )

    workflow = proxy_workflow_module.OpenAIProxyWorkflow(
        mode="inline",
        agent=_DummyAgent(),
        proxy_addr="http://proxy",
    )
    set_workflow_context(WorkflowContext(task_id=9))

    result = await workflow.arun_episode(engine=None, data={})
    assert result is None
