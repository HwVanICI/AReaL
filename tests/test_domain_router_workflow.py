from __future__ import annotations

import asyncio
import types

import pytest
import torch

from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    interactions_to_trajectory,
)
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.workflow.domain_router import (
    DomainRouterRolloutWorkflow,
    DomainRouterWorkflow,
)


class _EchoWorkflow:
    def __init__(self, prefix: str):
        self.prefix = prefix

    async def run(self, data, **extra_kwargs):
        return {
            "value": f"{self.prefix}:{data['value']}",
            "base_url": extra_kwargs.get("base_url"),
        }


class _SyncWorkflow:
    def run(self, data, **extra_kwargs):  # noqa: ARG002
        return data


class _EchoRolloutWorkflow(RolloutWorkflow):
    def __init__(self, prefix: str):
        self.prefix = prefix

    async def arun_episode(self, engine, data):  # noqa: ARG002
        return {"value": f"{self.prefix}:{data['value']}"}


class _AgentRolloutAdapter(RolloutWorkflow):
    def __init__(self, agent):
        self.agent = agent

    async def arun_episode(self, engine, data):  # noqa: ARG002
        return await self.agent.run(data)


class _InteractionRolloutWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):  # noqa: ARG002
        interaction = InteractionWithTokenLogpReward(
            _cache={
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            }
        )
        return {"response-id": interaction}


def test_domain_router_routes_by_domain():
    """Router dispatches each sample to its configured domain workflow."""

    router = DomainRouterWorkflow(
        workflows={
            "math": {"workflow": _EchoWorkflow, "kwargs": {"prefix": "m"}},
            "tau2": {"workflow": _EchoWorkflow, "kwargs": {"prefix": "t"}},
        }
    )

    math_result = asyncio.run(
        router.run({"domain": "math", "value": "gsm8k"}, base_url="proxy")
    )
    tau2_result = asyncio.run(router.run({"domain": "tau2", "value": "task"}))

    assert math_result == {"value": "m:gsm8k", "base_url": "proxy"}
    assert tau2_result["value"] == "t:task"


def test_domain_router_rejects_unknown_domain():
    """Unknown domain values fail with a clear configuration error."""

    router = DomainRouterWorkflow(
        workflows={"math": {"workflow": _EchoWorkflow, "kwargs": {"prefix": "m"}}}
    )

    with pytest.raises(ValueError, match="No workflow configured"):
        asyncio.run(router.run({"domain": "tau2", "value": "task"}))


def test_domain_router_requires_async_run():
    """Agent workflows must expose async run()."""

    with pytest.raises(TypeError, match="run\\(\\) must be async"):
        DomainRouterWorkflow(workflows={"sync": _SyncWorkflow()})


def test_domain_router_requires_domain_key():
    """Samples must be explicitly tagged with the routing domain."""

    router = DomainRouterWorkflow(
        workflows={"math": {"workflow": _EchoWorkflow, "kwargs": {"prefix": "m"}}}
    )

    with pytest.raises(ValueError, match="Missing domain key"):
        asyncio.run(router.run({"value": "gsm8k"}))


def test_domain_router_rollout_routes_by_domain():
    """Rollout router dispatches arun_episode to the domain workflow."""

    router = DomainRouterRolloutWorkflow(
        workflows={
            "math": {"workflow": _EchoRolloutWorkflow, "kwargs": {"prefix": "m"}},
            "geometry3k": {
                "workflow": _EchoRolloutWorkflow,
                "kwargs": {"prefix": "g"},
            },
        }
    )

    result = asyncio.run(
        router.arun_episode(None, {"domain": "geometry3k", "value": "proof"})
    )

    assert result == {"value": "g:proof", "domain": "geometry3k"}


def test_domain_router_rollout_resolves_agent_children():
    """Inference engines can adapt agent children before router construction."""

    def resolver(workflow, kwargs):
        if workflow is _EchoWorkflow:
            return _EchoRolloutWorkflow(**kwargs)
        return workflow(**kwargs)

    router = DomainRouterRolloutWorkflow._from_workflow_resolver(
        workflow_resolver=resolver,
        workflows={
            "agent": {
                "workflow": _EchoWorkflow,
                "kwargs": {"prefix": "a"},
            },
            "rollout": {
                "workflow": _EchoRolloutWorkflow,
                "kwargs": {"prefix": "r"},
            },
        },
    )

    result = asyncio.run(
        router.arun_episode(None, {"domain": "agent", "value": "task"})
    )

    assert result == {"value": "a:task", "domain": "agent"}


def test_remote_engine_resolves_mixed_domain_router_children():
    """Remote workflow resolution wraps only the agent-style child."""

    engine = object.__new__(RemoteInfEngine)
    engine.logger = None

    def wrap_agent(self, agent, proxy_addr):
        assert proxy_addr == "http://proxy"
        return _AgentRolloutAdapter(agent)

    engine._wrap_openai_agent = types.MethodType(wrap_agent, engine)

    router = engine._resolve_workflow(
        "areal.workflow.DomainRouterRolloutWorkflow",
        {
            "workflows": {
                "agent": {
                    "workflow": _EchoWorkflow,
                    "kwargs": {"prefix": "a"},
                },
                "rollout": {
                    "workflow": _EchoRolloutWorkflow,
                    "kwargs": {"prefix": "r"},
                },
            }
        },
        proxy_addr="http://proxy",
    )

    agent_result = asyncio.run(
        router.arun_episode(None, {"domain": "agent", "value": "task"})
    )
    rollout_result = asyncio.run(
        router.arun_episode(None, {"domain": "rollout", "value": "task"})
    )

    assert agent_result == {
        "value": "a:task",
        "base_url": None,
        "domain": "agent",
    }
    assert rollout_result == {"value": "r:task", "domain": "rollout"}


def test_domain_router_preserves_domain_on_proxy_interactions():
    """Interaction trajectories retain domain metadata after conversion."""

    router = DomainRouterRolloutWorkflow(
        workflows={"math": _InteractionRolloutWorkflow()}
    )

    interactions = asyncio.run(
        router.arun_episode(None, {"domain": "math", "value": "task"})
    )
    trajectory = interactions_to_trajectory(interactions)

    assert trajectory["domain"] == "math"
