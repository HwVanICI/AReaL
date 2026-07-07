from __future__ import annotations

import asyncio

import pytest

from areal.api.workflow_api import RolloutWorkflow
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

    assert result == {"value": "g:proof"}
