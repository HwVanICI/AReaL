# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from areal.api.engine_api import InferenceEngine
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.dynamic_import import import_from_string


def _tag_result_domain(result: Any, domain_key: str, domain: str) -> Any:
    if isinstance(result, dict):
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        if result and all(
            isinstance(value, InteractionWithTokenLogpReward)
            for value in result.values()
        ):
            for interaction in result.values():
                interaction.trajectory_metadata.setdefault(domain_key, domain)
        else:
            result.setdefault(domain_key, domain)
    return result


@dataclass
class DomainWorkflowSpec:
    """Configuration for one domain-routed agent workflow."""

    workflow: Any
    kwargs: dict[str, Any] = field(default_factory=dict)


class DomainRouterWorkflow:
    """Route agent-style workflow calls by a domain key in each sample.

    This class intentionally does not inherit from RolloutWorkflow. AReaL treats
    classes with an async run() method as agent workflows and wraps them in the
    OpenAI proxy workflow.
    """

    def __init__(
        self,
        workflows: Mapping[str, Any],
        domain_key: str = "domain",
    ) -> None:
        self.domain_key = domain_key
        self.workflows = {
            domain: self._build_workflow(domain, spec)
            for domain, spec in workflows.items()
        }
        if len(self.workflows) == 0:
            raise ValueError("DomainRouterWorkflow requires at least one workflow.")

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> Any:
        domain = data.get(self.domain_key)
        if domain is None:
            raise ValueError(f"Missing domain key {self.domain_key!r} in sample")
        if domain not in self.workflows:
            raise ValueError(
                f"No workflow configured for domain {domain!r}. "
                f"Available domains: {sorted(self.workflows)}"
            )

        return await self.workflows[domain].run(data, **extra_kwargs)

    @staticmethod
    def _build_workflow(domain: str, spec: Any) -> Any:
        if isinstance(spec, Mapping):
            workflow = spec.get("workflow")
            if workflow is None:
                raise ValueError(f"Workflow spec for domain {domain!r} misses workflow")
            kwargs = spec.get("kwargs", spec.get("workflow_kwargs", {}))
        elif isinstance(spec, DomainWorkflowSpec):
            workflow = spec.workflow
            kwargs = spec.kwargs
        else:
            workflow = spec
            kwargs = {}

        workflow_obj = DomainRouterWorkflow._instantiate_workflow(workflow, kwargs)
        run = getattr(workflow_obj, "run", None)
        if not callable(run):
            raise TypeError(
                f"Workflow for domain {domain!r} must define callable async run()."
            )
        if not asyncio.iscoroutinefunction(run):
            raise TypeError(f"Workflow for domain {domain!r} run() must be async.")
        return workflow_obj

    @staticmethod
    def _instantiate_workflow(workflow: Any, kwargs: Mapping[str, Any]) -> Any:
        if isinstance(workflow, str):
            workflow = import_from_string(workflow)

        if isinstance(workflow, type):
            return workflow(**dict(kwargs))

        if kwargs:
            raise ValueError(
                "Workflow kwargs are only supported when workflow is a class or "
                f"import path, got {type(workflow).__name__}."
            )
        return workflow


class DomainRouterRolloutWorkflow(RolloutWorkflow):
    """Route rollout or agent workflow episodes by a sample domain.

    Agent-style children are resolved by the inference engine and wrapped in an
    OpenAI proxy workflow before this router is constructed.
    """

    def __init__(
        self,
        workflows: Mapping[str, Any],
        domain_key: str = "domain",
        _workflow_resolver: Callable[[Any, dict[str, Any]], RolloutWorkflow]
        | None = None,
    ) -> None:
        self.domain_key = domain_key
        self._workflow_resolver = _workflow_resolver
        self.workflows = {
            domain: self._build_workflow(domain, spec)
            for domain, spec in workflows.items()
        }
        if len(self.workflows) == 0:
            raise ValueError(
                "DomainRouterRolloutWorkflow requires at least one workflow."
            )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        domain = data.get(self.domain_key)
        if domain is None:
            raise ValueError(f"Missing domain key {self.domain_key!r} in sample")
        if domain not in self.workflows:
            raise ValueError(
                f"No workflow configured for domain {domain!r}. "
                f"Available domains: {sorted(self.workflows)}"
            )
        result = await self.workflows[domain].arun_episode(engine, data)
        return _tag_result_domain(result, self.domain_key, domain)

    @staticmethod
    def _parse_workflow_spec(domain: str, spec: Any) -> tuple[Any, dict[str, Any]]:
        if isinstance(spec, Mapping):
            workflow = spec.get("workflow")
            if workflow is None:
                raise ValueError(f"Workflow spec for domain {domain!r} misses workflow")
            kwargs = spec.get("kwargs", spec.get("workflow_kwargs", {}))
        elif isinstance(spec, DomainWorkflowSpec):
            workflow = spec.workflow
            kwargs = spec.kwargs
        else:
            workflow = spec
            kwargs = {}
        return workflow, dict(kwargs)

    def _build_workflow(self, domain: str, spec: Any) -> RolloutWorkflow:
        workflow, kwargs = self._parse_workflow_spec(domain, spec)
        if self._workflow_resolver is not None:
            workflow_obj = self._workflow_resolver(workflow, kwargs)
        else:
            workflow_obj = DomainRouterWorkflow._instantiate_workflow(workflow, kwargs)
        if not isinstance(workflow_obj, RolloutWorkflow):
            raise TypeError(
                f"Workflow for domain {domain!r} must be a RolloutWorkflow when "
                "DomainRouterRolloutWorkflow is constructed directly. Agent-style "
                "workflows are supported when the router is passed to an inference "
                f"engine, got {type(workflow_obj).__name__}."
            )
        return workflow_obj

    @classmethod
    def _from_workflow_resolver(
        cls,
        workflow_resolver: Callable[[Any, dict[str, Any]], RolloutWorkflow],
        **kwargs: Any,
    ) -> DomainRouterRolloutWorkflow:
        return cls(_workflow_resolver=workflow_resolver, **kwargs)

    @classmethod
    def _iter_workflow_specs(
        cls, workflow_kwargs: Mapping[str, Any]
    ) -> Iterable[tuple[Any, dict[str, Any]]]:
        workflows = workflow_kwargs.get("workflows", {})
        for domain, spec in workflows.items():
            yield cls._parse_workflow_spec(domain, spec)
