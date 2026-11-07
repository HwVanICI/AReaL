# SPDX-License-Identifier: Apache-2.0

"""Fork-teardown continuation orchestration, isolated from the generic workflows.

The step-0 actor-GPU fork feature's continuation logic -- route a re-submitted
partial to the resume path, package an evicted partial as a descriptor,
distribute per-sample continuations across a group, surface a group's
descriptors plus its already-complete rollouts -- was inlined in the generic
``RLVRWorkflow`` / ``GroupedRolloutWorkflow``. This module isolates that
orchestration so the generic workflows keep only thin delegate calls.

Lives in :mod:`areal.utils` (a leaf layer imported by both ``areal.workflow``
and ``areal.infra``) to avoid any ``areal.infra`` <-> ``areal.workflow`` import
edge. The continuation GENERATION (``RLVRWorkflow._arun_episode_continue``)
stays on ``RLVRWorkflow`` because it needs ``areal.workflow.reward_metrics``
(which imports ``areal.infra``); this module orchestrates and calls back into
``base._arun_episode_continue``.

All methods are static: they take the base workflow / logger as parameters
rather than holding state, so there is no instantiation or injection -- the
generic workflows call them directly (data-driven, inert for non-fork tasks).
"""

from __future__ import annotations

from typing import Any

from areal.utils import logging
from areal.utils.data import concat_padded_tensors
from areal.utils.fork_constants import (
    FORK_COMPLETE_KEY,
    FORK_CONTINUATION_KEY,
    FORK_CONTINUE_FIELD,
)

logger = logging.getLogger("ForkContinuationWorkflow")


class ForkContinuationWorkflow:
    """Stateless helpers for fork-teardown continuation orchestration.

    Used by ``RLVRWorkflow`` (per-sample) and ``GroupedRolloutWorkflow`` (group)
    via thin delegates; the bulk fork logic lives here.
    """

    @staticmethod
    async def route_continuation(
        base: Any, engine: Any, data: dict[str, Any]
    ) -> tuple[bool, Any]:
        """If ``data`` carries ``FORK_CONTINUE_FIELD`` (the controller re-submitting
        a partial to a surviving engine), route to the base workflow's
        continuation generation.

        Handles the ``group_size == 1`` case where the controller's per-sample
        list arrives unwrapped at the base workflow (``GroupedRolloutWorkflow``
        only wraps for ``group_size > 1``): take the sole element so the
        continuation generation receives a dict, not a list.

        Returns ``(handled, result)``. When not handled (no continuation field),
        the caller proceeds with normal generation.
        """
        cont = data.get(FORK_CONTINUE_FIELD)
        if isinstance(cont, list):
            cont = cont[0] if cont else None
        if cont is not None:
            return True, await base._arun_episode_continue(engine, data, cont)
        return False, None

    @staticmethod
    def build_continuation_descriptor(req: Any, resp: Any) -> dict[str, Any]:
        """Wrap a fork-evicted partial as a continuation descriptor (not a
        training trajectory). ``req`` is a ``ModelRequest``, ``resp`` a
        ``ModelResponse`` with ``stop_reason="abort"`` + partial tokens.
        """
        return {
            FORK_CONTINUATION_KEY: {
                "prompt_ids": list(resp.input_tokens),
                "partial_ids": list(resp.output_tokens),
                "partial_logprobs": list(resp.output_logprobs),
                "partial_versions": list(resp.output_versions),
                "rid": req.rid,
            }
        }

    @staticmethod
    async def arun_one_with_continue(
        base: Any, engine: Any, data: dict[str, Any], cont: dict[str, Any]
    ) -> Any:
        """Run one sample of a group with its per-sample continuation partial
        injected, delegating generation to ``base.arun_episode``.
        """
        d = dict(data)
        d[FORK_CONTINUE_FIELD] = cont
        return await base.arun_episode(engine, d)

    @staticmethod
    def surface_group_descriptor(
        results: list[Any], logger: Any
    ) -> dict[str, Any] | None:
        """Surface a partially-evicted group's in-flight descriptors AND its
        already-complete rollouts (merged into one group) so the controller can
        re-merge them with the continuations into one full n_samples group.

        Returns the ``{FORK_CONTINUATION_KEY, FORK_COMPLETE_KEY}`` descriptor if
        any result is a continuation descriptor, else ``None`` (caller proceeds
        with the normal merge). Fork co-occurs only with tensor-dict workflows
        (rlvr), not online agent mode (InteractionWithTokenLogpReward), so
        ``concat_padded_tensors`` on the complete half is safe.
        """
        descriptor_results = [
            r
            for r in results
            if r is not None and isinstance(r, dict) and FORK_CONTINUATION_KEY in r
        ]
        if not descriptor_results:
            return None
        complete_results = [
            r
            for r in results
            if r is not None and isinstance(r, dict) and FORK_CONTINUATION_KEY not in r
        ]
        complete_group = (
            concat_padded_tensors(complete_results) if complete_results else None
        )
        logger.info(
            f"[step0-fork] grouped workflow: surfacing "
            f"{len(descriptor_results)}/{len(results)} evicted sample(s) "
            "as a group continuation descriptor; "
            f"preserving {len(complete_results)} completed for re-merge"
        )
        return {
            FORK_CONTINUATION_KEY: [
                r[FORK_CONTINUATION_KEY] for r in descriptor_results
            ],
            FORK_COMPLETE_KEY: complete_group,
        }
