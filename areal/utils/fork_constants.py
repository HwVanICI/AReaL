# SPDX-License-Identifier: Apache-2.0

"""Shared constants for the step-0 actor-GPU fork feature.

These literals form the protocol between the controller
(:mod:`areal.infra.controller.rollout_controller`), the grouped workflow
(:class:`areal.infra.remote_inf_engine.GroupedRolloutWorkflow`), and the
rollout workflow (:class:`areal.workflow.rlvr.RLVRWorkflow`).

Centralized here -- in :mod:`areal.utils`, a leaf layer already imported by all
three -- rather than duplicated across modules, to avoid any
``areal.infra`` <-> ``areal.workflow`` import edge.
"""

# Reserved trajectory key marking a fork-teardown "continuation descriptor"
# (not a training trajectory). When an actor-GPU fork engine is evicted
# mid-generation, the workflow returns this descriptor carrying the partial
# tokens/logprobs/rid so the controller can resume generation on a surviving
# engine. Absent from normal trajectories -> the existing merge path is inert.
FORK_CONTINUATION_KEY = "__fork_continuation__"

# Per-sample continuation state injected into ``data`` by the controller when it
# re-submits a partial to a surviving engine. Value: dict with prompt_ids,
# partial_ids, partial_logprobs, partial_versions, rid.
FORK_CONTINUE_FIELD = "__continue__"

# Carries a partially-evicted group's already-complete rollouts (merged into one
# group) alongside FORK_CONTINUATION_KEY, so the controller can re-merge them
# with the continued rollouts into one full n_samples group. None when the
# whole group was in-flight at eviction.
FORK_COMPLETE_KEY = "__fork_complete__"

# Tags a continuation re-submit's ``data`` with its originating fork task_id, so
# the controller's accept branch can look up the parked already-complete half
# and re-merge.
FORK_MERGE_KEY = "__fork_merge__"
