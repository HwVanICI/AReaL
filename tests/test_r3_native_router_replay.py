# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function

from areal.engine.r3.discovery import NativeRouterReplayRef
from areal.engine.r3.orchestration import (
    clear_router_replay_action,
    clear_router_replay_state,
    enqueue_recorded_indices,
    set_router_replay_action,
)

try:
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
except ImportError:  # pragma: no cover - compatibility with older Megatron layouts
    from megatron.core.transformer.moe.router import RouterReplay, RouterReplayAction


@pytest.fixture(autouse=True)
def clear_native_router_replay_registry():
    if hasattr(RouterReplay, "clear_global_router_replay_instances"):
        RouterReplay.clear_global_router_replay_instances()
    yield
    if hasattr(RouterReplay, "clear_global_router_replay_instances"):
        RouterReplay.clear_global_router_replay_instances()


def _default_compute_topk(
    scores: torch.Tensor,
    topk: int,
    num_groups: int | None = None,
    group_topk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert num_groups is None
    assert group_topk is None
    return torch.topk(scores, k=topk, dim=1)


def test_native_router_replay_record_stores_live_topk_indices():
    replay = RouterReplay()
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.7, 0.4, 0.8, 0.2],
            [0.5, 0.6, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )

    replay.set_router_replay_action(RouterReplayAction.RECORD)
    probs, top_indices = replay.get_replay_topk(
        scores,
        topk=2,
        default_compute_topk=_default_compute_topk,
    )

    expected_probs, expected_indices = torch.topk(scores, k=2, dim=1)
    torch.testing.assert_close(top_indices, expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(probs, expected_probs, rtol=0, atol=0)
    torch.testing.assert_close(
        replay.get_recorded_indices(),
        expected_indices,
        rtol=0,
        atol=0,
    )


def test_native_router_replay_forward_and_backward_use_target_fifo():
    replay = RouterReplay()
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.7, 0.4, 0.8, 0.2],
            [0.5, 0.6, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    first_target = torch.tensor([[3, 1], [0, 2], [1, 0]], dtype=torch.long)
    second_target = torch.tensor([[2, 0], [3, 1], [0, 2]], dtype=torch.long)

    replay.set_target_indices(first_target)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    forward_probs, forward_indices = replay.get_replay_topk(
        scores,
        topk=2,
        default_compute_topk=_default_compute_topk,
    )

    torch.testing.assert_close(forward_indices, first_target, rtol=0, atol=0)
    torch.testing.assert_close(
        forward_probs,
        scores.gather(1, first_target),
        rtol=0,
        atol=0,
    )

    replay.set_target_indices(second_target)
    replay.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    backward_probs_0, backward_indices_0 = replay.get_replay_topk(
        scores,
        topk=2,
        default_compute_topk=_default_compute_topk,
    )
    backward_probs_1, backward_indices_1 = replay.get_replay_topk(
        scores,
        topk=2,
        default_compute_topk=_default_compute_topk,
    )

    torch.testing.assert_close(backward_indices_0, first_target, rtol=0, atol=0)
    torch.testing.assert_close(
        backward_probs_0,
        scores.gather(1, first_target),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(backward_indices_1, second_target, rtol=0, atol=0)
    torch.testing.assert_close(
        backward_probs_1,
        scores.gather(1, second_target),
        rtol=0,
        atol=0,
    )


def test_finish_r3_forward_only_releases_native_backward_queue():
    """Logprob microbatches must not retain targets for an unused backward."""
    finish = _load_function(
        "areal/engine/megatron_engine.py",
        "_finish_r3_microbatch_replay",
        {
            "clear_router_replay_state": clear_router_replay_state,
            "clear_router_replay_action": clear_router_replay_action,
            "enqueue_recorded_indices": enqueue_recorded_indices,
            "set_router_replay_action": set_router_replay_action,
        },
    )
    replay = RouterReplay()
    ref = NativeRouterReplayRef(0, "router", torch.nn.Identity(), replay)
    replay.set_target_indices(torch.ones(4, 2, dtype=torch.long))
    set_router_replay_action([ref], "REPLAY_FORWARD")

    finish(None, [ref], "replay", forward_only=True)

    assert replay.replay_backward_list == []
    assert replay.target_topk_idx is None
    assert replay.router_replay_action is None
