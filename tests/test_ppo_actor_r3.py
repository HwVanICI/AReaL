# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor
from areal.utils import stats_tracker
from areal.utils.data import TrajBatchMeta, split_training_batch_into_microbatches
from areal.utils.stats_tracker import DistributedStatsTracker


class _FakeR3Engine:
    def __init__(self, forward_result=None, *, r3_enabled=True, require_r3=True):
        self._r3_enabled = r3_enabled
        self._r3_pending_routed_experts = None
        self._r3_pending_valid = None
        self.forward_result = forward_result
        self.require_r3 = require_r3

    def eval(self):
        self.eval_called = True

    def forward(self, input_, aggregate_fn):
        assert "routed_experts" not in input_
        assert "r3_routing_valid" not in input_
        if self.require_r3:
            assert self._r3_pending_routed_experts is not None
            assert self._r3_pending_valid is not None
        if self.forward_result is not None:
            return self.forward_result
        return aggregate_fn([torch.ones(1), torch.ones(1)])


def _reset_stats_tracker():
    stats_tracker.export(reset=True)


@pytest.mark.parametrize("device", ["cpu", pytest.param("npu", marks=pytest.mark.npu)])
def test_ppo_update_keeps_routing_on_cpu_before_concat(monkeypatch, device):
    """Replacing RPC input dict values releases the original accelerator routes."""
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("Ascend NPU is not available")
    routes = torch.tensor([[[[0, 1]]]], dtype=torch.int32, device=device)
    data = [
        {
            "routed_experts": routes,
            "r3_routing_valid": torch.tensor([True], device=device),
            "input_ids": torch.tensor([[1]], device=device),
        }
    ]
    actor = PPOActor(PPOActorConfig(), _FakeR3Engine())
    seen = []

    def concat(actual):
        assert actual is data
        assert actual[0]["routed_experts"].device.type == "cpu"
        assert actual[0]["r3_routing_valid"].device.type == "cpu"
        assert actual[0]["input_ids"].device.type == device
        seen.append("concat")
        return {}, None

    monkeypatch.setattr(actor_module, "concat_batch", concat)
    monkeypatch.setattr(actor, "_ppo_update", lambda *args: seen.append("update"))
    actor.ppo_update(data)

    assert seen == ["concat", "update"]
    assert data[0]["routed_experts"].device.type == "cpu"
    torch.testing.assert_close(data[0]["routed_experts"], routes.cpu(), rtol=0, atol=0)


def test_compute_logp_uses_r3_side_channel_without_forwarding_keys():
    _reset_stats_tracker()
    engine = _FakeR3Engine()
    actor = PPOActor(PPOActorConfig(), engine)
    routed = torch.ones(2, 4, 3, 2, dtype=torch.int32)
    valid = torch.tensor([True, False])
    data = {
        "input_ids": torch.ones(2, 4, dtype=torch.int64),
        "attention_mask": torch.ones(2, 4, dtype=torch.bool),
        "routed_experts": routed,
        "r3_routing_valid": valid,
    }

    result = actor._compute_logp(data)

    torch.testing.assert_close(result, torch.ones(2), rtol=0, atol=0)
    assert engine._r3_pending_routed_experts is None
    assert engine._r3_pending_valid is None
    assert data["routed_experts"] is routed
    assert data["r3_routing_valid"] is valid


def test_compute_logp_logs_rollout_train_r3_metrics():
    _reset_stats_tracker()
    train_logp = torch.tensor([[-0.1, -1.0, 0.0, 0.0]], dtype=torch.float32)
    engine = _FakeR3Engine(forward_result=train_logp)
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 4, dtype=torch.int64),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.tensor([[9.0, -0.3, -0.6, 8.0]], dtype=torch.float32),
        "loss_mask": torch.tensor([[False, True, True, False]]),
        "routed_experts": torch.ones(1, 4, 3, 2, dtype=torch.int32),
        "r3_routing_valid": torch.tensor([True]),
    }

    result = actor._compute_logp(data)

    torch.testing.assert_close(result, train_logp, rtol=0, atol=0)
    exported = stats_tracker.export(reset=True)
    delta = torch.tensor([0.2, -0.4], dtype=torch.float32)
    assert exported["compute_logp/r3/enabled"] == 1.0
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_abs_diff/avg"]),
        delta.abs().mean(),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_sq_diff/avg"]),
        delta.square().mean(),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_k3_kl/avg"]),
        (torch.expm1(delta) - delta).mean(),
        rtol=0,
        atol=1e-6,
    )
    assert exported["compute_logp/r3/rollout_train_extreme_frac_tau2/avg"] == 0.0
    assert exported["compute_logp/r3/rollout_train_extreme_frac_tau5/avg"] == 0.0


def test_compute_logp_r3_metrics_do_not_wrap_loss_mask_last_token():
    _reset_stats_tracker()
    train_logp = torch.tensor([[-0.5, 10.0, 10.0, 10.0]], dtype=torch.float32)
    engine = _FakeR3Engine(forward_result=train_logp)
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 4, dtype=torch.int64),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "logprobs": torch.tensor([[99.0, -0.25, 77.0, 88.0]], dtype=torch.float32),
        "loss_mask": torch.tensor([[True, True, False, False]]),
        "routed_experts": torch.ones(1, 4, 3, 2, dtype=torch.int32),
        "r3_routing_valid": torch.tensor([True]),
    }

    actor._compute_logp(data)

    exported = stats_tracker.export(reset=True)
    torch.testing.assert_close(
        torch.tensor(exported["compute_logp/r3/rollout_train_logp_abs_diff/avg"]),
        torch.tensor(0.25),
        rtol=0,
        atol=1e-6,
    )


def test_compute_logp_logs_r3_disabled_for_rollout_train_metrics():
    _reset_stats_tracker()
    engine = _FakeR3Engine(
        forward_result=torch.zeros(1, 3),
        r3_enabled=False,
        require_r3=False,
    )
    actor = PPOActor(PPOActorConfig(), engine)
    data = {
        "input_ids": torch.ones(1, 3, dtype=torch.int64),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "logprobs": torch.zeros(1, 3),
        "loss_mask": torch.tensor([[False, True, False]]),
    }

    actor._compute_logp(data)

    exported = stats_tracker.export(reset=True)
    assert exported["compute_logp/r3/enabled"] == 0.0


def test_compute_logp_r3_metrics_retain_only_compact_summaries():
    _reset_stats_tracker()
    length = 20000
    engine = _FakeR3Engine(
        forward_result=torch.full((1, length), -1.0), require_r3=False
    )
    actor = PPOActor(PPOActorConfig(), engine)
    mask = torch.ones(1, length, dtype=torch.bool)
    mask[:, 0] = False

    actor._compute_logp(
        {
            "logprobs": torch.full((1, length), -2.0),
            "loss_mask": mask,
        }
    )

    for key, values in stats_tracker.get().stats.items():
        if key.startswith("compute_logp/r3/"):
            assert all(
                not torch.is_tensor(value) or value.numel() <= 4 for value in values
            )
    exported = stats_tracker.export(reset=True)
    assert exported["compute_logp/r3/n_valid_tokens"] == length - 1
    assert exported["compute_logp/r3/rollout_train_logp_abs_diff/avg"] == 1.0


def test_split_r3_minibatches_follows_forward_indices():
    actor = PPOActor(PPOActorConfig(), _FakeR3Engine())
    routed = torch.arange(3 * 4 * 2 * 2).reshape(3, 4, 2, 2)
    valid = torch.tensor([True, False, True])
    mb_inputs = SimpleNamespace(
        forward_indices=[2, 0, 1],
        mbs=[
            {"input_ids": torch.ones(2, 4)},
            {"input_ids": torch.ones(1, 4)},
        ],
    )

    r3_mbs = actor._split_r3_minibatches(routed, valid, mb_inputs)

    torch.testing.assert_close(r3_mbs[0][0], routed[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[0][1], valid[[2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][0], routed[[1]], rtol=0, atol=0)
    torch.testing.assert_close(r3_mbs[1][1], valid[[1]], rtol=0, atol=0)


@pytest.mark.parametrize("n_minibatches", [1, 2, 8])
def test_optimizer_metadata_preserves_default_batches_and_row_order(n_minibatches):
    data = {
        "input_ids": torch.arange(4).unsqueeze(1).expand(4, 5),
        "attention_mask": torch.arange(5).unsqueeze(0)
        < torch.tensor([4, 1, 2, 5]).unsqueeze(1),
    }

    batches = split_training_batch_into_microbatches(data, n_minibatches)
    metadata = split_training_batch_into_microbatches(
        data, n_minibatches, return_metadata=True
    )

    assert len(batches) == len(metadata.mbs) == min(n_minibatches, 4)
    rows = torch.cat([batch["input_ids"][:, 0] for batch in batches]).tolist()
    assert metadata.forward_indices == rows
    for batch, metadata_batch in zip(batches, metadata.mbs, strict=True):
        for key in data:
            torch.testing.assert_close(batch[key], metadata_batch[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean", "constant"]
)
@pytest.mark.parametrize("n_minibatches", [1, 2, 8])
def test_ppo_update_keeps_routing_and_prepared_loss_aligned(
    mode, n_minibatches, monkeypatch
):
    monkeypatch.setattr(actor_module, "stats_tracker", DistributedStatsTracker())
    mask = torch.arange(5).unsqueeze(0) < torch.tensor([4, 1, 2, 5]).unsqueeze(1)
    routed = torch.arange(4 * 5 * 2 * 2, dtype=torch.int32).reshape(4, 5, 2, 2)
    valid = torch.tensor([True, False, True, True])
    engine = _FakeR3Engine()
    engine.data_parallel_group = None
    engine.device = "cpu"
    engine.train = lambda: None
    engine.get_version = lambda: 0
    observed_rows = []

    def train_batch(batch, loss_fn, loss_weight_fn):
        rows = batch["input_ids"][:, 0]
        observed_rows.extend(rows.tolist())
        torch.testing.assert_close(
            engine._r3_pending_routed_experts, routed[rows], rtol=0, atol=0
        )
        torch.testing.assert_close(
            engine._r3_pending_valid, valid[rows], rtol=0, atol=0
        )
        assert "routed_experts" not in batch
        step = loss_fn.keywords["pg_reduction"]
        assert step.mode == mode
        assert loss_weight_fn == step.normalizer_fn
        if mode in ("traj-mean", "prompt-mean"):
            denominators = torch.tensor(
                [5, 5, 2, 5] if mode == "traj-mean" else [7, 7, 7, 5]
            )
            expected_weights = mask.float() / denominators.unsqueeze(1)
            torch.testing.assert_close(
                batch["unit_weights"], expected_weights[rows], rtol=0, atol=0
            )
            units = 3 if mode == "traj-mean" else 2
            assert step._step_scale == min(n_minibatches, 4) / units
            assert step._global_rows == len(rows)
        return {}

    engine.train_batch = train_batch
    actor = PPOActor(
        PPOActorConfig(
            path="dummy",
            loss_aggregation=mode,
            loss_aggregation_divisor=16.0 if mode == "constant" else None,
            ppo_n_minibatches=n_minibatches,
        ),
        engine,
    )
    data = {
        "input_ids": torch.arange(4).unsqueeze(1).expand(4, 5),
        "attention_mask": mask,
        "loss_mask": mask,
        "begin_of_trajectory": torch.tensor([1, 0, 1, 1]),
        "rewards": torch.ones(4),
        "advantages": torch.zeros(4, 5),
        "kl_rewards": torch.zeros(4, 5),
        "tot_rewards": torch.zeros(4, 5),
        "routed_experts": routed,
        "r3_routing_valid": valid,
    }

    actor._ppo_update(data, TrajBatchMeta(2, [3, 1], [5, 5]))

    assert sorted(observed_rows) == [0, 1, 2, 3]
    if n_minibatches == 2:
        assert observed_rows != [0, 1, 2, 3]
    assert engine._r3_pending_routed_experts is None
    assert engine._r3_pending_valid is None
