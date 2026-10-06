from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore

from areal.api.cli_args import (
    MicroBatchSpec,
    NormConfig,
    PPOActorConfig,
    RejectionSamplingConfig,
)
from areal.engine.core.train_engine import compute_total_loss_weight
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor, _resolve_proximal_logp
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import MicroBatchList, Normalization
from areal.utils.functional import ppo_actor_loss_fn


@pytest.fixture
def fake_dp_group():
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=1)
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


def _rollout_batch(logp_differences: torch.Tensor) -> dict[str, torch.Tensor]:
    n_rows = logp_differences.shape[0]
    recomputed = torch.full((n_rows, 6), -1.0, dtype=torch.float32)
    generation = recomputed.clone()
    generation[:, 1:4] -= logp_differences
    # Prompt/padding logprobs must not affect the error metric.
    generation[:, [0, 4, 5]] = -1000.0
    return {
        "input_ids": torch.zeros((n_rows, 6), dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]]).expand(n_rows, -1),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 1, 0]]).expand(n_rows, -1).clone(),
        "logprobs": torch.roll(generation, shifts=1, dims=-1),
        "prox_logp": recomputed,
        "rewards": torch.tensor([1.0] + [0.0] * (n_rows - 1)),
    }


@pytest.mark.parametrize("decoupled", [True, False])
def test_sequence_error_masks_whole_rows_using_generation_logprobs(decoupled):
    """Absolute logp errors cannot cancel, even if standard PPO overwrites logprobs."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=decoupled,
        recompute_logprob=True,
        seq_logprob_error_threshold=2.0,
        kl_ctl=0.0,
    )
    actor = PPOActor(config, MagicMock())
    differences = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0], [-2.0, -2.0, -2.0]])
    data = _rollout_batch(differences)
    expected_mask = torch.tensor(
        [[0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )

    result = actor.compute_advantages([data])[0]

    torch.testing.assert_close(result["loss_mask"], expected_mask, rtol=0, atol=0)
    assert torch.isfinite(result["advantages"]).all()


def test_sequence_error_keeps_equal_threshold_and_empty_rows():
    """Equality is in bounds, and rows with no generated tokens remain masked."""
    config = PPOActorConfig(
        backend="fsdp:d1", seq_logprob_error_threshold=2.0, kl_ctl=0.0
    )
    data = _rollout_batch(torch.full((2, 3), torch.log(torch.tensor(2.0))))
    data["loss_mask"][1].zero_()
    expected_mask = torch.roll(data["loss_mask"].float(), -1, -1)

    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]

    torch.testing.assert_close(result["loss_mask"], expected_mask, rtol=0, atol=0)


def test_sequence_error_preserves_rewards_in_leave_one_out_baseline():
    """A masked winner still participates in its peers' reward statistics."""
    norm = NormConfig(
        mean_level="group",
        mean_leave1out=True,
        std_level="group",
        std_leave1out=True,
        group_size=4,
        eps=1e-6,
    )
    config = PPOActorConfig(
        backend="fsdp:d1",
        seq_logprob_error_threshold=2.0,
        reward_norm=norm,
        kl_ctl=0.0,
    )
    data = _rollout_batch(torch.tensor([[2.0] * 3] + [[0.0] * 3] * 3))
    expected_reward = Normalization(norm)(data["rewards"].clone())

    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]

    torch.testing.assert_close(result["loss_mask"][0], torch.zeros(6), rtol=0, atol=0)
    torch.testing.assert_close(
        result["advantages"][1:, 1:4],
        expected_reward[1:, None].expand(-1, 3),
        rtol=1e-6,
        atol=1e-6,
    )


def test_sequence_error_with_reuse_keeps_ratio_one_and_correction_gradient():
    """The extra filtering forward does not replace reused proximal logprobs."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        prox_logp_method="reuse",
        ppo_n_minibatches=1,
        seq_logprob_error_threshold=2.0,
        kl_ctl=0.0,
        rejection_sampling=RejectionSamplingConfig(upper=5.0),
    )
    data = _rollout_batch(torch.zeros((1, 3)))
    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]
    logprobs = torch.full((1, 6), -2.0, requires_grad=True)
    proximal = _resolve_proximal_logp(
        prox_logp_gt=result["prox_logp"],
        prox_logp_method=config.prox_logp_method,
        old_logp=result["logprobs"],
        logprobs=logprobs,
        versions=None,
        current_version=None,
    )
    mask = result["loss_mask"].bool()
    loss, stat = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=result["logprobs"],
        advantages=result["advantages"],
        eps_clip=config.eps_clip,
        loss_mask=mask,
        rejection_sampling=config.rejection_sampling,
    )
    loss.backward()

    torch.testing.assert_close(
        stat["importance_weight"][mask], torch.ones(3), rtol=0, atol=0
    )
    correction = torch.exp(logprobs.detach() - result["logprobs"])
    torch.testing.assert_close(
        logprobs.grad[mask], -correction[mask] / 3, rtol=0, atol=0
    )


@pytest.mark.parametrize("method", ["reuse", "recompute", "loglinear", "metrics"])
def test_sequence_error_requires_recomputation_for_every_proximal_method(method):
    """Filtering always uses a real pre-update policy logprob forward."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        prox_logp_method=method,
        ppo_n_minibatches=1,
        seq_logprob_error_threshold=2.0,
    )
    assert config.should_compute_prox_logp()


def test_sequence_error_preserves_loglinear_proximal_approximation():
    """The filtering-only forward must not override loglinear's approximation."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        prox_logp_method="loglinear",
        seq_logprob_error_threshold=2.0,
        kl_ctl=0.0,
    )
    data = _rollout_batch(torch.zeros((1, 3)))
    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]
    training_logprobs = torch.full((1, 6), -3.0)

    proximal = _resolve_proximal_logp(
        prox_logp_gt=result["prox_logp"],
        prox_logp_method=config.prox_logp_method,
        old_logp=result["logprobs"],
        logprobs=training_logprobs,
        versions=torch.zeros((1, 6), dtype=torch.long),
        current_version=2,
    )

    torch.testing.assert_close(
        proximal[result["loss_mask"].bool()], torch.full((3,), -2.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("filter_all", [True, False])
@pytest.mark.parametrize(
    "mode", ["token-mean", "seq-mean", "traj-mean", "prompt-mean", "constant"]
)
def test_sequence_error_all_filtered_batch_has_safe_global_normalization(
    filter_all, mode, fake_dp_group
):
    """A single-rank fully filtered batch normalizes safely with zero gradient."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        prox_logp_method="reuse",
        ppo_n_minibatches=1,
        seq_logprob_error_threshold=2.0,
        kl_ctl=0.0,
        loss_aggregation=mode,
        loss_aggregation_divisor=3.0 if mode == "constant" else None,
    )
    difference = 2.0 if filter_all else 0.0
    data = _rollout_batch(torch.full((1, 3), difference))
    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]
    prepared = prepare_policy_gradient_batch(
        result,
        mode=mode,
        divisor=config.loss_aggregation_divisor,
        group_sizes=[1] if mode in ("traj-mean", "prompt-mean") else None,
    )
    reduction = prepared.for_steps([result], dp_group=fake_dp_group, device="cpu")[0]
    mb_list = MicroBatchList(
        data=result, mb_spec=MicroBatchSpec(n_mbs=1), mbs=[result], group_lens=[1]
    )
    total_weight = compute_total_loss_weight(
        mb_list, reduction.normalizer_fn, fake_dp_group
    )
    logprobs = result["logprobs"].clone().requires_grad_(True)
    loss, _ = ppo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=logprobs.detach(),
        old_logprobs=result["logprobs"],
        advantages=result["advantages"],
        eps_clip=config.eps_clip,
        loss_mask=result["loss_mask"].bool(),
        pg_reduction=reduction,
        unit_weights=result.get("unit_weights"),
    )
    scaled_loss = loss * reduction.normalizer_fn(result) / total_weight
    scaled_loss.backward()

    expected_weight = 1.0 if filter_all or mode != "token-mean" else 3.0
    torch.testing.assert_close(
        total_weight, torch.tensor(expected_weight), rtol=0, atol=0
    )
    if filter_all:
        torch.testing.assert_close(logprobs.grad, torch.zeros((1, 6)), rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            logprobs.grad[result["loss_mask"].bool()],
            torch.full((3,), -1 / 3),
            rtol=0,
            atol=0,
        )


def test_safe_global_normalization_preserves_positive_fractional_weights(fake_dp_group):
    """Only a zero total gets a fallback; positive engine weights are unchanged."""
    mb_list = MicroBatchList(
        data={}, mb_spec=MicroBatchSpec(n_mbs=1), mbs=[{}], group_lens=[1]
    )
    total_weight = compute_total_loss_weight(
        mb_list, lambda _: torch.tensor(0.25), fake_dp_group
    )

    torch.testing.assert_close(total_weight, torch.tensor(0.25), rtol=0, atol=0)


@pytest.mark.parametrize("filter_all", [False, True])
def test_sequence_error_batch_advantage_norm_joins_remote_collectives(
    monkeypatch, filter_all
):
    """Active and fully masked shards follow the same batch-normalization collectives."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        seq_logprob_error_threshold=2.0,
        adv_norm=NormConfig(),
        kl_ctl=0.0,
    )
    actor = PPOActor(config, MagicMock())
    data = _rollout_batch(torch.full((1, 3), 2.0 if filter_all else 0.0))
    remote = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    global_mean = remote.sum() / 3 if filter_all else (remote.sum() + 3) / 6
    remote_addends = [
        torch.tensor(3.0),
        remote.sum(),
        torch.tensor(3.0),
        (remote - global_mean).square().sum(),
    ]
    addends = iter(remote_addends)
    calls = []

    def all_reduce(tensor, op, group):
        calls.append((op, group))
        tensor.add_(next(addends).reshape_as(tensor))

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    result = actor.compute_advantages([data])[0]

    assert calls == [(dist.ReduceOp.SUM, None)] * 4
    if filter_all:
        torch.testing.assert_close(
            result["advantages"], torch.zeros((1, 6)), rtol=0, atol=0
        )
    else:
        global_values = torch.cat([torch.ones(3, dtype=torch.float64), remote])
        expected = (1 - global_mean) / (global_values.std() + config.adv_norm.eps)
        torch.testing.assert_close(
            result["advantages"][result["loss_mask"].bool()],
            expected.float().expand(3),
            rtol=1e-6,
            atol=1e-6,
        )


def test_sequence_error_preserves_prompt_length_statistics(monkeypatch):
    """Filtering changes loss eligibility, not the recorded prompt token count."""
    config = PPOActorConfig(
        backend="fsdp:d1", seq_logprob_error_threshold=2.0, kl_ctl=0.0
    )
    actor = PPOActor(config, MagicMock())
    data = _rollout_batch(torch.tensor([[0.0] * 3, [2.0] * 3]))
    result = actor.compute_advantages([data])[0]
    tracker = MagicMock()
    monkeypatch.setattr(actor_module, "stats_tracker", tracker)
    monkeypatch.setattr(
        actor_module,
        "split_training_batch_into_microbatches",
        lambda data, **kwargs: SimpleNamespace(mbs=[data]),
    )

    actor._ppo_update(result)

    recorded = next(
        call.kwargs["prompt_len"]
        for call in tracker.stat.call_args_list
        if "prompt_len" in call.kwargs
    )
    torch.testing.assert_close(recorded, torch.tensor([2.0, 2.0]), rtol=0, atol=0)


def test_disabled_sequence_error_preserves_loss_mask_without_recomputation():
    """Default configuration does not require or inspect recomputed logprobs."""
    config = PPOActorConfig(backend="fsdp:d1", kl_ctl=0.0)
    data = _rollout_batch(torch.full((1, 3), 2.0))
    data.pop("prox_logp")
    expected_mask = torch.roll(data["loss_mask"].float(), -1, -1)

    result = PPOActor(config, MagicMock()).compute_advantages([data])[0]

    assert not config.should_compute_prox_logp()
    torch.testing.assert_close(result["loss_mask"], expected_mask, rtol=0, atol=0)


def test_sequence_error_without_recomputed_logprobs_raises():
    """A missing filtering forward must not silently use rollout logprobs."""
    config = PPOActorConfig(backend="fsdp:d1", seq_logprob_error_threshold=2.0)
    data = _rollout_batch(torch.zeros((1, 3)))
    data.pop("prox_logp")

    with pytest.raises(ValueError, match="requires recomputed prox_logp"):
        PPOActor(config, MagicMock()).compute_advantages([data])


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("inf"), float("nan")])
def test_sequence_error_threshold_must_be_positive_and_finite(threshold):
    """Invalid thresholds fail during configuration validation."""
    with pytest.raises(
        ValueError, match="seq_logprob_error_threshold must be positive"
    ):
        PPOActorConfig(seq_logprob_error_threshold=threshold)
