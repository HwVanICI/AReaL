import pytest
import torch
import torch.distributed as dist

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, TrajBatchMeta, normalize_rollout_rewards


@pytest.mark.parametrize("unbiased", [True, False])
@pytest.mark.parametrize("std_level", ["batch", "group"])
@pytest.mark.parametrize(
    "rewards",
    [[1.0] + [0.0] * 15, [0.0, 1.0, 2.0, 4.0], [1.0] * 4, [1.0, 0.0], [1.0]],
)
def test_leave_one_out_std_matches_peer_reward_statistics(rewards, unbiased, std_level):
    """Each sample uses only its peers, keeping zero-variance advantages unscaled."""
    x = torch.tensor(rewards, dtype=torch.float32)
    config = NormConfig(
        mean_level="group",
        mean_leave1out=True,
        std_level=std_level,
        std_leave1out=True,
        std_unbiased=unbiased,
        group_size=len(rewards),
        eps=1e-6,
    )
    expected = torch.zeros_like(x)
    for i in range(len(rewards)):
        peers = torch.cat([x[:i], x[i + 1 :]]).double()
        if peers.numel() == 0:
            expected[i] = x[i]
            continue
        advantage = x[i] - peers.mean()
        std = (
            peers.std(correction=int(unbiased)) if peers.numel() > int(unbiased) else 0
        )
        expected[i] = advantage / (std + config.eps) if std > 0 else advantage

    actual = Normalization(config)(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_leave_one_out_std_excludes_masked_rewards():
    """Masked rewards do not contribute to either peer mean or variance."""
    config = NormConfig(
        mean_level="group",
        mean_leave1out=True,
        std_level="group",
        std_leave1out=True,
        group_size=4,
        eps=1e-6,
    )
    x = torch.tensor([0.0, 1.0, 2.0, 100.0])
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    peer_std = torch.tensor(0.5).sqrt()
    expected = torch.tensor([-1.5, 0.0, 1.5, 0.0]) / (peer_std + config.eps)

    actual = Normalization(config)(x, mask)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_leave_one_out_std_groups_trajectories_not_exported_rows():
    """Multi-row trajectories and a partial tail retain prompt-group boundaries."""
    config = NormConfig(
        mean_level="group",
        mean_leave1out=True,
        std_level="group",
        std_leave1out=True,
        group_size=4,
        eps=1e-6,
    )
    normalize = Normalization(config)
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    row_to_trajectory = torch.tensor([0, 0, 1, 2, 3, 3, 3, 4, 5])
    starts = torch.tensor([1, 0, 1, 1, 1, 0, 0, 1, 1])
    expected = normalize(rewards)[row_to_trajectory]

    actual = normalize(rewards[row_to_trajectory], traj_starts=starts)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[-2:], torch.tensor([1.0, -1.0]), rtol=0, atol=0)


def test_default_std_preserves_existing_leave_one_out_normalization():
    """Without the new option, the existing shared group denominator is unchanged."""
    config = NormConfig(
        mean_level="group", mean_leave1out=True, std_level="group", group_size=16
    )
    x = torch.tensor([1.0] + [0.0] * 15)
    centered = x - (x.sum() - x) / 15
    expected = centered / ((centered.double().square().sum() / 15).sqrt() + config.eps)

    actual = Normalization(config)(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mean_level", [None, "batch", "group"])
@pytest.mark.parametrize("std_level", [None, "batch", "group"])
@pytest.mark.parametrize("mean_leave1out", [False, True])
@pytest.mark.parametrize("unbiased", [False, True])
def test_leave_one_out_std_is_independent_of_mean_configuration(
    mean_level, std_level, mean_leave1out, unbiased
):
    """Mean and LOO std select their own populations, including masked elements."""
    x = torch.tensor([0.0, 1.0, 2.0, 10.0, 12.0, 16.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    config = NormConfig(
        mean_level=mean_level,
        mean_leave1out=mean_leave1out,
        std_level=std_level,
        std_leave1out=True,
        std_unbiased=unbiased,
        group_size=3,
        eps=1e-6,
    )
    expected = torch.zeros_like(x)
    indices = torch.arange(x.numel())
    for i in indices[mask.bool()].tolist():
        group = indices // config.group_size == i // config.group_size
        baseline = 0.0
        if mean_level is not None:
            selected = mask.bool() & (group if mean_level == "group" else True)
            if mean_leave1out:
                selected &= indices != i
            peers = x[selected].double()
            baseline = peers.mean() if peers.numel() else 0.0
        advantage = x[i] - baseline
        std = 0.0
        if std_level is not None:
            selected = mask.bool() & (group if std_level == "group" else True)
            peers = x[selected & (indices != i)].double()
            if peers.numel() > int(unbiased):
                std = peers.std(correction=int(unbiased))
        expected[i] = advantage / (std + config.eps) if std > 0 else advantage

    actual = Normalization(config)(x, mask)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_disabled_std_ignores_leave_one_out_option():
    """With no std normalization, even singleton means keep their old behavior."""
    x = torch.tensor([1.0, 2.0])
    config = NormConfig(
        mean_level="group", mean_leave1out=True, std_level=None, group_size=1
    )
    expected = Normalization(config)(x)
    config.std_leave1out = True

    actual = Normalization(config)(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("all_masked", [False, True])
@pytest.mark.parametrize(
    "mean_level,std_level", [(None, "batch"), ("batch", "batch"), ("batch", "group")]
)
def test_batch_leave_one_out_std_uses_global_statistics(
    monkeypatch, all_masked, mean_level, std_level
):
    """Masked ranks join collectives; peer statistics include the remote shard."""
    x = torch.tensor([1.0, 3.0])
    remote = torch.tensor([5.0, 7.0, 9.0])
    mask = torch.zeros_like(x) if all_masked else torch.ones_like(x)
    config = NormConfig(
        mean_level=mean_level,
        mean_leave1out=True,
        std_level=std_level,
        std_leave1out=True,
        group_size=2,
        eps=1e-6,
    )
    remote_count = torch.tensor(float(remote.numel()))
    remote_sum = remote.sum()
    reductions = []
    if mean_level == "batch":
        reductions.extend([remote_count, remote_sum])
    if std_level == "batch":
        reductions.append(
            torch.stack([remote_count, remote_sum, remote.square().sum()])
        )
    calls = []
    remote_reductions = iter(reductions)
    group = object()

    def all_reduce(tensor, op, group):
        calls.append(group)
        tensor.add_(next(remote_reductions).reshape_as(tensor))

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    expected = torch.zeros_like(x)
    if not all_masked:
        for i in range(x.numel()):
            peers = torch.cat([x[:i], x[i + 1 :], remote]).double()
            advantage = x[i] - peers.mean() if mean_level == "batch" else x[i]
            expected[i] = (
                advantage / (peers.std() + config.eps)
                if std_level == "batch"
                else advantage
            )

    actual = Normalization(config)(x, mask, reduce_group=group)

    assert calls == [group] * len(reductions)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_leave_one_out_std_uses_explicit_group_sizes():
    """Actual group membership overrides the configured singleton fallback."""
    x = torch.tensor([0.0, 2.0, 4.0, 8.0, 16.0])
    config = NormConfig(
        mean_level="group",
        mean_leave1out=True,
        std_level="group",
        std_leave1out=True,
        group_size=1,
    )

    actual = Normalization(config)(x, group_sizes=[2, 3])
    expected = torch.tensor([-2.0, 2.0, -8.0, -2.0, 10.0])
    expected[2:] /= torch.tensor([32.0, 72.0, 8.0]).sqrt() + config.eps

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_leave_one_out_std_preserves_individual_scores_and_rollout_references():
    """A zero peer variance leaves discounted row scores centered but unscaled."""
    scores = torch.tensor([0.9, 1.0, 3.0])
    starts = torch.tensor([1, 0, 1])
    meta = TrajBatchMeta(
        n_trajs=1,
        traj_group_sizes=[3],
        traj_seqlens=[1],
        rollout_rewards=[[1.0, 1.0, None]],
    )
    norm = Normalization(
        NormConfig(mean_level="group", std_level="group", std_leave1out=True)
    )

    actual = normalize_rollout_rewards(scores, norm, meta, traj_starts=starts)

    torch.testing.assert_close(
        actual, torch.tensor([-1.1, -1.0, 1.0]), rtol=1e-6, atol=1e-6
    )
