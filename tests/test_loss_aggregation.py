# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec, NormConfig, PPOActorConfig
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import (
    PPOActor,
    _atomic_row_group_sizes,
    _trajectory_row_group_sizes,
    trajectory_token_weights,
)
from areal.utils.data import (
    _resolve_microbatch_sequence_groups,
    split_padded_tensor_dict_into_mb_list,
)
from areal.utils.functional.loss_aggregation import PolicyGradientReduction

# Three packed sequences of length 2, 3 and 5. Per-token losses are constant
# within a sequence so every expected value below can be read off by hand.
CU_SEQLENS = torch.tensor([0, 2, 5, 10], dtype=torch.int32)
LOSS = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0])
MASK = torch.ones(10, dtype=torch.bool)


def legacy_token_mean(loss, loss_mask):
    """The reduction ppo_actor_loss_fn used before loss_aggregation existed."""
    return torch.where(loss_mask, loss, 0).sum() / (loss_mask.count_nonzero() or 1)


class TestTokenMeanParity:
    """token-mean must stay bit-identical to the pre-feature reduction."""

    def test_matches_legacy_reduction(self):
        reduction = PolicyGradientReduction("token-mean")
        assert torch.equal(
            reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS),
            legacy_token_mean(LOSS, MASK),
        )

    def test_matches_legacy_reduction_with_holes(self):
        mask = MASK.clone()
        mask[0] = False
        mask[7] = False
        reduction = PolicyGradientReduction("token-mean")
        assert torch.equal(
            reduction.aggregate(LOSS, mask, cu_seqlens=CU_SEQLENS),
            legacy_token_mean(LOSS, mask),
        )

    def test_normalizer_matches_legacy_loss_weight_fn(self):
        reduction = PolicyGradientReduction("token-mean")
        data = {"loss_mask": MASK, "cu_seqlens": CU_SEQLENS}
        assert torch.equal(reduction.normalizer_fn(data), MASK.count_nonzero())

    def test_token_mean_needs_no_sequence_boundaries(self):
        reduction = PolicyGradientReduction("token-mean")
        assert reduction.aggregate(LOSS, MASK) == pytest.approx(2.8)


class TestSeqMean:
    def test_packed(self):
        # Per-sequence token means are 1, 2 and 4.
        reduction = PolicyGradientReduction("seq-mean")
        result = reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS)
        assert result.item() == pytest.approx(7 / 3)

    def test_padded_matches_packed(self):
        loss = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0, 4.0],
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        )
        reduction = PolicyGradientReduction("seq-mean")
        assert reduction.aggregate(loss, mask).item() == pytest.approx(7 / 3)

    def test_ignores_sequence_length(self):
        """A long and a short sequence with the same per-token loss tie."""
        cu_seqlens = torch.tensor([0, 1, 9], dtype=torch.int32)
        loss = torch.tensor([1.0] + [3.0] * 8)
        mask = torch.ones(9, dtype=torch.bool)
        reduction = PolicyGradientReduction("seq-mean")
        assert reduction.aggregate(loss, mask, cu_seqlens=cu_seqlens).item() == (
            pytest.approx(2.0)
        )

    def test_normalizer_counts_sequences(self):
        reduction = PolicyGradientReduction("seq-mean")
        data = {"loss_mask": MASK, "cu_seqlens": CU_SEQLENS}
        assert reduction.normalizer_fn(data).item() == pytest.approx(3.0)

    def test_packed_input_requires_cu_seqlens(self):
        reduction = PolicyGradientReduction("seq-mean")
        with pytest.raises(ValueError, match="requires cu_seqlens"):
            reduction.aggregate(LOSS, MASK)


class TestConstant:
    def test_divides_by_fixed_denominator(self):
        # Masked sum is 28 over 3 active sequences, divisor 2 -> 28 / 6.
        reduction = PolicyGradientReduction("constant", divisor=2.0)
        result = reduction.aggregate(LOSS, MASK, cu_seqlens=CU_SEQLENS)
        assert result.item() == pytest.approx(28 / 6)

    def test_divisor_is_required(self):
        with pytest.raises(ValueError, match="divisor must be a positive"):
            PolicyGradientReduction("constant")

    def test_divisor_rejected_by_other_modes(self):
        with pytest.raises(ValueError, match="only valid for"):
            PolicyGradientReduction("seq-mean", divisor=2.0)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="loss_aggregation must be one of"):
            PolicyGradientReduction("mean")


class TestNarrowedNumerator:
    """Rejection sampling narrows the numerator but keeps the denominator."""

    @pytest.mark.parametrize("mode", ["token-mean", "seq-mean"])
    def test_denominator_mask_is_honoured(self, mode):
        narrowed = MASK.clone()
        narrowed[0] = False
        reduction = PolicyGradientReduction(mode)
        kept = reduction.aggregate(
            LOSS, narrowed, denominator_mask=MASK, cu_seqlens=CU_SEQLENS
        )
        rescaled = reduction.aggregate(LOSS, narrowed, cu_seqlens=CU_SEQLENS)
        # Dropping a token must shrink the loss, not redistribute its weight.
        assert kept.item() < rescaled.item()

    def test_token_mean_denominator_is_the_original_count(self):
        narrowed = MASK.clone()
        narrowed[0] = False
        reduction = PolicyGradientReduction("token-mean")
        result = reduction.aggregate(
            LOSS, narrowed, denominator_mask=MASK, cu_seqlens=CU_SEQLENS
        )
        assert result.item() == pytest.approx((LOSS.sum().item() - 1.0) / 10)


class TestMicrobatchSequenceGroups:
    """No atomic groups must reproduce the pre-feature granularity split."""

    @pytest.mark.parametrize("bs,granularity", [(6, 1), (8, 2), (12, 3), (130, 1)])
    def test_matches_legacy_granularity_split(self, bs, granularity):
        groups = _resolve_microbatch_sequence_groups(bs, granularity, None)
        assert groups == [
            list(range(i * granularity, (i + 1) * granularity))
            for i in range(bs // granularity)
        ]

    def test_indivisible_batch_still_raises(self):
        with pytest.raises(RuntimeError, match="cannot divide granularity"):
            _resolve_microbatch_sequence_groups(7, 2, None)

    def test_explicit_ragged_groups(self):
        groups = _resolve_microbatch_sequence_groups(6, 1, [3, 1, 2])
        assert groups == [[0, 1, 2], [3], [4, 5]]

    def test_explicit_groups_accept_tensors(self):
        groups = _resolve_microbatch_sequence_groups(6, 1, torch.tensor([3, 1, 2]))
        assert groups == [[0, 1, 2], [3], [4, 5]]

    def test_group_sizes_must_cover_the_batch(self):
        with pytest.raises(ValueError, match="group_sizes sum to"):
            _resolve_microbatch_sequence_groups(6, 1, [3, 1, 1])

    def test_group_sizes_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            _resolve_microbatch_sequence_groups(6, 1, [3, 0, 3])


# Four padded rows forming two trajectories: the first spans three rows
# (2 + 3 + 5 = 10 masked tokens), the second is a single 4-token row.
TRAJ_BEGIN = torch.tensor([1, 0, 0, 1], dtype=torch.int32)
TRAJ_MASK = torch.tensor(
    [
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ],
    dtype=torch.bool,
)
TRAJ_LOSS = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 0.0, 0.0],
        [4.0, 4.0, 4.0, 4.0, 4.0],
        [8.0, 8.0, 8.0, 8.0, 0.0],
    ]
)


class TestTrajectoryTokenWeights:
    def test_weights_sum_to_the_trajectory_count(self):
        weights = trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN)
        assert weights.sum().item() == pytest.approx(2.0)

    def test_rows_of_one_trajectory_share_a_weight(self):
        weights = trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN)
        # First trajectory holds 10 masked tokens, second holds 4.
        assert weights[0][0].item() == pytest.approx(0.1)
        assert weights[2][0].item() == pytest.approx(0.1)
        assert weights[3][0].item() == pytest.approx(0.25)

    def test_masked_tokens_get_no_weight(self):
        weights = trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN)
        assert weights[0][2].item() == 0.0

    def test_missing_marker_falls_back_to_one_trajectory_per_row(self):
        weights = trajectory_token_weights(TRAJ_MASK, None)
        assert weights.sum().item() == pytest.approx(4.0)

    def test_marker_length_must_match(self):
        with pytest.raises(ValueError, match="begin_of_trajectory has"):
            trajectory_token_weights(TRAJ_MASK, torch.tensor([1, 0, 1]))

    def test_first_row_must_start_a_trajectory(self):
        with pytest.raises(ValueError, match="first row must begin"):
            trajectory_token_weights(
                TRAJ_MASK, torch.tensor([0, 0, 1, 1], dtype=torch.int32)
            )


class TestTrajectoryAtomicMinibatches:
    def test_row_group_sizes_follow_trajectory_markers(self):
        assert _trajectory_row_group_sizes(TRAJ_BEGIN, len(TRAJ_BEGIN)) == [3, 1]

    def test_missing_markers_make_each_row_atomic(self):
        assert _trajectory_row_group_sizes(None, 4) == [1, 1, 1, 1]

    def test_split_keeps_trajectory_rows_together(self):
        row_ids = torch.arange(4).reshape(-1, 1).expand(-1, 5)
        data = {
            "attention_mask": TRAJ_MASK,
            "input_ids": row_ids,
        }

        batches = split_padded_tensor_dict_into_mb_list(
            data,
            MicroBatchSpec(n_mbs=2),
            atomic_group_sizes=[3, 1],
        ).mbs

        row_sets = [{int(row) for row in mb["input_ids"][:, 0]} for mb in batches]
        assert {frozenset(rows) for rows in row_sets} == {
            frozenset({0, 1, 2}),
            frozenset({3}),
        }
        assert all("group_sizes" not in mb for mb in batches)

    def test_single_minibatch_preserves_token_layout(self):
        row_ids = torch.arange(4).reshape(-1, 1).expand(-1, 5)
        data = {"attention_mask": TRAJ_MASK, "input_ids": row_ids}
        baseline = split_padded_tensor_dict_into_mb_list(
            data,
            MicroBatchSpec(n_mbs=1),
        )
        grouped = split_padded_tensor_dict_into_mb_list(
            data,
            MicroBatchSpec(n_mbs=1),
            atomic_group_sizes=[3, 1],
        )

        torch.testing.assert_close(
            grouped.mbs[0]["input_ids"], baseline.mbs[0]["input_ids"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            grouped.mbs[0]["attention_mask"],
            baseline.mbs[0]["attention_mask"],
            rtol=0,
            atol=0,
        )
        assert grouped.forward_indices == baseline.forward_indices
        assert grouped.backward_indices == baseline.backward_indices
        assert "group_sizes" not in grouped.mbs[0]

    @pytest.mark.parametrize("n_minibatches", [1, 2])
    def test_actor_uses_trajectory_groups_for_optimizer_minibatches(
        self, monkeypatch, n_minibatches
    ):
        captured = {}

        def capture_split(data, mb_spec, group=None, atomic_group_sizes=None):
            captured["n_mbs"] = mb_spec.n_mbs
            captured["atomic_group_sizes"] = atomic_group_sizes
            return SimpleNamespace(mbs=[])

        monkeypatch.setattr(
            actor_module, "split_padded_tensor_dict_into_mb_list", capture_split
        )
        actor = object.__new__(PPOActor)
        actor.config = SimpleNamespace(
            c_clip=None,
            eps_clip=0.2,
            log_agent_stats=False,
            loss_aggregation="traj-mean",
            loss_aggregation_divisor=None,
            mask_no_eos_with_zero=False,
            ppo_n_minibatches=n_minibatches,
            rejection_sampling=None,
        )
        actor.engine = SimpleNamespace(train=lambda: None, get_version=lambda: 0)
        actor._ppo_update(
            {
                "attention_mask": TRAJ_MASK,
                "loss_mask": TRAJ_MASK,
                "begin_of_trajectory": TRAJ_BEGIN,
                "rewards": torch.tensor([1.0, 1.0, 1.0, 0.0]),
                "advantages": torch.zeros_like(TRAJ_LOSS),
                "kl_rewards": torch.zeros_like(TRAJ_LOSS),
                "tot_rewards": torch.zeros_like(TRAJ_LOSS),
            }
        )

        assert captured == {
            "n_mbs": n_minibatches,
            "atomic_group_sizes": [3, 1],
        }


class TestTrajMean:
    def _weights(self):
        return trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN)

    def test_hand_computed_value(self):
        # Trajectory means are (2*1 + 3*2 + 5*4) / 10 = 2.8 and 8.0.
        reduction = PolicyGradientReduction("traj-mean")
        result = reduction.aggregate(TRAJ_LOSS, TRAJ_MASK, unit_weights=self._weights())
        assert result.item() == pytest.approx(5.4)

    @pytest.mark.parametrize(
        "partition",
        [
            [[0], [1], [2], [3]],
            [[0, 3], [1, 2]],
            [[2], [0], [3], [1]],
            [[0, 1, 2, 3]],
        ],
    )
    def test_partitioning_does_not_change_the_result(self, partition):
        """A trajectory may be split across forward microbatches in one step."""
        reduction = PolicyGradientReduction("traj-mean")
        weights = self._weights()
        numerator = 0.0
        denominator = 0.0
        for rows in partition:
            index = torch.tensor(rows)
            mask, loss, unit = TRAJ_MASK[index], TRAJ_LOSS[index], weights[index]
            local_mean = reduction.aggregate(loss, mask, unit_weights=unit)
            local_weight = reduction.normalizer_fn(
                {"loss_mask": mask, "unit_weights": unit}
            )
            numerator += local_mean.item() * local_weight.item()
            denominator += local_weight.item()
        assert numerator / denominator == pytest.approx(5.4, rel=1e-5)

    def test_matches_seq_mean_when_every_row_is_a_trajectory(self):
        weights = trajectory_token_weights(TRAJ_MASK, None)
        packed_loss = TRAJ_LOSS[TRAJ_MASK]
        packed_mask = TRAJ_MASK[TRAJ_MASK]
        packed_weights = weights[TRAJ_MASK]
        cu_seqlens = torch.tensor([0, 2, 5, 10, 14], dtype=torch.int32)
        traj = PolicyGradientReduction("traj-mean").aggregate(
            packed_loss, packed_mask, unit_weights=packed_weights
        )
        seq = PolicyGradientReduction("seq-mean").aggregate(
            packed_loss, packed_mask, cu_seqlens=cu_seqlens
        )
        assert traj.item() == pytest.approx(seq.item(), rel=1e-6)

    def test_requires_unit_weights(self):
        reduction = PolicyGradientReduction("traj-mean")
        with pytest.raises(ValueError, match="unit_weights are required"):
            reduction.aggregate(TRAJ_LOSS, TRAJ_MASK)

    def test_unit_weights_must_match_the_loss_shape(self):
        reduction = PolicyGradientReduction("traj-mean")
        with pytest.raises(ValueError, match="unit_weights shape"):
            reduction.aggregate(TRAJ_LOSS, TRAJ_MASK, unit_weights=torch.ones(3, 3))

    def test_normalizer_requires_unit_weights(self):
        reduction = PolicyGradientReduction("traj-mean")
        with pytest.raises(ValueError, match="unit_weights are required"):
            reduction.normalizer_fn({"loss_mask": TRAJ_MASK})

    def test_needs_no_sequence_boundaries(self):
        reduction = PolicyGradientReduction("traj-mean")
        packed_loss = TRAJ_LOSS[TRAJ_MASK]
        packed_mask = TRAJ_MASK[TRAJ_MASK]
        packed_weights = self._weights()[TRAJ_MASK]
        assert reduction.aggregate(
            packed_loss, packed_mask, unit_weights=packed_weights
        ).item() == pytest.approx(5.4)


class TestPromptMean:
    """One rollout group carries weight one, pooling its tokens.

    The fixture's two trajectories stand in for a group of two rollouts: 10
    masked tokens summing to 28, and 4 summing to 32.
    """

    def _weights(self):
        return trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN, 2)

    def test_weights_sum_to_the_group_count(self):
        assert self._weights().sum().item() == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # Tokens pool across the group: (28 + 32) / (10 + 4).
        reduction = PolicyGradientReduction("prompt-mean")
        result = reduction.aggregate(TRAJ_LOSS, TRAJ_MASK, unit_weights=self._weights())
        assert result.item() == pytest.approx(60.0 / 14.0)

    def test_pooling_differs_from_averaging_rollouts(self):
        """The longer rollout pulls the group mean; traj-mean gives 5.4."""
        prompt = PolicyGradientReduction("prompt-mean").aggregate(
            TRAJ_LOSS, TRAJ_MASK, unit_weights=self._weights()
        )
        traj = PolicyGradientReduction("traj-mean").aggregate(
            TRAJ_LOSS,
            TRAJ_MASK,
            unit_weights=trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN),
        )
        assert prompt.item() == pytest.approx(60.0 / 14.0)
        assert traj.item() == pytest.approx(5.4)

    def test_a_group_of_one_is_traj_mean(self):
        assert torch.allclose(
            trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN, 1),
            trajectory_token_weights(TRAJ_MASK, TRAJ_BEGIN),
        )

    @pytest.mark.parametrize(
        "partition",
        [
            [[0], [1], [2], [3]],
            [[0, 3], [1, 2]],
            [[2], [0], [3], [1]],
            [[0, 1, 2, 3]],
        ],
    )
    def test_partitioning_does_not_change_the_result(self, partition):
        """A group may be split across forward microbatches in one step."""
        reduction = PolicyGradientReduction("prompt-mean")
        weights = self._weights()
        numerator = 0.0
        denominator = 0.0
        for rows in partition:
            index = torch.tensor(rows)
            mask, loss, unit = TRAJ_MASK[index], TRAJ_LOSS[index], weights[index]
            local_mean = reduction.aggregate(loss, mask, unit_weights=unit)
            local_weight = reduction.normalizer_fn(
                {"loss_mask": mask, "unit_weights": unit}
            )
            numerator += local_mean.item() * local_weight.item()
            denominator += local_weight.item()
        assert numerator / denominator == pytest.approx(60.0 / 14.0, rel=1e-5)

    def test_atomic_rows_cover_the_whole_group(self):
        """A group spans its trajectories' rows, not one row per rollout."""
        assert _trajectory_row_group_sizes(TRAJ_BEGIN, 4) == [3, 1]
        assert _atomic_row_group_sizes(TRAJ_BEGIN, 4, 2) == [4]

    def test_group_of_one_leaves_atomic_rows_per_trajectory(self):
        assert _atomic_row_group_sizes(TRAJ_BEGIN, 4, 1) == [3, 1]

    def test_incomplete_group_is_rejected_not_silently_shifted(self):
        begin = torch.tensor([1, 0, 0, 1, 1], dtype=torch.int32)
        mask = torch.ones(5, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match="do not divide into groups"):
            trajectory_token_weights(mask, begin, 2)
        with pytest.raises(ValueError, match="do not divide into groups"):
            _atomic_row_group_sizes(begin, 5, 2)

    def test_requires_unit_weights(self):
        reduction = PolicyGradientReduction("prompt-mean")
        with pytest.raises(ValueError, match="unit_weights are required"):
            reduction.aggregate(TRAJ_LOSS, TRAJ_MASK)

    def test_normalizer_requires_unit_weights(self):
        reduction = PolicyGradientReduction("prompt-mean")
        with pytest.raises(ValueError, match="unit_weights are required"):
            reduction.normalizer_fn({"loss_mask": TRAJ_MASK})


class TestPromptMeanConfig:
    @staticmethod
    def _grouped_norm(group_size=8):
        return NormConfig(mean_level="group", std_level="group", group_size=group_size)

    def test_inherits_group_size_from_reward_norm(self):
        config = PPOActorConfig(
            path="dummy",
            loss_aggregation="prompt-mean",
            reward_norm=self._grouped_norm(),
        )
        assert config.loss_aggregation_group_size == 8

    def test_explicit_group_size_wins(self):
        config = PPOActorConfig(
            path="dummy",
            loss_aggregation="prompt-mean",
            loss_aggregation_group_size=4,
            reward_norm=self._grouped_norm(8),
        )
        assert config.loss_aggregation_group_size == 4

    def test_batch_level_norm_is_not_inherited(self):
        """group_size sits at its unused default there, and 1 is not a group."""
        with pytest.raises(ValueError, match="needs the number of"):
            PPOActorConfig(
                path="dummy",
                loss_aggregation="prompt-mean",
                reward_norm=NormConfig(mean_level="batch", std_level="batch"),
            )

    def test_no_reward_norm_needs_an_explicit_size(self):
        with pytest.raises(ValueError, match="needs the number of"):
            PPOActorConfig(path="dummy", loss_aggregation="prompt-mean")

    def test_group_size_rejected_for_other_modes(self):
        with pytest.raises(ValueError, match="only used when"):
            PPOActorConfig(
                path="dummy",
                loss_aggregation="traj-mean",
                loss_aggregation_group_size=8,
            )
