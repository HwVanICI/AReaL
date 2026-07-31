# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.ppo.actor import grpo_loss_fn, mopd_pg_loss_fn


def test_mopd_loss_uses_current_policy_for_advantage_and_old_policy_for_ratio():
    """MOPD should separate the current-policy gap from behavior correction."""
    logprobs = torch.tensor([[-1.0, -2.0, -3.0]])
    old_logprobs = torch.tensor([[-2.0, -1.0, -3.0]])
    teacher_logprobs = torch.tensor([[1.0, -5.0, 2.0]])
    loss_mask = torch.tensor([[True, True, False]])

    loss, stat = mopd_pg_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        teacher_logprobs=teacher_logprobs,
        loss_mask=loss_mask,
        adv_clip=2.5,
    )

    expected_raw_advantage = torch.tensor([[2.0, -3.0, 5.0]])
    expected_advantage = torch.tensor([[2.0, -2.5, 2.5]])
    expected_log_ratio = torch.tensor([[1.0, -1.0, 0.0]])
    expected_importance_weight = expected_log_ratio.exp()
    expected_token_loss = -expected_importance_weight * expected_advantage
    expected_loss = expected_token_loss[:, :2].mean()

    torch.testing.assert_close(
        stat["raw_advantage"], expected_raw_advantage, rtol=0, atol=0
    )
    torch.testing.assert_close(stat["advantage"], expected_advantage, rtol=0, atol=0)
    torch.testing.assert_close(stat["approx_kl"], expected_log_ratio, rtol=0, atol=0)
    torch.testing.assert_close(
        stat["importance_weight"], expected_importance_weight, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(stat["loss"], expected_token_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(loss, expected_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        stat["adv_clip_mask"],
        torch.tensor([[False, True, False]]),
        rtol=0,
        atol=0,
    )


def test_mopd_loss_stops_advantage_teacher_and_behavior_gradients():
    """Only the importance ratio should carry gradients to the current policy."""
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    old_logprobs = torch.tensor([[-2.0, -2.0]], requires_grad=True)
    teacher_logprobs = torch.tensor([[0.0, -4.0]], requires_grad=True)
    loss_mask = torch.tensor([[True, True]])

    loss, _ = mopd_pg_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        teacher_logprobs=teacher_logprobs,
        loss_mask=loss_mask,
        adv_clip=5.0,
    )
    loss.backward()

    expected_gradient = torch.tensor([[-torch.e / 2, 1.0]])
    torch.testing.assert_close(logprobs.grad, expected_gradient, rtol=1e-6, atol=1e-6)
    assert old_logprobs.grad is None
    assert teacher_logprobs.grad is None


def test_mopd_loss_with_empty_mask_returns_zero():
    """An empty token mask should produce a finite zero scalar loss."""
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)

    loss, _ = mopd_pg_loss_fn(
        logprobs=logprobs,
        old_logprobs=torch.tensor([[-1.0, -2.0]]),
        teacher_logprobs=torch.tensor([[0.0, -1.0]]),
        loss_mask=torch.tensor([[False, False]]),
        adv_clip=5.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=0)
    loss.backward()
    torch.testing.assert_close(
        logprobs.grad, torch.zeros_like(logprobs), rtol=0, atol=0
    )


def test_mopd_loss_applies_ppo_behavior_correction():
    """Rejected PPO tokens have zero weight in the MOPD objective."""

    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    proximal_logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    behavior_importance_weight = torch.tensor([[0.0, 1.0]], requires_grad=True)
    teacher_logprobs = torch.tensor([[-2.0, -3.0]])
    loss_mask = torch.tensor([[True, True]])

    loss, stat = mopd_pg_loss_fn(
        logprobs=logprobs,
        old_logprobs=torch.tensor([[-10.0, -2.0]]),
        teacher_logprobs=teacher_logprobs,
        loss_mask=loss_mask,
        adv_clip=5.0,
        proximal_logprobs=proximal_logprobs,
        behavior_importance_weight=behavior_importance_weight,
    )

    torch.testing.assert_close(
        stat["importance_weight"],
        torch.tensor([[0.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        stat["loss"],
        torch.tensor([[0.0, 1.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(loss, torch.tensor(0.5), rtol=0, atol=0)

    loss.backward()
    torch.testing.assert_close(
        logprobs.grad,
        torch.tensor([[0.0, 0.5]]),
        rtol=0,
        atol=0,
    )
    assert proximal_logprobs.grad is None
    assert behavior_importance_weight.grad is None


def test_mopd_loss_factorizes_clamped_behavior_correction():
    """MOPD combines current/proximal ratio with PPO's clamped correction."""

    logprobs = torch.tensor([[-0.5, -2.0]])
    proximal_logprobs = torch.tensor([[-1.0, -2.0]])
    behavior_importance_weight = torch.tensor([[5.0, 0.5]])

    _, stat = mopd_pg_loss_fn(
        logprobs=logprobs,
        old_logprobs=torch.tensor([[-10.0, -1.0]]),
        teacher_logprobs=torch.tensor([[-1.5, -3.0]]),
        loss_mask=torch.tensor([[True, True]]),
        adv_clip=5.0,
        proximal_logprobs=proximal_logprobs,
        behavior_importance_weight=behavior_importance_weight,
    )

    expected = torch.tensor([[5.0 * torch.exp(torch.tensor(0.5)), 0.5]])
    torch.testing.assert_close(
        stat["importance_weight"],
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_grpo_mopd_reuses_ppo_rejection_mask():
    """GRPO excludes the same divergent behavior token from PPO and MOPD."""

    input_data = {
        "input_ids": torch.tensor([[10, 11]]),
        "logprobs": torch.tensor([[-10.0, 0.0]]),
        "prox_logp": torch.zeros(1, 2),
        "advantages": torch.zeros(1, 2),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        "teacher_logp": torch.full((1, 2), -1.0),
        "rl_loss_weight": 0.0,
        "distill_loss_weight": 1.0,
        "distill_loss_type": "mopd_pg",
        "mopd_adv_clip": 5.0,
    }

    with patch("areal.trainer.ppo.actor.stats_tracker") as mock_tracker:
        mock_tracker.denominator = MagicMock()
        mock_tracker.stat = MagicMock()
        mock_tracker.scalar = MagicMock()
        mock_tracker.scope = MagicMock()
        mock_tracker.scope.return_value.__enter__ = MagicMock()
        mock_tracker.scope.return_value.__exit__ = MagicMock()

        loss = grpo_loss_fn(
            logprobs=torch.zeros(1, 2),
            entropy=torch.zeros(1, 2),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=RejectionSamplingConfig(
                metric="ratio",
                upper=5.0,
                action="mask",
            ),
            use_decoupled_loss=True,
        )

    torch.testing.assert_close(loss, torch.tensor(0.5), rtol=0, atol=0)
