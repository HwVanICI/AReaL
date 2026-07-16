from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import DistillationConfig
from areal.trainer.rl_trainer import PPOTrainer


class _FakeTeacher:
    def __init__(self, value: float):
        self.value = value
        self.seen_batches = []

    def compute_logp(self, batch):
        self.seen_batches.append(batch)
        return [torch.full_like(traj["loss_mask"], self.value) for traj in batch]


def _teacher_config(domain: str):
    return SimpleNamespace(domain=domain)


def _distillation_teacher():
    return SimpleNamespace(engine_type="rollout", weight=1.0, domain=None)


def test_distillation_config_owns_loss_settings():
    config = DistillationConfig(
        teachers=[_distillation_teacher()],
        distill_loss_type="mopd_pg",
        mopd_adv_clip=3.0,
    )

    assert config.distill_loss_type == "mopd_pg"
    assert config.mopd_adv_clip == 3.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"distill_loss_type": "invalid"}, "teacher.distill_loss_type"),
        ({"mopd_adv_clip": 0.0}, "teacher.mopd_adv_clip"),
    ],
)
def test_distillation_config_rejects_invalid_loss_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DistillationConfig(teachers=[_distillation_teacher()], **kwargs)


def test_assign_domain_teacher_logps_routes_each_trajectory_to_matching_teacher():
    """Domain routing scores each trajectory with exactly one teacher."""

    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        teacher=SimpleNamespace(
            routing_mode="domain",
            domain_key="domain",
            rl_loss_weight=1.0,
            distill_loss_weight=0.25,
            distill_loss_type="mopd_pg",
            mopd_adv_clip=3.0,
        )
    )
    trainer.teachers = [_FakeTeacher(1.0), _FakeTeacher(2.0)]
    trainer.teacher_configs = [_teacher_config("math"), _teacher_config("coding")]
    trainer.teacher_domain_to_idx = {"math": 0, "coding": 1}

    rollout_batch = [
        {
            "domain": "math",
            "loss_mask": torch.ones(1, 3),
        },
        {
            "domain": "coding",
            "loss_mask": torch.ones(1, 2),
        },
        {
            "domain": "math",
            "loss_mask": torch.ones(1, 4),
        },
    ]

    trainer._assign_teacher_logps(rollout_batch)

    assert len(trainer.teachers[0].seen_batches) == 1
    assert len(trainer.teachers[1].seen_batches) == 1
    math_lengths = [
        traj["loss_mask"].shape[-1] for traj in trainer.teachers[0].seen_batches[0]
    ]
    coding_lengths = [
        traj["loss_mask"].shape[-1] for traj in trainer.teachers[1].seen_batches[0]
    ]
    assert math_lengths == [3, 4]
    assert coding_lengths == [2]
    assert "domain" not in trainer.teachers[0].seen_batches[0][0]
    assert "domain" not in trainer.teachers[1].seen_batches[0][0]
    torch.testing.assert_close(
        rollout_batch[0]["teacher_logp"],
        torch.ones(1, 3),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rollout_batch[1]["teacher_logp"],
        torch.full((1, 2), 2.0),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        rollout_batch[2]["teacher_logp"],
        torch.ones(1, 4),
        rtol=0,
        atol=0,
    )
    assert rollout_batch[0]["distill_loss_weight"] == 0.25
    assert rollout_batch[0]["distill_loss_type"] == "mopd_pg"
    assert rollout_batch[0]["mopd_adv_clip"] == 3.0
    assert rollout_batch[1]["distill_loss_type"] == "mopd_pg"
