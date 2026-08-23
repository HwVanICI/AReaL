from types import SimpleNamespace
from unittest.mock import MagicMock, call

import examples.swe.rollout_swe as rollout_swe
from examples.swe.rollout_swe import (
    _configure_rollout_only,
    _run_rollouts,
)


def test_init_rollout_starts_v1_proxy(monkeypatch):
    class FakeRolloutController:
        def __init__(self):
            self.initialize = MagicMock()
            self.start_proxy = MagicMock()

    controller = FakeRolloutController()
    engine_cls = MagicMock()
    engine_cls.as_controller.return_value = controller
    allocation = SimpleNamespace(
        backend="sglang",
        parallel=SimpleNamespace(tp_size=2, pp_size=1),
    )
    config = SimpleNamespace(
        rollout=SimpleNamespace(
            backend="sglang:d1t2p1",
            return_routed_experts=False,
        ),
        sglang=SimpleNamespace(),
    )
    scheduler = MagicMock()

    monkeypatch.setattr(rollout_swe, "RolloutController", FakeRolloutController)
    monkeypatch.setattr(rollout_swe, "RemoteSGLangEngine", engine_cls)
    monkeypatch.setattr(
        rollout_swe.ModelAllocation,
        "from_str",
        MagicMock(return_value=allocation),
    )
    build_args = MagicMock(return_value={"model_path": "model"})
    monkeypatch.setattr(rollout_swe.SGLangConfig, "build_args", build_args)

    result = rollout_swe._init_rollout(config, scheduler)

    assert result is controller
    engine_cls.as_controller.assert_called_once_with(config.rollout, scheduler)
    controller.initialize.assert_called_once_with(
        role="rollout", server_args={"model_path": "model"}
    )
    controller.start_proxy.assert_called_once_with()


def test_configure_rollout_only_disables_training_limits_and_shuffle():
    config = SimpleNamespace(
        rollout=SimpleNamespace(max_head_offpolicyness=4),
        train_dataset=SimpleNamespace(shuffle=True, drop_last=True),
    )

    _configure_rollout_only(config)

    assert config.rollout.max_head_offpolicyness == int(1e12)
    assert config.train_dataset.shuffle is False
    assert config.train_dataset.drop_last is False


def test_run_rollouts_submits_finite_batches_once_and_logs_each_step(monkeypatch):
    config = SimpleNamespace(
        should_accept_fn="examples.swe.filter_function.should_accept",
        gconfig=SimpleNamespace(n_samples=4),
    )
    data = [{"id": index} for index in range(5)]
    dataloader = [data[:3], data[3:]]
    rollout = MagicMock()
    rollout.wait.side_effect = [
        [
            {"interactions": [{}, {}]},
            None,
            {"interactions": [{}]},
        ],
        [{"interactions": [{}, {}, {}]}, None],
    ]
    rollout.export_stats.side_effect = [
        {
            "rollout/reward": 0.25,
            "rollout/accepted": 1.0,
            "rollout/rejected": 1.0,
        },
        {"rollout/reward": 0.5, "rollout/accepted": 1.0},
    ]
    tabulate_stats = MagicMock(return_value="stats table")
    monkeypatch.setattr(rollout_swe, "tabulate_stats", tabulate_stats)
    workflow_kwargs = {"timeout": 10.0}

    _run_rollouts(config, dataloader, rollout, workflow_kwargs)

    assert [item.args[0] for item in rollout.submit.call_args_list] == data
    assert rollout.submit.call_count == 5
    rollout.submit.assert_has_calls(
        [
            call(
                item,
                workflow="examples.swe.agent.SWEAgentWorkflow",
                workflow_kwargs=workflow_kwargs,
                should_accept_fn="examples.swe.filter_function.should_accept",
                group_size=4,
            )
            for item in data
        ]
    )
    assert rollout.wait.call_args_list == [
        call(3, timeout=None),
        call(2, timeout=None),
    ]
    rollout.prepare_batch.assert_not_called()
    assert rollout.export_stats.call_count == 2
    assert tabulate_stats.call_args_list == [
        call(
            {
                "rollout/reward": 0.25,
                "rollout/accepted": 1.0,
                "rollout/rejected": 1.0,
                "rollout/data_count": 3,
                "rollout/trajectory_count": 3,
            }
        ),
        call(
            {
                "rollout/reward": 0.5,
                "rollout/accepted": 1.0,
                "rollout/data_count": 2,
                "rollout/trajectory_count": 3,
            }
        ),
    ]


def test_run_rollouts_covers_481_items_with_partial_final_step(monkeypatch):
    config = SimpleNamespace(
        should_accept_fn=None,
        gconfig=SimpleNamespace(n_samples=1),
    )
    data = [{"id": index} for index in range(481)]
    dataloader = [data[offset : offset + 64] for offset in range(0, len(data), 64)]
    rollout = MagicMock()
    rollout.wait.side_effect = lambda count, timeout: [
        {"interactions": [{}]} for _ in range(count)
    ]
    rollout.export_stats.return_value = {}
    monkeypatch.setattr(rollout_swe, "tabulate_stats", MagicMock(return_value="stats"))

    _run_rollouts(config, dataloader, rollout, {})

    assert [item.args[0]["id"] for item in rollout.submit.call_args_list] == list(
        range(481)
    )
    assert rollout.submit.call_count == 481
    assert [item.args[0] for item in rollout.wait.call_args_list] == [
        64,
        64,
        64,
        64,
        64,
        64,
        64,
        33,
    ]
    rollout.prepare_batch.assert_not_called()
    assert rollout.export_stats.call_count == 8
