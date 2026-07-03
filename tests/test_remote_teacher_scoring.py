from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from areal.infra.remote_inf_engine import RemoteInfEngine


class _ScoreBackend:
    def build_score_request(
        self,
        input_ids,
        target_len,
        with_lora,  # noqa: ARG002
        version,  # noqa: ARG002
    ):
        return SimpleNamespace(
            method="POST",
            endpoint="/score",
            payload={"input_ids": input_ids, "target_len": target_len},
        )

    def parse_score_response(self, payload, target_len):  # noqa: ARG002
        return payload["token_logps"]


class _ScoreResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        target_len = self._payload["target_len"]
        return {
            "token_logps": [
                float(token) for token in self._payload["input_ids"][-target_len:]
            ],
        }


def _make_engine(max_concurrent_rollouts=1):
    engine = object.__new__(RemoteInfEngine)
    engine.config = SimpleNamespace(
        request_timeout=10,
        use_lora=False,
        max_concurrent_rollouts=max_concurrent_rollouts,
        consumer_batch_size=1,
        routing_strategy="round_robin",
    )
    engine.backend = _ScoreBackend()
    engine.addresses = ["server-0"]
    engine.server_idx = 0
    engine.lock = threading.Lock()
    engine._version = 0
    return engine


def test_compute_logp_scores_concurrently_and_aligns_with_student(monkeypatch):
    """Teacher targets are ordered and shifted to student prediction positions."""

    engine = _make_engine(max_concurrent_rollouts=2)

    active_lock = threading.Lock()
    two_requests_started = threading.Event()
    active_requests = 0

    def request(method, url, json, timeout):  # noqa: ARG001
        nonlocal active_requests
        with active_lock:
            active_requests += 1
            if active_requests == 2:
                two_requests_started.set()
        assert two_requests_started.wait(timeout=1)
        try:
            return _ScoreResponse(json)
        finally:
            with active_lock:
                active_requests -= 1

    monkeypatch.setattr("areal.infra.remote_inf_engine.requests.request", request)

    trajectories = [
        {
            "input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]]),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        },
        {
            "input_ids": torch.tensor([[3, 4, 5], [4, 5, 6]]),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        },
    ]

    results = engine.compute_logp(trajectories)

    torch.testing.assert_close(
        results[0],
        torch.tensor([[2.0, 3.0, 0.0], [3.0, 4.0, 0.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        results[1],
        torch.tensor([[4.0, 5.0, 0.0], [5.0, 6.0, 0.0]]),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    (
        "input_ids",
        "loss_mask",
        "attention_mask",
        "expected_logps",
        "expected_request_ids",
        "expected_score_len",
    ),
    [
        (
            [10, 11, 12, 13],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [11.0, 0.0, 13.0, 0.0],
            [10, 11, 12, 13],
            3,
        ),
        (
            [10, 11, 12, 13, 14],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [11.0, 12.0, 0.0, 0.0, 0.0],
            [10, 11, 12],
            2,
        ),
        (
            [10, 11, 12, 0, 0],
            [0, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [11.0, 12.0, 0.0, 0.0, 0.0],
            [10, 11, 12],
            2,
        ),
        (
            [0, 0, 10, 11, 12],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0.0, 0.0, 11.0, 12.0, 0.0],
            [10, 11, 12],
            2,
        ),
    ],
)
def test_compute_logp_maps_masked_tokens_to_prediction_positions(
    monkeypatch,
    input_ids,
    loss_mask,
    attention_mask,
    expected_logps,
    expected_request_ids,
    expected_score_len,
):
    """Remote scoring handles mask gaps, trailing tokens, and sequence padding."""

    engine = _make_engine()
    seen_payloads = []

    def request(method, url, json, timeout):  # noqa: ARG001
        seen_payloads.append(json)
        return _ScoreResponse(json)

    monkeypatch.setattr("areal.infra.remote_inf_engine.requests.request", request)

    result = engine.compute_logp(
        [
            {
                "input_ids": torch.tensor([input_ids]),
                "loss_mask": torch.tensor([loss_mask]),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.bool),
            }
        ]
    )

    torch.testing.assert_close(
        result[0],
        torch.tensor([expected_logps]),
        rtol=0,
        atol=0,
    )
    assert seen_payloads == [
        {
            "input_ids": expected_request_ids,
            "target_len": expected_score_len,
        }
    ]


@pytest.mark.parametrize(
    ("loss_mask", "attention_mask", "message"),
    [
        (
            [0, 0, 1],
            [1, 1, 0],
            "loss_mask can only select tokens included by attention_mask",
        ),
        (
            [1, 0, 0],
            [1, 1, 1],
            "loss_mask cannot select the first active token",
        ),
    ],
)
def test_compute_logp_rejects_unscorable_targets(
    loss_mask,
    attention_mask,
    message,
):
    """Invalid target coordinates fail instead of wrapping or scoring padding."""

    engine = _make_engine()
    trajectory = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "loss_mask": torch.tensor([loss_mask]),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.bool),
    }

    with pytest.raises(ValueError, match=message):
        engine.compute_logp([trajectory])
