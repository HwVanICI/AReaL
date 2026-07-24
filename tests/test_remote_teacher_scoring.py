from __future__ import annotations

import threading
from types import SimpleNamespace

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
        token = self._payload["input_ids"][0]
        return {
            "token_logps": [float(token)] * self._payload["target_len"],
        }


def test_compute_logp_scores_sequences_concurrently_and_preserves_order(monkeypatch):
    """Teacher scoring overlaps HTTP requests and restores trajectory order."""

    engine = object.__new__(RemoteInfEngine)
    engine.config = SimpleNamespace(
        request_timeout=10,
        use_lora=False,
        max_concurrent_rollouts=2,
        consumer_batch_size=1,
        routing_strategy="round_robin",
    )
    engine.backend = _ScoreBackend()
    engine.addresses = ["server-0"]
    engine.server_idx = 0
    engine.lock = threading.Lock()
    engine._version = 0

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
        torch.tensor([[0.0, 1.0, 1.0], [0.0, 2.0, 2.0]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        results[1],
        torch.tensor([[0.0, 3.0, 3.0], [0.0, 4.0, 4.0]]),
        rtol=0,
        atol=0,
    )
