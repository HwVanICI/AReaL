# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api import WeightUpdateMeta
from areal.engine.vllm_remote import VLLMBackend


@pytest.mark.parametrize(
    "colocated,comm_backend,validation_steps,expected_level",
    [
        (None, "file", 0, 1),
        (False, "file", 0, 1),
        (True, "file", 0, 2),
        (True, "nccl", 0, 2),
        (True, "hccl", 0, 2),
        (True, "file", 1, 2),
        (True, "nccl", 1, 1),
        (True, "hccl", 1, 1),
    ],
)
def test_offload_sleep_level_preserves_awex_validation(
    colocated, comm_backend, validation_steps, expected_level
):
    backend = VLLMBackend()
    if colocated is not None:
        meta = WeightUpdateMeta.from_awex(
            meta_server_addr="localhost:1234",
            enable_colocate_mode=colocated,
            comm_backend=comm_backend,
            weights_validation_steps=validation_steps,
        )
        backend.build_awex_init_request(meta, engine_rank=0, num_engines=1)

    request = backend.get_offload_request()

    assert request.endpoint == f"/sleep?level={expected_level}"
    assert request.method == "POST"
    assert request.payload == {}
