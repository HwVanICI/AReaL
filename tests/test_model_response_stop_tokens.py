# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from transformers import GenerationConfig

from areal.api.io_struct import ModelResponse


@pytest.mark.parametrize(
    "eos_ids,output_tokens",
    [(7, [42, 7]), ([7, 8], [42, 8]), ([7, 8], [42, 8, 7, 0])],
)
def test_response_generation_eos_is_stripped_without_changing_raw_tokens(
    tmp_path, eos_ids, output_tokens
):
    """Strip configured EOS/PAD tokens while keeping the raw rollout intact."""
    GenerationConfig(eos_token_id=eos_ids).save_pretrained(tmp_path)
    tokenizer = SimpleNamespace(
        eos_token_id=7, pad_token_id=0, name_or_path=str(tmp_path)
    )
    response = ModelResponse(output_tokens=output_tokens.copy(), tokenizer=tokenizer)

    assert response.end_with_stop
    assert response.output_tokens_without_stop == [42]
    assert response.output_tokens == output_tokens
