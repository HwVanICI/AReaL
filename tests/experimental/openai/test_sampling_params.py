from unittest.mock import AsyncMock, MagicMock

import pytest

from areal.api import ModelResponse
from areal.experimental.openai import ArealOpenAI


@pytest.mark.asyncio
async def test_chat_completion_puts_sampling_params_in_model_request():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = {"input_ids": [1, 2]}
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.decode.return_value = "done"

    captured_requests = []

    async def agenerate(request):
        captured_requests.append(request)
        return ModelResponse(
            input_tokens=request.input_ids,
            output_tokens=[3],
            output_logprobs=[-0.1],
            output_versions=[0],
            stop_reason="length",
            tokenizer=tokenizer,
        )

    engine = MagicMock()
    engine.agenerate = AsyncMock(side_effect=agenerate)
    client = ArealOpenAI(engine=engine, tokenizer=tokenizer, api_key="test-key")

    await client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=8,
        temperature=0.7,
        top_p=0.8,
        top_k=50,
    )

    assert len(captured_requests) == 1
    gconfig = captured_requests[0].gconfig
    assert gconfig.temperature == 0.7
    assert gconfig.top_p == 0.8
    assert gconfig.top_k == 50
