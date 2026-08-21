"""Tests for the Qwen3.5 token-classification compatibility models."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from areal.api.cli_args import TrainEngineConfig
from areal.models.transformers.qwen3_5 import (
    QWEN3_5_TOKEN_CLASSIFICATION_MODELS,
    Qwen3_5ForTokenClassification,
    Qwen3_5MoeForTokenClassification,
    get_qwen3_5_token_classification_model,
)


def test_qwen3_5_token_classification_forward_returns_per_token_values():
    """A dense Qwen3.5 critic produces one scalar for every input token."""
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
        num_labels=1,
    )
    model = Qwen3_5ForTokenClassification(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False)

    assert output.logits.shape == torch.Size([1, 4, 1])
    assert "score.weight" in model.state_dict()


def test_qwen3_5_multimodal_config_uses_text_hidden_size_for_score():
    """The full Qwen3.5 config gets its critic width from text_config."""
    text_config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        out_hidden_size=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    config = Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        num_labels=1,
    )

    model = Qwen3_5ForTokenClassification(config)

    assert model.score.in_features == config.text_config.hidden_size
    assert model.score.out_features == 1
    assert hasattr(model.model, "visual")


@pytest.mark.parametrize(
    ("model_type", "expected_class"),
    [
        ("qwen3_5", Qwen3_5ForTokenClassification),
        ("qwen3_5_text", Qwen3_5ForTokenClassification),
        ("qwen3_5_moe", Qwen3_5MoeForTokenClassification),
        ("qwen3_5_moe_text", Qwen3_5MoeForTokenClassification),
    ],
)
def test_qwen3_5_token_classification_model_lookup_supports_family(
    model_type, expected_class
):
    """The FSDP critic lookup covers dense, MoE, root, and text configs."""
    assert get_qwen3_5_token_classification_model(model_type) is expected_class


def test_qwen3_5_token_classification_model_lookup_ignores_other_models():
    """Non-Qwen3.5 configs continue through the Transformers auto-model path."""
    assert get_qwen3_5_token_classification_model("qwen3") is None


def test_fsdp_critic_uses_qwen3_5_backport(monkeypatch):
    """Qwen3.5 critics bypass the unsupported Transformers auto mapping."""
    import areal.engine.fsdp_engine as fsdp_module

    calls = []

    class FakeModel:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, **kwargs):
            calls.append((pretrained_model_name_or_path, kwargs))
            return object()

    monkeypatch.setattr(
        fsdp_module.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="qwen3_5_text"),
    )
    monkeypatch.setattr(fsdp_module, "is_valid_vision_model", lambda *_args: False)
    monkeypatch.setitem(QWEN3_5_TOKEN_CLASSIFICATION_MODELS, "qwen3_5_text", FakeModel)
    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test-experiment",
        trial_name="trial0",
        path="test-model",
        is_critic=True,
    )

    model = fsdp_module.FSDPEngine(config)._create_llm_actor_or_critic()

    assert model is not None
    assert calls[0][0] == "test-model"
    assert calls[0][1]["num_labels"] == 1
