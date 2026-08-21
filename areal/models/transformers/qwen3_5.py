# SPDX-License-Identifier: Apache-2.0

"""Compatibility models for Qwen3.5 on older Transformers releases."""

from torch import nn
from transformers import AutoModel
from transformers.modeling_layers import GenericForTokenClassification
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel,
)


def _init_token_classification_model(model, config, pretrained_model_class) -> None:
    """Initialize a classifier for flat text or nested multimodal configs."""
    pretrained_model_class.__init__(model, config)
    model.num_labels = config.num_labels
    setattr(model, model.base_model_prefix, AutoModel.from_config(config))

    text_config = getattr(config, "text_config", config)
    classifier_dropout = getattr(config, "classifier_dropout", None)
    if classifier_dropout is None:
        hidden_dropout = getattr(text_config, "hidden_dropout", None)
        classifier_dropout = hidden_dropout if hidden_dropout is not None else 0.1
    model.dropout = nn.Dropout(classifier_dropout)
    model.score = nn.Linear(text_config.hidden_size, config.num_labels)
    model.post_init()


class Qwen3_5ForTokenClassification(
    GenericForTokenClassification, Qwen3_5PreTrainedModel
):
    """Backport of the upstream Qwen3.5 token-classification model."""

    def __init__(self, config):
        _init_token_classification_model(self, config, Qwen3_5PreTrainedModel)


class Qwen3_5MoeForTokenClassification(
    GenericForTokenClassification, Qwen3_5MoePreTrainedModel
):
    """Backport of the upstream Qwen3.5-MoE token-classification model."""

    def __init__(self, config):
        _init_token_classification_model(self, config, Qwen3_5MoePreTrainedModel)


QWEN3_5_TOKEN_CLASSIFICATION_MODELS = {
    "qwen3_5": Qwen3_5ForTokenClassification,
    "qwen3_5_text": Qwen3_5ForTokenClassification,
    "qwen3_5_moe": Qwen3_5MoeForTokenClassification,
    "qwen3_5_moe_text": Qwen3_5MoeForTokenClassification,
}


def get_qwen3_5_token_classification_model(model_type: str):
    """Return the Qwen3.5 token-classification class, if applicable."""
    return QWEN3_5_TOKEN_CLASSIFICATION_MODELS.get(model_type)
