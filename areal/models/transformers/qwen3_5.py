# SPDX-License-Identifier: Apache-2.0

"""Compatibility models for Qwen3.5 on older Transformers releases."""

from transformers.modeling_layers import GenericForTokenClassification
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoePreTrainedModel,
)


class Qwen3_5ForTokenClassification(
    GenericForTokenClassification, Qwen3_5PreTrainedModel
):
    """Backport of the upstream Qwen3.5 token-classification model."""


class Qwen3_5MoeForTokenClassification(
    GenericForTokenClassification, Qwen3_5MoePreTrainedModel
):
    """Backport of the upstream Qwen3.5-MoE token-classification model."""


QWEN3_5_TOKEN_CLASSIFICATION_MODELS = {
    "qwen3_5": Qwen3_5ForTokenClassification,
    "qwen3_5_text": Qwen3_5ForTokenClassification,
    "qwen3_5_moe": Qwen3_5MoeForTokenClassification,
    "qwen3_5_moe_text": Qwen3_5MoeForTokenClassification,
}


def get_qwen3_5_token_classification_model(model_type: str):
    """Return the Qwen3.5 token-classification class, if applicable."""
    return QWEN3_5_TOKEN_CLASSIFICATION_MODELS.get(model_type)
