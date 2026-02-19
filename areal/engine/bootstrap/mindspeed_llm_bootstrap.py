from __future__ import annotations

def create_gpt_model_from_mindspeed_args(
    *,
    pre_process: bool,
    post_process: bool,
):
    """Build model via MindSpeed-LLM BaseTrainer.model_provider."""
    from mindspeed_llm.tasks.posttrain.base.base_trainer import BaseTrainer

    # model_provider is self-free in practice; call it directly to stay aligned
    # with MindSpeed-LLM model construction logic.
    return BaseTrainer.model_provider(
        object(),
        pre_process=pre_process,
        post_process=post_process,
    )
