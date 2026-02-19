from __future__ import annotations

from megatron.training import get_args


def create_gpt_model_from_mindspeed_args(
    *,
    tf_config,
    hf_config,
    pre_process: bool,
    post_process: bool,
):
    """Build GPTModel using the same args semantics as MindSpeed-LLM BaseTrainer."""
    args = get_args()
    spec = getattr(args, "spec", None)
    if not spec:
        raise ValueError(
            "MindSpeed-LLM spec mode requires `--spec ...` in mindspeed_llm.extra_cli_args."
        )

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.spec_utils import import_module

    transformer_layer_spec = import_module(spec)
    use_te = getattr(args, "transformer_impl", None) == "transformer_engine"
    mtp_block_spec = None
    if getattr(args, "mtp_num_layers", None) is not None:
        mtp_block_spec = get_gpt_mtp_block_spec(
            tf_config,
            transformer_layer_spec,
            use_transformer_engine=use_te,
        )

    return GPTModel(
        config=tf_config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=getattr(args, "fp16_lm_cross_entropy", False),
        parallel_output=True,
        share_embeddings_and_output_weights=not getattr(
            args, "untie_embeddings_and_output_weights", True
        ),
        position_embedding_type=getattr(args, "position_embedding_type", "rope"),
        rotary_percent=getattr(args, "rotary_percent", 1.0),
        rotary_base=getattr(
            args, "rotary_base", getattr(hf_config, "rope_theta", 10000.0)
        ),
        seq_len_interpolation_factor=getattr(
            args, "rotary_seq_len_interpolation_factor", None
        ),
        mtp_block_spec=mtp_block_spec,
    )
