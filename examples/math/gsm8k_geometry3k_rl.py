import sys

import torch

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.workflow.rlvr import RLVRWorkflow


class GSM8KTextWorkflowForMixedVLM(RLVRWorkflow):
    """GSM8K RLVR workflow with empty multimodal fields for mixed VLM batches."""

    async def arun_episode(self, engine, data):
        result = await super().arun_episode(engine, data)
        if result is None:
            return None

        result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"])
        batch_size = result["input_ids"].shape[0]
        result["multi_modal_input"] = [{} for _ in range(batch_size)]
        return result


def _router_workflow_kwargs(config, tokenizer_path: str, processor_path: str):
    return {
        "domain_key": "domain",
        "workflows": {
            "gsm8k": {
                "workflow": (
                    "examples.math.gsm8k_geometry3k_rl.GSM8KTextWorkflowForMixedVLM"
                ),
                "kwargs": {
                    "reward_fn": "areal.reward.gsm8k.gsm8k_reward_fn",
                    "gconfig": config.gconfig,
                    "tokenizer": tokenizer_path,
                    "enable_thinking": False,
                },
            },
            "geometry3k": {
                "workflow": "areal.workflow.vision_rlvr.VisionRLVRWorkflow",
                "kwargs": {
                    "reward_fn": ("examples.vlm.geometry3k_grpo.geometry3k_reward_fn"),
                    "gconfig": config.gconfig,
                    "tokenizer": tokenizer_path,
                    "processor": processor_path,
                    "enable_thinking": False,
                },
            },
        },
    }


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            split="test",
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
            processor=processor,
        )

    workflow_kwargs = _router_workflow_kwargs(
        config,
        tokenizer_path=config.tokenizer_path,
        processor_path=config.tokenizer_path,
    )

    eval_config = config.gconfig.new(temperature=0.6)
    eval_workflow_kwargs = _router_workflow_kwargs(
        config,
        tokenizer_path=config.tokenizer_path,
        processor_path=config.tokenizer_path,
    )
    for spec in eval_workflow_kwargs["workflows"].values():
        spec["kwargs"]["gconfig"] = eval_config

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.DomainRouterRolloutWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.DomainRouterRolloutWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
