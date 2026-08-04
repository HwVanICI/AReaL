import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

enable_thinking = False


def _router_workflow_kwargs(config, tokenizer_path: str, processor_path: str):
    return {
        "domain_key": "domain",
        "workflows": {
            "aime": {
                "workflow": ("areal.workflow.rlvr.RLVRWorkflow"),
                "kwargs": {
                    "reward_fn": "areal.reward.aime.aime_reward_fn",
                    "gconfig": config.gconfig,
                    "tokenizer": tokenizer_path,
                    "enable_thinking": enable_thinking,
                },
            },
            "leetcode": {
                "workflow": ("areal.workflow.openai.code_agent.CodeAgent"),
                "kwargs": {
                    "temperature": config.gconfig.temperature,
                    "top_p": config.gconfig.top_p,
                    "max_tokens": config.gconfig.max_tokens,
                    "max_completion_tokens": config.gconfig.max_new_tokens,
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
        if "kwargs" in spec and "gconfig" in spec["kwargs"]:
            spec["kwargs"]["gconfig"] = eval_config
        if "kwargs" in spec and "temperature" in spec["kwargs"]:
            spec["kwargs"]["temperature"] = eval_config.temperature

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
