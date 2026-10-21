import sys

from areal import MultiLoRATrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.trainer.multi_task.manager import MultiTaskManager

from copy import deepcopy

def load_gsm8k(n_tasks: int, args=None, multi_task_manager=None, num_registered=0):
    
    if args is None:
        args=['--config', '/home/tim/AReaL/examples/multilora/gsm8k_grpo_npu.yaml']
            
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    if multi_task_manager is None:
        multi_task_manager = MultiTaskManager()
        
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    base_workflow_kwargs = dict(
        # temperature=config.gconfig.temperature,
        # top_p=config.gconfig.top_p,
        # max_tokens=config.gconfig.max_tokens,
        # max_completion_tokens=config.gconfig.max_new_tokens,
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    base_eval_workflow_kwargs = base_workflow_kwargs.copy()
    base_eval_workflow_kwargs["temperature"] = 0.6

    register_task_list = []

    # Create 10 LoRA tasks
    for i in range(1, n_tasks+1):
        idx = num_registered+i
        lora_name = f"lora-gsm8k{i}"
        
        workflow_kwargs = deepcopy(base_workflow_kwargs)
        eval_workflow_kwargs = deepcopy(base_eval_workflow_kwargs)
        
        print(f"{workflow_kwargs=}")
        # workflow_kwargs["lora_name"] = lora_name
        # eval_workflow_kwargs["lora_name"] = lora_name
        
        register_task_list.append({
            "lora_int_id": idx+1,
            "lora_name": lora_name,
            # "workflow": "areal.workflow.openai.math_agent.MathAgent",
            # "valid_workflow": "areal.workflow.openai.math_agent.MathAgent",
            "workflow": "areal.workflow.rlvr.RLVRWorkflow",
            "valid_workflow": "areal.workflow.rlvr.RLVRWorkflow",
            "config": deepcopy(config),
            "train_dataset": deepcopy(train_dataset),
            "workflow_kwargs": workflow_kwargs,
            "eval_workflow_kwargs": eval_workflow_kwargs,
            "max_staleness": 2,
            "valid_dataset": deepcopy(valid_dataset),
        })
        
        print(f"{workflow_kwargs=}")
    
    # Register tasks
    for task_instance in register_task_list:
        multi_task_manager.register_task(
            lora_int_id=task_instance["lora_int_id"],
            lora_name=task_instance["lora_name"],
            workflow=task_instance["workflow"],
            valid_workflow=task_instance["valid_workflow"],
            config=task_instance["config"],
            train_dataset=task_instance["train_dataset"],
            workflow_kwargs=task_instance["workflow_kwargs"],
            eval_workflow_kwargs=task_instance["eval_workflow_kwargs"],
            max_staleness=task_instance["max_staleness"],
            valid_dataset=task_instance["valid_dataset"],
        )
    
    return config, multi_task_manager

    with MultiLoRATrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            # workflow="areal.workflow.openai.math_agent.MathAgent",
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            # eval_workflow="areal.workflow.openai.math_agent.MathAgent",
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


def main(args):
    n_tasks = 2
    config, multi_task_manager = load_gsm8k(n_tasks=n_tasks, args=args)
    
    with MultiLoRATrainer(
        config=config,
        multi_task_manager=multi_task_manager,
        lora_model_path="/home/tim/models/Qwen3-0.6B-lora",
        lora_model_base_path="/home/model/Qwen3-0.6B",
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
