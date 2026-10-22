import argparse
import sys
import os

from areal import MultiLoRATrainerAsync
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.trainer.multi_task.manager import MultiTaskManager

from copy import deepcopy

def load_search(n_tasks: int, args=None, multi_task_manager=None, num_registered=0, num_steps=100):
    
    if args is None:
        args = ['--config', './examples/multi_lora/_large_experiment_wiki']  # <-- update this
 
    config, _ = load_expr_config(args, GRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    ## Verify URL is set
    url = os.getenv("WIKI_JUDGE_URL")
    if not url:
        raise ValueError("WIKI_JUDGE_URL environment variable is not set")

    if multi_task_manager is None:
        multi_task_manager = MultiTaskManager()
        
    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    
    valid_dataset = None  # keep consistent with your original search script

    base_workflow_kwargs = dict(
        reward_fn="areal.reward.wiki.wiki_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    base_eval_workflow_kwargs = base_workflow_kwargs.copy()
    base_eval_workflow_kwargs["temperature"] = 0.6

    register_task_list = []

    # Create n_tasks LoRAs
    for i in range(1, n_tasks+1):
        idx = num_registered+i
        lora_name = f"lora-wiki-{i}"
        
        workflow_kwargs = deepcopy(base_workflow_kwargs)
        eval_workflow_kwargs = deepcopy(base_eval_workflow_kwargs)
        
        register_task_list.append({
            "lora_int_id": idx+1,
            "lora_name": lora_name,
            "workflow": "areal.workflow.rlvr.RLVRWorkflow",
            "valid_workflow": "areal.workflow.rlvr.RLVRWorkflow",
            "config": deepcopy(config),
            "train_dataset": deepcopy(train_dataset),
            "workflow_kwargs": workflow_kwargs,
            "eval_workflow_kwargs": eval_workflow_kwargs,
            "max_staleness": 2,
            "valid_dataset": deepcopy(valid_dataset),
        })
        
    # Register tasks in the management layer
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
            num_steps=num_steps,
        )
    
    return config, multi_task_manager


def main(args):
    
    num_steps =  50 ## max number of steps
    
    reward_stop_thresholds = {
        "lora-wiki-1": 1000,
    }
    
    parser = argparse.ArgumentParser(add_help=False)
    custom_args, hydra_args = parser.parse_known_args(args)

    n_tasks = 1

    config, multi_task_manager = load_search(
        n_tasks=n_tasks,
        args=hydra_args,
        num_steps=num_steps,
    )

    with MultiLoRATrainerAsync(
        config=config,
        multi_task_manager=multi_task_manager,
        ## where to save initial LoRA Path
        lora_model_path="/tmp/initial_lora/llama_init",
        lora_model_base_path=config.actor.path,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            reward_stop_thresholds=reward_stop_thresholds,
        )

if __name__ == "__main__":
    main(sys.argv[1:])
