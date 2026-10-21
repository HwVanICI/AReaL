

import sys
from copy import deepcopy

from areal import MultiLoRATrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.trainer.multi_task.manager import MultiTaskManager

from examples.multi_lora.amc12 import load_amc12
from examples.multi_lora.gsm8k import load_gsm8k
from examples.multi_lora.wiki import load_search

def main(args=None):

    n_tasks = 1
    num_steps =  50 ## max number of steps
    
    reward_stop_thresholds = {
        "lora-gsm8k1": 10000,
        "lora-amc12-1": 10000,
        "lora-wiki-1": 10000,
    }
    
    config, multi_task_manager = load_amc12(n_tasks=n_tasks, args=args, num_steps=num_steps)
    config, multi_task_manager = load_gsm8k(n_tasks=n_tasks, args=args, multi_task_manager=multi_task_manager, num_registered=1, num_steps=num_steps)
    config, multi_task_manager = load_search(n_tasks=n_tasks, args=args, multi_task_manager=multi_task_manager, num_registered=2, num_steps=num_steps)
    
    with MultiLoRATrainer(
        config=config,
        multi_task_manager=multi_task_manager,
        ## where to save initial LoRA Path
        lora_model_path="/tmp/initial_lora",
        lora_model_base_path=config.actor.path,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            reward_stop_thresholds=reward_stop_thresholds
            # workflow="areal.workflow.openai.code_agent.CodeAgent"
        )


if __name__ == "__main__":
    main()
