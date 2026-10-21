from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict


@dataclass
class TaskState:
    # Identity
    lora_int_id: int
    lora_name: str

    # Workflows
    workflow: str
    valid_workflow: str

    # Config
    config: Any

    # Training
    train_dataset: Any
    workflow_kwargs: Dict[str, Any]

    # Validation / Evaluation
    valid_dataset: Any
    eval_workflow_kwargs: Dict[str, Any]

    # Rollout / staleness control
    max_staleness: int = 2
    current_rollouts: int = 0
    
    num_steps: int = 100

    lock: Lock = field(default_factory=Lock)

    # Internal iterator for stepwise batch submission
    _train_iterator: Any = field(default=None, init=False)

    # ---------------------------
    # Staleness Management
    # ---------------------------

    def can_submit(self, n: int = 1) -> bool:
        with self.lock:
            return (self.current_rollouts + n) <= self.max_staleness + 1

    def register_submission(self, n: int = 1):
        with self.lock:
            self.current_rollouts += n

    def register_completion(self, n: int = 1):
        with self.lock:
            self.current_rollouts = max(0, self.current_rollouts - n)

    def remaining_capacity(self) -> int:
        with self.lock:
            return self.max_staleness + 1 - self.current_rollouts

    # ---------------------------
    # Dataloader / batch access
    # ---------------------------

    def _register_dataloaders(self, dataloader, split: str):
        if split == "train":
            self.train_dataloader = dataloader
        elif split == "validation":
            self.valid_dataloader = dataloader
        else:
            raise ValueError("split must be 'train' or 'validation'")

    def _initialize_step_config(self):
        self.total_epochs = self.config.total_train_epochs
        self.steps_per_epoch = len(self.train_dataloader)
        self.max_steps = self.total_epochs * self.steps_per_epoch
        self.global_step = 0

    def _task_status(self) -> bool:
        # True if there are still training steps left
        return self.global_step < self.max_steps


    def next_train_batch(self):
        if self._train_iterator is None:
            self._train_iterator = iter(self.train_dataloader)
        try:
            batch = next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            batch = next(self._train_iterator)
        return batch


    def __repr__(self):
        return (
            f"TaskState("
            f"id={self.lora_int_id}, "
            f"name={self.lora_name}, "
            f"workflow={self.workflow}, "
            f"rollouts={self.current_rollouts}/{self.max_staleness})"
        )


class MultiTaskManager:
    def __init__(self):
        self.tasks = {}
        self.lock = Lock()
        self.task_mapping = {}

    def has_running_tasks(self) -> bool:
        for task_id in self.tasks:
            task = self.get_task(task_id)
            if task._task_status():
                return True
        return False

    def register_task(
        self,
        lora_int_id: int,
        lora_name: str,
        workflow: str,
        valid_workflow: str,
        config,
        train_dataset,
        workflow_kwargs,
        eval_workflow_kwargs,
        max_staleness: int = 32,
        valid_dataset=None,
        num_steps=100,
    ):
        with self.lock:
            if lora_int_id in self.tasks:
                raise ValueError(f"Task {lora_int_id} already registered")

            self.tasks[lora_int_id] = TaskState(
                lora_int_id=lora_int_id,
                lora_name=lora_name,
                workflow=workflow,
                valid_workflow=valid_workflow,
                config=config,
                train_dataset=train_dataset,
                workflow_kwargs=workflow_kwargs,
                eval_workflow_kwargs=eval_workflow_kwargs,
                max_staleness=max_staleness,
                valid_dataset=valid_dataset,
                num_steps=num_steps,
            )
            self.task_mapping[lora_name] = lora_int_id

    def get_task(self, task_id) -> TaskState:
        if isinstance(task_id, str):
            task_id = self.task_mapping[task_id]
        return self.tasks[task_id]

    # Staleness API
    def can_submit(self, lora_int_id: int, n: int = 1) -> bool:
        return self.get_task(lora_int_id).can_submit(n)

    def register_submission(self, lora_int_id: int, n: int = 1):
        self.get_task(lora_int_id).register_submission(n)

    def register_completion(self, lora_int_id: int, n: int = 1):
        self.get_task(lora_int_id).register_completion(n)

    def get_remaining_capacity(self, lora_int_id: int) -> int:
        return self.get_task(lora_int_id).remaining_capacity()

    def summary(self):
        return {k: repr(v) for k, v in self.tasks.items()}

    def __repr__(self):
        with self.lock:
            if not self.tasks:
                return "MultiTaskManager(num_tasks=0)"

            task_lines = []
            total_rollouts = 0
            total_capacity = 0

            for task in self.tasks.values():
                with task.lock:
                    total_rollouts += task.current_rollouts
                    total_capacity += task.max_staleness
                    task_lines.append(
                        f"{task.lora_name}"
                        f"(id={task.lora_int_id}, "
                        f"rollouts={task.current_rollouts}/{task.max_staleness})"
                    )

            task_str = ", ".join(task_lines)

            return (
                f"MultiTaskManager("
                f"num_tasks={len(self.tasks)}, "
                f"total_rollouts={total_rollouts}/{total_capacity}, "
                f"tasks=[{task_str}]"
                f")"
            )