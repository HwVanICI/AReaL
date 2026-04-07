# Eval Only

This directory provides standardized `eval_only` entrypoints that submit validation
or test samples to an eval rollout controller and dump trajectories into
`eval-rollout/` without running PPO training.

## Entry Points

### General Workflow

Use this for standard rollout workflows such as:

- `areal.workflow.rlvr.RLVRWorkflow`
- `areal.workflow.vision_rlvr.VisionRLVRWorkflow`

```bash
python -m offline_eval.eval_only.general_eval \
  --config offline_eval/eval_only/examples/general_eval_only.yaml \
  --max-items 8
```

The bundled general example is tuned for a local single-GPU demo rather than the
smallest possible smoke test. Short `max_new_tokens` often truncates GSM8K answers
and drives reward to zero; very small `max-items` also makes pass@k estimates noisy.

### Agent Workflow

Use this for agent-style workflows such as:

- `examples.openai_agents.train_agents.OpenAIAgentWorkflow`
- `areal.workflow.openai.*`
- `areal.workflow.openai_agent.*`

```bash
OPENAI_API_KEY=placeholder \
python -m offline_eval.eval_only.agent_eval \
  --config offline_eval/eval_only/examples/agent_openai_eval_only.yaml \
  --max-items 4
```

Recommended first choice:

- [examples/agent_openai_eval_only.yaml](./examples/agent_openai_eval_only.yaml)

This uses `areal.workflow.openai.math_agent.MathAgent`, which is the lightest
agent-style path we validated locally for the offline eval-only flow.

Because this workflow uses the OpenAI Python SDK, the simplest way to satisfy the
client-side API key requirement is to set a dummy value such as
`OPENAI_API_KEY=placeholder`.

Heavier examples such as [examples/agent_eval_only.yaml](./examples/agent_eval_only.yaml)
still exist, but they depend on a richer proxy/OpenAI-agents stack and are less
robust as a first smoke test.

## Config Contract

The eval-only scripts load a dedicated `EvalOnlyConfig`. The minimal config needs:

- `experiment_name`
- `trial_name`
- `tokenizer_path`
- `workflow`
- `workflow_kwargs`
- `valid_dataset`
- `rollout`
- `gconfig`
- `scheduler`
- `allocation_mode`
- `sglang` or `vllm`

Optional fields:

- `eval_workflow`
- `eval_workflow_kwargs`
- `processor_path`
- `eval_split`

## Important Settings

### Dump Output

Make sure:

```yaml
rollout:
  dump_to_file: true
```

The scripts always submit with `is_eval=True`, so trajectories are written into the
`eval-rollout/` subdirectory.

## Custom `eval_only.py`

If a user writes their own `eval_only.py`, YAML alone does not guarantee that the
resulting JSONL will be usable by `pass@k`.

The distinction is:

- If the custom script still uses AReaL's standard rollout dump path, then the YAML
  is usually enough.
- If the custom script bypasses AReaL's dump path and writes JSONL itself, then the
  script must satisfy the dump schema explicitly.

### Case 1: Custom Script Still Uses AReaL Dumping

If the script eventually does the equivalent of:

- create a rollout controller
- call `controller.submit(..., is_eval=True)`
- enable `rollout.dump_to_file: true`
- wait for completion through the normal controller / workflow executor path

then AReaL will automatically dump standard JSONL under:

```text
<fileroot>/logs/<user>/<experiment_name>/<trial_name>/eval-rollout/
```

In this case, the main requirements are:

- `rollout.dump_to_file: true`
- `is_eval=True` on submitted requests
- `gconfig.n_samples > 1` if you want meaningful `pass@k`
- workflow output must still be a standard AReaL trajectory, not an arbitrary dict

If those conditions hold, a custom script can still be analyzed later with:

```bash
python -m offline_eval.pass_at_k.cli \
  --source <eval-rollout-dir> \
  --output-dir <analysis-dir> \
  --emit-csv \
  --emit-json \
  --emit-task-csv
```

### Case 2: Custom Script Writes JSONL Itself

If the script does not go through AReaL's standard dump path, then the YAML does not
automatically make the output valid. The emitted JSONL must satisfy the schema in
[../pass_at_k/SCHEMA.md](../pass_at_k/SCHEMA.md).

The minimum required fields are:

- `task_id`
- `sample_idx`
- `reward`
- `tail_version`

The required semantics are:

- one `(tail_version, task_id)` group = one dataset problem
- different `sample_idx` rows inside that group = different sampled candidate answers
- `reward` should already be binary, or be convertible later with `--success-threshold`

Optional but strongly recommended fields are:

- `prompt`
- `completion`
- `head_version`

These are not required by `pass@k`, but they make debugging much easier.

### Agent / Multi-Turn Caveat

For agent-style or multi-turn workflows, one dumped row must correspond to one final
candidate solution.

Do not dump intermediate tool calls, retries, or internal reasoning steps as if they
were independent samples unless you also define a folding rule first. Otherwise
`pass@k` will over-count candidates and the result will be meaningless.

### Practical Checklist

Before running later analysis, verify:

- the files are under `eval-rollout/` or another directory passed to `--source`
- each JSONL row has `task_id`, `sample_idx`, `reward`, and `tail_version`
- the same question is grouped by `(tail_version, task_id)`
- multiple candidate answers are separated by `sample_idx`
- rewards are binary, or you know the threshold you want to apply
- for agent workflows, each row is a final candidate, not an intermediate trace

### Workflow Kwargs

`workflow_kwargs` can reference top-level config nodes such as `${gconfig}` and
`${tokenizer_path}`. For example:

```yaml
workflow_kwargs:
  reward_fn: areal.reward.gsm8k.gsm8k_reward_fn
  gconfig: ${gconfig}
  tokenizer: ${tokenizer_path}
  enable_thinking: false
```

### Agent Workflow Notes

For agent workflows, the most important requirement is semantic clarity of dumped
records. The `pass@k` tool assumes each `(tail_version, task_id)` corresponds to one
problem, and each emitted `sample_idx` corresponds to one candidate solution.

Be careful with:

- `rollout.openai.export_style=individual`
- multi-turn agent runs that emit multiple intermediate interactions

For the MVP, prefer workflows where a single task ends with one final reward-bearing
candidate solution.

## Example YAML

See:

- [examples/general_eval_only.yaml](./examples/general_eval_only.yaml)
- [examples/agent_eval_only.yaml](./examples/agent_eval_only.yaml)
- [examples/agent_openai_eval_only.yaml](./examples/agent_openai_eval_only.yaml)
