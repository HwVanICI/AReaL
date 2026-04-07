# pass@k Dump Schema

The offline `pass@k` tool depends on a small subset of dumped JSONL fields.

## Required Fields

- `task_id`
- `sample_idx`
- `reward`
- `tail_version`

## Optional Fields

- `prompt`
- `completion`
- `head_version`

## Semantic Contract

The tool interprets records as follows:

- One `(tail_version, task_id)` group represents one dataset item.
- Multiple rows with different `sample_idx` values inside that group represent
  multiple sampled candidate solutions for that dataset item.
- `reward` must either:
  - already be binary, or
  - be convertible to success/failure using `success_threshold`

## Important Caveat for Agent Workflows

For `agent_workflow`, this contract is valid only if each dumped row represents one
independent candidate solution. If the dump contains intermediate tool calls, retries,
or internal reasoning steps as separate rows, you must first define how those rows
should be folded into one final candidate solution per task.
