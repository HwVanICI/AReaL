# pass@k

This tool reads dumped `eval-rollout` JSONL files and aggregates dataset-level
`pass@k` by checkpoint version (`tail_version`).

## What It Assumes

- One `(tail_version, task_id)` corresponds to one dataset item.
- Multiple `sample_idx` values under the same `(tail_version, task_id)` correspond to
  multiple sampled candidate solutions for that item.
- `reward` is either binary (`0/1`) or can be converted to binary with
  `--success-threshold`.

## Basic Usage

```bash
python -m offline_eval.pass_at_k.cli \
  --source /tmp/areal/experiments/logs/$USER/gsm8k-eval-only/trial0/eval-rollout
```

## Export Summary Files

```bash
python -m offline_eval.pass_at_k.cli \
  --source /tmp/areal/experiments/logs/$USER/gsm8k-eval-only/trial0/eval-rollout \
  --output-dir /tmp/pass_at_k_out \
  --emit-csv \
  --emit-json
```

## Plot pass@k

```bash
python -m offline_eval.pass_at_k.cli \
  --source /tmp/areal/experiments/logs/$USER/gsm8k-eval-only/trial0/eval-rollout \
  --output-dir /tmp/pass_at_k_out \
  --emit-csv \
  --emit-json \
  --plot
```

The generated plot uses:

- x-axis: `k`
- y-axis: `pass@k`

If one source contains multiple `tail_version` summaries, each version is plotted as a
separate curve.

## Plot Multiple Experiments Together

```bash
python -m offline_eval.pass_at_k.cli \
  --source \
    /tmp/exp_a/eval-rollout \
    /tmp/exp_b/eval-rollout \
  --labels exp_a,exp_b \
  --output-dir /tmp/pass_at_k_compare \
  --emit-csv \
  --emit-json \
  --plot
```

When multiple sources are provided:

- the CLI prints and exports a combined summary table
- `summary.csv` / `summary.json` include a `source` column
- the plot overlays one curve per source, or one curve per `(source, tail_version)`
  when a source contains multiple versions

## Key Flags

- `--ks 1,2,4,8`
- `--success-threshold 0.5`
- `--strict`
- `--emit-task-csv`

If `--output-dir` is provided and no explicit emit flags are set, the CLI writes:

- `summary.csv`
- `summary.json`

If `--plot` is set, it also writes:

- `pass_at_k.png`

If `--emit-task-csv` is set, it also writes:

- `task_metrics.csv`

## Agent Workflow Boundary

`agent_workflow` is supported only when dumped rows truly correspond to independent
candidate solutions.

Use caution when:

- one task contains multiple intermediate interactions
- `export_style=individual`
- rewards are attached to internal tool or reasoning steps

In those cases, you may need to change the export style or only keep the final
reward-bearing candidate before using this tool.

See [SCHEMA.md](./SCHEMA.md) for the exact record contract.
