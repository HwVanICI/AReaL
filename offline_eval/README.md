# Offline Eval

This directory contains two layers of offline evaluation utilities:

1. `eval_only/`
   Generate standardized `eval-rollout` dumps without launching PPO training.
2. `pass_at_k/`
   Read dumped JSONL records and aggregate dataset-level `pass@k`.

If you write your own `eval_only.py`, see
[eval_only/README.md](./eval_only/README.md) for the exact conditions under which a
custom script will still produce JSONL that `pass_at_k` can consume directly.

## Typical Flow

1. Run `eval_only` to generate dump files under:

```text
<fileroot>/logs/<user>/<experiment_name>/<trial_name>/eval-rollout/
```

2. Run the `pass_at_k` CLI on that directory:

```bash
python -m offline_eval.pass_at_k.cli \
  --source /tmp/areal/experiments/logs/$USER/gsm8k-eval-only/trial0/eval-rollout \
  --output-dir /tmp/pass_at_k_out \
  --emit-csv \
  --emit-json \
  --plot
```

## Entry Points

### General Workflow

```bash
python -m offline_eval.eval_only.general_eval \
  --config offline_eval/eval_only/examples/general_eval_only.yaml \
  --max-items 32
```

### Agent Workflow

```bash
OPENAI_API_KEY=placeholder \
python -m offline_eval.eval_only.agent_eval \
  --config offline_eval/eval_only/examples/agent_openai_eval_only.yaml \
  --max-items 4
```

See [eval_only/README.md](./eval_only/README.md) and
[pass_at_k/README.md](./pass_at_k/README.md) for details.
