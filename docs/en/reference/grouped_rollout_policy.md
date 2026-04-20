# Grouped Rollout Policy Reference

This document explains the two grouped rollout configuration fields:

- `rollout.grouped_result_policy`
- `rollout.grouped_expected_samples_per_subrollout`

They control:

1. how invalid sub-rollout results are handled inside a grouped task;
2. how many samples a sub-rollout is expected to emit;
3. whether incomplete grouped tasks are dropped entirely or partially retained.

## When these configs take effect

They only matter when grouped rollout is enabled, for example:

- `gconfig.n_samples > 1`, or
- `prepare_batch(..., group_size > 1)` / `rollout_batch(..., group_size > 1)`.

If `group_size == 1`, these options are effectively inactive.

## Config fields

### `rollout.grouped_result_policy`

Choices:

- `drop_group`
- `allow_partial`

#### `drop_group`

Semantics:

- If any sub-rollout inside a grouped task is invalid, the whole grouped task returns `None`
- The outer executor treats it as rejected
- When `dynamic_bs=false`, the system keeps fetching new grouped tasks until enough accepted tasks are collected

The following are treated as invalid:

- `None`
- empty result
- exception
- sample count mismatch against `grouped_expected_samples_per_subrollout`

Recommended when:

- you want strict group semantics;
- training depends on complete groups;
- each sub-rollout is expected to emit a fixed number of samples, typically 1 in `concat` mode.

#### `allow_partial`

Semantics:

- Invalid sub-rollouts are filtered out
- Valid sub-rollouts are kept and concatenated
- Each emitted sample carries grouped metadata:
  - `group_ids`
  - `group_expected_size`
  - `group_actual_size`
  - `group_is_complete`

Recommended when:

- partial grouped tasks are acceptable;
- or sub-rollouts naturally emit variable numbers of samples.

### `rollout.grouped_expected_samples_per_subrollout`

Type:

- positive integer
- or `null`

Semantics:

- declares how many samples a sub-rollout is expected to emit under normal conditions;
- it does not control sampling, only grouped-result validation.

#### Set it to `1`

Use this when:

- `rollout.openai.export_style=concat`
- a sub-rollout is expected to yield exactly one training sample

#### Set it to `null`

Use this when:

- output cardinality is intentionally variable;
- `rollout.openai.export_style=individual`;
- multi-leaf / multi-interaction outputs are expected.

## Recommended configurations

### Scenario 1: strict complete groups

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: drop_group
  grouped_expected_samples_per_subrollout: 1
```

### Scenario 2: tolerate missing sub-rollouts, but still expect 1 sample

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: 1
```

### Scenario 3: variable-size sub-rollout output is expected

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: null
```

## Interaction with `dynamic_bs`

### `drop_group + dynamic_bs=false`

This is the most stable combination:

- bad grouped tasks are rejected;
- the scheduler fetches replacement groups;
- accepted batches are more likely to preserve complete-group semantics.

### `allow_partial + dynamic_bs=false`

Accepted task count is still replenished, but individual grouped tasks may contribute variable numbers of samples.

### `allow_partial + dynamic_bs=true`

This is the loosest and most variable combination.

## Interaction with reward / advantage normalization

If you use group-based normalization, for example:

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
```

Grouped rollout attaches `group_ids`, and normalization regroups by `group_ids`.

Current implementation assumes:

- samples belonging to the same `group_id` remain contiguous in batch order.

The built-in grouped rollout path preserves this property. Do not arbitrarily shuffle samples within a group before normalization.

## Runtime metrics

Grouped rollout surfaces the following actor-side metrics:

- `ppo_actor/grouped_n_groups`
- `ppo_actor/grouped_complete_ratio/*`
- `ppo_actor/grouped_expected_size/*`
- `ppo_actor/grouped_actual_size/*`
- `ppo_actor/grouped_partial_groups`

## Common templates

### Agent workflow + concat + strict complete groups

```yaml
gconfig:
  n_samples: 4

rollout:
  openai:
    export_style: concat
  grouped_result_policy: drop_group
  grouped_expected_samples_per_subrollout: 1
```

### Agent workflow + individual + variable-size outputs

```yaml
gconfig:
  n_samples: 4

rollout:
  openai:
    export_style: individual
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: null
```

## Related docs

- [RolloutWorkflow Reference](./rollout_workflow.md)
- [Agent Workflow Reference](./agent_workflow.md)
- [PPO, GRPO, and Related Algorithms](../algorithms/grpo_series.md)
