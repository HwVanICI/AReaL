# Grouped Rollout Policy Reference

本文档说明 grouped rollout 相关的两个配置：

- `rollout.grouped_result_policy`
- `rollout.grouped_expected_samples_per_subrollout`

它们用于控制：

1. grouped task 中子 rollout 无效时如何处理；
2. 每个子 rollout 正常应当产出多少条样本；
3. 不完整 grouped task 是整组丢弃，还是保留部分结果。

## 什么时候这些配置会生效

它们只在 grouped rollout 启用时生效，也就是：

- `gconfig.n_samples > 1`，或
- 调用 `prepare_batch(..., group_size > 1)` / `rollout_batch(..., group_size > 1)`。

如果 `group_size == 1`，这两个配置基本不参与决策。

## 配置项

### `rollout.grouped_result_policy`

可选值：

- `drop_group`
- `allow_partial`

#### `drop_group`

语义：

- 只要某个 grouped task 中任一 sub-rollout 无效，整个 grouped task 直接返回 `None`
- 外层 executor 会把这组样本视为 rejected
- 当 `dynamic_bs=false` 时，系统会继续补新的 grouped task

以下情况会被视为无效：

- sub-rollout 返回 `None`
- sub-rollout 返回空结果
- sub-rollout 抛异常
- sub-rollout 的样本数不满足 `grouped_expected_samples_per_subrollout`

适合：

- 你希望 group 语义严格；
- 训练依赖完整 group；
- `concat` 模式下，每个 sub-rollout 正常应只产出 1 条样本。

#### `allow_partial`

语义：

- 仅过滤无效 sub-rollout
- 有效 sub-rollout 继续进入训练
- 每条样本附带 grouped metadata：
  - `group_ids`
  - `group_expected_size`
  - `group_actual_size`
  - `group_is_complete`

适合：

- 你允许 grouped task 部分缺样本；
- 或者 sub-rollout 天生就是 variable-size 输出。

### `rollout.grouped_expected_samples_per_subrollout`

类型：

- 正整数
- 或 `null`

语义：

- 声明“每个子 rollout 正常应该产出多少条样本”
- 它不控制采样，只用于 grouped result 校验

#### 设为 `1`

表示：

- 每个 sub-rollout 正常应当产出 1 条样本

典型场景：

- `rollout.openai.export_style=concat`
- 单 leaf agent workflow
- 标准 grouped GRPO / GSPO

#### 设为 `null`

表示：

- 不对每个 sub-rollout 的固定样本条数做检查

典型场景：

- `rollout.openai.export_style=individual`
- 多 interaction / 多 leaf 输出是设计预期

## 推荐配置

### 场景 1：严格完整组

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: drop_group
  grouped_expected_samples_per_subrollout: 1
```

### 场景 2：允许缺样本，但正常预期仍为 1 条

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: 1
```

### 场景 3：输出条数本来就可变

```yaml
gconfig:
  n_samples: 8

rollout:
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: null
```

## 与 `dynamic_bs` 的关系

### `drop_group + dynamic_bs=false`

这是最稳的组合：

- 坏组直接 rejected
- 调度器继续补新的 grouped task
- batch 更容易保持完整 group

### `allow_partial + dynamic_bs=false`

accepted task 数会补齐，但单个 grouped task 的样本数可能变化。

### `allow_partial + dynamic_bs=true`

batch 大小和 group 完整性都会更不稳定。

## 与 reward norm / adv norm 的关系

如果你使用 group-based normalization，例如：

```yaml
actor:
  reward_norm:
    mean_level: group
    std_level: group
    group_size: ${gconfig.n_samples}
```

grouped rollout 会给样本附加 `group_ids`，normalization 会按 `group_ids` regroup。

当前实现要求：

- 同一个 `group_id` 在 batch 中必须保持连续块

内置 grouped rollout 路径会保持这个性质；不要在训练前手动重排同组样本内部顺序。

## 可观测指标

actor 侧会记录：

- `ppo_actor/grouped_n_groups`
- `ppo_actor/grouped_complete_ratio/*`
- `ppo_actor/grouped_expected_size/*`
- `ppo_actor/grouped_actual_size/*`
- `ppo_actor/grouped_partial_groups`

## 常见模板

### Agent workflow + concat + 严格完整组

```yaml
gconfig:
  n_samples: 4

rollout:
  openai:
    export_style: concat
  grouped_result_policy: drop_group
  grouped_expected_samples_per_subrollout: 1
```

### Agent workflow + individual + variable-size 输出

```yaml
gconfig:
  n_samples: 4

rollout:
  openai:
    export_style: individual
  grouped_result_policy: allow_partial
  grouped_expected_samples_per_subrollout: null
```

## 相关文档

- [RolloutWorkflow 参考](./rollout_workflow.md)
- [Agent Workflow 参考](./agent_workflow.md)
- [PPO、GRPO及相关算法](../algorithms/grpo_series.md)
