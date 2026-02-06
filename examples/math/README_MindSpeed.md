## AReaL with MindSpeed on Ascend NPU

This tutorial describes how to run multi-node GRPO training with the **MindSpeed**
backend using the **Qwen3-30B-A3B** model and the **Boba** dataset on Ascend NPU.

______________________________________________________________________

### 1. Initialize Ray on All Nodes

#### Head Node

```bash
ray start --head
```

#### Worker Nodes

```bash
# Replace with the IP address of the head node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

______________________________________________________________________

### 2. Update Configuration (YAML)

Before launching, update the YAML configuration file as needed.

**Update:**

- `actor.path`
- `train_dataset.path`

**If using a different number of nodes, also update:**

- `n_nodes`
- `allocation_mode`

Ensure all values match your cluster setup.

______________________________________________________________________

### 3. Launch the Boba Training Workload

#### Single-Controller Mode

```bash
python3 boba_grpo.py \
  --config examples/math/boba_grpo_megatron_npu.yaml
```

#### SPMD Mode

```bash
python3 -m areal.launcher.ray boba_grpo.py \
  --config examples/math/boba_grpo_megatron_npu.yaml
```
