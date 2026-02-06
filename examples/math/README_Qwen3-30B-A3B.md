## Running Qwen3-30B-A3B on Multiple Nodes with BOBA Workload

This guide describes how to run **Qwen3-30B-A3B** on multiple nodes using the **BOBA**
workload with **Ray**.

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
- `allocation_model`

Ensure all values match your cluster setup.

______________________________________________________________________

### 3. Launch the BOBA Workload

#### Single Controller Mode

```bash
python3 boba_grpo.py \
  --config examples/math/boba_grpo_megatron.yaml
```

#### SPMD Mode (Ray Launcher)

```bash
python3 -m areal.launcher.ray boba_grpo.py \
  --config examples/math/boba_grpo_megatron.yaml
```
