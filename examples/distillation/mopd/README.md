# Multi-teacher on-policy distillation with AIME and LeetCode

This example demonstrates multi-teacher on-policy distillation across mathematical
reasoning and code-generation tasks.

The provided configuration was tested on two Atlas A3 nodes with a total of 32 Ascend
910C dies. Adapt `n_nodes`, `n_gpus_per_node`, engine backends, and scheduling
specifications for your hardware.

This example implements
[MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training](https://arxiv.org/abs/2606.30406).

## How the example works

### Domain routing

Every source in `mopd_multi.yaml` has a `domain` value. Dataset loading preserves this
value on every sample, and `DomainRouterRolloutWorkflow` uses it to select the matching
rollout workflow:

- `domain: aime` uses `RLVRWorkflow` and the AIME mathematical reward.
- `domain: leetcode` uses `CodeAgent` and executes the generated solution against the
  test cases stored in the sample.

The router also attaches the domain to the resulting trajectory. Because
`teacher.routing_mode` is `domain`, the trainer sends each trajectory only to the
teacher with the same `teacher.teachers[].domain`. The AIME teacher therefore scores
only AIME trajectories, while the LeetCode teacher scores only LeetCode trajectories.
This differs from `routing_mode: mixture`, where every teacher scores every trajectory
and AReaL combines their token log-probabilities using a weighted log-sum-exp.

The domain names must match exactly across these three places:

```text
dataset source domain → workflow routing key → teacher domain
```

### MOPD algorithm and loss

MOPD performs on-policy distillation: the student generates each trajectory, and the
domain expert evaluates the probability of the same generated tokens. For each response
token, AReaL computes the stopped-gradient teacher-student log-probability gap

$$
A_t^{\text{MOPD}} =
\operatorname{clip}\left(
\log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t),
-c,
c
\right)
$$

where `c` is `teacher.mopd_adv_clip`. The implementation applies an importance ratio
between the current policy and the rollout policy, producing the per-token
policy-gradient loss

$$
L_{\text{MOPD}} =
-\mathbb{E}_t\left[
\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
A_t^{\text{MOPD}}
\right]
$$

The final actor objective is

$$
L = w_{\text{RL}} L_{\text{RL}} + w_{\text{distill}} L_{\text{MOPD}}.
$$

In the supplied configuration, `rl_loss_weight: 0.0`,
`distill_loss_weight: 1.0`, and `distill_loss_type: mopd_pg`, so actor updates use only
the MOPD distillation signal. Set a positive `rl_loss_weight` to combine the task reward
objective with distillation.

### Training workflow

One training iteration follows this sequence:

1. The mixed dataset yields an AIME or LeetCode sample tagged with its domain.
1. `DomainRouterRolloutWorkflow` selects the corresponding rollout implementation.
1. The student produces an on-policy trajectory and the domain workflow computes its
   task reward.
1. The trainer routes the trajectory to the matching frozen teacher and obtains
   token-level `teacher_logp` values.
1. The actor computes the MOPD advantage and updates the student using `mopd_pg`.
1. Metrics, checkpoints, and evaluation are handled by `PPOTrainer` using the common
   utility configuration in `mopd_multi.yaml`.

The task reward is still generated and logged when `rl_loss_weight` is zero, but it does
not contribute to the actor loss.

## Dataset preparation

The example uses AIME 2025 for training, AIME 2026 for evaluation, and the medium
difficulty subset of the LeetCode dataset. Create separate AIME and LeetCode directories,
place each preparation script in its corresponding directory, and run it from there.

### AIME

Save the following as `prepare_aime.py`, then run `python prepare_aime.py` from the
AIME data directory:

```python
import pandas as pd
from datasets import Dataset
import subprocess
from pathlib import Path


aime26_url = "https://huggingface.co/datasets/MathArena/aime_2026/resolve/main/data/train-00000-of-00001.parquet"
aime25_url = "https://huggingface.co/datasets/MathArena/aime_2025/resolve/main/data/train-00000-of-00001.parquet"


def download(url, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-q", url, "-O", out], check=True)


def process(path, out):

    df = pd.read_parquet(path)

    df = df.rename(columns={
        "problem": "question",
    })

    df = df[["question", "answer"]]

    dataset = Dataset.from_pandas(df)
    dataset.to_parquet(out)
    print(f"Prepared {len(dataset)} rows in {out}")


# run
download(aime25_url, "aime25.parquet")
download(aime26_url, "aime26.parquet")

process("aime25.parquet", "aime_train.parquet")
process("aime26.parquet", "aime_test.parquet")
```

### LeetCode

Save the following as `prepare_leetcode.py`, then run `python prepare_leetcode.py` from
the LeetCode data directory:

```python
import pandas as pd
from datasets import Dataset
import subprocess
import re
import ast

# Download dataset
train_url = "https://huggingface.co/datasets/newfacade/LeetCodeDataset/resolve/main/LeetCodeDataset-train.jsonl"
test_url = "https://huggingface.co/datasets/newfacade/LeetCodeDataset/resolve/main/LeetCodeDataset-test.jsonl"

# REQUIRED_MODULES = ["heapq", "collections", "math", "bisect", "itertools"]

ALLOWED_IMPORTS = {
    "math", "heapq", "itertools", "collections", "functools",
    "string", "bisect", "operator", "random", "datetime",
    "typing"
}
SUBSETS = ["Medium"]


FLOAT_ASSERT_RE = re.compile(
    r"assert\s+(.*?)==\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)"
)


SKIP_ASSERT_KEYWORDS = [
    "tree_node",
    "is_same_tree",
    "list_node",
    "is_same_list",
    "linked_list_to_list"
]


LIST_COMPARATOR = """
def _eq_list(a, b):
    if a == b:
        return True

    if isinstance(a, list) and isinstance(b, list):
        try:
            def normalize(x):
                if isinstance(x, list):
                    return tuple(sorted(normalize(i) for i in x))
                return x

            return sorted(normalize(i) for i in a) == sorted(normalize(i) for i in b)
        except:
            pass

    return False
"""


LIST_ASSERT_RE = re.compile(
    r"assert\s+(.*?)==\s*(\[[^\n]*\])"
)
def should_skip_assert(lhs: str, rhs: str) -> bool:
    text = lhs + " " + rhs
    return any(k in text for k in SKIP_ASSERT_KEYWORDS)

def fix_float_asserts(code: str) -> str:
    def repl(match):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()

        if should_skip_assert(lhs, rhs):
            return match.group(0)

        return f"assert abs({lhs} - ({rhs})) <= 1e-6"

    return FLOAT_ASSERT_RE.sub(repl, code)

def fix_list_asserts(code: str) -> str:
    def repl(match):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()

        # skip trees / linked structures
        if should_skip_assert(lhs, rhs):
            return match.group(0)

        return f"assert _eq_list({lhs}, {rhs})"

    return LIST_ASSERT_RE.sub(repl, code)

def ensure_modules(code: str) -> str:
    for mod in ALLOWED_IMPORTS:
        if f"{mod}." in code and f"import {mod}" not in code:
            code = f"import {mod}\n" + code
    return code

def is_inplace_problem(starter_code: str) -> bool:
    if not isinstance(starter_code, str):
        return False
    return "-> None" in starter_code

# remove unallowed imports from the testing code
# primarily just sortedcontainers
def filter_imports(code: str) -> str:
    tree = ast.parse(code)

    new_body = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
            if all(name in ALLOWED_IMPORTS for name in names):
                new_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            module = node.module.split(".")[0] if node.module else ""
            if module in ALLOWED_IMPORTS:
                new_body.append(node)

        else:
            new_body.append(node)

    tree.body = new_body
    return ast.unparse(tree)


def clean_imports(code: str) -> str:
    try:
        code = filter_imports(code)
    except Exception:
        return ""
    # code = ensure_modules(code)
    return code

def download(url):
    subprocess.run(["wget", "-q", url, "-O", url.split("/")[-1]], check=True)


def add_test_check(row):
    test_str = row["tests"]
    entry = row["entry_point"]

    if not isinstance(test_str, str):
        return ""

    if "def test_check():" in test_str:
        return test_str

    if not test_str.endswith("\n"):
        test_str += "\n"

    # 1. floats
    test_str = fix_float_asserts(test_str)

    # 2. lists
    new_test_str = fix_list_asserts(test_str)

    needs_list_comparator = new_test_str != test_str

    test_str = new_test_str

    # 3. prepend comparator if needed
    if needs_list_comparator:
        test_str = LIST_COMPARATOR + "\n" + test_str

    # 4. append test_check
    test_str += f"""
def test_check():
    check({entry})
"""

    return test_str

def strip_use_this_format(text: str) -> str:
    marker = "use this format:"
    idx = text.lower().find(marker)
    if idx != -1:
        return text[:idx].rstrip()
    return text

def process(path, out_name):

    df = pd.read_json(path, lines=True)
    df = df.rename(columns={
        "problem_description": "question",
        "test": "tests"
    })

    print("\n=== Difficulty distribution ===")
    print(df["difficulty"].value_counts())

    df = df[df["difficulty"].isin(SUBSETS)]
    df = df[~df["starter_code"].apply(is_inplace_problem)]

    df["tests"] = df.apply(add_test_check, axis=1)

    df["imports"] = df["prompt"].apply(clean_imports)
    df["question"] = df["question"] + "\nThe following classes and imports are defined: \n" + \
        df["imports"].fillna("") + "\n\nUse this format for the solution: \n" + df["starter_code"].fillna("")
    # df["question"] = df["question"].apply(strip_use_this_format)
    df = df[["question", "tests", "imports"]]
    dataset = Dataset.from_pandas(df)
    dataset.to_parquet(out_name)
    print(f"Prepared {len(dataset)} rows in {out_name}")


# run
download(train_url)
download(test_url)

process("LeetCodeDataset-train.jsonl", "train.parquet")
process("LeetCodeDataset-test.jsonl", "test.parquet")
```

After preparation, the directories must have this layout:

```text
/path/to/aime/
├── aime_train.parquet
└── aime_test.parquet

/path/to/leetcode/
├── train.parquet
└── test.parquet
```

Update the `train_dataset.path` and `valid_dataset.path` values in the single-domain
configurations and the paths under `train_dataset.datasets` and
`valid_dataset.datasets` in `mopd_multi.yaml`.

The mixed training configuration sets `upsample_to_largest: true`. AReaL repeats the
smaller source until both domains contribute the same number of training samples. This
balances domains by sample count; it does not account for differences in response length
or task difficulty.

The LeetCode preparation currently keeps only medium-difficulty problems and excludes
methods whose starter signature returns `None`. It also rewrites selected floating-point
and list assertions, and skips special comparison handling for some tree and linked-list
problems. Each retained test program must define `test_check()`, because the reward
runner calls that function after loading the generated solution and tests.

## Code-execution safety

LeetCode evaluation executes model-generated Python. **The tested configuration uses the
local subprocess evaluator, which is not a security sandbox. Run it only inside an
isolated, disposable environment without credentials or sensitive filesystem access.**

## Train the domain experts

Before launching, update `actor.path` in each configuration. The AIME configuration may
also point to an existing checkpoint if training is being resumed.

The student and teachers must use compatible token IDs. In practice, use checkpoints
from the same model family with the same tokenizer and vocabulary. Teacher scoring is
performed on the token sequence generated by the student, so sharing only similar text
formatting is not sufficient.

Train the AIME expert:

```bash
python examples/math/aime_rl.py --config examples/distillation/mopd/mopd_rl_aime.yaml
```

Train the LeetCode expert:

```bash
python examples/code/train.py --config examples/distillation/mopd/mopd_rl_leetcode.yaml
```

Use a checkpoint that performs well on the corresponding validation set rather than
automatically selecting the last checkpoint. Record the selected checkpoint paths and
evaluation results before starting multi-teacher training.

### Teacher vLLM memory headroom

Use a lower `gpu_memory_utilization` for each teacher than you would for a standalone
inference server. This setting controls vLLM's memory planning, but it is not a strict
upper bound on every allocation. Prefill can require additional transient memory for
model activations, attention workspaces, and communication buffers, especially with long
student trajectories or concurrent scoring requests. Peak usage can therefore exceed
the amount anticipated from the configured utilization target and cause an out-of-memory
error during teacher scoring.

The supplied configuration starts each teacher at:

```yaml
teacher:
  teachers:
    - vllm:
        gpu_memory_utilization: 0.5
        max_num_batched_tokens: 7000
```

Treat these values as starting points rather than universal defaults. If teacher prefill
runs out of memory, lower `gpu_memory_utilization`, `max_num_batched_tokens`, rollout
concurrency, or the maximum sequence length. Increase utilization only after observing
stable peak memory during representative long-sequence prefills. Do not copy the actor's
higher vLLM utilization directly into the teacher configurations without measuring the
teacher workload.

## Run multi-teacher MOPD

Set the two `teacher.teachers[].path` values in `mopd_multi.yaml` to the AIME and
LeetCode expert checkpoints. Also verify `actor.path`, the dataset paths, and the cluster
configuration.

Launch training with:

```bash
python examples/distillation/mopd/mopd_rl.py \
  --config examples/distillation/mopd/mopd_multi.yaml
```

## Preflight checks

Before launching the full two-node run:

1. Load both prepared datasets and confirm that their columns match the expected loader
   inputs.
1. Run a small AIME rollout and verify that the mathematical reward is nonzero for a
   known correct answer.
1. Run a small LeetCode rollout in the isolated execution environment and verify that a
   known correct solution receives reward `1`.
1. Confirm that every trajectory retains either `domain: aime` or `domain: leetcode`.
1. Confirm that each teacher checkpoint can score student token IDs.
1. Start with reduced concurrency and batch sizes before scaling to the supplied
   hardware configuration.

## Metrics to monitor

Monitor both task quality and the distillation signal:

- `rollout/reward/aime` and `rollout/reward/leetcode` for domain-specific rewards.
- `rollout/n_seqs/aime` and `rollout/n_seqs/leetcode` for the effective domain mix.
- `mopd_loss`, `mopd_advantage`, and `mopd_raw_advantage` for the distillation update.
- `mopd_importance_weight` for drift between the current and rollout policies.
- `mopd_clipped_advantage` to detect frequent clipping by `mopd_adv_clip`.

Metric names may receive a trainer-specific prefix in the configured logging backend.

## Troubleshooting

| Symptom                            | Likely cause                                                | Check                                                                     |
| ---------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------- |
| Missing-domain error               | Dataset source was not tagged or the routing key changed    | Ensure every source defines `domain` and `domain_key` remains `domain`    |
| No teacher configured for a domain | Dataset and teacher names differ                            | Match dataset, workflow, and teacher domain strings exactly               |
| LeetCode rewards are always zero   | Invalid generated code, missing imports, or malformed tests | Inspect dumped rollouts and verify that every test defines `test_check()` |
| AIME rewards are always zero       | Answer field or boxed-answer format is incorrect            | Inspect the processed `answer` column and decoded completions             |
| Teacher scoring fails              | Teacher and student token IDs are incompatible              | Use the same tokenizer and vocabulary across checkpoints                  |
| Teacher scoring runs out of memory | Teacher parallelism or concurrency is too aggressive        | Reduce teacher concurrency, batch size, or GPU memory utilization         |
| Validation resembles training      | Validation sources use the wrong split                      | Ensure all entries under `valid_dataset.datasets` use `split: test`       |

## Other Comments

For the general distillation implementation and other teacher modes, see
[`docs/en/algorithms/distillation.md`](../../../docs/en/algorithms/distillation.md).
