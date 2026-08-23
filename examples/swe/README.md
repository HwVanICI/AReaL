# SWE-bench RL training with AReaL-SWEAgent

This example runs SWE-bench coding-agent RL (GRPO) in AReaL. The actual agent loop,
sandboxing and reward computation live in the bundled or external
[AReaL-SWEAgent](https://github.com/areal-project/AReaL-SWEAgent): for each SWE-bench
instance it edits code inside an isolated sandbox, grades the patch, and returns a
reward for the RL loop. AReaL serves the policy being trained and drives rollouts
through its OpenAI-compatible proxy.

```
AReaL (this repo)                         AReaL-SWEAgent (separate checkout)
  PPOTrainer / rollout proxy  ──▶  run_agent_with_reward()  ──▶  sandbox + grade
        ▲                                                              │
        └──────────────── reward, stats ◀──────────────────────────────┘
```

## 1. Get AReaL-SWEAgent

This checkout includes `AReaL-SWEAgent/`. A sibling checkout remains supported. Install
the extra for the sandbox backend into the environment used by every AReaL worker:

```bash
cd AReaL-SWEAgent
pip install -e ".[aenv]"  # or: pip install -e ".[e2b]"
```

AReaL imports it as the Python package `aweagent`; the repository directory name
(`AReaL-SWEAgent`) and the package name (`aweagent`) intentionally differ.

## 2. Point AReaL at the checkout

AReaL discovers the checkout in this order (first match wins):

1. `econfig.agent_root` in your YAML (legacy alias: `econfig.swe_agent_root`).
1. `AWEAGENT_ROOT` / `SWE_AGENT_ROOT` environment variable.
1. `AReaL-SWEAgent` under the AReaL repo root, then a sibling checkout.

In a multi-node / containerized setup the checkout must be importable on every worker,
so put it on shared storage and set both the env var and `PYTHONPATH`, e.g. in the
worker `scheduling_spec.env_vars`:

```yaml
env_vars:
  AWEAGENT_ROOT: "/path/to/AReaL-SWEAgent"
  SWE_AGENT_ROOT: "/path/to/AReaL-SWEAgent"
  PYTHONPATH: "/path/to/AReaL-SWEAgent:/path/to/AReaL"
  # URL of the AEnvironment sandbox service used by AReaL-SWEAgent.
  AENV_SYSTEM_URL: "http://your-aenv-service:8080"
```

## 3. Configure the agent (econfig)

The SWE example reads an `econfig` block (see `examples/swe/utils.py`):

| Field                   | Meaning                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `agent_type`            | Agent to run: `swe` (built-in tool-use), `cc` (Claude Code), `codex`, or `opencode`.                        |
| `sandbox_backend`       | `aenv` or `e2b`; E2B supports `cc`, `codex`, and `opencode`, while AEnv supports `swe` and `cc`.            |
| `agent_config`          | Generic config name under `AReaL-SWEAgent/aweagent/configs/`. Overrides the per-type fields below when set. |
| `swe_agent_config`      | Config used when `agent_type=swe` (default `1_0_0/min-swe-agent-train-top1`).                               |
| `cc_agent_config`       | Config used when `agent_type=cc`.                                                                           |
| `codex_agent_config`    | Config used when `agent_type=codex` (default `train_codex_time3600`).                                       |
| `opencode_agent_config` | Config used when `agent_type=opencode` (default `train_opencode_time3600`).                                 |
| `agent_root`            | Path to the AReaL-SWEAgent checkout (see section 2).                                                        |
| `step_limit`            | Max agent interaction steps per episode.                                                                    |
| `max_tokens`            | Total context window advertised to OpenCode.                                                                |
| `max_completion_tokens` | Max completion tokens per agent LLM call.                                                                   |
| `timeout`               | Max wall-clock time per episode (seconds).                                                                  |

## 4. Dataset format

`train_swe_rl.py` loads a JSONL file (`train_dataset.path`) where each line is a
SWE-bench instance with at least:

- `instance_id` — e.g. `django__django-10097`
- `problem_statement` — the GitHub issue text
- `eval_script` — shell script used by AReaL-SWEAgent to grade the patch
- E2B only: `image` (metadata-routed OCI image) and optional `workdir` (default
  `/testbed`)

## 5. Run

Edit the placeholder paths in `qwen3_30b_a3b_grpo.yaml` (model, dataset,
`AWEAGENT_ROOT`, caches, sandbox/W&B endpoints), then launch:

```bash
python -m examples.swe.train_swe_rl --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

With `scheduler.type=slurm`, AReaL launches the rollout / actor / proxy workers; each
rollout calls into AReaL-SWEAgent, which runs the agent in a sandbox and returns the
reward.

To run the same dataset and workflow without launching the actor or performing any
training, use the rollout-only entry point with the same YAML and overrides:

```bash
python -m examples.swe.rollout_swe --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

This launches only the rollout engine and proxy workers. It traverses the dataset
exactly once in source order, including the final partial batch, regardless of the
dataset `shuffle` setting. It ignores the training-only `total_train_epochs`,
`total_train_steps`, and rollout staleness limits. Trajectories are written when
`rollout.dump_to_file=true`.

## 6. Set up the AEnvironment backend

`AENV_SYSTEM_URL` (section 2) must point at a running
[AEnvironment](https://github.com/inclusionAI/AEnvironment) platform. AReaL never talks
to a sandbox directly — AReaL-SWEAgent asks the AEnvironment API service to spin up two
kinds of environment per rollout:

| Role                                                                    | Environment AReaL-SWEAgent requests                          | Ships in the AEnvironment repo?               |
| ----------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **Task sandbox** — where the agent runs shell commands and edits code   | `persistent-bash-env` (swe agent) / `cc-bash-env` (cc agent) | No — provided by your AEnvironment deployment |
| **Grading sandbox** — where the patch is scored resolved / not-resolved | `swebench` (overridable via `SWEBENCH_EVAL_ENV`)             | Yes — `aenv/builtin-envs/swebench`            |

### 6.1 Deploy the platform

AEnvironment deploys on Kubernetes via Helm (see its
[deployment guide](https://github.com/inclusionAI/AEnvironment/blob/main/docs/getting_started/deployment.md)):

```bash
git clone https://github.com/inclusionAI/AEnvironment.git
cd AEnvironment
helm install aenv-platform ./deploy \
  --namespace aenv --create-namespace --wait --timeout 10m
```

The API service is then reachable in-cluster at port `8080` — this is the value for
`AENV_SYSTEM_URL`:

```
AENV_SYSTEM_URL = http://api-service.aenv.svc.cluster.local:8080
```

### 6.2 Register the `swebench` grading environment (guaranteed example)

The grading environment is the reward source and is the one piece shipped in the
AEnvironment repo, so it always works out of the box. Install the CLI, point it at your
deployment, then build and publish the bundled `swebench` environment:

```bash
pip install aenvironment
aenv config init
aenv config set hub_backend http://envhub.aenv.svc.cluster.local:8083/
export AENV_SYSTEM_URL=http://api-service.aenv.svc.cluster.local:8080

cd AEnvironment/aenv/builtin-envs/swebench   # config.json: name=swebench, version=1.0.0
aenv build --push                            # requires a configured image registry
aenv push                                    # publish metadata to EnvHub
aenv list                                    # `swebench` should now appear
```

Its reward entrypoint is `swebench_reward(instance, model_patch, timeout)`, which is
exactly the `call_reward({"instance", "model_patch", "timeout"})` contract
AReaL-SWEAgent uses to grade patches.

> **Version note:** the repo ships `swebench@1.0.0`, while AReaL-SWEAgent defaults to
> `swebench@1.0.4`. Either export `SWEBENCH_EVAL_ENV=swebench@1.0.0` in the worker
> `env_vars`, or bump the `version` in `config.json` before `aenv build`.

### 6.3 Provide the task sandbox

The `persistent-bash-env` / `cc-bash-env` task sandboxes are **not** shipped in the
AEnvironment repo. Use the image already registered in your AEnvironment cluster, or
build your own environment that exposes a shell (the repo's
`aenv/builtin-envs/mini-terminal` is the closest reference) and register it the same way
as section 6.2. AReaL-SWEAgent selects it through the agent config's `aenv_version`
field.

## 7. Train the Claude Code (cc) agent

Section 5 runs the built-in `swe` agent. To train the **Claude Code** agent instead —
Claude Code solves the task in a `cc-bash-env` sandbox and the same `swebench`
environment grades the resulting patch — switch `econfig.agent_type` to `cc` and select
a cc config:

```yaml
econfig:
  agent_type: cc
  cc_agent_config: train_cc_time3600   # a config under AReaL-SWEAgent/aweagent/configs/
  agent_root: /path/to/AReaL-SWEAgent
  timeout: 3600.0
```

The cc agent's LLM calls are still routed through AReaL's OpenAI-compatible proxy (AReaL
injects the per-rollout `base_url`/`api_key`), so training stays on-policy exactly like
the swe path.

### 7.1 Environment variables for cc

In addition to the section 2 variables (`AWEAGENT_ROOT`, `PYTHONPATH`,
`AENV_SYSTEM_URL`), the cc agent reads these from the worker `env_vars` — they are
consumed by the cc config in AReaL-SWEAgent (`aweagent/configs/1_0_0/cc.yaml`), not by
AReaL itself:

| Variable                                 | Purpose                                                    |
| ---------------------------------------- | ---------------------------------------------------------- |
| `SWEBENCH_EVAL_ENV`                      | Grading environment name, e.g. `swebench@1.0.0` (see 6.2). |
| `REMOTE_PROXY_SERVICE_URL`               | Gateway that fronts the Claude CLI traffic (proxy mode).   |
| `REMOTE_PROXY_API_KEY`                   | API key for that proxy gateway.                            |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` | Used instead when running the CLI in direct mode.          |

The concrete agent behaviour — system prompt, `cli_flags`, tool allow-list,
thinking-token budgets — lives entirely in the AReaL-SWEAgent cc config; see that
repository for the available `cc_agent_config` values and their fields.

## 8. Use the E2B-compatible Claude Code backend

This path does not import or call AEnvironment. It follows Slime's metadata-routed E2B
API and starts a remote sandbox for Claude Code plus a fresh sandbox for grading:

```bash
export SWE_AGENT_TYPE=cc
export SWE_SANDBOX_BACKEND=e2b
export E2B_API_KEY=...
export AWEAGENT_E2B_IMAGE_METADATA_KEY=your.gateway/image
export AWEAGENT_NODE_TARBALL=/shared/node-v22-linux-x64.tar.xz
export AWEAGENT_CC_TARBALL=/shared/claude-code.tgz

python -m examples.swe.train_swe_rl \
  --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

The E2B sandbox must be able to connect to the `base_url` advertised by AReaL's proxy.
Claude Code receives that URL and the trajectory-specific API key as
`ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`; requests to `/v1/messages` are recorded by
AReaL and become the on-policy trajectory used by GRPO.

## 9. Use the E2B-compatible Codex backend

Codex uses the same E2B task metadata, fresh grading sandbox, AReaL workflow, and
training YAML as section 8. Select the Codex agent and provide its npm tarball:

```bash
export SWE_AGENT_TYPE=codex
export SWE_SANDBOX_BACKEND=e2b
export E2B_API_KEY=...
export AWEAGENT_E2B_IMAGE_METADATA_KEY=your.gateway/image
export AWEAGENT_NODE_TARBALL=/shared/node-v22-linux-x64.tar.xz
export AWEAGENT_CODEX_TARBALL=/shared/codex.tgz

python -m examples.swe.train_swe_rl \
  --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

`E2BCodex` writes a custom provider into `/home/agent/.codex/config.toml`, then runs
`codex exec --skip-git-repo-check` in the task repository. The provider's base URL is
the trajectory-specific AReaL proxy URL with `/v1` appended, and the proxy session key
is passed as `OPENAI_API_KEY`. As with Claude Code, the E2B sandbox must be able to
reach the proxy URL directly.

## 10. Use the E2B-compatible OpenCode backend

OpenCode uses the same E2B task metadata, clean grading sandbox, workflow, and training
YAML. Select `opencode` and provide the npm package tarball:

```bash
export SWE_AGENT_TYPE=opencode
export SWE_SANDBOX_BACKEND=e2b
export E2B_API_KEY=...
export AWEAGENT_E2B_IMAGE_METADATA_KEY=your.gateway/image
export AWEAGENT_NODE_TARBALL=/shared/node-v22-linux-x64.tar.xz
export AWEAGENT_OPENCODE_TARBALL=/shared/opencode-ai.tgz

python -m examples.swe.train_swe_rl \
  --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

`E2BOpenCode` configures an `@ai-sdk/openai-compatible` provider and runs
`opencode run --model <provider>/<model>` in the task repository. The provider uses the
trajectory-specific AReaL proxy URL and `OPENAI_API_KEY`; the E2B sandbox must be able
to reach that URL directly.
