# OpenHands / R2E-Gym Bring-Up

This path runs the reference OpenHands `CodeActAgent` through AReaL's proxy in an
E2B-compatible sandbox, then evaluates its patch with the official R2E-Gym grader in a
fresh sandbox. It is the first integration stage, not a complete MLPerf Closed
submission recipe. Loss alignment, grouped pass@4, offline checkpoint evaluation and
MLPerf timing/logging remain separate work. The bring-up path rejects reference outputs
with agent errors, including turn or context exhaustion. The reference training recipe
retains some of these samples with a sample-loss mask; that distinction still needs
alignment before a Closed convergence run.

## Runtime Images

Each task image must preserve the official task repository, dependencies and hidden
tests. Add the agent/grader runtime during image preparation, not during each rollout.
Do not replace task dependencies with those of the runtime.

The reference sources used here are:

| Component | Revision                                   |
| --------- | ------------------------------------------ |
| NeMo-RL   | `f987c0596af2bd6cfb9958e768deb8df951d7504` |
| NeMo Gym  | `610a08ab5fe9f8f5fb5fff36b170429ea67f0f92` |
| OpenHands | `0d766ad06b2be64a42e6f0175b9ebcc4a06599d9` |
| R2E-Gym   | `6823e64f94ae645f5265c03af0eb2e8523530a0d` |

Inside the task-image build environment, provide the reference NeMo-RL checkout and the
inner AReaL-SWEAgent checkout, then run:

```bash
bash AReaL-SWEAgent/tools/prepare_mlperf_runtime.sh /path/to/reference/RL
```

This reuses the reference setup scripts, including their network downloads and build
prerequisites. It prepares OpenHands at `/opt/openhands` and R2E-Gym at `/opt/r2e-gym`,
copies the byte-identical benchmark prompts, and applies the reference R2E
runtime-isolation patch. Preserve these absolute paths in the image because the virtual
environments contain absolute references. The helper is for fresh image builds, not
repeated rollout initialization. Build for the same CPU architecture as the task image
and validate dynamic library compatibility.

Publish the prepared images through your E2B gateway. The supplied official SIF images
cannot be passed as OCI image names. An image manifest maps each instance ID to the
corresponding prepared image; preparation must not change its base repository or grader
tests. For example:

```json
{"owner__repo-<40-character-commit>": "registry/task-with-runtime:tag"}
```

## Select One Task

Use the qualified 700-row training or 251-row validation JSONL. The preparation command
verifies the source SHA-256 before selecting a task and changing image routing; its
output keeps grader metadata outside the model's task text.

```bash
python -m examples.swe.prepare_mlperf_data \
  --dataset /data/benchmark_r2e_gym_easy_train.jsonl \
  --split train --image-map /data/runtime-images.json \
  --instance-id '<INSTANCE_ID>' --output /data/one-task.jsonl
```

Set `E2B_API_KEY` and `AWEAGENT_E2B_IMAGE_METADATA_KEY` for your gateway. Before using
any model, verify the task image and official grader:

```bash
python -m examples.swe.check_r2e_env \
  --dataset /data/one-task.jsonl --instance-id '<INSTANCE_ID>'
```

This runs `gold` and `empty` predictions in separate sandboxes. Gold must resolve and
empty must fail. A missing report, runtime error or evaluator deadline raises instead of
becoming a negative reward. Completed negative grader reports remain valid reward-zero
samples. The scorer's comparison logic is unchanged. A small launcher checks the test
runner exit status because the reference scorer otherwise ignores it: test failures
remain reward zero, while timeout, killed or missing runners reject.

## Rollout

Configure `AREAL_MODEL_PATH`, `AREAL_ROLLOUT_BACKEND`, `AREAL_ACTOR_BACKEND`,
`AREAL_WORKER_IMAGE`, `AREAL_ROOT`, `AWEAGENT_ROOT` and `MLPERF_PREPARED_DATA` for the
actual deployment. Backend allocations and worker images are site-specific; the sample
recipe does not assume a viable 397B GPU allocation. Override the inherited
cluster/scheduler/storage settings as appropriate.

```bash
python -m examples.swe.rollout_swe \
  --config examples/swe/mlperf_openhands.yaml
```

The harness uses the reference driver, prompt templates, 30-iteration CodeAct controller
and 60-second command timeout. Its only model transport change replaces the fork's NeMo
Gym client with a per-trajectory authenticated AReaL client at `/chat/completions`.
Interpreter subprocesses inherit this adapter; no agent control loop or tool
implementation is re-created. Driver-level task retries and iterative retries are
disabled so a failed attempt cannot contaminate a later attempt's proxy trajectory. This
is a documented failure-handling deviation. Qwen thinking is disabled in the proxy
tokenizer. OpenCode-specific message preprocessors are not enabled.

Hidden R2E tests are removed only from the agent sandbox. Patch collection uses the
reference driver's artifact, including its workspace staging behavior, rather than a
second worktree diff. A new evaluator uses the untouched image and reads the official
report's `resolved` value. Agent/evaluator infrastructure failures reject the rollout;
they do not receive a policy-training reward.

Verify successful and failing patches and inspect the recorded model messages against
the reference before scaling. The same workflow can be launched with `train_swe_rl.py`,
but this single-sample bring-up YAML is not a GRPO convergence recipe: group size, loss,
optimizer and benchmark accounting still need alignment.
