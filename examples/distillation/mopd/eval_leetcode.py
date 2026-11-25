#!/usr/bin/env python3
"""Offline LeetCode evaluation using AReaL's native v1 rollout controller.

Place in examples/code/. Requires a qualified Ascend AReaL checkout.
Uses the unmodified CodeAgent via AReaL OpenAI proxy workers.
Qualify the effective proxy generation settings before official scoring.
Run only in an isolated disposable environment (generated code is executed).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

# Set before importing any AReaL/torch/vLLM modules; inherited by RPC workers.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

N_PROBLEMS = 101
N_SAMPLES = 8
SEED = 1
TEST_SHA256 = "c3c7d01aefa54c40d752c56d2f3d5af7be14e32128c4b52045e7cddeec9470f4"
TARGET_SCORE = 52

def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def supports(fn, name):
    return name in inspect.signature(fn).parameters


def summarize(rewards):
    check(len(rewards) == N_PROBLEMS * N_SAMPLES, f"Expected 808 rewards, got {len(rewards)}")
    check(all(r in (0, 1) for r in rewards), "Expected binary, unnormalized rewards")
    correct = int(sum(rewards))
    return dict(num_problems=N_PROBLEMS, samples_per_problem=N_SAMPLES,
                num_responses=len(rewards), num_correct=correct,
                mean_reward=correct / len(rewards), target=TARGET_SCORE / 100,
                passed=correct * 100 >= TARGET_SCORE * len(rewards))


def read_dump(root, expected_task_ids):
    """Verify one complete group per submitted task, using native dump records."""
    groups = {}
    for path in sorted(root.rglob("*.jsonl")):
        rows = read_jsonl(path)
        check(len(rows) == N_SAMPLES, f"Incomplete or duplicate rollout group: {path}")
        check({r["sample_idx"] for r in rows} == set(range(N_SAMPLES)), f"Invalid sample indices: {path}")
        ids = {str(r["task_id"]) for r in rows}
        check(len(ids) == 1, f"Mixed task IDs: {path}")
        task_id = ids.pop()
        check(task_id not in groups, f"Duplicate task ID: {task_id}")
        check(len({r["prompt"] for r in rows}) == 1, f"Mixed prompts: {path}")
        groups[task_id] = rows
    check(set(groups) == set(expected_task_ids),
          f"Dump coverage differs from submitted tasks: {len(groups)}/{len(expected_task_ids)} groups in {root}")
    return [r for task_id in expected_task_ids for r in groups[task_id]]


def evaluate(args, areal_args):
    import numpy as np
    from areal.api.alloc_mode import ModelAllocation
    from areal.api.cli_args import GRPOConfig, load_expr_config, vLLMConfig
    from areal.dataset.leetcode import get_leetcode_rl_dataset
    from areal.engine.vllm_remote import RemotevLLMEngine
    from areal.infra import LocalScheduler
    from areal.utils.hf_utils import load_hf_tokenizer
    from areal.utils.stats_logger import StatsLogger

    check(args.isolated_reward_environment, "Run only in an isolated disposable environment; acknowledge with --isolated-reward-environment")
    checkpoint = args.checkpoint.resolve()
    tokenizer_path = args.tokenizer.resolve()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    check((checkpoint / "config.json").is_file(), "Expected an HF checkpoint directory")
    check(sha256(data_dir / "test.parquet") == TEST_SHA256, "Official validation checksum mismatch")
    check(not output.exists(), f"Output already exists: {output}")

    # Resolve AReaL's own config with CLI overrides, including interpolation.
    config, _ = load_expr_config(areal_args, GRPOConfig)
    check(config.scheduler.type == "local", "This evaluator uses the single-node LocalScheduler")
    check(config.cluster.n_nodes == 1, "Expected one node")
    config.seed = SEED
    random.seed(SEED)
    np.random.seed(SEED)
    config.actor.path = str(checkpoint)
    config.tokenizer_path = str(tokenizer_path)
    config.vllm.model = str(checkpoint)
    config.vllm.seed = SEED
    config.vllm.dtype = "bfloat16"
    config.vllm.max_model_len = 16384
    config.valid_dataset.path = str(data_dir)
    config.rollout.tokenizer_path = str(tokenizer_path)
    config.rollout.max_head_offpolicyness = 1000000
    config.rollout.dump_to_file = True
    # Private output/name resolution namespace; don't reuse training RPC names.
    config.experiment_name = "ict_leetcode_offline_eval"
    config.trial_name = "eval0"
    config.cluster.fileroot = str(output)
    config.rollout.experiment_name = config.experiment_name
    config.rollout.trial_name = config.trial_name
    config.rollout.fileroot = str(output)
    if config.cluster.name_resolve.type == "nfs":
        config.cluster.name_resolve.nfs_record_root = str(output / "name_resolve")
    for spec in config.rollout.scheduling_spec:
        if spec.env_vars is None:
            spec.env_vars = {}
        spec.env_vars["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    tokenizer = load_hf_tokenizer(str(tokenizer_path))
    dataset = get_leetcode_rl_dataset(path=str(data_dir), split="test", tokenizer=tokenizer, max_length=None)
    check(len(dataset) == N_PROBLEMS, f"Expected 101 validation problems, got {len(dataset)}")
    items = [dict(row) for row in dataset]
    # CodeAgent consumes messages directly; the proxy owns tokenization.
    check(all("messages" in row and "tests" in row for row in items),
          "Validation rows must contain messages and tests")
    check(config.rollout.agent is not None, "CodeAgent requires rollout.agent configuration")
    check(config.rollout.agent.mode in ("inline", "subproc"), "Expected an offline agent mode")
    config.rollout.agent.chat_template_kwargs = {
        **(config.rollout.agent.chat_template_kwargs or {}),
        "enable_thinking": False,
    }
    # CodeAgent removes max_tokens from its kwargs. Use the standard OpenAI
    # max_completion_tokens field instead; verify the proxy honors it in the
    # qualified checkout and compare against train.py's eval_workflow_kwargs.
    workflow_kwargs = dict(
        sandbox_backend="subprocess",
        temperature=0.6,
        top_p=1.0,
        max_completion_tokens=8000,
    )

    # API changed across AReaL revisions: support the two documented layouts.
    if supports(ModelAllocation.from_str, "name"):
        alloc = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    else:
        alloc = ModelAllocation.from_str(config.rollout.backend)
    check(alloc.backend == "vllm", "Expected rollout.backend=vllm:...")
    parallel = getattr(alloc, "parallel", alloc)
    server_args = vLLMConfig.build_args(vllm_config=config.vllm,
                                      tp_size=parallel.tp_size, pp_size=parallel.pp_size)
    output.mkdir(parents=True, exist_ok=False)
    if config.cluster.name_resolve.type == "nfs":
        Path(config.cluster.name_resolve.nfs_record_root).mkdir(parents=True, exist_ok=True)
    scheduler = LocalScheduler(exp_config=config)
    rollout = RemotevLLMEngine.as_controller(config.rollout, scheduler)
    initialized = False
    start = time.monotonic()
    try:
        init_kwargs = dict(role="eval-rollout", server_args=server_args)
        if supports(rollout.initialize, "rollout_alloc"):
            init_kwargs["rollout_alloc"] = alloc
        rollout.initialize(**init_kwargs)
        initialized = True
        # The trainer normally starts proxies lazily for agent workflows.
        # With a standalone v1 controller, start them explicitly.
        check(callable(getattr(rollout, "start_proxy", None)),
              "This rollout controller lacks start_proxy(); check the qualified AReaL revision")
        rollout.start_proxy()
        submit_kwargs = dict(
            workflow="areal.workflow.openai.code_agent.CodeAgent",
            workflow_kwargs=workflow_kwargs,
            group_size=N_SAMPLES,
        )
        if supports(rollout.submit, "is_eval"):
            submit_kwargs["is_eval"] = True
        if supports(rollout.submit, "reward_normalization"):
            submit_kwargs["reward_normalization"] = False
        if supports(rollout.submit, "drop_incomplete_group"):
            submit_kwargs["drop_incomplete_group"] = False
        expected_task_ids = []
        batch_size = config.valid_dataset.batch_size

        for offset in range(0, len(items), batch_size):
            batch = items[offset:offset + batch_size]
            for row in batch:
                task_id = rollout.submit(row, **submit_kwargs)
                check(task_id is not None, "Controller did not return a task ID")
                expected_task_ids.append(str(task_id))
            results = rollout.wait(len(batch), timeout=10800)
            check(len(results) == len(batch) and all(r is not None for r in results),
                  "One or more evaluation groups failed; no official score produced")
            print(f"Completed {offset + len(batch)}/{N_PROBLEMS} groups", flush=True)
        check(len(set(expected_task_ids)) == N_PROBLEMS, "Duplicate submitted task IDs")
        print("AReaL evaluation stats:", rollout.export_stats(), flush=True)
        log_path = Path(StatsLogger.get_log_path(
            experiment_name=config.experiment_name,
            trial_name=config.trial_name,
            fileroot=str(output),
        ))
        roots = [log_path / "eval-rollout", log_path / "rollout"]
        present = [p for p in roots if p.is_dir() and any(p.rglob("*.jsonl"))]
        check(len(present) == 1, f"Expected one native trajectory dump directory under {log_path}")
        records = read_dump(present[0], expected_task_ids)
        summary = summarize([r["reward"] for r in records])
        summary.update(checkpoint=str(checkpoint), test_sha256=TEST_SHA256,
                       rollout_backend=config.rollout.backend, seed=SEED,
                       workflow="areal.workflow.openai.code_agent.CodeAgent",
                       workflow_kwargs=workflow_kwargs,
                       evaluation_seconds=time.monotonic() - start)
    finally:
        try:
            if initialized and hasattr(rollout, "destroy"):
                rollout.destroy()
        finally:
            if hasattr(scheduler, "delete_workers"):
                # LocalScheduler signatures vary. Prefer native controller cleanup;
                # only use zero-argument scheduler cleanup when explicitly supported.
                sig = inspect.signature(scheduler.delete_workers)
                if all(p.default is not inspect.Parameter.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in sig.parameters.values()):
                    scheduler.delete_workers()
    write_json(output / "results.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def sweep(args, areal_args):
    check(args.isolated_reward_environment, "Use an isolated disposable environment")
    rows = read_jsonl(args.manifest)
    check(bool(rows), "Empty checkpoint manifest")
    run_id = rows[0]["run_id"]
    last_time = 0.0
    paths = set()
    for i, row in enumerate(rows, 1):
        check(row["run_id"] == run_id and row["completed_step"] == 10 * i, "Manifest must contain one run and steps 10, 20, ...")
        elapsed = float(row["elapsed_seconds"])
        check(math.isfinite(elapsed) and elapsed > last_time, "Invalid elapsed time")
        last_time = elapsed
        ckpt = Path(row["checkpoint"])
        if not ckpt.is_absolute():
            ckpt = args.manifest.resolve().parent / ckpt
        ckpt = ckpt.resolve()
        check(ckpt not in paths and (ckpt / "config.json").is_file(), f"Invalid or duplicate checkpoint: {ckpt}")
        paths.add(ckpt)
        row["checkpoint"] = str(ckpt)
    args.output.mkdir(parents=True, exist_ok=False)
    evaluated = []
    qualifying = None
    for row in rows:
        out = args.output / f"step_{row['completed_step']:04d}"
        cmd = [sys.executable, str(Path(__file__).resolve()), "evaluate", "--checkpoint", row["checkpoint"],
               "--tokenizer", str(args.tokenizer), "--data-dir", str(args.data_dir), "--output", str(out),
               "--isolated-reward-environment", *areal_args]
        subprocess.run(cmd, check=True)
        result = json.loads((out / "results.json").read_text())
        entry = {**row, "mean_reward": result["mean_reward"], "num_correct": result["num_correct"], "passed": result["passed"]}
        evaluated.append(entry)
        if entry["passed"]:
            qualifying = entry
            break
    summary = dict(run_id=run_id, passed=qualifying is not None,
                   qualifying_checkpoint=qualifying, evaluated_checkpoints=evaluated,
                   time_to_target_seconds=qualifying["elapsed_seconds"] if qualifying else None)
    write_json(args.output / "sweep_results.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog="Additional arguments are forwarded to AReaL, e.g. --config train.yaml rollout.backend=vllm:d4t2")
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("evaluate", "sweep"):
        p = sub.add_parser(mode)
        p.add_argument("--checkpoint" if mode == "evaluate" else "--manifest", type=Path, required=True)
        p.add_argument("--tokenizer", type=Path, required=True)
        p.add_argument("--data-dir", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--isolated-reward-environment", action="store_true")
    args, areal_args = parser.parse_known_args()
    check("--config" in areal_args or any(a.startswith("--config=") for a in areal_args), "Supply the qualified AReaL YAML with --config")
    (evaluate if args.mode == "evaluate" else sweep)(args, areal_args)


if __name__ == "__main__":
    main()
