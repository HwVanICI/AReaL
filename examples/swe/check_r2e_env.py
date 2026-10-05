"""Check official gold/empty patch outcomes in separate E2B evaluator sandboxes."""

import argparse
import asyncio
from pathlib import Path

from examples.swe.agent import _ensure_aweagent_importable
from examples.swe.dataset import get_swe_dataset


async def check_task(data: dict, mode: str, timeout: int = 180) -> None:
    from aweagent.envs.e2b import _enter_e2b_sandbox
    from aweagent.grading.r2e import evaluate_r2e_patch

    modes = ("gold", "empty") if mode == "both" else (mode,)
    for current in modes:
        sandbox = await _enter_e2b_sandbox(data["image"])
        try:
            reward, output = await evaluate_r2e_patch(
                sandbox,
                data,
                "",
                timeout,
                mode=current,
            )
            expected = 1.0 if current == "gold" else 0.0
            if reward != expected:
                raise RuntimeError(
                    f"{current}: expected {expected}, got {reward}\n{output}"
                )
        finally:
            await sandbox.__aexit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--agent-root", default="")
    parser.add_argument("--mode", choices=("gold", "empty", "both"), default="both")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    _ensure_aweagent_importable(args.agent_root)
    matches = [
        row
        for row in get_swe_dataset(str(args.dataset))
        if row["instance_id"] == args.instance_id
    ]
    if len(matches) != 1:
        raise ValueError("Select exactly one official R2E task")
    asyncio.run(check_task(matches[0], args.mode, args.timeout))


if __name__ == "__main__":
    main()
