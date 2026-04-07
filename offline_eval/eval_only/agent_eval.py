from __future__ import annotations

import sys

from .common import run_eval


def main(args: list[str]) -> dict[str, float]:
    return run_eval(
        args,
        logger_name="AgentEvalOnly",
        log_filename="agent_eval.log",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
