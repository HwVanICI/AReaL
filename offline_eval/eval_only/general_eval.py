from __future__ import annotations

import sys

from .common import run_eval


def main(args: list[str]) -> dict[str, float]:
    return run_eval(
        args,
        logger_name="GeneralEvalOnly",
        log_filename="general_eval.log",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
