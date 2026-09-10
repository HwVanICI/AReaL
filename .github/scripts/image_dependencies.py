"""Collect image dependency metadata without importing accelerator libraries."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from pathlib import Path


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def collect() -> dict[str, dict[str, str]]:
    python = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        value = dist.version
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        revision = direct.get("vcs_info", {}).get("commit_id")
        if revision:
            value += f" @ {revision}"
        python[name] = value

    os_packages = {}
    for line in command(
        "dpkg-query", "-W", "-f=${binary:Package}\t${Version}\t${db:Status-Status}\n"
    ).splitlines():
        name, version, status = line.split("\t")
        if status == "installed":
            os_packages[name] = version

    sources = {}
    root = Path("/areal-workspace")
    if root.exists():
        for path in sorted(root.iterdir()):
            if path.is_symlink() or not (path / ".git").exists():
                continue
            revision = command("git", "-C", str(path), "rev-parse", "HEAD")
            patch = subprocess.check_output(
                ["git", "-C", str(path), "diff", "--binary", "HEAD", "--"]
            )
            if patch:
                revision += f" + patch sha256:{hashlib.sha256(patch).hexdigest()}"
            sources[path.name] = revision

    ascend = {}
    root = Path("/usr/local/Ascend")
    # Resolve aliases such as latest/ and ascend-toolkit/ to avoid duplicates.
    files = {
        path.resolve()
        for pattern in ("**/version.info", "**/version.cfg", "**/*install.info")
        for path in root.glob(pattern)
        if path.is_file()
    }
    for path in sorted(files):
        versions = []
        for line in path.read_text(errors="replace").splitlines():
            if re.match(r"\s*(?:[\w.-]*version[\w.-]*)\s*[:=]", line, re.I):
                versions.append(line.strip())
        if versions:
            ascend[str(path.relative_to(root))] = "; ".join(versions)

    return {
        "Python": python,
        "OS packages": os_packages,
        "Git sources (revision and tracked patches)": sources,
        "Ascend version metadata": ascend,
        "Runtime": {"Python": platform.python_version()},
    }


def report(previous: dict | None, current: dict) -> str:
    lines = ["## Image dependency changes", ""]
    for title, inventory in (("Previous", previous), ("Current", current)):
        if inventory is None:
            lines.append(f"{title}: unavailable; no comparison baseline.")
        else:
            lines.append(f"{title}: `{inventory['image']}`")
    lines += [
        "",
        "Coverage: installed Python distributions, dpkg packages, Git checkouts under "
        "`/areal-workspace` (including tracked patches), Ascend version metadata, "
        "and Python runtime. Unmanaged binaries and untracked source files are not inventoried.",
        "",
    ]
    if previous is None:
        lines.append(
            "Current inventory attached as an artifact; changes cannot be determined."
        )
        return "\n".join(lines) + "\n"
    changes = 0
    for section in sorted(
        previous["dependencies"].keys() | current["dependencies"].keys()
    ):
        old = previous["dependencies"].get(section, {})
        new = current["dependencies"].get(section, {})
        rows = [
            f"{name}: {old.get(name, '(absent)')} -> {new.get(name, '(absent)')}"
            for name in sorted(old.keys() | new.keys())
            if old.get(name) != new.get(name)
        ]
        if rows:
            changes += len(rows)
            lines += [f"### {section} ({len(rows)})", "", "```text", *rows, "```", ""]
    if not changes:
        lines.append("No dependency changes detected.")
    return "\n".join(lines) + "\n"


def snapshot(image: str, output: Path) -> None:
    command("docker", "pull", "--quiet", image)
    try:
        digest = json.loads(command("docker", "image", "inspect", image))[0][
            "RepoDigests"
        ][0]
        result = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network=none",
                "--read-only",
                "-i",
                "--entrypoint",
                "python3",
                digest,
                "-",
                "collect",
            ],
            input=Path(__file__).read_bytes(),
        )
        output.write_text(
            json.dumps({"image": digest, "dependencies": json.loads(result)}, indent=2)
            + "\n"
        )
    finally:
        subprocess.run(["docker", "image", "rm", image], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "snapshot", "report"))
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--current", type=Path)
    args = parser.parse_args()
    if args.mode == "collect":
        # stdout is the inventory protocol, including when executed inside an image.
        print(json.dumps(collect(), sort_keys=True))
    elif args.mode == "snapshot":
        snapshot(args.image, args.output)
    else:
        previous = (
            json.loads(args.previous.read_text()) if args.previous.exists() else None
        )
        args.output.write_text(report(previous, json.loads(args.current.read_text())))


if __name__ == "__main__":
    main()
