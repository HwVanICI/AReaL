"""Collect image dependency metadata without importing accelerator libraries."""

import argparse
import hashlib
import importlib.metadata
import io
import json
import platform
import re
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZipFile


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


def image_reference(image: str, digest: str) -> str:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError(f"Invalid image digest: {digest}")
    return f"{image.rsplit(':', 1)[0]}@{digest}"


def restore(image: str, repository: str, accelerator: str, output: Path) -> None:
    output.unlink(missing_ok=True)
    manifest = json.loads(
        command(
            "docker",
            "buildx",
            "imagetools",
            "inspect",
            image,
            "--format",
            "{{json .Manifest}}",
        )
    )
    digest = manifest["digest"]
    reference = image_reference(image, digest)
    # The second name supports inventories uploaded before digest-based naming.
    names = (
        f"image-dependencies-{accelerator}-{digest[7:]}",
        f"image-dependencies-{accelerator}",
    )
    for name in names:
        query = urlencode({"name": name, "per_page": 100})
        artifacts = json.loads(
            command("gh", "api", f"repos/{repository}/actions/artifacts?{query}")
        )["artifacts"]
        for artifact in sorted(artifacts, key=lambda item: item["id"], reverse=True):
            if artifact["expired"]:
                continue
            data = subprocess.check_output(
                [
                    "gh",
                    "api",
                    f"repos/{repository}/actions/artifacts/{artifact['id']}/zip",
                ]
            )
            with ZipFile(io.BytesIO(data)) as archive:
                if "current.json" not in archive.namelist():
                    continue
                inventory = json.loads(archive.read("current.json"))
            if inventory.get("image") == reference:
                output.write_text(json.dumps(inventory, indent=2) + "\n")
                return
    print(
        f"No retained inventory matches {reference}; this build will establish a baseline."
    )


def prepare(
    dockerfile: str,
    image: str,
    cache: str,
    revision: str,
    repository: str,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "group": {"default": {"targets": ["runtime", "inventory"]}},
        "target": {
            "runtime": {
                "context": str(Path.cwd()),
                "dockerfile": dockerfile,
                "tags": [image],
                "labels": {
                    "org.opencontainers.image.revision": revision,
                    "org.opencontainers.image.source": f"https://github.com/{repository}",
                },
                "cache-from": [f"type=registry,ref={cache}"],
                "cache-to": [f"type=registry,ref={cache},mode=min"],
                "output": ["type=registry"],
            },
            "inventory": {
                "context": str(output.resolve()),
                "contexts": {
                    "runtime": "target:runtime",
                    "dependency-tools": str(Path(__file__).resolve().parent),
                },
                "dockerfile-inline": (
                    "FROM runtime AS collect\n"
                    "RUN --network=none --mount=type=bind,from=dependency-tools,"
                    "source=image_dependencies.py,target=/tmp/image_dependencies.py "
                    "python3 /tmp/image_dependencies.py collect > /dependencies.json\n"
                    "FROM scratch\n"
                    "COPY --from=collect /dependencies.json /dependencies.json\n"
                ),
                "output": [f"type=local,dest={output.resolve() / 'export'}"],
            },
        },
    }
    (output / "bake.json").write_text(json.dumps(configuration, indent=2) + "\n")


def record(image: str, metadata: Path, inventory: Path, output: Path) -> None:
    digest = json.loads(metadata.read_text())["runtime"]["containerimage.digest"]
    output.write_text(
        json.dumps(
            {
                "image": image_reference(image, digest),
                "dependencies": json.loads(inventory.read_text()),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("collect", "restore", "prepare", "record", "report")
    )
    parser.add_argument("--image")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--accelerator", choices=("a2", "a3"))
    parser.add_argument("--dockerfile")
    parser.add_argument("--cache")
    parser.add_argument("--revision")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    if args.mode == "collect":
        # stdout is the inventory protocol, including when executed inside an image.
        print(json.dumps(collect(), sort_keys=True))
    elif args.mode == "restore":
        restore(args.image, args.repository, args.accelerator, args.output)
    elif args.mode == "prepare":
        prepare(
            args.dockerfile,
            args.image,
            args.cache,
            args.revision,
            args.repository,
            args.output,
        )
    elif args.mode == "record":
        record(args.image, args.metadata, args.inventory, args.output)
    else:
        previous = (
            json.loads(args.previous.read_text()) if args.previous.exists() else None
        )
        args.output.write_text(report(previous, json.loads(args.current.read_text())))


if __name__ == "__main__":
    main()
