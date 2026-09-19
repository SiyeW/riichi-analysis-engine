from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [
        ROOT / item
        for item in result.stdout.decode("utf-8").split("\0")
        if item and (ROOT / item).is_file()
    ]


def main() -> None:
    manifest = json.loads((ROOT / "engine.json").read_text(encoding="utf-8"))
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_source = (ROOT / "src/riichi_analysis_engine/__init__.py").read_text(
        encoding="utf-8"
    )
    server_source = (ROOT / "src/riichi_analysis_engine/server.py").read_text(
        encoding="utf-8"
    )
    package_version = re.search(r'^__version__ = "([^"]+)"$', package_source, re.MULTILINE)
    server_version = re.search(r'^ENGINE_VERSION = "([^"]+)"$', server_source, re.MULTILINE)
    require(package_version is not None, "package version is missing")
    require(server_version is not None, "server version is missing")
    canonical_version = project["project"]["version"]
    display_version = canonical_version.replace(".dev", "-dev.")
    require(
        package_version.group(1) == canonical_version,
        "package and project versions differ",
    )
    require(
        manifest["version"] == display_version
        and server_version.group(1) == display_version,
        "manifest, server, and project versions differ",
    )
    require(manifest["id"] == "org.riichi.analysis", "unexpected engine id")
    require(manifest["name"] == "Riichi Analysis Engine", "unexpected engine name")
    require(
        manifest["protocol"]
        == {"name": "riichi-engine-protocol", "major": 2, "minor": 2},
        "unexpected protocol version",
    )
    require(
        manifest["entrypoints"]["windows-x64"]["executable"]
        == "runtime/riichi-analysis-engine.exe",
        "unexpected Windows entrypoint",
    )

    files = public_files()
    forbidden_suffixes = {".ckpt", ".mjson", ".npy", ".npz", ".onnx", ".pt", ".pth", ".zip"}
    forbidden = [path for path in files if path.suffix.lower() in forbidden_suffixes]
    require(not forbidden, "generated data or weights found in the source tree")

    environment_files = [
        path
        for path in files
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")
    ]
    require(not environment_files, "local environment file found in the source tree")

    local_path = re.compile(r"(?i)\b[a-z]:[\\/]")
    private_training_markers = (
        "data_" + "provenance.md",
        "ten" + "hou-to-mjai",
        "nikke" + "tryhard",
        "train-" + "2025",
        "holdout-" + "2025",
        "validation-" + "2026",
        "2023-" + "2025",
    )
    for path in files:
        if path.suffix.lower() not in {".json", ".md", ".ps1", ".py", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        require(not local_path.search(text), f"local Windows path found in {relative}")
        lowered = text.lower()
        exposed = [marker for marker in private_training_markers if marker in lowered]
        require(
            not exposed,
            f"private training provenance found in {relative}: {exposed}",
        )
        retired_name = "uni" + "fied"
        require(retired_name not in lowered, f"retired project name found in {relative}")

    print("OK: manifest, project name, public paths, data, and weight boundaries")


if __name__ == "__main__":
    main()
