from __future__ import annotations

import json
import re
import subprocess
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
        if item
    ]


def main() -> None:
    manifest = json.loads((ROOT / "engine.json").read_text(encoding="utf-8"))
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

    local_path = re.compile(r"(?i)\b[a-z]:\\")
    for path in files:
        if path.suffix.lower() not in {".json", ".md", ".ps1", ".py", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)
        require(not local_path.search(text), f"local Windows path found in {relative}")
        retired_name = "uni" + "fied"
        require(retired_name not in text.lower(), f"retired project name found in {relative}")

    print("OK: manifest, project name, public paths, data, and weight boundaries")


if __name__ == "__main__":
    main()
