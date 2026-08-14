from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def stable_fraction(paths: list[Path], numerator: int, denominator: int) -> list[Path]:
    """Select an exact deterministic fraction without depending on directory order."""
    if not 0 < numerator <= denominator:
        raise ValueError("fraction must satisfy 0 < numerator <= denominator")
    ordered = sorted(
        paths,
        key=lambda path: (hashlib.sha256(path.name.encode("utf-8")).digest(), path.name),
    )
    return ordered[: len(ordered) * numerator // denominator]


def mortal_validation_members(names: list[str], train_ratio: float = 0.98) -> list[str]:
    selected: list[str] = []
    for name in names:
        digest = hashlib.md5(name.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "little") / 2**64
        if bucket >= train_ratio:
            selected.append(name)
    return sorted(selected)


def write_manifest(path: Path, records: list[dict[str, object]], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"manifest": metadata}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic 2025/2026 manifests.")
    parser.add_argument("--raw-2025", type=Path, required=True)
    parser.add_argument("--zip-2026", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--train-numerator", type=int, default=1)
    parser.add_argument("--train-denominator", type=int, default=20)
    args = parser.parse_args()

    raw_2025 = args.raw_2025.resolve()
    zip_2026 = args.zip_2026.resolve()
    files_2025 = list(raw_2025.glob("*.mjson"))
    selected_2025 = stable_fraction(files_2025, args.train_numerator, args.train_denominator)

    with zipfile.ZipFile(zip_2026) as archive:
        members_2026 = [name for name in archive.namelist() if name.endswith(".mjson")]
    validation_2026 = mortal_validation_members(members_2026)

    train_records = [
        {"sourceId": path.name, "path": str(path.resolve())}
        for path in selected_2025
    ]
    val_records = [
        {
            "sourceId": member,
            "path": f"zip://{zip_2026}!{member}",
        }
        for member in validation_2026
    ]
    train_selection_hash = hashlib.sha256(
        "\n".join(record["sourceId"] for record in train_records).encode("utf-8")
    ).hexdigest()
    val_selection_hash = hashlib.sha256(
        "\n".join(record["sourceId"] for record in val_records).encode("utf-8")
    ).hexdigest()

    write_manifest(
        args.output / "train-2025-1of20.jsonl",
        train_records,
        {
            "split": "train",
            "year": 2025,
            "sourceCount": len(files_2025),
            "selectedCount": len(train_records),
            "fraction": f"{args.train_numerator}/{args.train_denominator}",
            "selection": "sha256-filename-order-v1",
            "selectionHash": train_selection_hash,
        },
    )
    write_manifest(
        args.output / "validation-2026.jsonl",
        val_records,
        {
            "split": "validation",
            "year": 2026,
            "sourceCount": len(members_2026),
            "selectedCount": len(val_records),
            "selection": "mortal-md5-bucket-gte-0.98",
            "selectionHash": val_selection_hash,
        },
    )
    print(f"2025 train: {len(train_records)} / {len(files_2025)}")
    print(f"2026 validation: {len(val_records)} / {len(members_2026)}")


if __name__ == "__main__":
    main()
