from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def mortal_validation_members(names: list[str], train_ratio: float = 0.98) -> list[str]:
    """Select the deterministic 2026 evaluation members used since the first comparison."""

    selected: list[str] = []
    for name in names:
        digest = hashlib.md5(name.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "little") / 2**64
        if bucket >= train_ratio:
            selected.append(name)
    return sorted(selected)


def stable_source_order(names: list[str], seed: int) -> list[str]:
    """Order source names by a seeded hash so no split depends on directory order."""

    return sorted(
        names,
        key=lambda name: (
            hashlib.sha256(f"{seed}\0{name}".encode("utf-8")).digest(),
            name,
        ),
    )


def split_holdout(names: list[str], count: int, seed: int) -> tuple[list[str], list[str]]:
    """Return the held-out names and the remaining names, both in stable source order."""

    if count < 0:
        raise ValueError("holdout count must not be negative")
    if count > len(names):
        raise ValueError(f"holdout count {count} exceeds the {len(names)} available names")
    ordered = stable_source_order(names, seed)
    return ordered[:count], ordered[count:]


def selection_hash(names: list[str]) -> str:
    """Fingerprint one split so a later run can prove it rebuilt the same selection."""

    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def write_manifest(path: Path, records: list[dict[str, object]], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"manifest": metadata}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def raw_records(directory: Path, names: list[str]) -> list[dict[str, object]]:
    return [{"sourceId": name, "path": str((directory / name).resolve())} for name in names]


def zip_records(archive: Path, names: list[str]) -> list[dict[str, object]]:
    return [
        {"sourceId": name, "path": f"zip://{archive}!{name}"}
        for name in names
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the deterministic 2025/2026 manifests.")
    parser.add_argument("--raw-2025", type=Path, required=True)
    parser.add_argument("--zip-2026", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests"))
    parser.add_argument("--source-order-seed", type=int, default=20252026)
    parser.add_argument("--holdout-2025-count", type=int, default=2048)
    parser.add_argument("--test-2026-count", type=int, default=1000)
    args = parser.parse_args()

    raw_2025 = args.raw_2025.resolve()
    zip_2026 = args.zip_2026.resolve()
    names_2025 = sorted(path.name for path in raw_2025.glob("*.mjson"))
    if not names_2025:
        raise FileNotFoundError(f"no 2025 games under {raw_2025}")
    holdout_2025, train_2025 = split_holdout(
        names_2025, args.holdout_2025_count, args.source_order_seed
    )

    with zipfile.ZipFile(zip_2026) as archive:
        members_2026 = sorted(
            name for name in archive.namelist() if name.endswith(".mjson")
        )
    if not members_2026:
        raise FileNotFoundError(f"no 2026 games in {zip_2026}")
    validation_2026 = mortal_validation_members(members_2026)
    validation_set = set(validation_2026)
    remaining_2026 = [name for name in members_2026 if name not in validation_set]
    test_2026, _unused = split_holdout(
        remaining_2026, args.test_2026_count, args.source_order_seed
    )

    common = {
        "sourceCount": 0,
        "sourceOrderSeed": args.source_order_seed,
        "selection": "seeded-sha256-source-order-v1",
    }

    write_manifest(
        args.output / "train-2025.jsonl",
        raw_records(raw_2025, train_2025),
        {
            **common,
            "split": "train",
            "year": 2025,
            "sourceCount": len(names_2025),
            "selectedCount": len(train_2025),
            "selectionHash": selection_hash(train_2025),
        },
    )
    write_manifest(
        args.output / "holdout-2025.jsonl",
        raw_records(raw_2025, holdout_2025),
        {
            **common,
            "split": "validation",
            "year": 2025,
            "sourceCount": len(names_2025),
            "selectedCount": len(holdout_2025),
            "selectionHash": selection_hash(holdout_2025),
        },
    )
    write_manifest(
        args.output / "validation-2026.jsonl",
        zip_records(zip_2026, validation_2026),
        {
            **common,
            "split": "validation",
            "year": 2026,
            "sourceCount": len(members_2026),
            "selectedCount": len(validation_2026),
            "selection": "mortal-md5-bucket-gte-0.98",
            "selectionHash": selection_hash(validation_2026),
        },
    )
    write_manifest(
        args.output / "test-2026.jsonl",
        zip_records(zip_2026, test_2026),
        {
            **common,
            "split": "test",
            "year": 2026,
            "sourceCount": len(remaining_2026),
            "selectedCount": len(test_2026),
            "selectionHash": selection_hash(test_2026),
        },
    )

    print(f"2025 train: {len(train_2025)} / {len(names_2025)}")
    print(f"2025 holdout: {len(holdout_2025)} / {len(names_2025)}")
    print(f"2026 validation: {len(validation_2026)} / {len(members_2026)}")
    print(f"2026 test: {len(test_2026)} / {len(remaining_2026)} remaining")


if __name__ == "__main__":
    main()
