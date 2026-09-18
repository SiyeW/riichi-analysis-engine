import argparse

import pytest

from riichi_analysis_engine.benchmark_conversion import (
    BenchmarkCase,
    parse_case,
    validate_comparable,
)


def test_parse_conversion_benchmark_case() -> None:
    assert parse_case("2:6") == BenchmarkCase(workers=2, compression_level=6)


@pytest.mark.parametrize("value", ["", "2", "0:1", "1:-1", "1:10", "a:1"])
def test_rejects_invalid_conversion_benchmark_case(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_case(value)


def test_conversion_benchmark_requires_comparable_outputs() -> None:
    rows = [
        {
            "case": "workers-1-compression-1",
            "status": "complete",
            "converterStatus": "complete",
            "completedGames": 4,
            "failures": [],
            "samples": 100,
        },
        {
            "case": "workers-2-compression-1",
            "status": "complete",
            "converterStatus": "complete",
            "completedGames": 4,
            "failures": [],
            "samples": 101,
        },
    ]

    with pytest.raises(ValueError, match="different sample counts"):
        validate_comparable(rows, expected_games=4)
