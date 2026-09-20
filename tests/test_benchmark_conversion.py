import argparse

import pytest

from riichi_analysis_engine.benchmark_conversion import (
    BenchmarkCase,
    default_cases,
    parse_case,
    validate_comparable,
)


def test_parse_conversion_benchmark_case() -> None:
    assert parse_case("2:6") == BenchmarkCase(workers=2, compression_level=6)
    assert parse_case("2:6:128") == BenchmarkCase(
        workers=2, compression_level=6, chunk_samples=128
    )


def test_default_conversion_benchmark_cases_are_bounded_by_work() -> None:
    assert default_cases(max_games=16, logical_cpu_count=28) == [
        BenchmarkCase(1, 1),
        BenchmarkCase(2, 1),
        BenchmarkCase(4, 1),
        BenchmarkCase(8, 1),
        BenchmarkCase(16, 1),
        BenchmarkCase(16, 6),
        BenchmarkCase(16, 1, 64),
        BenchmarkCase(16, 1, 128),
        BenchmarkCase(16, 1, 256),
    ]
    assert default_cases(max_games=4, logical_cpu_count=32) == [
        BenchmarkCase(1, 1),
        BenchmarkCase(2, 1),
        BenchmarkCase(4, 1),
        BenchmarkCase(4, 6),
        BenchmarkCase(4, 1, 64),
        BenchmarkCase(4, 1, 128),
        BenchmarkCase(4, 1, 256),
    ]


@pytest.mark.parametrize(
    "value", ["", "2", "0:1", "1:-1", "1:10", "a:1", "1:1:0", "1:1:2:3"]
)
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
