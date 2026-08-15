from unittest import mock

import pytest

from riichi_analysis_engine.server import Engine, ProtocolError


@pytest.mark.parametrize(
    ("host_minor", "negotiated_minor"),
    [(0, 0), (1, 1), (2, 1)],
)
def test_hello_negotiates_the_highest_common_minor(
    host_minor: int,
    negotiated_minor: int,
) -> None:
    with mock.patch("riichi_analysis_engine.server.torch.cuda.is_available", return_value=False):
        hello = Engine().hello({
            "protocol": {
                "name": "riichi-engine-protocol",
                "major": 2,
                "minor": host_minor,
            }
        })

    assert hello["protocol"] == {
        "name": "riichi-engine-protocol",
        "major": 2,
        "minor": negotiated_minor,
    }


@pytest.mark.parametrize(
    "protocol",
    [
        {"name": "another-protocol", "major": 2, "minor": 1},
        {"name": "riichi-engine-protocol", "major": 3, "minor": 1},
        {"name": "riichi-engine-protocol", "major": 2, "minor": -1},
        {"name": "riichi-engine-protocol", "major": 2, "minor": True},
        {"name": "riichi-engine-protocol", "major": 2, "minor": "1"},
    ],
)
def test_hello_rejects_incompatible_or_invalid_protocols(protocol: dict) -> None:
    with pytest.raises(ProtocolError, match="not compatible"):
        Engine().hello({"protocol": protocol})
