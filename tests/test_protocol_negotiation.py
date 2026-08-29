from unittest import mock

import pytest

from riichi_analysis_engine.server import Engine, ProtocolError


@pytest.mark.parametrize(
    ("host_minor", "negotiated_minor"),
    [(0, 0), (1, 1), (2, 2), (3, 2)],
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


def test_hello_only_declares_features_from_the_negotiated_minor() -> None:
    with mock.patch("riichi_analysis_engine.server.torch.cuda.is_available", return_value=False):
        old = Engine().hello(
            {
                "protocol": {
                    "name": "riichi-engine-protocol",
                    "major": 2,
                    "minor": 0,
                }
            }
        )
        current = Engine().hello(
            {
                "protocol": {
                    "name": "riichi-engine-protocol",
                    "major": 2,
                    "minor": 2,
                }
            }
        )
        previous = Engine().hello(
            {
                "protocol": {
                    "name": "riichi-engine-protocol",
                    "major": 2,
                    "minor": 1,
                }
            }
        )

    old_outputs = {item["id"]: item for item in old["outputContracts"]}
    current_outputs = {item["id"]: item for item in current["outputContracts"]}
    previous_outputs = {item["id"]: item for item in previous["outputContracts"]}
    assert "kyoku-outcome" not in old_outputs
    assert "point-estimate" not in old_outputs["opponent-dora-count"]["representations"]
    assert "kyoku-outcome" in current_outputs
    assert "version" not in current_outputs["kyoku-outcome"]
    assert previous_outputs["kyoku-outcome"]["version"] == 1
    assert all(item["version"] == 1 for item in old["outputContracts"])
    assert all("version" not in item for item in current["weightSlots"][0]["requiredForOutputs"])
    assert "point-estimate" in current_outputs["opponent-dora-count"]["representations"]


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
