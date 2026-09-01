from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .runtime import AnalysisRuntime

PROTOCOL = {"name": "riichi-engine-protocol", "major": 2, "minor": 2}
ENGINE_VERSION = "0.1.0-dev.6"
OUTPUT_IDS = [
    "action-recommendation",
    "opponent-shanten",
    "opponent-deal-in-probability",
    "opponent-concealed-tile-count",
    "wall-tile-count",
    "opponent-dora-count",
    "opponent-score",
    "kyoku-outcome",
    "kyoku-score-delta",
    "match-placement",
    "match-score",
]
NUMERIC_REPRESENTATIONS = {
    "opponent-concealed-tile-count": ["distribution", "expected-value"],
    "wall-tile-count": ["distribution", "expected-value"],
    "kyoku-score-delta": ["expected-value"],
    "match-placement": ["distribution", "expected-value"],
    "match-score": ["expected-value"],
}
OUTPUT_INTRODUCED = {
    "kyoku-outcome": 1,
    "kyoku-score-delta": 1,
    "match-placement": 1,
    "match-score": 1,
}
POLICY_METRIC = {
    "id": "policy",
    "title": {"default": "Policy", "zh-CN": "策略概率", "ja-JP": "方策確率"},
    "format": "percentage",
    "fractionDigits": 2,
    "preferredDirection": "higher",
}


class ProtocolError(Exception):
    def __init__(self, message: str, error_code: str, code: int = -32000) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.code = code


def available_output_ids(protocol_minor: int) -> list[str]:
    return [
        output_id
        for output_id in OUTPUT_IDS
        if OUTPUT_INTRODUCED.get(output_id, 0) <= protocol_minor
    ]


def numeric_representations(output_id: str, protocol_minor: int) -> list[str] | None:
    if output_id == "opponent-dora-count":
        if protocol_minor >= 2:
            return ["distribution", "expected-value", "point-estimate"]
        return ["expected-value"]
    if output_id == "opponent-score":
        if protocol_minor >= 2:
            return ["distribution", "expected-value", "point-estimate"]
        return ["distribution", "expected-value"]
    return NUMERIC_REPRESENTATIONS.get(output_id)


def output_reference(output_id: str, protocol_minor: int) -> dict[str, Any]:
    reference: dict[str, Any] = {"id": output_id}
    if protocol_minor < 2:
        reference["version"] = 1
    return reference


def requested_output_ids(values: Any, protocol_minor: int) -> list[str] | None:
    if not isinstance(values, list):
        return None
    result: list[str] = []
    for item in values:
        if not isinstance(item, dict):
            return None
        output_id = item.get("id")
        if not isinstance(output_id, str) or not output_id:
            return None
        if protocol_minor < 2:
            if item.get("version") != 1:
                return None
        elif "version" in item:
            return None
        result.append(output_id)
    return result


def output_declaration(
    output_id: str,
    protocol_minor: int,
    *,
    initialized: bool = False,
    representations: list[str] | None = None,
) -> dict[str, Any]:
    result = output_reference(output_id, protocol_minor)
    representations = (
        numeric_representations(output_id, protocol_minor)
        if representations is None
        else representations
    )
    if representations is not None:
        result["representations"] = representations
    if output_id == "action-recommendation":
        result["metrics"] = [POLICY_METRIC]
        if initialized:
            result["primaryMetricId"] = "policy"
            result["recommendationMetricId"] = "policy"
    return result


class Engine:
    def __init__(self) -> None:
        self.runtime: AnalysisRuntime | None = None
        self.enabled: set[str] = set()
        self.state = "starting"
        self.protocol_minor: int | None = None

    def hello(self, params: dict[str, Any]) -> dict[str, Any]:
        protocol = params.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("name") != PROTOCOL["name"]
            or protocol.get("major") != 2
            or isinstance(protocol.get("minor"), bool)
            or not isinstance(protocol.get("minor"), int)
            or protocol.get("minor") < 0
        ):
            raise ProtocolError("protocol version is not compatible", "PROTOCOL_MISMATCH")
        self.protocol_minor = min(protocol["minor"], PROTOCOL["minor"])
        outputs = available_output_ids(self.protocol_minor)
        devices = [{"type": "cpu", "title": {"default": "CPU"}}]
        if torch.cuda.is_available():
            devices.append({"type": "cuda", "title": {"default": "NVIDIA CUDA"}})
        return {
            "protocol": {**PROTOCOL, "minor": self.protocol_minor},
            "engine": {"id": "org.riichi.analysis", "name": "Riichi Analysis Engine", "version": ENGINE_VERSION},
            "outputContracts": [
                output_declaration(value, self.protocol_minor) for value in outputs
            ],
            "weightSlots": [
                {
                    "id": "model",
                    "title": {"default": "Model weights", "zh-CN": "模型权重", "ja-JP": "モデルの重み"},
                    "formats": [{"id": "riichi-analysis-pytorch-v1", "extensions": [".pt"]}],
                    "requiredForOutputs": [
                        output_reference(value, self.protocol_minor)
                        for value in outputs
                    ],
                }
            ],
            "devices": devices,
            "runtimeCapabilities": {"multipleSessions": True, "concurrentRequests": False, "cancellation": False},
            "optionsSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}, "additionalProperties": False},
        }

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.protocol_minor is None:
            raise ProtocolError("engine.hello is required", "PROTOCOL_MISMATCH")
        enabled = params.get("enabledOutputs")
        if not isinstance(enabled, list) or not enabled:
            raise ProtocolError("enabledOutputs must be non-empty", "UNSUPPORTED_OUTPUT")
        ids = requested_output_ids(enabled, self.protocol_minor)
        available = set(available_output_ids(self.protocol_minor))
        if ids is None or len(set(ids)) != len(ids) or not set(ids).issubset(available):
            raise ProtocolError("enabledOutputs contains an unavailable output", "UNSUPPORTED_OUTPUT")
        weights = params.get("weights")
        if (
            not isinstance(weights, list)
            or len(weights) != 1
            or weights[0].get("slotId") != "model"
            or weights[0].get("format") != "riichi-analysis-pytorch-v1"
        ):
            raise ProtocolError("model weights are required", "INVALID_WEIGHTS")
        if params.get("options") != {}:
            raise ProtocolError("options must be empty", "INVALID_OPTIONS")
        path = Path(weights[0].get("path", ""))
        device_type = params.get("device", {}).get("type")
        if device_type not in {"cpu", "cuda"} or device_type == "cuda" and not torch.cuda.is_available():
            raise ProtocolError("requested device is unavailable", "UNSUPPORTED_DEVICE")
        self.state = "loading"
        self.runtime = AnalysisRuntime(path, device_type)
        self.enabled = set(ids)
        self.state = "ready"
        return {
            "outputs": [
                output_declaration(
                    value,
                    self.protocol_minor,
                    initialized=True,
                    representations=(
                        self.runtime.representations(value, self.protocol_minor)
                        if value in {"opponent-dora-count", "opponent-score"}
                        else None
                    ),
                )
                for value in ids
            ],
            "device": {"type": device_type},
            "effectiveOptions": {},
        }

    def analyze(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.runtime is None:
            raise ProtocolError("engine is not initialized", "ENGINE_NOT_INITIALIZED")
        seat = params.get("controlledSeat")
        events = params.get("events")
        requested = params.get("outputs")
        if isinstance(seat, bool) or not isinstance(seat, int) or not 0 <= seat <= 3:
            raise ProtocolError("controlledSeat must be 0..3", "INVALID_HISTORY")
        if params.get("inputMode") != "standard" or not isinstance(events, list) or not events:
            raise ProtocolError("standard non-empty history is required", "INVALID_HISTORY")
        if not isinstance(requested, list) or not requested:
            raise ProtocolError("outputs must be non-empty", "UNSUPPORTED_OUTPUT")
        ids = requested_output_ids(requested, self.protocol_minor or 0)
        if ids is None or len(set(ids)) != len(ids) or not set(ids).issubset(self.enabled):
            raise ProtocolError("outputs contains an unavailable output", "UNSUPPORTED_OUTPUT")
        try:
            data, elapsed = self.runtime.predict(
                events, seat, requested, self.protocol_minor or 0
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(str(error), "INVALID_HISTORY") from error
        return {
            "outputs": [
                {
                    **output_reference(value, self.protocol_minor or 0),
                    "data": data[value],
                }
                for value in ids
            ],
            "timing": {"totalMs": elapsed},
        }


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def error_payload(error: Exception) -> dict[str, Any]:
    if isinstance(error, ProtocolError):
        return {"code": error.code, "message": str(error), "data": {"errorCode": error.error_code, "recoverable": False}}
    return {"code": -32603, "message": str(error) or type(error).__name__, "data": {"errorCode": "INTERNAL_ERROR", "recoverable": False}}


def main() -> None:
    engine = Engine()
    for line in sys.stdin:
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
                raise ProtocolError("invalid request", "INVALID_REQUEST", -32600)
            method = request["method"]
            params = request.get("params", {})
            with contextlib.redirect_stdout(sys.stderr):
                if method == "engine.hello":
                    result = engine.hello(params)
                    engine.state = "uninitialized"
                elif method == "engine.initialize":
                    result = engine.initialize(params)
                elif method == "analysis.run":
                    result = engine.analyze(params)
                elif method == "engine.getStatus":
                    result = {"state": engine.state, "activeTasks": 0, "queuedTasks": 0, "lastError": None}
                elif method in {"session.reset", "session.close"} or method == "engine.shutdown":
                    result = {"ok": True}
                else:
                    raise ProtocolError("method not found", "METHOD_NOT_FOUND", -32601)
            if "id" in request:
                emit({"jsonrpc": "2.0", "id": request["id"], "result": result})
            if method == "engine.shutdown":
                break
        except Exception as error:  # noqa: BLE001 -- JSON-RPC errors belong on the wire
            if not isinstance(request, dict) or "id" in request:
                emit({"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None, "error": error_payload(error)})


if __name__ == "__main__":
    main()
