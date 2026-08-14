from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from .runtime import AnalysisRuntime

PROTOCOL = {"name": "riichi-engine-protocol", "major": 2, "minor": 1}
ENGINE_VERSION = "0.1.0-dev.0"
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
    "opponent-dora-count": ["expected-value"],
    "opponent-score": ["expected-value"],
    "kyoku-score-delta": ["expected-value"],
    "match-placement": ["distribution", "expected-value"],
    "match-score": ["expected-value"],
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


def output_declaration(output_id: str, *, initialized: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"id": output_id, "version": 1}
    if output_id in NUMERIC_REPRESENTATIONS:
        result["representations"] = NUMERIC_REPRESENTATIONS[output_id]
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

    def hello(self, params: dict[str, Any]) -> dict[str, Any]:
        protocol = params.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("name") != PROTOCOL["name"]
            or protocol.get("major") != 2
            or isinstance(protocol.get("minor"), bool)
            or not isinstance(protocol.get("minor"), int)
            or protocol.get("minor") < 1
        ):
            raise ProtocolError("protocol version is not compatible", "PROTOCOL_MISMATCH")
        devices = [{"type": "cpu", "title": {"default": "CPU"}}]
        if torch.cuda.is_available():
            devices.append({"type": "cuda", "title": {"default": "NVIDIA CUDA"}})
        return {
            "protocol": PROTOCOL,
            "engine": {"id": "org.riichi.analysis", "name": "Riichi Analysis Engine", "version": ENGINE_VERSION},
            "outputContracts": [output_declaration(value) for value in OUTPUT_IDS],
            "weightSlots": [
                {
                    "id": "model",
                    "title": {"default": "Model weights", "zh-CN": "模型权重", "ja-JP": "モデルの重み"},
                    "formats": [{"id": "riichi-analysis-pytorch-v1", "extensions": [".pt"]}],
                    "requiredForOutputs": [{"id": value, "version": 1} for value in OUTPUT_IDS],
                }
            ],
            "devices": devices,
            "runtimeCapabilities": {"multipleSessions": True, "concurrentRequests": False, "cancellation": False},
            "optionsSchema": {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {}, "additionalProperties": False},
        }

    def initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        enabled = params.get("enabledOutputs")
        if not isinstance(enabled, list) or not enabled:
            raise ProtocolError("enabledOutputs must be non-empty", "UNSUPPORTED_OUTPUT")
        ids = [item.get("id") for item in enabled if isinstance(item, dict) and item.get("version") == 1]
        if len(ids) != len(enabled) or len(set(ids)) != len(ids) or not set(ids).issubset(OUTPUT_IDS):
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
            "outputs": [output_declaration(value, initialized=True) for value in ids],
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
        ids = [item.get("id") for item in requested if isinstance(item, dict) and item.get("version") == 1]
        if len(ids) != len(requested) or len(set(ids)) != len(ids) or not set(ids).issubset(self.enabled):
            raise ProtocolError("outputs contains an unavailable output", "UNSUPPORTED_OUTPUT")
        try:
            data, elapsed = self.runtime.predict(events, seat, requested)
        except (KeyError, TypeError, ValueError) as error:
            raise ProtocolError(str(error), "INVALID_HISTORY") from error
        return {
            "outputs": [{"id": value, "version": 1, "data": data[value]} for value in ids],
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
