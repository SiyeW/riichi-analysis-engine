from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelArchitecture:
    """The trainable shape of the current model format.

    Architecture is weight metadata, rather than an implicit collection of
    constructor defaults.  A weight file can therefore be reconstructed on a
    different machine without guessing the bottleneck widths it was trained
    with.
    """

    observation_version: int = 4
    analysis_channels: int = 192
    analysis_blocks: int = 36
    analysis_latent_width: int = 768
    state_width: int = 768
    future_width: int = 640
    policy_context_channels: int = 96
    policy_context_blocks: int = 4
    policy_context_width: int = 256
    policy_width: int = 640

    def __post_init__(self) -> None:
        if self.observation_version != 4:
            raise ValueError("only observation version 4 is supported")
        positive = {
            "analysis_channels": self.analysis_channels,
            "analysis_blocks": self.analysis_blocks,
            "analysis_latent_width": self.analysis_latent_width,
            "state_width": self.state_width,
            "future_width": self.future_width,
            "policy_context_channels": self.policy_context_channels,
            "policy_context_blocks": self.policy_context_blocks,
            "policy_context_width": self.policy_context_width,
            "policy_width": self.policy_width,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"architecture dimensions must be positive: {', '.join(invalid)}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> "ModelArchitecture":
        if not isinstance(value, dict):
            raise ValueError("model architecture must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(
                "model architecture fields do not match "
                f"(missing={missing}, extra={extra})"
            )
        if not all(type(item) is int for item in value.values()):
            raise ValueError("model architecture values must be integers")
        return cls(**value)
