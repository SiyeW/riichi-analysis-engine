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
    analysis_channels: int = 288
    analysis_blocks: int = 54
    analysis_latent_width: int = 1152
    state_width: int = 1024
    future_width: int = 1024
    policy_context_channels: int = 144
    policy_context_blocks: int = 6
    policy_context_width: int = 384
    policy_width: int = 1024

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
            raise ValueError(
                f"architecture dimensions must be positive: {', '.join(invalid)}"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> ModelArchitecture:
        if not isinstance(value, dict):
            raise TypeError("model architecture must be an object")
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


@dataclass(frozen=True)
class StructuredModelArchitecture:
    """Trainable shape of the v8 family-tower model.

    The shared trunk learns low-level public-state features. Each prediction
    family then owns residual depth and task-specific heads, so unrelated
    objectives cannot force every useful abstraction through one shallow
    adapter.
    """

    observation_version: int = 4
    shared_channels: int = 256
    shared_blocks: int = 30
    family_latent_width: int = 768
    opponent_latent_width: int = 1024
    policy_latent_width: int = 1024
    opponent_blocks: int = 24
    hidden_blocks: int = 8
    value_blocks: int = 6
    kyoku_blocks: int = 6
    match_blocks: int = 4
    policy_blocks: int = 24
    task_width: int = 512
    tile_width: int = 128
    policy_context_channels: int = 144
    policy_context_blocks: int = 6
    policy_context_width: int = 384
    policy_width: int = 1024

    def __post_init__(self) -> None:
        if self.observation_version != 4:
            raise ValueError("only observation version 4 is supported")
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name != "observation_version"
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"architecture dimensions must be positive: {', '.join(invalid)}"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> StructuredModelArchitecture:
        if not isinstance(value, dict):
            raise TypeError("model architecture must be an object")
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


@dataclass(frozen=True)
class SemanticModelArchitecture:
    """Shape of the shared semantic contract used by the v12 candidates.

    CNN and Transformer candidates deliberately share every field except the
    backbone name. This keeps comparisons on one input, decoder and target
    contract instead of accidentally comparing two data pipelines.
    """

    backbone: str = "cnn"
    observation_version: int = 4
    width: int = 256
    stem_width: int = 384
    event_width: int = 192
    backbone_blocks: int = 8
    event_blocks: int = 4
    decoder_width: int = 512
    attention_heads: int = 8
    transformer_ff_multiplier: int = 4
    transformer_tile_prior_blocks: int = 0
    transformer_event_prior_blocks: int = 0

    def __post_init__(self) -> None:
        if self.backbone not in {"cnn", "transformer"}:
            raise ValueError("semantic backbone must be cnn or transformer")
        if self.observation_version != 4:
            raise ValueError("only observation version 4 is supported")
        positive = {
            name: value
            for name, value in asdict(self).items()
            if name
            not in {
                "backbone",
                "observation_version",
                "transformer_tile_prior_blocks",
                "transformer_event_prior_blocks",
            }
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f"architecture dimensions must be positive: {', '.join(invalid)}"
            )
        non_negative = {
            "transformer_tile_prior_blocks": self.transformer_tile_prior_blocks,
            "transformer_event_prior_blocks": self.transformer_event_prior_blocks,
        }
        invalid = [name for name, value in non_negative.items() if value < 0]
        if invalid:
            raise ValueError(
                "architecture block counts must be non-negative: "
                f"{', '.join(invalid)}"
            )
        if self.width % self.attention_heads:
            raise ValueError("semantic width must be divisible by attention heads")

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> SemanticModelArchitecture:
        if not isinstance(value, dict):
            raise TypeError("model architecture must be an object")
        expected = set(cls.__dataclass_fields__)
        optional = {
            "transformer_ff_multiplier",
            "transformer_tile_prior_blocks",
            "transformer_event_prior_blocks",
        }
        missing = expected - set(value)
        extra = set(value) - expected
        if extra or missing - optional:
            raise ValueError(
                "model architecture fields do not match "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )
        if type(value["backbone"]) is not str or not all(
            type(item) is int
            for name, item in value.items()
            if name != "backbone"
        ):
            raise ValueError("model architecture values have invalid types")
        # v12 checkpoints written before the prior-enhanced Transformer did not
        # carry these Transformer-only controls.  Their historical topology was
        # a 4x FFN without explicit tile or event prior blocks.
        compatible = {
            "transformer_ff_multiplier": 4,
            "transformer_tile_prior_blocks": 0,
            "transformer_event_prior_blocks": 0,
            **value,
        }
        return cls(**compatible)
