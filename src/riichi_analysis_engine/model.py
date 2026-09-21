from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import torch
from torch import Tensor, nn

from .architecture import (
    ModelArchitecture,
    SemanticModelArchitecture,
    StructuredModelArchitecture,
)
from .constants import ACTION_SPACE, MORTAL_OBS_CHANNELS, OBS_CHANNELS, TILE_TYPES
from .kyoku_outcome import OUTCOME_COUNT
from .model_input import (
    ANALYSIS_CHANNELS as V9_ANALYSIS_CHANNELS,
)
from .model_input import (
    POLICY_CONTEXT_CHANNELS as V9_POLICY_CONTEXT_CHANNELS,
)
from .model_input import (
    split_model_input,
)
from .observation_layout import (
    ANALYSIS_CHANNELS as LEGACY_ANALYSIS_CHANNELS,
)
from .observation_layout import (
    POLICY_CONTEXT_CHANNELS as LEGACY_POLICY_CONTEXT_CHANNELS,
)
from .observation_layout import (
    POLICY_CONTEXT_START as LEGACY_POLICY_CONTEXT_START,
)
from .prediction_values import DORA_VALUES, SCORE_VALUES
from .semantic_model import SemanticRiichiModel


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // ratio)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.Mish(inplace=True),
            nn.Linear(hidden, channels),
        )
        for layer in self.shared_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)

    def forward(self, value: Tensor) -> Tensor:
        avg = self.shared_mlp(value.mean(dim=-1))
        maximum = self.shared_mlp(value.amax(dim=-1))
        weight = torch.sigmoid(avg + maximum).unsqueeze(-1)
        return value * weight


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        norm = partial(nn.BatchNorm1d, channels, momentum=0.01, eps=1e-3)
        self.residual = nn.Sequential(
            norm(),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            norm(),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
        )
        self.attention = ChannelAttention(channels)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.attention(self.residual(value))


class ResidualEncoder(nn.Module):
    """Mortal-style residual encoder with an explicit input boundary."""

    def __init__(
        self,
        input_channels: int,
        *,
        channels: int,
        blocks: int,
        latent_width: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(input_channels, channels, 3, padding=1, bias=False),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(blocks))
        layers.extend(
            [
                nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
                nn.Mish(inplace=True),
                nn.Conv1d(channels, 32, 3, padding=1),
                nn.Mish(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * TILE_TYPES, latent_width),
                nn.Mish(inplace=True),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)


class SpatialTrunk(nn.Module):
    """Shared low-level encoder that keeps the 34-tile axis intact."""

    def __init__(
        self,
        input_channels: int,
        *,
        channels: int,
        blocks: int,
        finalize: bool = True,
    ) -> None:
        super().__init__()
        self.input = nn.Conv1d(input_channels, channels, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.output = (
            nn.Sequential(
                nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
                nn.Mish(inplace=True),
            )
            if finalize
            else nn.Identity()
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.output(self.blocks(self.input(observation)))


class FamilyTower(nn.Module):
    """Private residual reasoning for one related family of predictions."""

    def __init__(self, channels: int, *, blocks: int, latent_width: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.summary = nn.Sequential(
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, 32, 3, padding=1),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * TILE_TYPES, latent_width),
            nn.Mish(inplace=True),
        )

    def forward(self, shared: Tensor) -> tuple[Tensor, Tensor]:
        spatial = self.blocks(shared)
        return spatial, self.summary(spatial)


class DensePredictionHead(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, output_width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.Mish(inplace=True),
            nn.Linear(hidden_width, output_width),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)


class TilePredictionHead(nn.Module):
    def __init__(
        self, input_channels: int, hidden_channels: int, output_channels: int
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, 1),
            nn.Mish(inplace=True),
            nn.Conv1d(hidden_channels, output_channels, 1),
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spatial: Tensor) -> Tensor:
        return self.net(spatial)


class MortalV4Encoder(ResidualEncoder):
    """Legacy full-observation encoder retained for v1--v5 weights."""

    def __init__(self, channels: int = 256, blocks: int = 54) -> None:
        super().__init__(
            MORTAL_OBS_CHANNELS,
            channels=channels,
            blocks=blocks,
            latent_width=1024,
        )


@dataclass(frozen=True)
class HeadDimensions:
    shanten: int = 3 * 7
    furiten_no_yaku: int = 3
    deal_in_tile: int = 3 * 34
    concealed_count: int = 3 * 34 * 5
    wall_count: int = 34 * 5
    concealed_red_count: int = 3 * 3 * 2
    wall_red_count: int = 3 * 2
    dora_distribution: int = 3 * len(DORA_VALUES)
    dora_point: int = 3
    score_distribution: int = 3 * len(SCORE_VALUES)
    score_point: int = 3
    outcome_any_win: int = 1
    outcome_winner: int = 4
    deal_in_player: int = 4
    target: int = 4 * 4
    kyoku_delta: int = 4
    placement: int = 24
    match_score: int = 4

    @property
    def state_total(self) -> int:
        return (
            self.shanten
            + self.furiten_no_yaku
            + self.deal_in_tile
            + self.concealed_count
            + self.wall_count
            + self.concealed_red_count
            + self.wall_red_count
        )

    @property
    def future_total(self) -> int:
        return (
            self.dora_distribution
            + self.dora_point
            + self.score_distribution
            + self.score_point
            + self.outcome_any_win
            + self.outcome_winner
            + self.deal_in_player
            + self.target
            + self.kyoku_delta
            + self.placement
            + self.match_score
        )


@dataclass(frozen=True)
class HeadDimensionsV2(HeadDimensions):
    concealed_red_count: int = 0
    wall_red_count: int = 0


@dataclass(frozen=True)
class HeadDimensionsV4(HeadDimensions):
    target: int = 0
    outcome: int = OUTCOME_COUNT

    @property
    def future_total(self) -> int:
        return super().future_total + self.outcome


@dataclass(frozen=True)
class HeadDimensionsV5(HeadDimensionsV4):
    outcome_any_win: int = 0
    outcome_winner: int = 0
    deal_in_player: int = 0


@dataclass(frozen=True)
class HeadDimensionsV7(HeadDimensionsV5):
    score_point: int = 0


@dataclass(frozen=True)
class HeadDimensionsV1:
    shanten: int = 3 * 7
    furiten_no_yaku: int = 3
    deal_in_tile: int = 3 * 34
    concealed_count: int = 3 * 34 * 5
    wall_count: int = 34 * 5
    dora: int = 3
    score: int = 3
    outcome: int = 16
    deal_in_player: int = 4
    target: int = 4 * 4
    kyoku_delta: int = 4
    placement: int = 24
    match_score: int = 4

    @property
    def state_total(self) -> int:
        return (
            self.shanten
            + self.furiten_no_yaku
            + self.deal_in_tile
            + self.concealed_count
            + self.wall_count
        )

    @property
    def future_total(self) -> int:
        return (
            self.dora
            + self.score
            + self.outcome
            + self.deal_in_player
            + self.target
            + self.kyoku_delta
            + self.placement
            + self.match_score
        )


class RiichiAnalysisModel(nn.Module):
    """Versioned multi-task model with preserved legacy loading paths.

    Formats v1--v7 preserve exported artifacts. New v8 training keeps a shared
    low-level spatial trunk, then gives each related task family private
    residual reasoning and task-specific output capacity.
    """

    def __init__(
        self,
        *,
        channels: int | None = None,
        blocks: int | None = None,
        state_width: int | None = None,
        future_width: int | None = None,
        format_version: int = 8,
        architecture: (
            ModelArchitecture
            | StructuredModelArchitecture
            | SemanticModelArchitecture
            | None
        ) = None,
    ) -> None:
        super().__init__()
        if format_version not in {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}:
            raise ValueError(f"unsupported model format version: {format_version}")
        if architecture is not None and format_version not in {
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
        }:
            raise ValueError(
                "only model formats v6 and later accept architecture metadata"
            )
        self.format_version = format_version
        self.architecture: (
            ModelArchitecture
            | StructuredModelArchitecture
            | SemanticModelArchitecture
            | None
        ) = None
        if format_version in {12, 13}:
            if any(
                value is not None
                for value in (channels, blocks, state_width, future_width)
            ):
                raise ValueError(
                    "semantic architecture must be configured through its metadata"
                )
            configured = architecture or SemanticModelArchitecture()
            if not isinstance(configured, SemanticModelArchitecture):
                raise TypeError(
                    "semantic model formats require SemanticModelArchitecture"
                )
            self.architecture = configured
            self.semantic_model = SemanticRiichiModel(
                configured, shared_rule_context=format_version == 13
            )
            return
        if format_version in {8, 9, 10, 11}:
            if any(
                value is not None
                for value in (channels, blocks, state_width, future_width)
            ):
                raise ValueError(
                    "structured architecture must be configured through its metadata"
                )
            configured = architecture or StructuredModelArchitecture()
            if not isinstance(configured, StructuredModelArchitecture):
                raise TypeError(
                    "model formats v8 through v11 require StructuredModelArchitecture"
                )
            self._init_v8(
                configured,
                analysis_channels=(
                    V9_ANALYSIS_CHANNELS
                    if format_version in {9, 10, 11}
                    else LEGACY_ANALYSIS_CHANNELS
                ),
                policy_context_channels=(
                    V9_POLICY_CONTEXT_CHANNELS
                    if format_version in {9, 10, 11}
                    else LEGACY_POLICY_CONTEXT_CHANNELS
                ),
                preserve_residual_boundary=format_version in {10, 11},
                shanten_hidden_width=(
                    configured.opponent_latent_width
                    if format_version in {10, 11}
                    else configured.task_width
                ),
            )
            if format_version in {10, 11}:
                self._init_reference_weights()
            return
        self.dimensions = (
            HeadDimensionsV1()
            if format_version == 1
            else HeadDimensionsV2()
            if format_version == 2
            else HeadDimensionsV7()
            if format_version == 7
            else HeadDimensionsV5()
            if format_version in {5, 6}
            else HeadDimensionsV4()
            if format_version == 4
            else HeadDimensions()
        )
        if format_version in {6, 7}:
            configured = architecture or ModelArchitecture()
            overrides: dict[str, int] = {}
            if channels is not None:
                overrides["analysis_channels"] = channels
            if blocks is not None:
                overrides["analysis_blocks"] = blocks
            if state_width is not None:
                overrides["state_width"] = state_width
            if future_width is not None:
                overrides["future_width"] = future_width
            self.architecture = replace(configured, **overrides)
            architecture = self.architecture
            self.encoder = ResidualEncoder(
                LEGACY_ANALYSIS_CHANNELS,
                channels=architecture.analysis_channels,
                blocks=architecture.analysis_blocks,
                latent_width=architecture.analysis_latent_width,
            )
            self.policy_context = ResidualEncoder(
                LEGACY_POLICY_CONTEXT_CHANNELS,
                channels=architecture.policy_context_channels,
                blocks=architecture.policy_context_blocks,
                latent_width=architecture.policy_context_width,
            )
            latent_width = architecture.analysis_latent_width
            state_width = architecture.state_width
            future_width = architecture.future_width
            self.policy_adapter = nn.Sequential(
                nn.Linear(
                    architecture.analysis_latent_width
                    + architecture.policy_context_width,
                    architecture.policy_width,
                ),
                nn.Mish(inplace=True),
            )
            self.policy_head = nn.Linear(architecture.policy_width, ACTION_SPACE)
        else:
            legacy_channels = 256 if channels is None else channels
            legacy_blocks = 54 if blocks is None else blocks
            latent_width = 1024
            state_width = 1024 if state_width is None else state_width
            future_width = 768 if future_width is None else future_width
            self.encoder = MortalV4Encoder(
                channels=legacy_channels, blocks=legacy_blocks
            )
        self.state_adapter = nn.Sequential(
            nn.Linear(latent_width, state_width),
            nn.Mish(inplace=True),
        )
        self.state_head = nn.Linear(state_width, self.dimensions.state_total)
        self.future_adapter = nn.Sequential(
            nn.Linear(latent_width, future_width),
            nn.Mish(inplace=True),
        )
        self.future_head = nn.Linear(future_width, self.dimensions.future_total)
        if format_version not in {6, 7}:
            self.policy_head = nn.Linear(latent_width, ACTION_SPACE)

        nn.init.zeros_(self.state_head.bias)
        nn.init.zeros_(self.future_head.bias)
        nn.init.zeros_(self.policy_head.bias)

    def _init_v8(
        self,
        architecture: StructuredModelArchitecture,
        *,
        analysis_channels: int,
        policy_context_channels: int,
        preserve_residual_boundary: bool = False,
        shanten_hidden_width: int | None = None,
    ) -> None:
        self.architecture = architecture
        channels = architecture.shared_channels
        family_latent = architecture.family_latent_width
        opponent_latent = architecture.opponent_latent_width
        policy_latent = architecture.policy_latent_width
        task = architecture.task_width
        self.shared_trunk = SpatialTrunk(
            analysis_channels,
            channels=channels,
            blocks=architecture.shared_blocks,
            finalize=not preserve_residual_boundary,
        )
        self.opponent_tower = FamilyTower(
            channels,
            blocks=architecture.opponent_blocks,
            latent_width=opponent_latent,
        )
        self.hidden_tower = FamilyTower(
            channels, blocks=architecture.hidden_blocks, latent_width=family_latent
        )
        self.value_tower = FamilyTower(
            channels, blocks=architecture.value_blocks, latent_width=family_latent
        )
        self.kyoku_tower = FamilyTower(
            channels, blocks=architecture.kyoku_blocks, latent_width=family_latent
        )
        self.match_tower = FamilyTower(
            channels, blocks=architecture.match_blocks, latent_width=family_latent
        )
        self.policy_tower = FamilyTower(
            channels,
            blocks=architecture.policy_blocks,
            latent_width=policy_latent,
        )
        self.policy_context = ResidualEncoder(
            policy_context_channels,
            channels=architecture.policy_context_channels,
            blocks=architecture.policy_context_blocks,
            latent_width=architecture.policy_context_width,
        )

        self.shanten_head = DensePredictionHead(
            opponent_latent, shanten_hidden_width or task, 3 * 7
        )
        self.furiten_head = DensePredictionHead(opponent_latent, task, 3)
        # The successful shanten/deal-in model decoded every wait from one
        # complete Mortal-style state.  A tile-local output shortcut regressed
        # that design and lets the matching input column dominate.  Keep the
        # 34 outputs independent, but let every one read the full opponent
        # representation.
        self.wait_head = nn.Linear(opponent_latent, 3 * TILE_TYPES)
        nn.init.zeros_(self.wait_head.bias)
        self.hidden_source_head = TilePredictionHead(
            channels, architecture.tile_width, 4
        )
        self.hidden_red_head = DensePredictionHead(family_latent, task, 3 * 4)
        self.dora_head = DensePredictionHead(family_latent, task, 3 * len(DORA_VALUES))
        self.dora_tail_head = DensePredictionHead(family_latent, task, 3)
        self.score_head = DensePredictionHead(
            family_latent, task, 3 * len(SCORE_VALUES)
        )
        self.outcome_head = DensePredictionHead(family_latent, task, OUTCOME_COUNT)
        self.kyoku_account_head = DensePredictionHead(family_latent, task, 5)
        self.placement_head = DensePredictionHead(family_latent, task, 24)
        self.match_score_head = DensePredictionHead(family_latent, task, 4)
        self.policy_head = DensePredictionHead(
            policy_latent + architecture.policy_context_width,
            architecture.policy_width,
            ACTION_SPACE,
        )

    def _init_reference_weights(self) -> None:
        """Match the explicit initialization used by the validated reference model."""

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _split(value: Tensor, dimensions: tuple[int, ...]) -> tuple[Tensor, ...]:
        return value.split(dimensions, dim=-1)

    def forward(
        self,
        observation: Tensor,
        event_tokens: Tensor | None = None,
        event_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.format_version in {12, 13}:
            if event_tokens is None or event_mask is None:
                raise ValueError("model format v12 requires semantic event memory")
            return self.semantic_model(observation, event_tokens, event_mask)
        if self.format_version in {9, 10, 11}:
            return self._forward_v9(observation)
        if self.format_version == 8:
            return self._forward_v8(observation)
        if self.format_version in {6, 7}:
            if observation.shape[1:] != (OBS_CHANNELS, TILE_TYPES):
                raise ValueError(f"wrong observation shape: {tuple(observation.shape)}")
            latent = self.encoder(observation[:, :LEGACY_ANALYSIS_CHANNELS])
            policy_context = self.policy_context(
                observation[:, LEGACY_POLICY_CONTEXT_START:]
            )
            policy = self.policy_head(
                self.policy_adapter(torch.cat((latent, policy_context), dim=-1))
            )
        else:
            latent = self.encoder(observation)
            policy = self.policy_head(latent)
        d = self.dimensions
        state = self.state_head(self.state_adapter(latent))
        future = self.future_head(self.future_adapter(latent))

        state_dimensions = (
            d.shanten,
            d.furiten_no_yaku,
            d.deal_in_tile,
            d.concealed_count,
            d.wall_count,
        )
        if self.format_version >= 3:
            assert isinstance(d, HeadDimensions)
            state_dimensions += (d.concealed_red_count, d.wall_red_count)
        state_parts = self._split(state, state_dimensions)
        shanten, furiten, deal_in, concealed, wall = state_parts[:5]
        batch = observation.shape[0]
        outputs = {
            "shanten": shanten.view(batch, 3, 7),
            "furiten_no_yaku": furiten.view(batch, 3),
            "deal_in_tile": deal_in.view(batch, 3, 34),
            "concealed_count": concealed.view(batch, 3, 34, 5),
            "wall_count": wall.view(batch, 34, 5),
            "policy": policy,
        }
        if self.format_version >= 3:
            concealed_red, wall_red = state_parts[5:]
            outputs.update(
                {
                    "concealed_red_count": concealed_red.view(batch, 3, 3, 2),
                    "wall_red_count": wall_red.view(batch, 3, 2),
                }
            )
        if self.format_version == 1:
            assert isinstance(d, HeadDimensionsV1)
            dora, score, outcome, deal_player, target, delta, placement, match_score = (
                self._split(
                    future,
                    (
                        d.dora,
                        d.score,
                        d.outcome,
                        d.deal_in_player,
                        d.target,
                        d.kyoku_delta,
                        d.placement,
                        d.match_score,
                    ),
                )
            )
            outputs.update(
                {
                    "dora": dora.view(batch, 3),
                    "score": score.view(batch, 3),
                    "outcome": outcome.view(batch, 16),
                    "deal_in_player": deal_player.view(batch, 4),
                    "target": target.view(batch, 4, 4),
                    "kyoku_delta": delta.view(batch, 4),
                    "placement": placement.view(batch, 24),
                    "match_score": match_score.view(batch, 4),
                }
            )
            return outputs

        assert isinstance(d, HeadDimensions)
        if self.format_version == 7:
            assert isinstance(d, HeadDimensionsV7)
            (
                dora_distribution,
                dora_point,
                score_distribution,
                outcome,
                delta,
                placement,
                match_score,
            ) = self._split(
                future,
                (
                    d.dora_distribution,
                    d.dora_point,
                    d.score_distribution,
                    d.outcome,
                    d.kyoku_delta,
                    d.placement,
                    d.match_score,
                ),
            )
            outputs.update(
                {
                    "dora_distribution": dora_distribution.view(
                        batch, 3, len(DORA_VALUES)
                    ),
                    "dora_point": dora_point.view(batch, 3),
                    "score_distribution": score_distribution.view(
                        batch, 3, len(SCORE_VALUES)
                    ),
                    "outcome": outcome.view(batch, OUTCOME_COUNT),
                    "kyoku_delta": delta.view(batch, 4),
                    "placement": placement.view(batch, 24),
                    "match_score": match_score.view(batch, 4),
                }
            )
            return outputs

        if self.format_version in {5, 6}:
            assert isinstance(d, HeadDimensionsV5)
            (
                dora_distribution,
                dora_point,
                score_distribution,
                score_point,
                outcome,
                delta,
                placement,
                match_score,
            ) = self._split(
                future,
                (
                    d.dora_distribution,
                    d.dora_point,
                    d.score_distribution,
                    d.score_point,
                    d.outcome,
                    d.kyoku_delta,
                    d.placement,
                    d.match_score,
                ),
            )
            outputs.update(
                {
                    "dora_distribution": dora_distribution.view(
                        batch, 3, len(DORA_VALUES)
                    ),
                    "dora_point": dora_point.view(batch, 3),
                    "score_distribution": score_distribution.view(
                        batch, 3, len(SCORE_VALUES)
                    ),
                    "score_point": score_point.view(batch, 3),
                    "outcome": outcome.view(batch, OUTCOME_COUNT),
                    "kyoku_delta": delta.view(batch, 4),
                    "placement": placement.view(batch, 24),
                    "match_score": match_score.view(batch, 4),
                }
            )
            return outputs

        if self.format_version == 4:
            assert isinstance(d, HeadDimensionsV4)
            (
                dora_distribution,
                dora_point,
                score_distribution,
                score_point,
                outcome_any_win,
                outcome_winner,
                deal_player,
                outcome,
                delta,
                placement,
                match_score,
            ) = self._split(
                future,
                (
                    d.dora_distribution,
                    d.dora_point,
                    d.score_distribution,
                    d.score_point,
                    d.outcome_any_win,
                    d.outcome_winner,
                    d.deal_in_player,
                    d.outcome,
                    d.kyoku_delta,
                    d.placement,
                    d.match_score,
                ),
            )
            outputs.update(
                {
                    "dora_distribution": dora_distribution.view(
                        batch, 3, len(DORA_VALUES)
                    ),
                    "dora_point": dora_point.view(batch, 3),
                    "score_distribution": score_distribution.view(
                        batch, 3, len(SCORE_VALUES)
                    ),
                    "score_point": score_point.view(batch, 3),
                    "outcome_any_win": outcome_any_win.view(batch),
                    "outcome_winner": outcome_winner.view(batch, 4),
                    "deal_in_player": deal_player.view(batch, 4),
                    "outcome": outcome.view(batch, OUTCOME_COUNT),
                    "kyoku_delta": delta.view(batch, 4),
                    "placement": placement.view(batch, 24),
                    "match_score": match_score.view(batch, 4),
                }
            )
            return outputs

        (
            dora_distribution,
            dora_point,
            score_distribution,
            score_point,
            outcome_any_win,
            outcome_winner,
            deal_player,
            target,
            delta,
            placement,
            match_score,
        ) = self._split(
            future,
            (
                d.dora_distribution,
                d.dora_point,
                d.score_distribution,
                d.score_point,
                d.outcome_any_win,
                d.outcome_winner,
                d.deal_in_player,
                d.target,
                d.kyoku_delta,
                d.placement,
                d.match_score,
            ),
        )
        outputs.update(
            {
                "dora_distribution": dora_distribution.view(batch, 3, len(DORA_VALUES)),
                "dora_point": dora_point.view(batch, 3),
                "score_distribution": score_distribution.view(
                    batch, 3, len(SCORE_VALUES)
                ),
                "score_point": score_point.view(batch, 3),
                "outcome_any_win": outcome_any_win.view(batch),
                "outcome_winner": outcome_winner.view(batch, 4),
                "deal_in_player": deal_player.view(batch, 4),
                "target": target.view(batch, 4, 4),
                "kyoku_delta": delta.view(batch, 4),
                "placement": placement.view(batch, 24),
                "match_score": match_score.view(batch, 4),
            }
        )
        return outputs

    def _forward_v8(self, observation: Tensor) -> dict[str, Tensor]:
        if observation.shape[1:] != (OBS_CHANNELS, TILE_TYPES):
            raise ValueError(f"wrong observation shape: {tuple(observation.shape)}")
        shared = self.shared_trunk(observation[:, :LEGACY_ANALYSIS_CHANNELS])
        return self._forward_v8_from_shared(observation, shared)

    def _forward_v9(self, observation: Tensor) -> dict[str, Tensor]:
        analysis, policy = split_model_input(observation)
        shared = self.shared_trunk(analysis)
        return self._forward_structured_from_shared(shared, self.policy_context(policy))

    def forward_with_shared(
        self, observation: Tensor
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Return v8 outputs and the shared feature boundary for diagnostics."""

        if self.format_version in {9, 10, 11}:
            analysis, policy = split_model_input(observation)
            shared = self.shared_trunk(analysis)
            return (
                self._forward_structured_from_shared(
                    shared, self.policy_context(policy)
                ),
                shared,
            )
        if self.format_version != 8:
            raise RuntimeError(
                "shared-feature diagnostics require model formats v8 through v11"
            )
        if observation.shape[1:] != (OBS_CHANNELS, TILE_TYPES):
            raise ValueError(f"wrong observation shape: {tuple(observation.shape)}")
        shared = self.shared_trunk(observation[:, :LEGACY_ANALYSIS_CHANNELS])
        return self._forward_v8_from_shared(observation, shared), shared

    def _forward_v8_from_shared(
        self, observation: Tensor, shared: Tensor
    ) -> dict[str, Tensor]:
        return self._forward_structured_from_shared(
            shared,
            self.policy_context(observation[:, LEGACY_POLICY_CONTEXT_START:]),
        )

    def _forward_structured_from_shared(
        self, shared: Tensor, policy_context: Tensor
    ) -> dict[str, Tensor]:
        batch = len(shared)
        opponent_spatial, opponent = self.opponent_tower(shared)
        analysis = opponent_spatial if self.format_version == 11 else shared
        hidden_spatial, hidden = self.hidden_tower(analysis)
        _value_spatial, value = self.value_tower(analysis)
        _kyoku_spatial, kyoku = self.kyoku_tower(analysis)
        _match_spatial, match = self.match_tower(analysis)
        _policy_spatial, policy_analysis = self.policy_tower(shared)
        wait = self.wait_head(opponent)
        hidden_source = self.hidden_source_head(hidden_spatial)
        return {
            "shanten": self.shanten_head(opponent).view(batch, 3, 7),
            "furiten_no_yaku": self.furiten_head(opponent).view(batch, 3),
            "deal_in_tile": wait.view(batch, 3, 34),
            "hidden_source_affinity": hidden_source.view(batch, 4, 34),
            "hidden_red_source": self.hidden_red_head(hidden).view(batch, 3, 4),
            "dora_distribution": self.dora_head(value).view(batch, 3, len(DORA_VALUES)),
            "dora_tail": self.dora_tail_head(value).view(batch, 3),
            "score_distribution": self.score_head(value).view(
                batch, 3, len(SCORE_VALUES)
            ),
            "outcome": self.outcome_head(kyoku).view(batch, OUTCOME_COUNT),
            "kyoku_accounts": self.kyoku_account_head(kyoku).view(batch, 5),
            "placement": self.placement_head(match).view(batch, 24),
            "match_score": self.match_score_head(match).view(batch, 4),
            "policy": self.policy_head(
                torch.cat((policy_analysis, policy_context), dim=-1)
            ),
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    if model.format_version in {12, 13}:
        groups = {
            "input": model.semantic_model.input,
            "backbone": model.semantic_model.backbone,
            "decoder": model.semantic_model.decoder,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["total"] = sum(parameter.numel() for parameter in model.parameters())
        return counts
    if model.format_version in {8, 9, 10, 11}:
        groups: dict[str, nn.Module] = {
            "shared": model.shared_trunk,
            "opponent": nn.ModuleList(
                [
                    model.opponent_tower,
                    model.shanten_head,
                    model.furiten_head,
                    model.wait_head,
                ]
            ),
            "hidden": nn.ModuleList(
                [model.hidden_tower, model.hidden_source_head, model.hidden_red_head]
            ),
            "value": nn.ModuleList(
                [
                    model.value_tower,
                    model.dora_head,
                    model.dora_tail_head,
                    model.score_head,
                ]
            ),
            "kyoku": nn.ModuleList(
                [model.kyoku_tower, model.outcome_head, model.kyoku_account_head]
            ),
            "match": nn.ModuleList(
                [model.match_tower, model.placement_head, model.match_score_head]
            ),
            "policy_context": model.policy_context,
            "policy": nn.ModuleList([model.policy_tower, model.policy_head]),
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in groups.items()
        }
        counts["total"] = sum(parameter.numel() for parameter in model.parameters())
        return counts
    groups: dict[str, nn.Module] = {
        "encoder": model.encoder,
        "state": nn.ModuleList([model.state_adapter, model.state_head]),
        "future": nn.ModuleList([model.future_adapter, model.future_head]),
    }
    if model.format_version in {6, 7}:
        groups["policy_context"] = model.policy_context
        groups["policy"] = nn.ModuleList([model.policy_adapter, model.policy_head])
    else:
        groups["policy"] = model.policy_head
    counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in groups.items()
    }
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    return counts
