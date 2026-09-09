from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import torch
from torch import Tensor, nn

from .constants import ACTION_SPACE, MORTAL_OBS_CHANNELS, OBS_CHANNELS, TILE_TYPES
from .architecture import ModelArchitecture
from .kyoku_outcome import OUTCOME_COUNT
from .observation_layout import ANALYSIS_CHANNELS, POLICY_CONTEXT_CHANNELS, POLICY_CONTEXT_START
from .prediction_values import DORA_VALUES, SCORE_VALUES


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
    """Shared state analysis with a separate decision-only feature path.

    Formats v1--v5 preserve the old full-observation architecture solely for
    loading already exported artifacts.  New training uses v6, where policy
    phase/action features and single-player EV tables no longer consume the
    shared prediction encoder's capacity.
    """

    def __init__(
        self,
        *,
        channels: int | None = None,
        blocks: int | None = None,
        state_width: int | None = None,
        future_width: int | None = None,
        format_version: int = 6,
        architecture: ModelArchitecture | None = None,
    ) -> None:
        super().__init__()
        if format_version not in {1, 2, 3, 4, 5, 6}:
            raise ValueError(f"unsupported model format version: {format_version}")
        if architecture is not None and format_version != 6:
            raise ValueError("only model format v6 accepts explicit architecture metadata")
        self.format_version = format_version
        self.architecture: ModelArchitecture | None = None
        self.dimensions = (
            HeadDimensionsV1()
            if format_version == 1
            else HeadDimensionsV2()
            if format_version == 2
            else HeadDimensionsV5()
            if format_version in {5, 6}
            else HeadDimensionsV4()
            if format_version == 4
            else HeadDimensions()
        )
        if format_version == 6:
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
                ANALYSIS_CHANNELS,
                channels=architecture.analysis_channels,
                blocks=architecture.analysis_blocks,
                latent_width=architecture.analysis_latent_width,
            )
            self.policy_context = ResidualEncoder(
                POLICY_CONTEXT_CHANNELS,
                channels=architecture.policy_context_channels,
                blocks=architecture.policy_context_blocks,
                latent_width=architecture.policy_context_width,
            )
            latent_width = architecture.analysis_latent_width
            state_width = architecture.state_width
            future_width = architecture.future_width
            self.policy_adapter = nn.Sequential(
                nn.Linear(
                    architecture.analysis_latent_width + architecture.policy_context_width,
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
            self.encoder = MortalV4Encoder(channels=legacy_channels, blocks=legacy_blocks)
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
        if format_version != 6:
            self.policy_head = nn.Linear(latent_width, ACTION_SPACE)

        nn.init.zeros_(self.state_head.bias)
        nn.init.zeros_(self.future_head.bias)
        nn.init.zeros_(self.policy_head.bias)

    @staticmethod
    def _split(value: Tensor, dimensions: tuple[int, ...]) -> tuple[Tensor, ...]:
        return value.split(dimensions, dim=-1)

    def forward(self, observation: Tensor) -> dict[str, Tensor]:
        if self.format_version == 6:
            if observation.shape[1:] != (OBS_CHANNELS, TILE_TYPES):
                raise ValueError(f"wrong observation shape: {tuple(observation.shape)}")
            latent = self.encoder(observation[:, :ANALYSIS_CHANNELS])
            policy_context = self.policy_context(observation[:, POLICY_CONTEXT_START:])
            policy = self.policy_head(self.policy_adapter(torch.cat((latent, policy_context), dim=-1)))
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
            dora, score, outcome, deal_player, target, delta, placement, match_score = self._split(
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
                    "dora_distribution": dora_distribution.view(batch, 3, len(DORA_VALUES)),
                    "dora_point": dora_point.view(batch, 3),
                    "score_distribution": score_distribution.view(batch, 3, len(SCORE_VALUES)),
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
                    "dora_distribution": dora_distribution.view(batch, 3, len(DORA_VALUES)),
                    "dora_point": dora_point.view(batch, 3),
                    "score_distribution": score_distribution.view(batch, 3, len(SCORE_VALUES)),
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
                "score_distribution": score_distribution.view(batch, 3, len(SCORE_VALUES)),
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


def count_parameters(model: nn.Module) -> dict[str, int]:
    groups: dict[str, nn.Module] = {
        "encoder": model.encoder,
        "state": nn.ModuleList([model.state_adapter, model.state_head]),
        "future": nn.ModuleList([model.future_adapter, model.future_head]),
    }
    if model.format_version == 6:
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
