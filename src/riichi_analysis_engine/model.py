from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor, nn

from .constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = channels // ratio
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


class MortalV4Encoder(nn.Module):
    """Mortal v4-style residual encoder for public observations."""

    def __init__(self, channels: int = 256, blocks: int = 54) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(OBS_CHANNELS, channels, 3, padding=1, bias=False),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(blocks))
        layers.extend(
            [
                nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
                nn.Mish(inplace=True),
                nn.Conv1d(channels, 32, 3, padding=1),
                nn.Mish(inplace=True),
                nn.Flatten(),
                nn.Linear(32 * TILE_TYPES, 1024),
                nn.Mish(inplace=True),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, observation: Tensor) -> Tensor:
        return self.net(observation)


@dataclass(frozen=True)
class HeadDimensions:
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
    """One shared encoder with separate state, future and policy adapters."""

    def __init__(
        self,
        *,
        channels: int = 256,
        blocks: int = 54,
        state_width: int = 1024,
        future_width: int = 768,
    ) -> None:
        super().__init__()
        self.dimensions = HeadDimensions()
        self.encoder = MortalV4Encoder(channels=channels, blocks=blocks)
        self.state_adapter = nn.Sequential(
            nn.Linear(1024, state_width),
            nn.Mish(inplace=True),
        )
        self.state_head = nn.Linear(state_width, self.dimensions.state_total)
        self.future_adapter = nn.Sequential(
            nn.Linear(1024, future_width),
            nn.Mish(inplace=True),
        )
        self.future_head = nn.Linear(future_width, self.dimensions.future_total)
        self.policy_head = nn.Linear(1024, ACTION_SPACE)

        nn.init.zeros_(self.state_head.bias)
        nn.init.zeros_(self.future_head.bias)
        nn.init.zeros_(self.policy_head.bias)

    @staticmethod
    def _split(value: Tensor, dimensions: tuple[int, ...]) -> tuple[Tensor, ...]:
        return value.split(dimensions, dim=-1)

    def forward(self, observation: Tensor) -> dict[str, Tensor]:
        latent = self.encoder(observation)
        d = self.dimensions
        state = self.state_head(self.state_adapter(latent))
        future = self.future_head(self.future_adapter(latent))

        shanten, furiten, deal_in, concealed, wall = self._split(
            state,
            (d.shanten, d.furiten_no_yaku, d.deal_in_tile, d.concealed_count, d.wall_count),
        )
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
        batch = observation.shape[0]
        return {
            "shanten": shanten.view(batch, 3, 7),
            "furiten_no_yaku": furiten.view(batch, 3),
            "deal_in_tile": deal_in.view(batch, 3, 34),
            "concealed_count": concealed.view(batch, 3, 34, 5),
            "wall_count": wall.view(batch, 34, 5),
            "dora": dora.view(batch, 3),
            "score": score.view(batch, 3),
            "outcome": outcome.view(batch, 16),
            "deal_in_player": deal_player.view(batch, 4),
            "target": target.view(batch, 4, 4),
            "kyoku_delta": delta.view(batch, 4),
            "placement": placement.view(batch, 24),
            "match_score": match_score.view(batch, 4),
            "policy": self.policy_head(latent),
        }


def count_parameters(model: nn.Module) -> dict[str, int]:
    groups = {
        "encoder": model.encoder,
        "state": nn.ModuleList([model.state_adapter, model.state_head]),
        "future": nn.ModuleList([model.future_adapter, model.future_head]),
        "policy": model.policy_head,
    }
    counts = {
        name: sum(parameter.numel() for parameter in module.parameters())
        for name, module in groups.items()
    }
    counts["total"] = sum(parameter.numel() for parameter in model.parameters())
    return counts
