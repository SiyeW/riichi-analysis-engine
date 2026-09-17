import pytest
import torch

from riichi_analysis_engine.architecture import (
    ModelArchitecture,
    StructuredModelArchitecture,
)
from riichi_analysis_engine.constants import ACTION_SPACE, OBS_CHANNELS, TILE_TYPES
from riichi_analysis_engine.losses import (
    LOSS_TERMS,
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    _masked_mean,
    masked_score_logits,
    multitask_loss,
    opponent_dealer_mask,
    score_class_indices,
)
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.model_input import MODEL_INPUT_CHANNELS
from riichi_analysis_engine.observation_layout import JIKAZE_CHANNEL, WIND_TILE_START
from riichi_analysis_engine.prediction_values import SCORE_VALUES, score_class_mask
from riichi_analysis_engine.structured_outputs import (
    conditional_deal_in_probabilities,
    fixed_total_values,
    zero_sum_accounts,
)
from riichi_analysis_engine.train import validate


def test_masked_mean_does_not_multiply_non_finite_unselected_values() -> None:
    values = torch.tensor([2.0, float("inf")])
    mask = torch.tensor([True, False])

    assert _masked_mean(values, mask).item() == 2.0


def test_masked_mean_of_empty_selection_is_zero() -> None:
    values = torch.tensor([1.0, 2.0])
    mask = torch.tensor([False, False])

    assert _masked_mean(values, mask).item() == 0.0


def test_score_class_indices_use_actual_settlement_values() -> None:
    indices = score_class_indices(torch.tensor([1000, 7700, 288000]))
    assert indices.tolist() == sorted(indices.tolist())


def test_score_class_indices_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="700"):
        score_class_indices(torch.tensor([700]))


def test_dealer_status_is_derived_from_existing_observation() -> None:
    observation = torch.zeros(4, OBS_CHANNELS, TILE_TYPES)
    for wind in range(4):
        observation[wind, JIKAZE_CHANNEL, WIND_TILE_START + wind] = 1

    assert opponent_dealer_mask(observation).tolist() == [
        [False, False, False],
        [False, False, True],
        [False, True, False],
        [True, False, False],
    ]


def test_score_logits_mask_impossible_dealer_classes() -> None:
    observation = torch.zeros(2, OBS_CHANNELS, TILE_TYPES)
    observation[0, JIKAZE_CHANNEL, WIND_TILE_START] = 1
    observation[1, JIKAZE_CHANNEL, WIND_TILE_START + 3] = 1
    outputs = {"score_distribution": torch.zeros(2, 3, len(SCORE_VALUES))}

    logits = masked_score_logits(outputs, observation)

    assert torch.isfinite(logits[0, 0]).tolist() == list(score_class_mask(dealer=False))
    assert torch.isfinite(logits[1, 0]).tolist() == list(score_class_mask(dealer=True))
    assert torch.isfinite(logits[1, 1]).tolist() == list(score_class_mask(dealer=False))


def test_uncertainty_balancer_learns_only_from_active_terms() -> None:
    balancer = LearnedUncertaintyBalancer()
    losses = {name: torch.tensor(2.0, requires_grad=True) for name in LOSS_TERMS}
    active = {name: name != "dora_point" for name in LOSS_TERMS}

    total, weights = balancer(losses, active)
    total.backward()

    assert set(weights) == set(LOSS_TERMS)
    inactive_index = LOSS_TERMS.index("dora_point")
    assert balancer.log_variance.grad is not None
    assert balancer.log_variance.grad[inactive_index].item() == 0.0
    assert balancer.log_variance.grad[LOSS_TERMS.index("policy")].item() != 0.0


def test_balancer_can_select_a_bounded_ablation_loss_set() -> None:
    outputs = {
        "policy": torch.zeros(1, ACTION_SPACE, requires_grad=True),
        "shanten": torch.zeros(1, 3, 7, requires_grad=True),
        "furiten_no_yaku": torch.zeros(1, 3, requires_grad=True),
        "deal_in_tile": torch.zeros(1, 3, 34, requires_grad=True),
    }
    # Complete the legacy output schema with tensors that the loss builder
    # evaluates but that the selected balancer does not optimize.
    outputs.update(
        {
            "concealed_count": torch.zeros(1, 3, 34, 5, requires_grad=True),
            "concealed_red_count": torch.zeros(1, 3, 3, 2, requires_grad=True),
            "wall_count": torch.zeros(1, 34, 5, requires_grad=True),
            "wall_red_count": torch.zeros(1, 3, 2, requires_grad=True),
            "dora_distribution": torch.zeros(1, 3, 8, requires_grad=True),
            "dora_point": torch.zeros(1, 3, requires_grad=True),
            "score_distribution": torch.zeros(
                1, 3, len(SCORE_VALUES), requires_grad=True
            ),
            "outcome": torch.zeros(1, 33, requires_grad=True),
            "kyoku_delta": torch.zeros(1, 4, requires_grad=True),
            "placement": torch.zeros(1, 24, requires_grad=True),
            "match_score": torch.zeros(1, 4, requires_grad=True),
        }
    )
    batch = {
        "obs": torch.zeros(1, OBS_CHANNELS, TILE_TYPES),
        "action_mask": torch.ones(1, ACTION_SPACE, dtype=torch.bool),
        "policy": torch.zeros(1, dtype=torch.long),
        "shanten": torch.zeros(1, 3, dtype=torch.long),
        "furiten_no_yaku": torch.zeros(1, 3),
        "deal_in_tile": torch.zeros(1, 3, 34),
        "concealed_count": torch.zeros(1, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(1, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(1, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(1, 3, dtype=torch.long),
        "dora": torch.zeros(1, 3, dtype=torch.long),
        "score": torch.zeros(1, 3, dtype=torch.long),
        "winner_mask": torch.zeros(1, 3, dtype=torch.bool),
        "outcome": torch.zeros(1, dtype=torch.long),
        "kyoku_delta": torch.zeros(1, 4),
        "placement": torch.zeros(1, dtype=torch.long),
        "match_score": torch.zeros(1, 4),
    }
    names = ("shanten", "furiten_no_yaku", "deal_in_tile")

    _total, losses, active, weights = multitask_loss(
        outputs, batch, LearnedUncertaintyBalancer(names)
    )

    assert tuple(losses) == names
    assert tuple(active) == names
    assert tuple(weights) == names


def test_rank_augmented_model_and_balanced_losses_backpropagate() -> None:
    """Keep the v6 observation split, heads, labels, and balancer connected."""

    architecture = ModelArchitecture(
        analysis_channels=8,
        analysis_blocks=1,
        analysis_latent_width=16,
        state_width=16,
        future_width=16,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    model = RiichiAnalysisModel(format_version=7, architecture=architecture)
    batch_size = 2
    action_mask = torch.ones(batch_size, ACTION_SPACE, dtype=torch.bool)
    batch = {
        "obs": torch.zeros(batch_size, OBS_CHANNELS, TILE_TYPES),
        "action_mask": action_mask,
        "policy": torch.tensor([0, 1]),
        "shanten": torch.zeros(batch_size, 3, dtype=torch.long),
        "furiten_no_yaku": torch.zeros(batch_size, 3),
        "deal_in_tile": torch.zeros(batch_size, 3, 34),
        "concealed_count": torch.zeros(batch_size, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(batch_size, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(batch_size, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(batch_size, 3, dtype=torch.long),
        "dora": torch.zeros(batch_size, 3, dtype=torch.long),
        "score": torch.zeros(batch_size, 3, dtype=torch.long),
        "winner_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
        "outcome": torch.zeros(batch_size, dtype=torch.long),
        "kyoku_delta": torch.zeros(batch_size, 4),
        "placement": torch.zeros(batch_size, dtype=torch.long),
        "match_score": torch.zeros(batch_size, 4),
    }

    total, losses, active, weights = multitask_loss(
        model(batch["obs"]), batch, LearnedUncertaintyBalancer()
    )
    assert torch.isfinite(total)
    assert set(losses) == set(LOSS_TERMS)
    assert set(weights) == set(LOSS_TERMS)
    assert not active["dora_distribution"]
    assert not active["score_distribution"]
    total.backward()
    assert model.encoder.net[0].weight.grad is not None


def test_conditional_deal_in_is_bounded_by_tenpai_and_eligibility() -> None:
    outputs = {
        "shanten": torch.tensor([[[0.0, 2.0, 0, 0, 0, 0, 0]]]).expand(1, 3, 7),
        "furiten_no_yaku": torch.zeros(1, 3),
        "deal_in_tile": torch.full((1, 3, 34), 10.0),
    }

    risk = conditional_deal_in_probabilities(outputs)
    tenpai = outputs["shanten"].softmax(-1)[..., 0]
    eligible = 1.0 - outputs["furiten_no_yaku"].sigmoid()

    assert torch.all(risk <= tenpai.unsqueeze(-1) + 1e-7)
    assert torch.all(risk <= eligible.unsqueeze(-1) + 1e-7)
    assert risk[0, 0].sum() > tenpai[0, 0]


def test_account_projections_enforce_their_conservation_laws() -> None:
    accounts = zero_sum_accounts(torch.tensor([[2.0, -1.0, 7.0, 3.0, -4.0]]))
    scores = fixed_total_values(
        torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([10.5])
    )

    assert torch.allclose(accounts.sum(-1), torch.zeros(1), atol=1e-6)
    assert torch.allclose(scores.sum(-1), torch.tensor([10.5]), atol=1e-6)


def test_v8_structured_losses_backpropagate() -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=8,
        shared_blocks=1,
        family_latent_width=16,
        opponent_latent_width=20,
        policy_latent_width=24,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=12,
        tile_width=6,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    model = RiichiAnalysisModel(format_version=8, architecture=architecture)
    batch_size = 2
    batch = {
        "obs": torch.zeros(batch_size, OBS_CHANNELS, TILE_TYPES),
        "action_mask": torch.ones(batch_size, ACTION_SPACE, dtype=torch.bool),
        "policy": torch.tensor([0, 1]),
        "shanten": torch.zeros(batch_size, 3, dtype=torch.long),
        "furiten_no_yaku": torch.zeros(batch_size, 3, dtype=torch.long),
        "deal_in_tile": torch.zeros(batch_size, 3, 34),
        "concealed_count": torch.zeros(batch_size, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(batch_size, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(batch_size, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(batch_size, 3, dtype=torch.long),
        "dora": torch.zeros(batch_size, 3, dtype=torch.long),
        "score": torch.zeros(batch_size, 3, dtype=torch.long),
        "winner_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
        "outcome": torch.zeros(batch_size, dtype=torch.long),
        "kyoku_delta": torch.zeros(batch_size, 4),
        "placement": torch.zeros(batch_size, dtype=torch.long),
        "match_score": torch.full((batch_size, 4), 25_000),
    }
    outputs = model(batch["obs"])
    balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8)

    total, losses, active, weights = multitask_loss(outputs, batch, balancer)

    assert torch.isfinite(total)
    assert tuple(losses) == LOSS_TERMS_V8
    assert tuple(weights) == LOSS_TERMS_V8
    assert not active["dora_distribution"]
    assert not active["dora_tail"]
    total.backward()
    assert model.shared_trunk.input.weight.grad is not None


def test_structured_analysis_losses_use_only_the_canonical_frame_rows() -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=8,
        shared_blocks=1,
        family_latent_width=16,
        opponent_latent_width=20,
        policy_latent_width=24,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=12,
        tile_width=6,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    model = RiichiAnalysisModel(format_version=10, architecture=architecture)
    batch = {
        "obs": torch.zeros(2, MODEL_INPUT_CHANNELS, TILE_TYPES),
        "action_mask": torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        "policy": torch.tensor([0, 1]),
        "analysis_active": torch.tensor([True, False]),
        "shanten": torch.tensor([[0, 1, 2], [6, 6, 6]]),
        "furiten_no_yaku": torch.zeros(2, 3, dtype=torch.long),
        "deal_in_tile": torch.zeros(2, 3, 34),
        "concealed_count": torch.zeros(2, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(2, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(2, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(2, 3, dtype=torch.long),
        "dora": torch.zeros(2, 3, dtype=torch.long),
        "score": torch.zeros(2, 3, dtype=torch.long),
        "winner_mask": torch.zeros(2, 3, dtype=torch.bool),
        "outcome": torch.tensor([0, 32]),
        "kyoku_delta": torch.zeros(2, 4),
        "placement": torch.tensor([0, 23]),
        "match_score": torch.full((2, 4), 25_000),
    }
    outputs = model(batch["obs"])
    _total, masked_losses, masked_active, _weights = multitask_loss(
        outputs, batch, LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    )

    first_batch = {
        name: value[:1] for name, value in batch.items() if name != "analysis_active"
    }
    first_outputs = {name: value[:1] for name, value in outputs.items()}
    _total, first_losses, _active, _weights = multitask_loss(
        first_outputs, first_batch, LearnedUncertaintyBalancer(LOSS_TERMS_V8)
    )

    for name in LOSS_TERMS_V8[1:]:
        assert masked_active[name] == _active[name]
        assert torch.allclose(masked_losses[name], first_losses[name])


def test_validation_metrics_use_only_the_canonical_analysis_rows() -> None:
    architecture = StructuredModelArchitecture(
        shared_channels=8,
        shared_blocks=1,
        family_latent_width=16,
        opponent_latent_width=20,
        policy_latent_width=24,
        opponent_blocks=1,
        hidden_blocks=1,
        value_blocks=1,
        kyoku_blocks=1,
        match_blocks=1,
        policy_blocks=1,
        task_width=12,
        tile_width=6,
        policy_context_channels=4,
        policy_context_blocks=1,
        policy_context_width=8,
        policy_width=16,
    )
    model = RiichiAnalysisModel(format_version=10, architecture=architecture)
    batch = {
        "obs": torch.zeros(2, MODEL_INPUT_CHANNELS, TILE_TYPES),
        "action_mask": torch.ones(2, ACTION_SPACE, dtype=torch.bool),
        "policy": torch.tensor([0, 1]),
        "analysis_active": torch.tensor([True, False]),
        "shanten": torch.tensor([[0, 1, 2], [6, 6, 6]]),
        "furiten_no_yaku": torch.zeros(2, 3, dtype=torch.long),
        "deal_in_tile": torch.zeros(2, 3, 34),
        "concealed_count": torch.zeros(2, 3, 34, dtype=torch.long),
        "concealed_red_count": torch.zeros(2, 3, 3, dtype=torch.long),
        "wall_count": torch.zeros(2, 34, dtype=torch.long),
        "wall_red_count": torch.zeros(2, 3, dtype=torch.long),
        "dora": torch.zeros(2, 3, dtype=torch.long),
        "score": torch.zeros(2, 3, dtype=torch.long),
        "winner_mask": torch.zeros(2, 3, dtype=torch.bool),
        "draw": torch.ones(2),
        "win": torch.zeros(2, 4),
        "deal_in_player": torch.zeros(2, 4),
        "outcome": torch.tensor([0, 32]),
        "kyoku_delta": torch.zeros(2, 4),
        "placement": torch.tensor([0, 23]),
        "match_score": torch.full((2, 4), 25_000),
    }
    model.eval()
    with torch.no_grad():
        expected = torch.nn.functional.cross_entropy(
            model(batch["obs"])["shanten"][:1].reshape(-1, 7),
            batch["shanten"][:1].reshape(-1),
        )

    metrics = validate(
        model,
        LearnedUncertaintyBalancer(LOSS_TERMS_V8),
        [batch],
        torch.device("cpu"),
    )

    assert metrics["metric/shantenNll"] == pytest.approx(float(expected))
