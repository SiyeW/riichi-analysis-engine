import torch

from riichi_analysis_engine.hidden_transport import (
    FIVE_TILE_INDICES,
    balanced_source_probabilities,
    count_marginals,
    hidden_count_distribution_loss,
    hidden_transport_nll,
    physical_count_marginals,
    physical_hidden_counts,
    projected_count_distributions,
    theoretical_count_baseline,
)


def _example() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    concealed = torch.zeros(1, 3, 34, dtype=torch.long)
    wall = torch.zeros(1, 34, dtype=torch.long)
    concealed[0, 0, 0] = 1
    concealed[0, 0, 4] = 2
    concealed[0, 1, 4] = 1
    concealed[0, 2, 7] = 1
    wall[0, 7] = 1
    concealed_red = torch.zeros(1, 3, 3, dtype=torch.long)
    wall_red = torch.zeros(1, 3, dtype=torch.long)
    concealed_red[0, 1, 0] = 1
    return concealed, wall, concealed_red, wall_red


def test_physical_counts_separate_red_fives_without_double_counting() -> None:
    physical, inventory, capacities = physical_hidden_counts(*_example())

    assert physical.shape == (1, 4, 37)
    assert physical[0, :, FIVE_TILE_INDICES[0]].tolist() == [2, 0, 0, 0]
    assert physical[0, :, 34].tolist() == [0, 1, 0, 0]
    assert inventory.sum().item() == capacities.sum().item() == 6


def test_transport_has_exact_inventory_and_capacity_expectations() -> None:
    physical, inventory, capacities = physical_hidden_counts(*_example())
    affinity = torch.randn(1, 4, 37, requires_grad=True)

    probabilities = balanced_source_probabilities(
        affinity, inventory, capacities, iterations=64
    )
    expected = probabilities * inventory.unsqueeze(1)

    assert torch.allclose(
        probabilities.sum(1)[inventory > 0],
        torch.ones_like(inventory[inventory > 0], dtype=torch.float),
        atol=1e-5,
    )
    assert torch.allclose(expected.sum(-1), capacities.float(), atol=1e-4)
    loss = hidden_transport_nll(probabilities, physical)
    assert torch.isfinite(loss)
    loss.backward()
    assert affinity.grad is not None
    assert torch.isfinite(affinity.grad).all()


def test_protocol_marginals_conserve_tiles_and_keep_red_inside_fives() -> None:
    physical, inventory, capacities = physical_hidden_counts(*_example())
    probabilities = balanced_source_probabilities(
        torch.randn(1, 4, 37), inventory, capacities, iterations=64
    )

    total, red = count_marginals(probabilities, inventory)
    values = torch.arange(5, dtype=total.dtype)
    expected_total = (total * values).sum(-1)

    total_inventory = physical.sum(1)[:, :34]
    for suit, tile in enumerate(FIVE_TILE_INDICES):
        total_inventory[:, tile] += inventory[:, 34 + suit]
    assert torch.allclose(expected_total.sum(1), total_inventory.float(), atol=1e-5)
    assert torch.allclose(red.sum(-1), torch.ones_like(red[..., 0]), atol=1e-6)
    assert torch.all(
        red[..., 1] <= expected_total[:, :, list(FIVE_TILE_INDICES)] + 1e-6
    )


def test_protocol_marginals_support_a_real_training_batch() -> None:
    """The marginal decoder must index batch, source, and tile independently."""

    _physical, inventory, capacities = physical_hidden_counts(*_example())
    inventory = inventory.repeat(32, 1)
    capacities = capacities.repeat(32, 1)
    probabilities = balanced_source_probabilities(
        torch.randn(32, 4, 37), inventory, capacities, iterations=16
    )

    total, red = count_marginals(probabilities, inventory)

    assert total.shape == (32, 4, 34, 5)
    assert red.shape == (32, 4, 3, 2)
    assert torch.allclose(total.sum(-1), torch.ones_like(total[..., 0]), atol=1e-6)
    assert torch.allclose(red.sum(-1), torch.ones_like(red[..., 0]), atol=1e-6)


def test_theoretical_baseline_is_normalized_and_already_has_exact_margins() -> None:
    _physical, inventory, capacities = physical_hidden_counts(*_example())

    baseline = theoretical_count_baseline(inventory, capacities)
    values = torch.arange(5, dtype=baseline.dtype)
    expected = (baseline * values).sum(-1)

    assert torch.allclose(baseline.sum(-1), torch.ones_like(baseline[..., 0]))
    assert torch.allclose(expected.sum(-1), capacities.float(), atol=1e-5)
    assert torch.allclose(expected.sum(1), inventory.float(), atol=1e-5)


def test_zero_residual_reproduces_the_theoretical_baseline() -> None:
    _physical, inventory, capacities = physical_hidden_counts(*_example())
    residual = torch.zeros(1, 4, 37, 5)

    probability, baseline = projected_count_distributions(
        residual, inventory, capacities
    )

    assert torch.allclose(probability, baseline, atol=1e-6)


def test_complete_count_residual_can_represent_a_non_binomial_shape() -> None:
    inventory = torch.zeros(1, 37, dtype=torch.long)
    inventory[0, 0] = 2
    inventory[0, 1] = 2
    capacities = torch.tensor([[2, 2, 0, 0]])
    residual = torch.zeros(1, 4, 37, 5, requires_grad=True)
    with torch.no_grad():
        residual[0, 0, 0, 0] = 4
        residual[0, 0, 0, 1] = -4
        residual[0, 0, 0, 2] = 4
        residual[0, 1, 0, 0] = 4
        residual[0, 1, 0, 1] = -4
        residual[0, 1, 0, 2] = 4

    probability, _baseline = projected_count_distributions(
        residual, inventory, capacities, iterations=48
    )
    selected = probability[0, 0, 0]

    assert selected[0] > selected[1]
    assert selected[2] > selected[1]
    values = torch.arange(5, dtype=probability.dtype)
    expected = (probability * values).sum(-1)
    assert torch.allclose(expected.sum(-1), capacities.float(), atol=1e-4)
    assert torch.allclose(expected.sum(1), inventory.float(), atol=1e-4)
    probability[..., 0].mean().backward()
    assert residual.grad is not None
    assert torch.isfinite(residual.grad).all()


def test_count_projection_stays_stable_for_large_early_residuals() -> None:
    _physical, inventory, capacities = physical_hidden_counts(*_example())
    generator = torch.Generator().manual_seed(20260921)
    residual = torch.randn(1, 4, 37, 5, generator=generator) * 4

    probability, _baseline = projected_count_distributions(
        residual, inventory, capacities
    )
    values = torch.arange(5, dtype=probability.dtype)
    expected = (probability * values).sum(-1)

    assert torch.isfinite(probability).all()
    assert torch.allclose(expected.sum(-1), capacities.float(), atol=1e-4)
    assert torch.allclose(expected.sum(1), inventory.float(), atol=1e-4)


def test_anchor_loss_uses_theory_instead_of_the_sampled_allocation() -> None:
    physical, inventory, capacities = physical_hidden_counts(*_example())
    residual = torch.randn(1, 4, 37, 5)
    probability, baseline = projected_count_distributions(
        residual, inventory, capacities
    )
    alternate = physical.roll(1, dims=1)

    first = hidden_count_distribution_loss(
        probability, baseline, physical, inventory, torch.tensor([True])
    )
    second = hidden_count_distribution_loss(
        probability, baseline, alternate, inventory, torch.tensor([True])
    )

    assert torch.allclose(first, second)

    exact, exact_baseline = projected_count_distributions(
        torch.zeros(1, 4, 37, 5), inventory, capacities
    )
    exact_loss = hidden_count_distribution_loss(
        exact, exact_baseline, physical, inventory, torch.tensor([True])
    )
    assert exact_loss.abs() < 1e-6


def test_physical_count_marginals_add_red_fives_to_the_base_family() -> None:
    _physical, inventory, capacities = physical_hidden_counts(*_example())
    distribution, _baseline = projected_count_distributions(
        torch.zeros(1, 4, 37, 5), inventory, capacities
    )

    total, red = physical_count_marginals(distribution)
    values = torch.arange(5, dtype=total.dtype)
    expected_total = (total * values).sum(-1)
    total_inventory = inventory[:, :34].clone()
    for suit, tile in enumerate(FIVE_TILE_INDICES):
        total_inventory[:, tile] += inventory[:, 34 + suit]

    assert torch.allclose(expected_total.sum(1), total_inventory.float(), atol=1e-5)
    assert torch.allclose(red.sum(-1), torch.ones_like(red[..., 0]), atol=1e-6)
