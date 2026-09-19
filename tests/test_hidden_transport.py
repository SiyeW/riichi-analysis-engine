import torch

from riichi_analysis_engine.hidden_transport import (
    FIVE_TILE_INDICES,
    balanced_source_probabilities,
    count_marginals,
    hidden_transport_nll,
    physical_hidden_counts,
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
