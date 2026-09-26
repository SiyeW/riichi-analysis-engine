import pytest
import torch

from riichi_analysis_engine.count_acceleration import CudaGraphJointCounts
from riichi_analysis_engine.joint_counts import project_joint_counts


def test_replay_rejects_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        CudaGraphJointCounts(4, torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_replay_dynamic_rows_values_gradients_and_optimizer():
    torch.manual_seed(9)
    replay = CudaGraphJointCounts(4, torch.device("cuda"))
    saved = []
    for i, rows in enumerate((4, 1, 3, 2, 4)):
        inventory = torch.full((rows, 37), 4, device="cuda")
        inventory[:, (4, 13, 22)] = 3
        inventory[:, 34:] = 1
        inventory[:, 0] -= i
        capacities = torch.tensor([[13, 13, 13, 97 - i]], device="cuda").expand(
            rows, -1
        )
        if i == 4:
            inventory.zero_()
            capacities = torch.zeros_like(capacities)
        x = torch.nn.Parameter(torch.randn(rows, 4, 34, 10, device="cuda") * 0.2)
        y = torch.nn.Parameter(x.detach().clone())
        ox = torch.optim.AdamW([x], lr=1e-5)
        oy = torch.optim.AdamW([y], lr=1e-5)
        weights = torch.rand_like(x)
        expected = project_joint_counts(x, inventory, capacities)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            actual = replay(y, inventory, capacities)
        torch.testing.assert_close(
            actual.probability, expected.probability, rtol=0, atol=0
        )
        torch.testing.assert_close(actual.baseline, expected.baseline, rtol=0, atol=0)
        saved.append((actual.probability, actual.probability.detach().clone()))
        ex_loss = (expected.probability * weights).square().sum()
        ac_loss = (actual.probability * weights).square().sum()
        with pytest.raises(RuntimeError, match="before another forward"):
            replay(y, inventory, capacities)
        ex_loss.backward()
        ac_loss.backward(retain_graph=True)
        torch.testing.assert_close(y.grad, x.grad, rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="one backward"):
            ac_loss.backward()
        ox.step()
        oy.step()
        torch.testing.assert_close(y, x, rtol=0, atol=0)
    for actual, snapshot in saved:
        torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
    with torch.no_grad(), pytest.raises(ValueError, match="training only"):
        replay(y, inventory, capacities)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_replay_rejects_oversized_batch():
    replay = CudaGraphJointCounts(1, torch.device("cuda"))
    with pytest.raises(ValueError, match="capacity"):
        replay(torch.zeros(2, 4, 34, 10, device="cuda", requires_grad=True), None, None)
