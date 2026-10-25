import json

import numpy as np
import pytest
import torch
from test_dataset import pack_directory

from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.validation_groups import ValidationGroups


def test_grouped_masked_sums_keep_per_label_denominators(tmp_path):
    groups = ValidationGroups()
    groups.add(
        "shantenNll", torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), torch.tensor([7, 8])
    )
    groups.add(
        "furitenBrier",
        torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        torch.tensor([7, 8]),
        torch.tensor([[True, False], [True, True]]),
    )
    path = tmp_path / "groups.json"
    groups.write(path, 2, "manifest")
    result = json.loads(path.read_text())
    assert result["groups"][0]["metrics"]["shantenNll"] == {"sum": 6.0, "count": 3}
    assert result["groups"][1]["metrics"]["furitenBrier"]["sum"] == pytest.approx(0.7)
    assert result["groups"][1]["metrics"]["furitenBrier"]["count"] == 2


def test_grouped_metrics_reject_nonfinite_and_wrong_row_axis():
    groups = ValidationGroups()
    with pytest.raises(ValueError, match="row axis"):
        groups.add("policyNll", torch.ones(3), torch.tensor([1, 2]))
    with pytest.raises(ValueError, match="nonfinite"):
        groups.add("policyNll", torch.tensor([float("nan")]), torch.tensor([1]))


def test_provenance_is_opt_in_and_never_changes_model_inputs(tmp_path):
    packs = pack_directory(tmp_path)
    normal = next(iter(PackDataset(packs, batch_size=8)))
    grouped = next(iter(PackDataset(packs, batch_size=8, include_source_game=True)))
    assert "source_game" not in normal and "source_game" in grouped
    for key, value in normal.items():
        assert np.array_equal(value.numpy(), grouped[key].numpy())
