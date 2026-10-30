from dataclasses import replace

import pytest
import torch
from riichi_analysis_engine.architecture import V18Architecture
from riichi_analysis_engine.architecture_growth import grow_v18
from riichi_analysis_engine.export_weights import sha256
from riichi_analysis_engine.grow_weights import publish_initialization
from riichi_analysis_engine.growth_mapping import (
    Replication,
    expand_linear,
    raw_reader_coefficients,
    split_outgoing_symmetry,
)
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.model_input import v16_model_input_metadata
from riichi_analysis_engine.prediction_values import DORA_VALUES, SCORE_VALUES
from riichi_analysis_engine.semantic_input import semantic_input_metadata
from riichi_analysis_engine.semantic_v18 import scale_raw_facts
from test_v17 import inputs


def test_linear_replication_and_zero_sum_symmetry():
    columns, rows = Replication.ordered(8, 12), Replication.ordered(5, 9)
    weight, value = torch.randn(5, 8), torch.randn(3, 8)
    grown = expand_linear(weight, rows, columns)
    split = split_outgoing_symmetry(grown, columns, seed=4, relative_scale=0.1)
    expected = (value @ weight.T)[:, rows.indices]
    torch.testing.assert_close(value[:, columns.indices] @ grown.T, expected)
    torch.testing.assert_close(value[:, columns.indices] @ split.T, expected)
    assert not torch.equal(split, grown)


def test_common_reader_handles_raw_zero_padding_and_replicated_learned_memory():
    axis = Replication.ordered(16, 24)
    coefficients = raw_reader_coefficients(axis, 8)
    weight, learned = torch.randn(16, 16), torch.randn(3, 16)
    raw = torch.zeros(3, 16)
    raw[:, 0] = 1
    raw[:, 1:8] = torch.randn(3, 7)
    wider = torch.nn.functional.pad(raw, (0, 8))
    grown = expand_linear(weight, axis, axis, column_coefficients=coefficients)
    torch.testing.assert_close(
        scale_raw_facts(wider) @ grown.T,
        (scale_raw_facts(raw) @ weight.T)[:, axis.indices],
    )
    torch.testing.assert_close(
        learned[:, axis.indices] @ grown.T,
        (learned @ weight.T)[:, axis.indices],
    )
    with pytest.raises(ValueError, match="every active"):
        raw_reader_coefficients(axis, 9)


def test_depth_only_growth_is_identity_and_parent_rng_are_unchanged():
    architecture = V18Architecture(
        width=128,
        blocks=1,
        attention_heads=4,
        feed_forward_width=256,
        decoder_width=256,
    )
    parent = RiichiAnalysisModel(format_version=18, architecture=architecture).eval()
    before = {key: value.clone() for key, value in parent.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model, report = grow_v18(parent, replace(architecture, blocks=3), seed=7)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(value, parent.state_dict()[key]) for key, value in before.items()
    )
    with torch.no_grad():
        old, new = parent(*inputs()), model.eval()(*inputs())
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert report["neutralNewBlocks"] == [1, 2]
    block = model.semantic_model.blocks[1]
    value, mask = torch.randn(1, 46, 128), torch.ones(1, 46, dtype=torch.bool)
    mask[:, -1] = False
    torch.testing.assert_close(block(value, mask), value * mask[..., None])


def test_actual_shape_and_mapping_coverage():
    parent = RiichiAnalysisModel(format_version=18)
    target = V18Architecture(
        width=384,
        blocks=6,
        attention_heads=12,
        feed_forward_width=1536,
        decoder_width=768,
    )
    model, report = grow_v18(parent, target, seed=42)
    assert count_parameters(model)["total"] == 29_135_002
    assert report["sourceTensors"] == 277
    assert report["targetTensors"] == 313
    assert report["exactFunctionPreservation"] is False
    assert report["optimizerStateMapped"] is False
    assert all(value.isfinite().all() for value in model.state_dict().values())
    # Query/task FFN replicas are initialized equally but outgoing columns differ.
    branch = model.semantic_model.decoder.query_branches["policy"]
    assert torch.equal(branch[1].weight[0], branch[1].weight[512])
    assert not torch.equal(branch[3].weight[:, 0], branch[3].weight[:, 512])
    with torch.no_grad():
        output = model.eval()(*inputs())
    assert all(value.isfinite().all() for value in output.values())
    features = torch.randn(1, 4, 92)
    mask = torch.tensor([[True, True, False, True]])
    order = torch.tensor([3, 1, 2, 0])
    with torch.no_grad():
        first = model(*inputs(), features, mask)["policy"]
        permuted = model(*inputs(), features[:, order], mask[:, order])["policy"]
    assert first[mask].isfinite().all() and first[~mask].isneginf().all()
    torch.testing.assert_close(permuted, first[:, order])


def test_rejects_wrong_heads_shrinking_and_nonfinite():
    parent = RiichiAnalysisModel(format_version=18)
    with pytest.raises(ValueError, match="head dimension"):
        grow_v18(parent, V18Architecture(width=384, attention_heads=8))
    with pytest.raises(ValueError, match="remove"):
        grow_v18(parent, V18Architecture(blocks=3))
    with torch.no_grad():
        next(parent.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        grow_v18(parent, V18Architecture())


def test_publication_preserves_parent_and_refuses_overwrite_or_bad_identity(tmp_path):
    architecture = V18Architecture(
        width=128,
        blocks=1,
        attention_heads=4,
        feed_forward_width=256,
        decoder_width=256,
    )
    model = RiichiAnalysisModel(format_version=18, architecture=architecture)
    path, output = tmp_path / "parent.pt", tmp_path / "initial.pt"
    torch.save(
        {
            "format": "riichi-analysis-model-v18",
            "model": model.state_dict(),
            "modelArchitecture": architecture.to_dict(),
            "modelInput": v16_model_input_metadata(),
            "semanticInput": semantic_input_metadata(),
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "step": 123,
            "samplesSeen": 1968,
        },
        path,
    )
    before = sha256(path)
    with pytest.raises(ValueError, match="checksum"):
        publish_initialization(path, output, architecture, expected_parent_sha256="bad")
    receipt = publish_initialization(
        path, output, replace(architecture, blocks=2), expected_parent_sha256=before
    )
    payload = torch.load(output, weights_only=True)
    assert receipt["parentCounters"]["step"] == 123
    assert receipt["newArchitectureUpdates"] == 0
    assert sha256(path) == before
    assert payload["resumeAllowed"] is False
    assert "optimizer" not in payload and "step" not in payload
    assert output.with_suffix(".pt.json").is_file()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        publish_initialization(
            path, output, architecture, expected_parent_sha256=before
        )
