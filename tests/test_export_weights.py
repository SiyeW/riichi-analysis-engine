from riichi_analysis_engine.export_weights import (
    public_dataset_metadata,
    training_source_revision,
)


def test_exported_provenance_omits_local_paths() -> None:
    checkpoint = {
        "datasets": {
            "train": {"path": "private-training-corpus", "games": 10},
            "validation": {"path": "private-validation-corpus", "games": 2},
        },
        "environment": {"sourceRevision": "abc123", "sourceDirty": False},
    }
    assert public_dataset_metadata(checkpoint) == {
        "train": {"games": 10},
        "validation": {"games": 2},
    }
    assert training_source_revision(checkpoint) == "abc123"
