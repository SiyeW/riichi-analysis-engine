import torch

from riichi_analysis_engine.architecture import ModelArchitecture
from riichi_analysis_engine.losses import LOSS_TERMS, LearnedUncertaintyBalancer
from riichi_analysis_engine.model import RiichiAnalysisModel
from riichi_analysis_engine.train import save_checkpoint, step_budget_reached


def test_v6_checkpoint_records_architecture_and_learned_loss_state(tmp_path) -> None:
    architecture = ModelArchitecture(
        analysis_channels=16,
        analysis_blocks=1,
        analysis_latent_width=32,
        state_width=24,
        future_width=20,
        policy_context_channels=16,
        policy_context_blocks=1,
        policy_context_width=16,
        policy_width=24,
    )
    model = RiichiAnalysisModel(architecture=architecture)
    balancer = LearnedUncertaintyBalancer()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": balancer.parameters(), "weight_decay": 0.0},
        ]
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    destination = tmp_path / "checkpoint.pt"

    save_checkpoint(
        destination,
        model,
        optimizer,
        scaler,
        balancer,
        architecture,
        epoch=1,
        batch_in_epoch=0,
        step=10,
        samples_seen=20,
        parameters={},
        datasets={},
        environment={},
        validation=None,
    )

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload["format"] == "riichi-analysis-model-v6"
    assert payload["modelArchitecture"] == architecture.to_dict()
    assert payload["lossBalancer"]["terms"] == list(LOSS_TERMS)


def test_step_budget_is_off_until_a_positive_limit_is_reached() -> None:
    assert not step_budget_reached(10, 0)
    assert not step_budget_reached(9, 10)
    assert step_budget_reached(10, 10)
