from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .architecture import ModelArchitecture, StructuredModelArchitecture
from .constants import OBS_CHANNELS
from .dataset import PackDataset
from .hidden_transport import (
    balanced_source_probabilities,
    count_marginals,
    physical_affinities,
    physical_hidden_counts,
)
from .kyoku_outcome import OUTCOME_DEAL_IN_INDICATORS, OUTCOME_WINNER_INDICATORS
from .losses import (
    LOSS_TERMS,
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    masked_score_logits,
    multitask_loss,
    score_class_indices,
)
from .model import RiichiAnalysisModel, count_parameters
from .model_input import (
    LEGACY_MODEL_INPUT_SCHEMA_ID,
    MODEL_INPUT_CHANNELS,
    MODEL_INPUT_SCHEMA_ID,
    model_input_metadata,
)
from .prediction_values import DORA_TAIL_START, DORA_VALUES, SCORE_VALUES
from .structured_outputs import (
    conditional_deal_in_probabilities,
    fixed_total_values,
    zero_sum_accounts,
)
from .training_schema import validate_v8_training_batch


class TrainingInterrupted(Exception):
    """Stop at a boundary whose state can be resumed without replaying data."""


def gradient_total_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Measure the global L2 norm without modifying any gradient."""

    squared = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def shared_gradient_geometry(
    losses: Mapping[str, torch.Tensor],
    active: Mapping[str, bool],
    shared: torch.Tensor,
) -> dict[str, float]:
    """Measure raw task-gradient norms and angles at the shared feature boundary.

    This intentionally measures each unweighted task loss before the learned
    loss balancer. It diagnoses representation conflict without materializing
    one full shared-parameter gradient vector per task.
    """

    gradients: dict[str, torch.Tensor] = {}
    metrics: dict[str, float] = {}
    for name, loss in losses.items():
        if not active.get(name, False) or not loss.requires_grad:
            continue
        gradient = torch.autograd.grad(
            loss, shared, retain_graph=True, allow_unused=True
        )[0]
        if gradient is None:
            continue
        vector = gradient.detach().float().reshape(-1)
        norm = vector.norm()
        gradients[name] = vector
        metrics[f"sharedGradientNorm/{name}"] = float(norm)

    cosines: list[float] = []
    names = list(gradients)
    for left_index, left_name in enumerate(names):
        left = gradients[left_name]
        left_norm = left.norm()
        for right_name in names[left_index + 1 :]:
            right = gradients[right_name]
            denominator = left_norm * right.norm()
            cosine = (
                float(torch.dot(left, right) / denominator)
                if float(denominator) > 0.0
                else 0.0
            )
            metrics[f"sharedGradientCosine/{left_name}__{right_name}"] = cosine
            cosines.append(cosine)
    if cosines:
        metrics["sharedGradientMeanCosine"] = sum(cosines) / len(cosines)
        metrics["sharedGradientConflictFraction"] = sum(
            cosine < 0.0 for cosine in cosines
        ) / len(cosines)
    return metrics


def source_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"sourceRevision": revision, "sourceDirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        revision = os.environ.get("RIICHI_ANALYSIS_SOURCE_REVISION", "").strip().lower()
        if len(revision) == 40 and all(
            character in "0123456789abcdef" for character in revision
        ):
            return {"sourceRevision": revision, "sourceDirty": False}
        return {"sourceRevision": None, "sourceDirty": None}


def environment_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cudaRuntime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        **source_metadata(),
    }
    if device.type == "cuda":
        metadata["deviceName"] = torch.cuda.get_device_name(device)
        metadata["deviceCapability"] = list(torch.cuda.get_device_capability(device))
    return metadata


def dataset_metadata(root: Path) -> dict[str, object]:
    """Record what a run trained on, without recording where the machine keeps it.

    This metadata reaches the exported weights, so it carries the manifest's
    own numbers and leaves out the local path the packing command was given.
    """

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"path": str(root.resolve())}
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    audit = manifest.get("audit")
    return {
        "path": str(root.resolve()),
        "format": manifest.get("format"),
        "seed": manifest.get("seed"),
        "packs": len(manifest.get("packs", [])),
        "samples": manifest.get("samples"),
        "chunks": manifest.get("chunks"),
        "sourceGames": manifest.get("sourceGames"),
        "modelInputSchema": manifest.get("modelInputSchema"),
        "observationChannels": manifest.get("observationChannels"),
        "verified": audit.get("verified") if isinstance(audit, dict) else None,
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def validate_dataset_input_contract(
    datasets: dict[str, dict[str, object]], model_format: int
) -> None:
    """Fail before training when packs cannot supply the selected model input."""

    expected_schema = (
        MODEL_INPUT_SCHEMA_ID
        if model_format in {9, 10, 11}
        else LEGACY_MODEL_INPUT_SCHEMA_ID
    )
    expected_channels = (
        MODEL_INPUT_CHANNELS if model_format in {9, 10, 11} else OBS_CHANNELS
    )
    for split, metadata in datasets.items():
        schema = metadata.get("modelInputSchema")
        channels = metadata.get("observationChannels")
        # Manifests written before the explicit metadata fields are v8 by
        # construction. They remain readable only by legacy model formats.
        if schema is None and channels is None and model_format not in {9, 10, 11}:
            continue
        if schema != expected_schema or channels != expected_channels:
            raise RuntimeError(f"{split} dataset uses a different model-input contract")


def move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def step_budget_reached(step: int, max_steps: int) -> bool:
    return max_steps > 0 and step >= max_steps


def single_pass_window(
    corpus_samples: int,
    next_sample: int,
    max_train_samples: int,
) -> tuple[int, int]:
    """Return the absolute pass limit and unread sample count for this run."""

    if corpus_samples <= 0:
        raise ValueError("training corpus must contain samples")
    if max_train_samples < 0:
        raise ValueError("maximum training samples must be non-negative")
    if not 0 <= next_sample <= corpus_samples:
        raise RuntimeError("resume cursor lies outside the training corpus")
    limit = (
        corpus_samples
        if max_train_samples == 0
        else min(corpus_samples, max_train_samples)
    )
    if next_sample >= limit:
        raise RuntimeError(
            "resume cursor has reached the sample limit; increase --max-train-samples"
        )
    return limit, limit - next_sample


def resume_training_cursor(
    checkpoint: dict[str, object],
    datasets: dict[str, dict[str, object]],
    batch_size: int,
    *,
    allow_complete: bool = False,
) -> tuple[int, int]:
    """Validate and return the next unread sample and completed update count."""

    cursor = checkpoint.get("trainingCursor")
    if not isinstance(cursor, dict) or cursor.get("type") != "single-pass-v1":
        raise RuntimeError(
            "resume checkpoint predates the single-pass cursor and cannot "
            "be resumed safely"
        )
    complete = cursor.get("complete")
    if complete not in (False, True):
        raise RuntimeError("resume checkpoint has an invalid pass-complete marker")
    if complete and not allow_complete:
        raise RuntimeError("resume checkpoint has already completed its training pass")
    if int(cursor.get("batchSize", -1)) != batch_size:
        raise RuntimeError("resume checkpoint uses a different training batch size")

    saved_datasets = checkpoint.get("datasets")
    if not isinstance(saved_datasets, dict):
        raise RuntimeError("resume checkpoint has no dataset identity")
    for split in ("train", "validation"):
        saved = saved_datasets.get(split)
        current = datasets.get(split)
        if not isinstance(saved, dict) or not isinstance(current, dict):
            raise RuntimeError(f"resume checkpoint has no {split} dataset identity")
        saved_digest = saved.get("manifestSha256")
        if not saved_digest or saved_digest != current.get("manifestSha256"):
            raise RuntimeError(f"resume checkpoint uses a different {split} manifest")

    next_sample = int(cursor.get("nextSample", -1))
    completed_steps = int(checkpoint.get("step", -1))
    if next_sample < 0 or completed_steps < 0:
        raise RuntimeError("resume checkpoint has an invalid single-pass cursor")
    if int(checkpoint.get("samplesSeen", -1)) != next_sample:
        raise RuntimeError("resume checkpoint has inconsistent sample counters")
    if int(cursor.get("batchesConsumed", -1)) != completed_steps:
        raise RuntimeError("resume checkpoint has inconsistent batch counters")
    return next_sample, completed_steps


@torch.no_grad()
def validate(
    model: RiichiAnalysisModel,
    balancer: LearnedUncertaintyBalancer,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_every: int = 100,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, float]:
    model.eval()
    loss_terms = balancer.names
    totals: dict[str, float] = {"total": 0.0, **{name: 0.0 for name in loss_terms}}
    # Label marginals, kept to build the null baseline each loss is compared
    # against: what a model that only knows the label distribution would score.
    policy_labels = np.zeros(46, dtype=np.int64)
    shanten_labels = np.zeros(7, dtype=np.int64)
    balance_totals: dict[str, float] = {name: 0.0 for name in loss_terms}
    batches = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    deal_in_positive_histogram = np.zeros(1_000, dtype=np.int64)
    deal_in_negative_histogram = np.zeros(1_000, dtype=np.int64)

    def add_metric(
        name: str, values: torch.Tensor, mask: torch.Tensor | None = None
    ) -> None:
        values = values.float()
        if mask is not None:
            values = values[mask]
        metric_sums[name] = metric_sums.get(name, 0.0) + float(values.sum())
        metric_counts[name] = metric_counts.get(name, 0) + values.numel()

    for batch in loader:
        if should_stop is not None and should_stop():
            raise TrainingInterrupted
        batch = move_batch(batch, device)
        outputs = model(batch["obs"].float())
        total, losses, _active, weights = multitask_loss(outputs, batch, balancer)
        totals["total"] += float(total)
        for name, value in losses.items():
            totals[name] += float(value)
        for name, value in weights.items():
            balance_totals[name] += float(value)
        policy_valid = batch["policy"] >= 0
        policy_labels += (
            torch.bincount(batch["policy"][policy_valid], minlength=46).cpu().numpy()
        )
        policy_logits = outputs["policy"].masked_fill(~batch["action_mask"], -torch.inf)
        add_metric(
            "policyAccuracy",
            policy_logits.argmax(-1) == batch["policy"],
            policy_valid,
        )
        analysis_rows = batch.get("analysis_active")
        if analysis_rows is None:
            analysis_rows = torch.ones(
                len(batch["policy"]), dtype=torch.bool, device=batch["policy"].device
            )
        else:
            analysis_rows = analysis_rows.bool()
        shanten_labels += (
            torch.bincount(batch["shanten"][analysis_rows].reshape(-1), minlength=7)
            .cpu()
            .numpy()
        )
        if not analysis_rows.any():
            batches += 1
            if progress_every > 0 and batches % progress_every == 0:
                print(json.dumps({"phase": "validation-progress", "batches": batches}))
            continue
        row_count = len(analysis_rows)
        outputs = {
            name: value[analysis_rows]
            for name, value in outputs.items()
            if name != "policy"
        }
        batch = {
            name: (
                value[analysis_rows]
                if value.ndim > 0 and len(value) == row_count
                else value
            )
            for name, value in batch.items()
        }
        add_metric(
            "shantenAccuracy",
            outputs["shanten"].argmax(-1) == batch["shanten"],
        )
        add_metric(
            "shantenNll",
            F.cross_entropy(
                outputs["shanten"].reshape(-1, 7),
                batch["shanten"].reshape(-1).long(),
                reduction="none",
            ),
        )
        add_metric(
            "furitenBrier",
            (
                outputs["furiten_no_yaku"].sigmoid() - batch["furiten_no_yaku"].float()
            ).square(),
            batch["shanten"] == 0,
        )
        structured = "hidden_source_affinity" in outputs
        deal_in_probability = (
            conditional_deal_in_probabilities(outputs)
            if structured
            else outputs["deal_in_tile"].sigmoid()
        )
        add_metric(
            "dealInTileBrier",
            (deal_in_probability - batch["deal_in_tile"].float()).square(),
        )
        deal_in_target = batch["deal_in_tile"].bool()
        add_metric("dealInPositiveMean", deal_in_probability, deal_in_target)
        add_metric("dealInNegativeMean", deal_in_probability, ~deal_in_target)
        deal_in_bins = (deal_in_probability.detach() * 1_000).long().clamp(0, 999)
        deal_in_positive_histogram += (
            torch.bincount(deal_in_bins[deal_in_target], minlength=1_000).cpu().numpy()
        )
        deal_in_negative_histogram += (
            torch.bincount(deal_in_bins[~deal_in_target], minlength=1_000).cpu().numpy()
        )
        if structured:
            _physical_counts, inventory, capacities = physical_hidden_counts(
                batch["concealed_count"],
                batch["wall_count"],
                batch["concealed_red_count"],
                batch["wall_red_count"],
            )
            source_probability = balanced_source_probabilities(
                physical_affinities(
                    outputs["hidden_source_affinity"], outputs["hidden_red_source"]
                ),
                inventory,
                capacities,
            )
            hidden_count, hidden_red = count_marginals(source_probability, inventory)
            expected_physical = source_probability * inventory.unsqueeze(1)
            add_metric(
                "hiddenSourceConservationError",
                (expected_physical.sum(-1) - capacities.float()).abs(),
            )
            add_metric(
                "hiddenInventoryConservationError",
                (expected_physical.sum(1) - inventory.float()).abs(),
            )
            add_metric(
                "concealedCountAccuracy",
                hidden_count[:, :3].argmax(-1) == batch["concealed_count"],
            )
            add_metric(
                "concealedRedCountAccuracy",
                hidden_red[:, :3].argmax(-1) == batch["concealed_red_count"],
            )
            add_metric(
                "wallCountAccuracy",
                hidden_count[:, 3].argmax(-1) == batch["wall_count"],
            )
            add_metric(
                "wallRedCountAccuracy",
                hidden_red[:, 3].argmax(-1) == batch["wall_red_count"],
            )
            dora_distribution = outputs["dora_distribution"].softmax(-1)
            dora_values = torch.arange(7, device=device, dtype=dora_distribution.dtype)
            dora_point = (dora_distribution[..., :7] * dora_values).sum(
                -1
            ) + dora_distribution[..., 7] * (7.0 + F.softplus(outputs["dora_tail"]))
        else:
            add_metric(
                "concealedCountAccuracy",
                outputs["concealed_count"].argmax(-1) == batch["concealed_count"],
            )
            add_metric(
                "concealedRedCountAccuracy",
                outputs["concealed_red_count"].argmax(-1)
                == batch["concealed_red_count"],
            )
            add_metric(
                "wallCountAccuracy",
                outputs["wall_count"].argmax(-1) == batch["wall_count"],
            )
            add_metric(
                "wallRedCountAccuracy",
                outputs["wall_red_count"].argmax(-1) == batch["wall_red_count"],
            )
            dora_point = F.softplus(outputs["dora_point"])
        add_metric(
            "doraMae",
            (dora_point - batch["dora"].float()).abs(),
            batch["winner_mask"].bool(),
        )
        winner_mask = batch["winner_mask"].bool()
        if winner_mask.any():
            add_metric(
                "doraDistributionAccuracy",
                outputs["dora_distribution"][winner_mask].argmax(-1)
                == batch["dora"].long()[winner_mask].clamp_max(DORA_TAIL_START),
            )
            add_metric(
                "scoreDistributionAccuracy",
                masked_score_logits(outputs, batch["obs"])[winner_mask].argmax(-1)
                == score_class_indices(batch["score"].long()[winner_mask]),
            )
            score_probabilities = masked_score_logits(outputs, batch["obs"]).softmax(-1)
            score_values = score_probabilities.new_tensor(SCORE_VALUES)
            add_metric(
                "scoreMaePoints",
                (
                    (score_probabilities * score_values).sum(-1)
                    - batch["score"].float()
                ).abs(),
                winner_mask,
            )
        outcome_probability = outputs["outcome"].softmax(-1)
        winner_indicators = outcome_probability.new_tensor(OUTCOME_WINNER_INDICATORS)
        deal_in_indicators = outcome_probability.new_tensor(OUTCOME_DEAL_IN_INDICATORS)
        win_probability = outcome_probability @ winner_indicators
        deal_in_probability = outcome_probability @ deal_in_indicators
        add_metric(
            "outcomeDrawBrier",
            (outcome_probability[:, 0] - batch["draw"].float()).square(),
        )
        add_metric(
            "outcomeWinnerBrier",
            (win_probability - batch["win"].float()).square(),
        )
        add_metric(
            "dealInPlayerBrier",
            (deal_in_probability - batch["deal_in_player"].float()).square(),
        )
        add_metric(
            "outcomeAccuracy",
            outputs["outcome"].argmax(-1) == batch["outcome"],
        )
        predicted_delta = (
            zero_sum_accounts(outputs["kyoku_accounts"])[..., :4]
            if structured
            else outputs["kyoku_delta"]
        )
        add_metric(
            "kyokuDeltaMaePoints",
            (predicted_delta * 10_000.0 - batch["kyoku_delta"].float()).abs(),
        )
        add_metric(
            "placementJointAccuracy",
            outputs["placement"].argmax(-1) == batch["placement"],
        )
        match_target = batch["match_score"].float() / 10_000.0
        predicted_match = (
            fixed_total_values(outputs["match_score"], match_target.sum(-1))
            if structured
            else outputs["match_score"]
        )
        add_metric(
            "matchScoreMaePoints",
            (predicted_match * 10_000.0 - batch["match_score"].float()).abs(),
        )
        batches += 1
        if progress_every > 0 and batches % progress_every == 0:
            print(json.dumps({"phase": "validation-progress", "batches": batches}))
    if should_stop is not None and should_stop():
        raise TrainingInterrupted
    result = {name: value / max(1, batches) for name, value in totals.items()}
    nulls = {
        "policy": _label_entropy(policy_labels),
        "shanten": _label_entropy(shanten_labels),
    }
    for name, value in nulls.items():
        result[f"null/{name}"] = value
    add_core_selection_metrics(result, nulls)
    result.update(
        {
            f"lossWeight/{name}": balance_totals[name] / max(1, batches)
            for name in loss_terms
        }
    )
    positive_descending = deal_in_positive_histogram[::-1].cumsum()
    negative_descending = deal_in_negative_histogram[::-1].cumsum()
    total_positive = int(deal_in_positive_histogram.sum())
    if total_positive:
        precision = positive_descending / np.maximum(
            1, positive_descending + negative_descending
        )
        recall_increment = deal_in_positive_histogram[::-1] / total_positive
        result["metric/dealInAveragePrecisionApprox"] = float(
            np.sum(precision * recall_increment)
        )
    result.update(
        {
            f"metric/{name}": metric_sums[name] / max(1, metric_counts[name])
            for name in metric_sums
        }
    )
    return result


def learning_rate_at(
    step: int,
    warmup_steps: int,
    cooldown_steps: int,
    peak_rate: float,
    base_rate: float,
) -> float:
    """Linear warmup to the peak rate, then a linear drop to the plateau rate.

    The shanten predictor project settled this shape: a fast warmup, a short
    cooldown, and a constant plateau instead of a decay to zero. The rate
    depends only on the step, so a resumed run needs no scheduler state. Both
    lengths default to zero, which leaves the plateau rate constant.
    """

    if warmup_steps > 0 and step <= warmup_steps:
        return peak_rate * step / warmup_steps
    if cooldown_steps > 0 and step <= warmup_steps + cooldown_steps:
        progress = (step - warmup_steps) / cooldown_steps
        return peak_rate + (base_rate - peak_rate) * progress
    return base_rate


def _label_entropy(counts: np.ndarray) -> float:
    """Entropy in nats of a label distribution, the score of a null predictor."""

    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def add_core_selection_metrics(
    result: dict[str, float], nulls: Mapping[str, float]
) -> None:
    """Add the joint selection score only when both constituent losses exist."""

    if not {"policy", "shanten"}.issubset(result):
        return
    if nulls["policy"] <= 0 or nulls["shanten"] <= 0:
        return
    score = 0.5 * (
        result["policy"] / nulls["policy"] + result["shanten"] / nulls["shanten"]
    )
    result["Selection/core_score"] = score
    result["Selection/core_skill"] = 1.0 - score


def write_pointer(run: Path, checkpoint: Path, **fields: object) -> None:
    """Record which checkpoint a resumed run should pick up, never moving back."""

    path = run / "latest_checkpoint.json"
    previous = -1
    if path.exists():
        try:
            previous = int(json.loads(path.read_text(encoding="utf-8")).get("step", -1))
        except (ValueError, TypeError):
            previous = -1
    if int(fields.get("step", 0)) < previous:
        return
    payload = {
        "path": str(checkpoint.resolve()),
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        **fields,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def prune_numbered_checkpoints(run: Path, keep: int) -> None:
    """Keep the most recent rolling checkpoints and leave the named ones alone."""

    if keep <= 0:
        return
    numbered: list[tuple[int, Path]] = []
    for path in run.glob("ckpt-*.pt"):
        try:
            numbered.append((int(path.stem.removeprefix("ckpt-")), path))
        except ValueError:
            continue
    for _, path in sorted(numbered)[:-keep]:
        path.unlink(missing_ok=True)


def resolve_resume_path(target: Path) -> Path:
    """Accept a checkpoint, a directory, or a run that only left a pointer."""

    if target.is_file():
        return target
    pointer = target / "latest_checkpoint.json"
    if pointer.exists():
        recorded = Path(str(json.loads(pointer.read_text(encoding="utf-8"))["path"]))
        if recorded.is_file():
            return recorded
    rolling: list[tuple[int, Path]] = []
    for path in target.glob("ckpt-*.pt"):
        try:
            rolling.append((int(path.stem.removeprefix("ckpt-")), path))
        except ValueError:
            continue
    if rolling:
        return max(rolling)[1]
    named = sorted(target.glob("best-*.pth"))
    if named:
        return named[0]
    raise FileNotFoundError(f"no checkpoint found under {target}")


def open_dashboard(run: Path) -> object | None:
    """TensorBoard writer for the run, or None when tensorboard is not installed."""

    try:
        from torch.utils.tensorboard.writer import SummaryWriter
    except ImportError:
        print("tensorboard is not installed; the run logs to metrics.jsonl only")
        return None
    return SummaryWriter(log_dir=str(run / "tensorboard"))


def write_dashboard(
    writer: object | None, metrics: dict[str, float], step: int
) -> None:
    """Write every finite validation metric under a stable TensorBoard tag."""

    if writer is None:
        return
    for name, value in metrics.items():
        if name == "total" or name.startswith("metric/"):
            tag = "Metrics/" + name.removeprefix("metric/")
        elif name.startswith("Selection/"):
            tag = name
        else:
            tag = "Validation/" + name
        if not isinstance(value, float) or value != value:
            continue
        writer.add_scalar(tag, value, step)


def close_dashboard(writer: object | None) -> None:
    if writer is None:
        return
    writer.flush()
    writer.close()


def capture_random_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "hasGauss": numpy_state[3],
            "cachedGaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state: object) -> None:
    if not isinstance(state, dict):
        raise RuntimeError("resume checkpoint has no random-number state")
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, dict) or not isinstance(
        numpy_state.get("keys"), torch.Tensor
    ):
        raise RuntimeError("resume checkpoint has incompatible NumPy random state")
    random.setstate(state["python"])
    np.random.set_state(
        (
            str(numpy_state["name"]),
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["hasGauss"]),
            float(numpy_state["cachedGaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(
    path: Path,
    model: RiichiAnalysisModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    balancer: LearnedUncertaintyBalancer,
    architecture: ModelArchitecture | StructuredModelArchitecture,
    *,
    step: int,
    samples_seen: int,
    analysis_samples_seen: int,
    batch_size: int,
    pass_complete: bool,
    parameters: dict[str, int],
    datasets: dict[str, dict[str, object]],
    environment: dict[str, object],
    validation: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": f"riichi-analysis-model-v{model.format_version}",
            "step": step,
            "samplesSeen": samples_seen,
            "analysisSamplesSeen": analysis_samples_seen,
            "trainingCursor": {
                "type": "single-pass-v1",
                "nextSample": samples_seen,
                "batchesConsumed": step,
                "batchSize": batch_size,
                "complete": pass_complete,
            },
            "model": model.state_dict(),
            "modelArchitecture": architecture.to_dict(),
            **(
                {"modelInput": model_input_metadata()}
                if model.format_version in {9, 10, 11}
                else {}
            ),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lossBalancer": {
                "terms": list(balancer.names),
                "state": balancer.state_dict(),
            },
            "randomState": capture_random_state(),
            "parameters": parameters,
            "predictionValues": {
                "dora": list(DORA_VALUES),
                "score": list(SCORE_VALUES),
            },
            "datasets": datasets,
            "environment": environment,
            "validation": validation,
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-task analysis model.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--loss-balance-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--model-format", type=int, choices=(7, 8, 9, 10, 11), default=11
    )
    parser.add_argument("--shared-channels", type=int, default=256)
    parser.add_argument("--shared-blocks", type=int, default=30)
    parser.add_argument("--family-latent-width", type=int)
    parser.add_argument("--opponent-latent-width", type=int, default=1024)
    parser.add_argument("--policy-latent-width", type=int, default=1024)
    parser.add_argument("--opponent-blocks", type=int, default=24)
    parser.add_argument("--hidden-blocks", type=int, default=8)
    parser.add_argument("--value-blocks", type=int, default=6)
    parser.add_argument("--kyoku-blocks", type=int, default=6)
    parser.add_argument("--match-blocks", type=int, default=4)
    parser.add_argument("--policy-blocks", type=int, default=24)
    parser.add_argument("--task-width", type=int)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--policy-context-channels", type=int, default=144)
    parser.add_argument("--policy-context-blocks", type=int, default=6)
    parser.add_argument("--policy-context-width", type=int, default=384)
    parser.add_argument("--policy-width", type=int, default=1024)
    parser.add_argument(
        "--analysis-channels", type=int, default=288, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--analysis-blocks", type=int, default=54, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--analysis-latent-width", type=int, default=1152, help=argparse.SUPPRESS
    )
    parser.add_argument("--state-width", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument(
        "--future-width", type=int, default=1024, help=argparse.SUPPRESS
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument(
        "--max-analysis-samples",
        type=int,
        default=0,
        help="stop after this many canonical analysis rows; zero disables it",
    )
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every-samples", type=int, default=1_000_000)
    parser.add_argument(
        "--gradient-diagnostics-every",
        type=int,
        default=0,
        help="measure v8 task-gradient geometry every N steps; zero disables it",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="linear warmup to the peak rate over this many steps",
    )
    parser.add_argument("--cooldown-steps", type=int, default=0)
    parser.add_argument(
        "--peak-learning-rate",
        type=float,
        default=0.0,
        help="rate the warmup reaches; defaults to the plateau rate",
    )
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate a saved checkpoint without consuming more training samples",
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--loss-term",
        action="append",
        default=[],
        help="train only this loss term; repeat for a controlled ablation",
    )
    args = parser.parse_args()
    if args.max_steps < 0:
        raise ValueError("max steps must be non-negative")
    if args.max_analysis_samples < 0:
        raise ValueError("maximum analysis samples must be non-negative")
    if args.checkpoint_every_samples < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("checkpoint retention must be non-negative")
    if args.gradient_diagnostics_every < 0:
        raise ValueError("gradient diagnostics interval must be non-negative")
    if args.gradient_diagnostics_every and args.model_format not in {8, 9, 10, 11}:
        raise ValueError("shared-gradient diagnostics require model formats 8 through 11")
    if args.validate_only and args.resume is None:
        raise ValueError("--validate-only requires --resume")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.float16 if device.type == "cuda" else None

    if args.model_format in {8, 9, 10, 11}:
        family_latent_width = args.family_latent_width or (
            1024 if args.model_format == 11 else 768
        )
        task_width = args.task_width or (1024 if args.model_format == 11 else 512)
        architecture: ModelArchitecture | StructuredModelArchitecture = (
            StructuredModelArchitecture(
                shared_channels=args.shared_channels,
                shared_blocks=args.shared_blocks,
                family_latent_width=family_latent_width,
                opponent_latent_width=args.opponent_latent_width,
                policy_latent_width=args.policy_latent_width,
                opponent_blocks=args.opponent_blocks,
                hidden_blocks=args.hidden_blocks,
                value_blocks=args.value_blocks,
                kyoku_blocks=args.kyoku_blocks,
                match_blocks=args.match_blocks,
                policy_blocks=args.policy_blocks,
                task_width=task_width,
                tile_width=args.tile_width,
                policy_context_channels=args.policy_context_channels,
                policy_context_blocks=args.policy_context_blocks,
                policy_context_width=args.policy_context_width,
                policy_width=args.policy_width,
            )
        )
        available_loss_terms = LOSS_TERMS_V8
    else:
        architecture = ModelArchitecture(
            analysis_channels=args.analysis_channels,
            analysis_blocks=args.analysis_blocks,
            analysis_latent_width=args.analysis_latent_width,
            state_width=args.state_width,
            future_width=args.future_width,
            policy_context_channels=args.policy_context_channels,
            policy_context_blocks=args.policy_context_blocks,
            policy_context_width=args.policy_context_width,
            policy_width=args.policy_width,
        )
        available_loss_terms = LOSS_TERMS
    if args.loss_term:
        if len(set(args.loss_term)) != len(args.loss_term):
            raise ValueError("loss terms must not be repeated")
        unknown = sorted(set(args.loss_term) - set(available_loss_terms))
        if unknown:
            raise ValueError(
                f"loss terms are unavailable for this model format: {unknown}"
            )
        loss_terms = tuple(args.loss_term)
    else:
        loss_terms = available_loss_terms
    model = RiichiAnalysisModel(
        format_version=args.model_format, architecture=architecture
    ).to(device)
    balancer = LearnedUncertaintyBalancer(loss_terms).to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.parameters(),
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": balancer.parameters(),
                "lr": args.loss_balance_learning_rate,
                "weight_decay": 0.0,
            },
        ]
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    step = 0
    samples_seen = 0
    analysis_samples_seen = 0
    datasets = {
        "train": dataset_metadata(args.train),
        "validation": dataset_metadata(args.validation),
    }
    validate_dataset_input_contract(datasets, args.model_format)
    environment = environment_metadata(device)
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume)
        print(json.dumps({"resumedFrom": str(resume_path)}))
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != f"riichi-analysis-model-v{args.model_format}":
            raise RuntimeError("resume checkpoint has an unsupported format")
        if checkpoint.get("modelArchitecture") != architecture.to_dict():
            raise RuntimeError("resume checkpoint uses a different model architecture")
        if (
            args.model_format in {9, 10, 11}
            and checkpoint.get("modelInput") != model_input_metadata()
        ):
            raise RuntimeError(
                "resume checkpoint uses a different model-input contract"
            )
        if checkpoint.get("predictionValues") != {
            "dora": list(DORA_VALUES),
            "score": list(SCORE_VALUES),
        }:
            raise RuntimeError("resume checkpoint uses different prediction values")
        model.load_state_dict(checkpoint["model"], strict=True)
        balance = checkpoint.get("lossBalancer")
        if not isinstance(balance, dict) or balance.get("terms") != list(loss_terms):
            raise RuntimeError("resume checkpoint has incompatible loss-balance state")
        balancer.load_state_dict(balance["state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_random_state(checkpoint.get("randomState"))
        samples_seen, step = resume_training_cursor(
            checkpoint,
            datasets,
            args.batch_size,
            allow_complete=args.validate_only,
        )
        analysis_samples_seen = int(
            checkpoint.get(
                "analysisSamplesSeen",
                samples_seen if args.model_format < 10 else -1,
            )
        )
        if analysis_samples_seen < 0 or analysis_samples_seen > samples_seen:
            raise RuntimeError(
                "resume checkpoint has an invalid analysis-sample counter"
            )
        if not args.validate_only and step_budget_reached(step, args.max_steps):
            raise RuntimeError(
                "resume budget is already exhausted; increase --max-steps"
            )
        if (
            not args.validate_only
            and args.max_analysis_samples > 0
            and analysis_samples_seen >= args.max_analysis_samples
        ):
            raise RuntimeError(
                "resume budget is already exhausted; increase --max-analysis-samples"
            )
    # The order comes from the training manifest: one pass over globally mixed
    # packs, with no shuffling left for the loader to do.
    train_manifest = PackDataset(args.train, batch_size=args.batch_size)
    if args.validate_only:
        sample_limit = samples_seen
        train_data = None
    else:
        sample_limit, unread_samples = single_pass_window(
            train_manifest.samples, samples_seen, args.max_train_samples
        )
        train_data = PackDataset(
            args.train,
            start_sample=samples_seen,
            max_samples=unread_samples,
            batch_size=args.batch_size,
        )
    validation_data = PackDataset(
        args.validation,
        max_samples=args.max_validation_samples,
        batch_size=args.batch_size,
    )
    train_loader = (
        DataLoader(
            train_data,
            batch_size=None,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        if train_data is not None
        else ()
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=None,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    try:
        fixture = next(iter(validation_loader))
    except StopIteration:
        fixture = None
    if args.model_format in {8, 9, 10, 11}:
        if args.validate_only:
            saved_contract = checkpoint.get("trainingContract")
            training_contract = (
                saved_contract
                if isinstance(saved_contract, dict)
                else {
                    "train": None,
                    "validation": (
                        validate_v8_training_batch(
                            fixture, require_analysis_active=args.model_format >= 10
                        )
                        if fixture is not None
                        else None
                    ),
                }
            )
        else:
            try:
                train_fixture = next(iter(train_loader))
            except StopIteration as error:
                raise RuntimeError("training data contains no samples") from error
            training_contract = {
                "train": validate_v8_training_batch(
                    train_fixture, require_analysis_active=args.model_format >= 10
                ),
                "validation": (
                    validate_v8_training_batch(
                        fixture, require_analysis_active=args.model_format >= 10
                    )
                    if fixture is not None
                    else None
                ),
            }
    else:
        training_contract = None

    args.run.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()),
        "run": str(args.run.resolve()),
        "effectiveDevice": str(device),
        "parameters": parameters,
        "modelArchitecture": architecture.to_dict(),
        "datasets": datasets,
        "trainingOrder": {
            "type": "globally-mixed-packs-v1",
            "pass": "single-pass-v1",
            "trainSeed": datasets["train"].get("seed"),
            "verified": datasets["train"].get("verified"),
            "startSample": samples_seen,
            "sampleLimit": sample_limit,
            "analysisSamplesSeen": analysis_samples_seen,
            "analysisSampleLimit": args.max_analysis_samples or None,
        },
        "environment": environment,
        "trainingContract": training_contract,
    }
    (args.run / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    log_path = args.run / "metrics.jsonl"
    writer = open_dashboard(args.run)
    atexit.register(close_dashboard, writer)

    interrupt_requested = False

    def request_interrupt(*_arguments: object) -> None:
        """Request a checkpoint at the next safe boundary between batches."""

        nonlocal interrupt_requested
        interrupt_requested = True

    def save_run_checkpoint(
        path: Path,
        *,
        validation: dict[str, float] | None = None,
        pass_complete: bool = False,
    ) -> None:
        save_checkpoint(
            path,
            model,
            optimizer,
            scaler,
            balancer,
            architecture,
            step=step,
            samples_seen=samples_seen,
            analysis_samples_seen=analysis_samples_seen,
            batch_size=args.batch_size,
            pass_complete=pass_complete,
            parameters=parameters,
            datasets=datasets,
            environment=environment,
            validation=validation,
        )

    signal.signal(signal.SIGINT, request_interrupt)
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "parameters": parameters}, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stop_reason: str | None = "validation-only" if args.validate_only else None
    last_log_time = started
    last_log_samples = samples_seen
    checkpoint_interval = args.checkpoint_every_samples
    next_checkpoint_sample = (
        ((samples_seen // checkpoint_interval) + 1) * checkpoint_interval
        if checkpoint_interval > 0
        else 0
    )
    model.train()
    for batch in train_loader:
        if interrupt_requested:
            stop_reason = "interrupted"
            break
        batch = move_batch(batch, device)
        peak = args.peak_learning_rate or args.learning_rate
        optimizer.param_groups[0]["lr"] = learning_rate_at(
            step, args.warmup_steps, args.cooldown_steps, peak, args.learning_rate
        )
        optimizer.param_groups[1]["lr"] = learning_rate_at(
            step,
            args.warmup_steps,
            args.cooldown_steps,
            args.loss_balance_learning_rate,
            args.loss_balance_learning_rate,
        )
        optimizer.zero_grad(set_to_none=True)
        gradient_geometry: dict[str, float] = {}
        diagnose_gradients = bool(
            args.gradient_diagnostics_every
            and (step + 1) % args.gradient_diagnostics_every == 0
        )
        try:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                if diagnose_gradients:
                    outputs, shared = model.forward_with_shared(batch["obs"].float())
                else:
                    outputs = model(batch["obs"].float())
                total, losses, active, weights = multitask_loss(
                    outputs, batch, balancer
                )
            if diagnose_gradients:
                gradient_geometry = shared_gradient_geometry(losses, active, shared)
            scaler.scale(total).backward()
        except torch.cuda.OutOfMemoryError:
            # The failed batch has not advanced the single-pass cursor, so a
            # resumed run will read it again rather than silently dropping it.
            save_run_checkpoint(args.run / "oom-interrupted.pt")
            write_pointer(
                args.run,
                args.run / "oom-interrupted.pt",
                step=step,
                samplesSeen=samples_seen,
            )
            print(json.dumps({"phase": "oom", "step": step}))
            raise
        scaler.unscale_(optimizer)
        gradient_norm = gradient_total_norm(list(model.parameters()))
        gradient_max = max(
            float(parameter.grad.abs().max())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
        step += 1
        samples_seen += len(batch["policy"])
        analysis_rows = batch.get("analysis_active")
        analysis_samples_seen += (
            len(batch["policy"])
            if analysis_rows is None
            else int(analysis_rows.bool().sum().item())
        )
        if step == 1 or step % 100 == 0 or gradient_geometry:
            now = time.perf_counter()
            interval_seconds = max(now - last_log_time, 1e-9)
            interval_samples = samples_seen - last_log_samples
            record = {
                "phase": "train",
                "step": step,
                "samples": samples_seen,
                "analysisSamples": analysis_samples_seen,
                "elapsedSeconds": now - started,
                "samplesPerSecond": interval_samples / interval_seconds,
                "total": float(total.detach()),
                "gradientNorm": gradient_norm,
                "gradientMax": gradient_max,
                **{name: float(value.detach()) for name, value in losses.items()},
                **{
                    f"lossWeight/{name}": float(value.detach())
                    for name, value in weights.items()
                },
                **gradient_geometry,
            }
            if device.type == "cuda":
                record["peakAllocatedMiB"] = (
                    torch.cuda.max_memory_allocated(device) / 2**20
                )
                record["peakReservedMiB"] = (
                    torch.cuda.max_memory_reserved(device) / 2**20
                )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))
            if writer is not None:
                writer.add_scalar("Loss/train_batch", float(total.detach()), step)
                for name, value in losses.items():
                    writer.add_scalar(
                        f"Loss/train_{name}_batch", float(value.detach()), step
                    )
                for name, value in weights.items():
                    writer.add_scalar(
                        f"LossBalance/{name}", float(value.detach()), step
                    )
                    if active[name]:
                        writer.add_scalar(
                            f"LossWeighted/{name}",
                            float((value * losses[name]).detach()),
                            step,
                        )
                writer.add_scalar("Gradient/norm", gradient_norm, step)
                writer.add_scalar("Gradient/max", gradient_max, step)
                for name, value in gradient_geometry.items():
                    writer.add_scalar(f"GradientShared/{name}", value, step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("Progress/samples", samples_seen, step)
                writer.add_scalar(
                    "Progress/analysis_samples", analysis_samples_seen, step
                )
                writer.add_scalar(
                    "Progress/samples_per_second", record["samplesPerSecond"], step
                )
                if device.type == "cuda":
                    writer.add_scalar(
                        "CUDA/allocated_mib", record["peakAllocatedMiB"], step
                    )
                    writer.add_scalar(
                        "CUDA/reserved_mib", record["peakReservedMiB"], step
                    )
                    writer.add_scalar("AMP/scale", scaler.get_scale(), step)
            last_log_time = now
            last_log_samples = samples_seen
        # A terminal boundary gets its named checkpoint immediately below.
        # Do not write the same model twice merely because it also lands on a
        # rolling cadence boundary.
        terminal_boundary = (
            samples_seen >= sample_limit
            or (
                args.max_analysis_samples > 0
                and analysis_samples_seen >= args.max_analysis_samples
            )
            or step_budget_reached(step, args.max_steps)
            or interrupt_requested
        )
        if (
            checkpoint_interval > 0
            and samples_seen >= next_checkpoint_sample
            and not terminal_boundary
        ):
            rolling = args.run / f"ckpt-{step:09d}.pt"
            save_run_checkpoint(rolling)
            write_pointer(args.run, rolling, step=step, samplesSeen=samples_seen)
            prune_numbered_checkpoints(args.run, args.keep_checkpoints)
            while next_checkpoint_sample <= samples_seen:
                next_checkpoint_sample += checkpoint_interval
        if step_budget_reached(step, args.max_steps):
            stop_reason = "max-steps"
            break
        if (
            args.max_analysis_samples > 0
            and analysis_samples_seen >= args.max_analysis_samples
        ):
            stop_reason = "max-analysis-samples"
            break
        if interrupt_requested:
            stop_reason = "interrupted"
            break

    if stop_reason == "interrupted":
        save_run_checkpoint(args.run / "interrupted.pth")
        write_pointer(
            args.run,
            args.run / "interrupted.pth",
            step=step,
            samplesSeen=samples_seen,
        )
        print(json.dumps({"phase": "interrupted", "step": step}), flush=True)
        close_dashboard(writer)
        atexit.unregister(close_dashboard)
        raise SystemExit(130)

    if stop_reason is None:
        if samples_seen != sample_limit:
            raise RuntimeError(
                "training stream ended at sample "
                f"{samples_seen}, expected {sample_limit}"
            )
        stop_reason = (
            "corpus-exhausted"
            if sample_limit == train_manifest.samples
            else "max-train-samples"
        )
    pass_complete = samples_seen == train_manifest.samples

    checkpoint_name = (
        "checkpoint-complete.pt" if pass_complete else f"checkpoint-step-{step}.pt"
    )
    final_checkpoint = args.run / checkpoint_name
    # The training boundary is durable before validation starts. A validation
    # error or interruption must never discard the newest trained samples.
    save_run_checkpoint(final_checkpoint, pass_complete=pass_complete)
    write_pointer(
        args.run,
        final_checkpoint,
        step=step,
        samplesSeen=samples_seen,
        passComplete=pass_complete,
        validationComplete=False,
    )

    try:
        metrics = validate(
            model,
            balancer,
            validation_loader,
            device,
            should_stop=lambda: interrupt_requested,
        )
    except TrainingInterrupted:
        print(json.dumps({"phase": "validation-interrupted", "step": step}), flush=True)
        close_dashboard(writer)
        atexit.unregister(close_dashboard)
        raise SystemExit(130) from None
    record = {
        "phase": "validation",
        "step": step,
        "stopReason": stop_reason,
        **metrics,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))
    write_dashboard(writer, metrics, step)
    if writer is not None and fixture is not None:
        # One fixed input, probed every validation: a metric can look steady
        # while the behaviour behind it drifts.
        with torch.no_grad():
            probe = model(fixture["obs"].float().to(device))
            policy = probe["policy"][0].softmax(-1)
            top = policy.topk(3)
            for rank, (value, action) in enumerate(
                zip(top.values.tolist(), top.indices.tolist(), strict=True), start=1
            ):
                writer.add_scalar(f"Fixture/policy_top{rank}_probability", value, step)
                writer.add_scalar(f"Fixture/policy_top{rank}_action", action, step)
            # Sample zero, first opponent: the head predicts one shanten
            # per opponent, so the probe keeps to a fixed slice of it.
            for shanten, probability in enumerate(
                probe["shanten"][0, 0].softmax(-1).tolist()
            ):
                writer.add_scalar(f"Fixture/shanten_p{shanten}", probability, step)
            writer.add_scalar("Fixture/max_policy", float(policy.max()), step)
    save_run_checkpoint(
        final_checkpoint,
        validation=metrics,
        pass_complete=pass_complete,
    )
    write_pointer(
        args.run,
        final_checkpoint,
        step=step,
        samplesSeen=samples_seen,
        passComplete=pass_complete,
        validationComplete=True,
    )
    close_dashboard(writer)
    atexit.unregister(close_dashboard)


if __name__ == "__main__":
    main()
