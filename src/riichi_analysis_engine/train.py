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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .architecture import ModelArchitecture
from .dataset import PackDataset
from .kyoku_outcome import OUTCOME_DEAL_IN_INDICATORS, OUTCOME_WINNER_INDICATORS
from .losses import LOSS_TERMS, LearnedUncertaintyBalancer, multitask_loss, score_class_indices
from .model import RiichiAnalysisModel, count_parameters
from .prediction_values import DORA_TAIL_START, DORA_VALUES, SCORE_VALUES


class TrainingInterrupted(Exception):
    """Stop at a boundary whose state can be resumed without replaying data."""


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
        "verified": audit.get("verified") if isinstance(audit, dict) else None,
        "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
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
    limit = corpus_samples if max_train_samples == 0 else min(corpus_samples, max_train_samples)
    if next_sample >= limit:
        raise RuntimeError(
            "resume cursor has reached the sample limit; increase --max-train-samples"
        )
    return limit, limit - next_sample


def resume_training_cursor(
    checkpoint: dict[str, object],
    datasets: dict[str, dict[str, object]],
    batch_size: int,
) -> tuple[int, int]:
    """Validate and return the next unread sample and completed update count."""

    cursor = checkpoint.get("trainingCursor")
    if not isinstance(cursor, dict) or cursor.get("type") != "single-pass-v1":
        raise RuntimeError(
            "resume checkpoint predates the single-pass cursor and cannot be resumed safely"
        )
    if cursor.get("complete") is not False:
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
    totals: dict[str, float] = {"total": 0.0, **{name: 0.0 for name in LOSS_TERMS}}
    # Label marginals, kept to build the null baseline each loss is compared
    # against: what a model that only knows the label distribution would score.
    policy_labels = np.zeros(46, dtype=np.int64)
    shanten_labels = np.zeros(7, dtype=np.int64)
    balance_totals: dict[str, float] = {name: 0.0 for name in LOSS_TERMS}
    batches = 0
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    def add_metric(name: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
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
        shanten_labels += (
            torch.bincount(batch["shanten"].reshape(-1), minlength=7).cpu().numpy()
        )
        policy_logits = outputs["policy"].masked_fill(~batch["action_mask"], -torch.inf)
        add_metric(
            "policyAccuracy",
            policy_logits.argmax(-1) == batch["policy"],
            policy_valid,
        )
        add_metric(
            "shantenAccuracy",
            outputs["shanten"].argmax(-1) == batch["shanten"],
        )
        add_metric(
            "furitenBrier",
            (outputs["furiten_no_yaku"].sigmoid() - batch["furiten_no_yaku"].float()).square(),
            batch["shanten"] == 0,
        )
        add_metric(
            "dealInTileBrier",
            (outputs["deal_in_tile"].sigmoid() - batch["deal_in_tile"].float()).square(),
        )
        add_metric(
            "concealedCountAccuracy",
            outputs["concealed_count"].argmax(-1) == batch["concealed_count"],
        )
        add_metric(
            "concealedRedCountAccuracy",
            outputs["concealed_red_count"].argmax(-1) == batch["concealed_red_count"],
        )
        add_metric(
            "wallCountAccuracy",
            outputs["wall_count"].argmax(-1) == batch["wall_count"],
        )
        add_metric(
            "wallRedCountAccuracy",
            outputs["wall_red_count"].argmax(-1) == batch["wall_red_count"],
        )
        add_metric(
            "doraMae",
            (F.softplus(outputs["dora_point"]) - batch["dora"].float()).abs(),
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
                outputs["score_distribution"][winner_mask].argmax(-1)
                == score_class_indices(batch["score"].long()[winner_mask]),
            )
        add_metric(
            "scoreMaePoints",
            (F.softplus(outputs["score_point"]) * 1000.0 - batch["score"].float()).abs(),
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
        add_metric(
            "kyokuDeltaMaePoints",
            (outputs["kyoku_delta"] * 10_000.0 - batch["kyoku_delta"].float()).abs(),
        )
        add_metric("placementJointAccuracy", outputs["placement"].argmax(-1) == batch["placement"])
        add_metric(
            "matchScoreMaePoints",
            (outputs["match_score"] * 10_000.0 - batch["match_score"].float()).abs(),
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
    if nulls["policy"] > 0 and nulls["shanten"] > 0:
        score = 0.5 * (
            result["policy"] / nulls["policy"] + result["shanten"] / nulls["shanten"]
        )
        result["Selection/core_score"] = score
        result["Selection/core_skill"] = 1.0 - score
    result.update(
        {
            f"lossWeight/{name}": balance_totals[name] / max(1, batches)
            for name in LOSS_TERMS
        }
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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def write_dashboard(writer: object | None, metrics: dict[str, float], step: int) -> None:
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
    if not isinstance(numpy_state, dict) or not isinstance(numpy_state.get("keys"), torch.Tensor):
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
    architecture: ModelArchitecture,
    *,
    step: int,
    samples_seen: int,
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
            "format": "riichi-analysis-model-v6",
            "step": step,
            "samplesSeen": samples_seen,
            "trainingCursor": {
                "type": "single-pass-v1",
                "nextSample": samples_seen,
                "batchesConsumed": step,
                "batchSize": batch_size,
                "complete": pass_complete,
            },
            "model": model.state_dict(),
            "modelArchitecture": architecture.to_dict(),
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
    parser.add_argument("--analysis-channels", type=int, default=192)
    parser.add_argument("--analysis-blocks", type=int, default=36)
    parser.add_argument("--analysis-latent-width", type=int, default=768)
    parser.add_argument("--state-width", type=int, default=768)
    parser.add_argument("--future-width", type=int, default=640)
    parser.add_argument("--policy-context-channels", type=int, default=96)
    parser.add_argument("--policy-context-blocks", type=int, default=4)
    parser.add_argument("--policy-context-width", type=int, default=256)
    parser.add_argument("--policy-width", type=int, default=640)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--checkpoint-every-samples", type=int, default=1_000_000)
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
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_steps < 0:
        raise ValueError("max steps must be non-negative")
    if args.checkpoint_every_samples < 0:
        raise ValueError("checkpoint interval must be non-negative")
    if args.keep_checkpoints < 0:
        raise ValueError("checkpoint retention must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.float16 if device.type == "cuda" else None

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
    model = RiichiAnalysisModel(architecture=architecture).to(device)
    balancer = LearnedUncertaintyBalancer().to(device)
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
    datasets = {
        "train": dataset_metadata(args.train),
        "validation": dataset_metadata(args.validation),
    }
    environment = environment_metadata(device)
    if args.resume is not None:
        resume_path = resolve_resume_path(args.resume)
        print(json.dumps({"resumedFrom": str(resume_path)}))
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != "riichi-analysis-model-v6":
            raise RuntimeError("resume checkpoint has an unsupported format")
        if checkpoint.get("modelArchitecture") != architecture.to_dict():
            raise RuntimeError("resume checkpoint uses a different model architecture")
        if checkpoint.get("predictionValues") != {
            "dora": list(DORA_VALUES),
            "score": list(SCORE_VALUES),
        }:
            raise RuntimeError("resume checkpoint uses different prediction values")
        model.load_state_dict(checkpoint["model"], strict=True)
        balance = checkpoint.get("lossBalancer")
        if not isinstance(balance, dict) or balance.get("terms") != list(LOSS_TERMS):
            raise RuntimeError("resume checkpoint has incompatible loss-balance state")
        balancer.load_state_dict(balance["state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_random_state(checkpoint.get("randomState"))
        samples_seen, step = resume_training_cursor(checkpoint, datasets, args.batch_size)
        if step_budget_reached(step, args.max_steps):
            raise RuntimeError("resume budget is already exhausted; increase --max-steps")
    # The order comes from the training manifest: one pass over globally mixed
    # packs, with no shuffling left for the loader to do.
    train_manifest = PackDataset(args.train, batch_size=args.batch_size)
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
    train_loader = DataLoader(
        train_data,
        batch_size=None,
        num_workers=0,
        pin_memory=device.type == "cuda",
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
        },
        "environment": environment,
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

    stop_reason: str | None = None
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
        try:
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                outputs = model(batch["obs"].float())
                total, losses, _active, weights = multitask_loss(outputs, batch, balancer)
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
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_([*model.parameters(), *balancer.parameters()], 5.0)
        )
        gradient_max = max(
            float(parameter.grad.abs().max())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        scaler.step(optimizer)
        scaler.update()
        step += 1
        samples_seen += len(batch["policy"])
        if step == 1 or step % 100 == 0:
            now = time.perf_counter()
            interval_seconds = max(now - last_log_time, 1e-9)
            interval_samples = samples_seen - last_log_samples
            record = {
                "phase": "train",
                "step": step,
                "samples": samples_seen,
                "elapsedSeconds": now - started,
                "samplesPerSecond": interval_samples / interval_seconds,
                "total": float(total.detach()),
                **{name: float(value.detach()) for name, value in losses.items()},
                **{
                    f"lossWeight/{name}": float(value.detach())
                    for name, value in weights.items()
                },
            }
            if device.type == "cuda":
                record["peakAllocatedMiB"] = torch.cuda.max_memory_allocated(device) / 2**20
                record["peakReservedMiB"] = torch.cuda.max_memory_reserved(device) / 2**20
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))
            if writer is not None:
                writer.add_scalar("Loss/train_batch", float(total.detach()), step)
                for name, value in losses.items():
                    writer.add_scalar(f"Loss/train_{name}_batch", float(value.detach()), step)
                for name, value in weights.items():
                    writer.add_scalar(f"LossBalance/{name}", float(value.detach()), step)
                    if _active[name]:
                        writer.add_scalar(
                            f"LossWeighted/{name}",
                            float((value * losses[name]).detach()),
                            step,
                        )
                writer.add_scalar("Gradient/norm", gradient_norm, step)
                writer.add_scalar("Gradient/max", gradient_max, step)
                writer.add_scalar("LR", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("Progress/samples", samples_seen, step)
                writer.add_scalar("Progress/samples_per_second", record["samplesPerSecond"], step)
                if device.type == "cuda":
                    writer.add_scalar("CUDA/allocated_mib", record["peakAllocatedMiB"], step)
                    writer.add_scalar("CUDA/reserved_mib", record["peakReservedMiB"], step)
                    writer.add_scalar("AMP/scale", scaler.get_scale(), step)
            last_log_time = now
            last_log_samples = samples_seen
        if checkpoint_interval > 0 and samples_seen >= next_checkpoint_sample:
            rolling = args.run / f"ckpt-{step:09d}.pt"
            save_run_checkpoint(rolling)
            write_pointer(args.run, rolling, step=step, samplesSeen=samples_seen)
            prune_numbered_checkpoints(args.run, args.keep_checkpoints)
            while next_checkpoint_sample <= samples_seen:
                next_checkpoint_sample += checkpoint_interval
        if step_budget_reached(step, args.max_steps):
            stop_reason = "max-steps"
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
                f"training stream ended at sample {samples_seen}, expected {sample_limit}"
            )
        stop_reason = (
            "corpus-exhausted"
            if sample_limit == train_manifest.samples
            else "max-train-samples"
        )
    pass_complete = samples_seen == train_manifest.samples

    checkpoint_name = "checkpoint-complete.pt" if pass_complete else f"checkpoint-step-{step}.pt"
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
        raise SystemExit(130)
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
