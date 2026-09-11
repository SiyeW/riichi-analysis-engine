from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def step_budget_reached(step: int, max_steps: int) -> bool:
    return max_steps > 0 and step >= max_steps


@torch.no_grad()
def validate(
    model: RiichiAnalysisModel,
    balancer: LearnedUncertaintyBalancer,
    loader: DataLoader,
    device: torch.device,
    *,
    progress_every: int = 100,
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
        batch = move_batch(batch, device)
        outputs = model(batch["obs"].float())
        total, losses, _active, weights = multitask_loss(outputs, batch, balancer)
        totals["total"] += float(total)
        for name, value in losses.items():
            totals[name] += float(value)
        for name, value in weights.items():
            balance_totals[name] += float(value)
        policy_valid = batch["policy"] >= 0
        policy_labels += np.bincount(
            batch["policy"][policy_valid].numpy(), minlength=46
        )
        shanten_labels += np.bincount(
            batch["shanten"].reshape(-1).numpy(), minlength=7
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
    """Write the validation dashboard, dropping metrics that carry no signal.

    A metric that never moves, or that repeats a value another tag already has,
    is noise on a dashboard that has to stay readable over days.
    """

    if writer is None:
        return
    seen: dict[float, str] = {}
    for name, value in metrics.items():
        if name == "total" or name.startswith("metric/"):
            tag = "Metrics/" + name.removeprefix("metric/")
        elif name.startswith("Selection/"):
            tag = name
        else:
            tag = "Validation/" + name
        if not isinstance(value, float) or value != value:
            continue
        if value in seen:
            continue
        seen[value] = tag
        writer.add_scalar(tag, value, step)


def save_checkpoint(
    path: Path,
    model: RiichiAnalysisModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    balancer: LearnedUncertaintyBalancer,
    architecture: ModelArchitecture,
    *,
    epoch: int,
    batch_in_epoch: int,
    step: int,
    samples_seen: int,
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
            "epoch": epoch,
            "batchInEpoch": batch_in_epoch,
            "step": step,
            "samplesSeen": samples_seen,
            "model": model.state_dict(),
            "modelArchitecture": architecture.to_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "lossBalancer": {
                "terms": list(balancer.names),
                "state": balancer.state_dict(),
            },
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
    parser.add_argument("--epochs", type=int, default=1)
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
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--keep-checkpoints", type=int, default=5)
    parser.add_argument("--best-metric", default="Selection/core_score")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_steps < 0:
        raise ValueError("max steps must be non-negative")

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
    start_epoch = 0
    resume_batch = 0
    step = 0
    samples_seen = 0
    live: dict[str, object] = {"step": 0, "samplesSeen": 0, "epoch": 0, "batch": 0}
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
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_batch = int(checkpoint.get("batchInEpoch", 0))
        step = int(checkpoint.get("step", 0))
        samples_seen = int(checkpoint.get("samplesSeen", 0))
    # The order comes from the training manifest: one pass over globally mixed
    # packs, with no shuffling left for the loader to do.
    train_data = PackDataset(
        args.train,
        max_samples=args.max_train_samples,
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
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=None,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    try:
        fixture = next(iter(validation_loader))
    except StopIteration:
        fixture = None

    args.run.mkdir(parents=True, exist_ok=True)
    datasets = {
        "train": dataset_metadata(args.train),
        "validation": dataset_metadata(args.validation),
    }
    environment = environment_metadata(device)
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
            "trainSeed": datasets["train"].get("seed"),
            "verified": datasets["train"].get("verified"),
        },
        "environment": environment,
    }
    (args.run / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    log_path = args.run / "metrics.jsonl"
    writer = open_dashboard(args.run)

    def save_interrupted(*_arguments: object) -> None:
        """Ctrl-C leaves a checkpoint behind instead of losing the run."""

        save_checkpoint(
            args.run / "interrupted.pth",
            model,
            optimizer,
            scaler,
            balancer,
            architecture,
            epoch=int(live["epoch"]),
            batch_in_epoch=int(live["batch"]),
            step=int(live["step"]),
            samples_seen=int(live["samplesSeen"]),
            parameters=parameters,
            datasets=datasets,
            environment=environment,
            validation=None,
        )
        print(json.dumps({"phase": "interrupted", "step": int(live["step"])}), flush=True)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, save_interrupted)
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "parameters": parameters}, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    best_score = None
    stopped_at_budget = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # A resumed epoch starts where the checkpoint stopped; the loader skips
        # the packs it already trained on instead of reading and discarding
        # them. Later epochs read the whole corpus again.
        resumed = epoch == start_epoch and resume_batch > 0
        train_data.skip_batches = resume_batch if resumed else 0
        for batch_in_epoch, batch in enumerate(
            train_loader, start=resume_batch if resumed else 1
        ):
            batch = move_batch(batch, device)
            samples_seen += len(batch["policy"])
            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    outputs = model(batch["obs"].float())
                    total, losses, _active, weights = multitask_loss(outputs, batch, balancer)
                scaler.scale(total).backward()
            except torch.cuda.OutOfMemoryError:
                # A long run should leave something to resume from when the GPU
                # runs out, not just a traceback.
                save_checkpoint(
                    args.run / "oom-interrupted.pt",
                    model,
                    optimizer,
                    scaler,
                    balancer,
                    architecture,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    step=step,
                    samples_seen=samples_seen,
                    parameters=parameters,
                    datasets=datasets,
                    environment=environment,
                    validation=None,
                )
                print(json.dumps({"phase": "oom", "step": step}))
                raise
            scaler.unscale_(optimizer)
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [*model.parameters(), *balancer.parameters()], 5.0
                )
            )
            gradient_max = max(
                float(parameter.grad.abs().max())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step == 1 or step % 100 == 0:
                record = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": step,
                    "samples": samples_seen,
                    "elapsedSeconds": time.perf_counter() - started,
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
                        writer.add_scalar(f"LossBalance/{name}", float(value), step)
                    writer.add_scalar("Gradient/norm", gradient_norm, step)
                    writer.add_scalar("Gradient/max", gradient_max, step)
                    writer.add_scalar(
                        "LR", optimizer.param_groups[0]["lr"], step
                    )
            live.update(step=step, samplesSeen=samples_seen, epoch=epoch, batch=batch_in_epoch)
            if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:
                rolling = args.run / f"ckpt-{step:09d}.pt"
                save_checkpoint(
                    rolling,
                    model,
                    optimizer,
                    scaler,
                    balancer,
                    architecture,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    step=step,
                    samples_seen=samples_seen,
                    parameters=parameters,
                    datasets=datasets,
                    environment=environment,
                    validation=None,
                )
                write_pointer(args.run, rolling, step=step, samplesSeen=samples_seen)
                prune_numbered_checkpoints(args.run, args.keep_checkpoints)
                save_checkpoint(
                    args.run / "checkpoint-latest.pt",
                    model,
                    optimizer,
                    scaler,
                    balancer,
                    architecture,
                    epoch=epoch,
                    batch_in_epoch=batch_in_epoch,
                    step=step,
                    samples_seen=samples_seen,
                    parameters=parameters,
                    datasets=datasets,
                    environment=environment,
                    validation=None,
                )
            if step_budget_reached(step, args.max_steps):
                stopped_at_budget = True
                break

        metrics = validate(model, balancer, validation_loader, device)
        record = {
            "phase": "validation",
            "epoch": epoch,
            "step": step,
            "stopReason": "max-steps" if stopped_at_budget else None,
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
                    writer.add_scalar(f"Fixture/policy_top{rank}_action{action}", value, step)
                # Sample zero, first opponent: the head predicts one shanten
                # per opponent, so the probe keeps to a fixed slice of it.
                for shanten, probability in enumerate(
                    probe["shanten"][0, 0].softmax(-1).tolist()
                ):
                    writer.add_scalar(f"Fixture/shanten_p{shanten}", probability, step)
                writer.add_scalar(
                    "Fixture/max_policy", float(policy.max()), step
                )
        score = metrics.get(args.best_metric)
        if score is not None and (best_score is None or float(score) < best_score):
            best_score = float(score)
            save_checkpoint(
                args.run / "best-core.pth",
                model,
                optimizer,
                scaler,
                balancer,
                architecture,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                step=step,
                samples_seen=samples_seen,
                parameters=parameters,
                datasets=datasets,
                environment=environment,
                validation=metrics,
            )
            write_pointer(
                args.run,
                args.run / "best-core.pth",
                step=step,
                samplesSeen=samples_seen,
                bestMetric=args.best_metric,
                bestScore=best_score,
            )
            print(json.dumps({"phase": "best", args.best_metric: best_score, "step": step}))
        checkpoint_name = (
            f"checkpoint-step-{step}.pt"
            if stopped_at_budget
            else f"checkpoint-epoch-{epoch + 1}.pt"
        )
        save_checkpoint(
            args.run / checkpoint_name,
            model,
            optimizer,
            scaler,
            balancer,
            architecture,
            epoch=epoch if stopped_at_budget else epoch + 1,
            batch_in_epoch=batch_in_epoch if stopped_at_budget else 0,
            step=step,
            samples_seen=samples_seen,
            parameters=parameters,
            datasets=datasets,
            environment=environment,
            validation=metrics,
        )
        if stopped_at_budget:
            break


if __name__ == "__main__":
    main()
