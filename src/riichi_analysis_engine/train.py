from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .dataset import ShardDataset
from .losses import DEFAULT_WEIGHTS, multitask_loss
from .model import RiichiAnalysisModel, count_parameters


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def validate(
    model: RiichiAnalysisModel,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"total": 0.0, **{name: 0.0 for name in DEFAULT_WEIGHTS}}
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
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            outputs = model(batch["obs"].float())
            total, losses = multitask_loss(outputs, batch)
        totals["total"] += float(total)
        for name, value in losses.items():
            totals[name] += float(value)
        policy_valid = batch["policy"] >= 0
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
            "wallCountAccuracy",
            outputs["wall_count"].argmax(-1) == batch["wall_count"],
        )
        add_metric(
            "doraMae",
            (F.softplus(outputs["dora"]) - batch["dora"].float()).abs(),
            batch["winner_mask"].bool(),
        )
        add_metric(
            "scoreMaePoints",
            (F.softplus(outputs["score"]) * 1000.0 - batch["score"].float()).abs(),
            batch["winner_mask"].bool(),
        )
        outcome_label = sum(
            batch["win"][:, player].long() << player for player in range(4)
        )
        add_metric("outcomeAccuracy", outputs["outcome"].argmax(-1) == outcome_label)
        add_metric(
            "dealInPlayerBrier",
            (outputs["deal_in_player"].sigmoid() - batch["deal_in_player"].float()).square(),
        )
        add_metric(
            "targetAccuracy",
            outputs["target"].argmax(-1) == batch["target"],
            batch["target"] >= 0,
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
    result = {name: value / max(1, batches) for name, value in totals.items()}
    result.update(
        {
            f"metric/{name}": metric_sums[name] / max(1, metric_counts[name])
            for name in metric_sums
        }
    )
    return result


def save_checkpoint(
    path: Path,
    model: RiichiAnalysisModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    step: int,
    parameters: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": "riichi-analysis-model-v1",
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "parameters": parameters,
            "lossWeights": DEFAULT_WEIGHTS,
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20252026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    amp_dtype = torch.float16 if device.type == "cuda" else None

    model = RiichiAnalysisModel().to(device)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    train_data = ShardDataset(
        args.train,
        shuffle=True,
        seed=args.seed,
        max_samples=args.max_train_samples,
        batch_size=args.batch_size,
    )
    validation_data = ShardDataset(
        args.validation,
        shuffle=False,
        seed=args.seed,
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
    args.run.mkdir(parents=True, exist_ok=True)
    config = {
        **vars(args),
        "train": str(args.train.resolve()),
        "validation": str(args.validation.resolve()),
        "run": str(args.run.resolve()),
        "effectiveDevice": str(device),
        "parameters": parameters,
    }
    (args.run / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    log_path = args.run / "metrics.jsonl"
    step = 0
    samples_seen = 0
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "parameters": parameters}, indent=2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            samples_seen += len(batch["policy"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                outputs = model(batch["obs"].float())
                total, losses = multitask_loss(outputs, batch)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
                }
                if device.type == "cuda":
                    record["peakAllocatedMiB"] = torch.cuda.max_memory_allocated(device) / 2**20
                    record["peakReservedMiB"] = torch.cuda.max_memory_reserved(device) / 2**20
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False))

        metrics = validate(model, validation_loader, device, amp_dtype)
        record = {"phase": "validation", "epoch": epoch, "step": step, **metrics}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False))
        save_checkpoint(
            args.run / f"checkpoint-epoch-{epoch + 1}.pt",
            model,
            optimizer,
            scaler,
            epoch=epoch + 1,
            step=step,
            parameters=parameters,
        )


if __name__ == "__main__":
    main()
