"""Short, read-only-pack execution benchmark; never resumes formal weights.

Each arm starts from the same random seed and reads the same bounded prefix.
Repeated arms are timing diagnostics, not epochs or model-quality comparisons.
Reports stay outside Git and no usable model checkpoint is produced.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from riichi_analysis_engine.count_acceleration import CudaGraphJointCounts
from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    BatchSupervision,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.optimizer_execution import make_adamw
from riichi_analysis_engine.train import forward_batch, gradients_are_finite, move_batch
from riichi_analysis_engine.training_schema import validate_v16_training_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=("baseline", "cpu-labels", "fused", "prefetch", "fp32"),
    )
    args = parser.parse_args()
    if args.output.exists() or not 16 <= args.steps <= 200 or not 1 <= args.rounds <= 3:
        parser.error("new output, 16..200 steps and 1..3 rounds required")
    if not torch.cuda.is_available():
        parser.error("CUDA device required for this benchmark")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    report = {
        "purpose": "execution timing only; no quality claim or formal checkpoint",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "batchSize": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "arms": [],
    }
    configurations = [
        ("baseline", "legacy", "foreach", 0, "amp"),
        ("cpu-labels", "cpu-metadata", "foreach", 0, "amp"),
        ("fused", "cpu-metadata", "fused", 0, "amp"),
        ("prefetch", "cpu-metadata", "fused", 1, "amp"),
        ("fp32", "cpu-metadata", "fused", 1, "fp32"),
    ]
    if args.arms is not None:
        configurations = [
            configuration
            for configuration in configurations
            if configuration[0] in args.arms
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(args.rounds):
        for name, loss_mode, backend, workers, precision in configurations[
            :: 1 if repetition % 2 == 0 else -1
        ]:
            torch.manual_seed(71)
            torch.cuda.manual_seed_all(71)
            model = RiichiAnalysisModel(format_version=18).to(device).train()
            balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8).to(device)
            optimizer, effective = make_adamw(
                [
                    {"params": model.parameters(), "lr": 2e-5, "weight_decay": 1e-4},
                    {"params": balancer.parameters(), "lr": 1e-3, "weight_decay": 0.0},
                ],
                device,
                backend,
            )
            parameters = [*model.parameters(), *balancer.parameters()]
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "amp")
            projector = CudaGraphJointCounts(args.batch_size, device)
            data = PackDataset(
                args.packs,
                batch_size=args.batch_size,
                max_samples=(args.steps + args.warmup) * args.batch_size,
            )
            options = {"batch_size": None, "num_workers": workers, "pin_memory": True}
            if workers:
                options.update(prefetch_factor=2, persistent_workers=True)
            loader = DataLoader(data, **options)
            iterator = iter(loader)
            signatures = []
            durations = {
                key: [] for key in ("copy", "forward", "loss", "backward", "optimizer")
            }
            host_seconds = {"loader": 0.0, "metadata": 0.0}
            raw_losses = []
            torch.cuda.reset_peak_memory_stats()
            started = None
            analysis_samples = 0
            amp_backoffs = 0
            measured_backoffs = 0
            for step in range(args.steps + args.warmup):
                if step == args.warmup:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                before = time.perf_counter()
                cpu = next(iterator)
                loaded = time.perf_counter()
                if step == 0:
                    validate_v16_training_batch(cpu)
                supervision = (
                    BatchSupervision.from_cpu(cpu) if loss_mode != "legacy" else None
                )
                prepared = time.perf_counter()
                if step >= args.warmup:
                    host_seconds["loader"] += loaded - before
                    host_seconds["metadata"] += prepared - loaded
                events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
                optimizer.zero_grad(set_to_none=True)
                events[0].record()
                batch = move_batch(cpu, device)
                events[1].record()
                for attempt in range(8):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.float16, enabled=precision == "amp"
                    ):
                        output = forward_batch(model, batch)
                        events[2].record()
                        total, losses, active, _ = multitask_loss(
                            output,
                            batch,
                            balancer,
                            supervision=supervision,
                            joint_count_projector=projector,
                        )
                        events[3].record()
                    scaler.scale(total).backward()
                    events[4].record()
                    scaler.unscale_(optimizer)
                    finite = gradients_are_finite(parameters)
                    previous_scale = scaler.get_scale() if not finite else None
                    scaler.step(optimizer)
                    scaler.update()
                    if finite:
                        break
                    amp_backoffs += 1
                    measured_backoffs += int(step >= args.warmup)
                    if attempt == 7 or scaler.get_scale() >= previous_scale:
                        raise FloatingPointError(
                            f"non-finite gradients in {name}/{step}; same-batch retries exhausted"
                        )
                analysis_samples += (
                    int(batch["analysis_active"].bool().sum().item())
                    if loss_mode == "legacy"
                    else supervision.analysis_count
                )
                events[5].record()
                if step >= args.warmup:
                    for index, key in enumerate(durations):
                        durations[key].append((events[index], events[index + 1]))
                # Keep CPU references and CUDA scalars; digest/readback AFTER timing.
                signatures.append(cpu)
                if step == args.steps + args.warmup - 1:
                    raw_losses = {key: value.detach() for key, value in losses.items()}
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            digest = hashlib.sha256()
            for cpu in signatures:
                for key in sorted(cpu):
                    digest.update(key.encode())
                    digest.update(cpu[key].numpy().tobytes())
            if not all(bool(torch.isfinite(p).all()) for p in parameters):
                raise FloatingPointError("non-finite model/balancer state")
            if not all(
                bool(torch.isfinite(value).all())
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ):
                raise FloatingPointError("non-finite optimizer state")
            result = {
                "name": name,
                "round": repetition,
                "lossExecution": loss_mode,
                "optimizerExecution": effective,
                "workers": workers,
                "precision": precision,
                "samplesPerSecond": args.steps * args.batch_size / elapsed,
                "seconds": elapsed,
                "hostSeconds": host_seconds,
                "gpuStageMsPerStep": {
                    key: sum(left.elapsed_time(right) for left, right in pairs)
                    / args.steps
                    for key, pairs in durations.items()
                },
                "peakAllocatedMiB": torch.cuda.max_memory_allocated() / 2**20,
                "peakReservedMiB": torch.cuda.max_memory_reserved() / 2**20,
                "inputDigest": digest.hexdigest(),
                "lastRawLoss": {key: float(value) for key, value in raw_losses.items()},
                "active": active,
                "parameters": count_parameters(model),
                "finite": True,
                "analysisSamples": analysis_samples,
                "ampBackoffs": amp_backoffs,
                "measuredAmpBackoffs": measured_backoffs,
            }
            report["arms"].append(result)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
            temporary.replace(args.output)
            print(
                json.dumps(
                    {
                        "name": name,
                        "round": repetition,
                        "samplesPerSecond": result["samplesPerSecond"],
                    }
                ),
                flush=True,
            )
            del (
                iterator,
                loader,
                projector,
                optimizer,
                scaler,
                model,
                balancer,
                output,
                batch,
                cpu,
                signatures,
                raw_losses,
                parameters,
                total,
                losses,
                durations,
                events,
                supervision,
            )
            gc.collect()
            torch.cuda.empty_cache()
    digests = {arm["inputDigest"] for arm in report["arms"]}
    if len(digests) != 1:
        raise RuntimeError("loader arms changed the input stream")
    report["complete"] = True
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
