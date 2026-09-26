"""Bounded, repeatable v13/v14 engineering benchmark on a read-only pack.

No authoritative training cursor is read or written. Models are randomly
initialized; these timings and finite-gradient checks do not measure quality.
Run from an environment with the package installed. Outputs are local artifacts.
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from riichi_analysis_engine.dataset import PackDataset
from riichi_analysis_engine.hidden_transport import (
    physical_count_marginals,
    physical_hidden_counts,
    projected_count_distributions,
)
from riichi_analysis_engine.joint_counts import project_joint_counts
from riichi_analysis_engine.losses import (
    LOSS_TERMS_V8,
    LearnedUncertaintyBalancer,
    multitask_loss,
)
from riichi_analysis_engine.model import RiichiAnalysisModel, count_parameters
from riichi_analysis_engine.train import forward_batch, move_batch
from riichi_analysis_engine.training_schema import validate_semantic_training_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("at least three repetitions are required")
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cpu_batch = next(
        iter(
            PackDataset(
                args.packs, batch_size=args.batch_size, max_samples=args.batch_size
            )
        )
    )
    validate_semantic_training_batch(cpu_batch, require_hidden_baseline_anchor=True)
    batch = move_batch(cpu_batch, device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(action, repeats):
        times = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            action()
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return {
            "medianMs": statistics.median(times),
            "minMs": min(times),
            "maxMs": max(times),
            "repeats": repeats,
        }

    report = {
        "purpose": "engineering only; random initialization; no quality claims",
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "torch": torch.__version__,
        "precision": "fp32",
        "batchSize": len(cpu_batch["obs"]),
        "eventLengths": cpu_batch["event_mask"].sum(-1).tolist(),
        "models": {},
    }
    for version in (13, 14):
        torch.manual_seed(71)
        model = RiichiAnalysisModel(format_version=version).to(device).eval()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = {"parameters": count_parameters(model)}
        one = {key: value[:1] for key, value in batch.items()}
        with torch.inference_mode():
            for _ in range(5):
                forward_batch(model, one)
            result["forwardBatch1"] = timed(
                lambda m=model, b=one: forward_batch(m, b), args.repeats
            )
            result["forwardBatchN"] = timed(
                lambda m=model: forward_batch(m, batch), args.repeats
            )
            output = {
                key: value.cpu() for key, value in forward_batch(model, one).items()
            }
            _, inventory, capacities = physical_hidden_counts(
                cpu_batch["concealed_count"][:1],
                cpu_batch["wall_count"][:1],
                cpu_batch["concealed_red_count"][:1],
                cpu_batch["wall_red_count"][:1],
            )

            def counts(
                version=version,
                output=output,
                inventory=inventory,
                capacities=capacities,
            ):
                if version == 14:
                    return project_joint_counts(
                        output["hidden_joint_residual"], inventory, capacities
                    ).marginals()
                return physical_count_marginals(
                    projected_count_distributions(
                        output["hidden_count_residual"], inventory, capacities
                    )[0]
                )

            counts()
            result["runtimeCpuCountsBatch1"] = timed(counts, args.repeats)
        model.train()
        balancer = LearnedUncertaintyBalancer(LOSS_TERMS_V8).to(device)
        optimizer = torch.optim.AdamW(
            [*model.parameters(), *balancer.parameters()], lr=1e-5
        )
        finite = []

        def step(model=model, optimizer=optimizer, balancer=balancer, finite=finite):
            optimizer.zero_grad(set_to_none=True)
            total, _, _, _ = multitask_loss(
                forward_batch(model, batch), batch, balancer
            )
            total.backward()
            finite.append(
                bool(torch.isfinite(total))
                and all(
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                    if p.grad is not None
                )
            )
            if not finite[-1]:
                raise RuntimeError("non-finite loss or gradient")
            optimizer.step()

        step()
        result["trainStep"] = timed(step, 3)
        result["finiteLossAndGradients"] = all(finite)
        result["peakAllocatedMiB"] = (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        )
        report["models"][str(version)] = result
        print(json.dumps({"version": version, **result}), flush=True)
        del step, counts, output, model, optimizer, balancer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
