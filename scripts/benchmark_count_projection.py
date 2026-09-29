"""Compare eager and CUDA replay using synthetic constraints, without training data."""

import argparse
import json
import statistics
import time

import torch

from riichi_analysis_engine.count_acceleration import CudaGraphJointCounts
from riichi_analysis_engine.joint_counts import project_joint_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("batch size/steps must be positive, warmup nonnegative")
    device = torch.device(args.device)
    torch.manual_seed(41)
    inventory = torch.full((args.batch_size, 37), 4, device=device)
    inventory[:, (4, 13, 22)] = 3
    inventory[:, 34:] = 1
    capacities = torch.tensor([[13, 13, 13, 97]], device=device).expand(
        args.batch_size, -1
    )
    residual = (
        torch.randn(args.batch_size, 4, 34, 10, device=device) * 0.2
    ).requires_grad_()
    weights = torch.rand_like(residual)
    arms = [("eager", project_joint_counts)]
    if device.type == "cuda":
        arms.extend(
            [
                ("cuda-graph", CudaGraphJointCounts(args.batch_size, device)),
                ("eager-repeat", project_joint_counts),
            ]
        )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    report = {
        "torch": torch.__version__,
        "device": str(device),
        "batchSize": args.batch_size,
        "steps": args.steps,
        "warmup": args.warmup,
        "arms": {},
    }
    reference = None
    for name, project in arms:
        durations = []
        for step in range(args.warmup + args.steps):
            residual.grad = None
            synchronize()
            started = time.perf_counter()
            probability = project(residual, inventory, capacities).probability
            (probability * weights).square().sum().backward()
            synchronize()
            if step >= args.warmup:
                durations.append(1000 * (time.perf_counter() - started))
        if reference is None:
            reference = (probability.detach().clone(), residual.grad.clone())
        torch.testing.assert_close(probability, reference[0], rtol=0, atol=0)
        torch.testing.assert_close(residual.grad, reference[1], rtol=0, atol=0)
        report["arms"][name] = {
            "medianForwardBackwardMs": statistics.median(durations),
            "bitwiseProbabilityAndGradient": True,
        }
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        report["peakAllocatedMiB"] = torch.cuda.max_memory_allocated(device) / 1048576
        report["peakReservedMiB"] = torch.cuda.max_memory_reserved(device) / 1048576
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
