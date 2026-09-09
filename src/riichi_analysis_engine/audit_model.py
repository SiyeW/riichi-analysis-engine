from __future__ import annotations

import argparse
import json

from .architecture import ModelArchitecture
from .model import RiichiAnalysisModel, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the model parameter budget.")
    parser.add_argument("--analysis-channels", type=int, default=192)
    parser.add_argument("--analysis-blocks", type=int, default=36)
    parser.add_argument("--analysis-latent-width", type=int, default=768)
    parser.add_argument("--state-width", type=int, default=768)
    parser.add_argument("--future-width", type=int, default=640)
    parser.add_argument("--policy-context-channels", type=int, default=96)
    parser.add_argument("--policy-context-blocks", type=int, default=4)
    parser.add_argument("--policy-context-width", type=int, default=256)
    parser.add_argument("--policy-width", type=int, default=640)
    args = parser.parse_args()
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
    model = RiichiAnalysisModel(architecture=architecture)
    counts = count_parameters(model)
    print(json.dumps({"architecture": architecture.to_dict(), "parameters": counts}, indent=2))


if __name__ == "__main__":
    main()
