from __future__ import annotations

import argparse
import json

from .architecture import ModelArchitecture, StructuredModelArchitecture
from .model import RiichiAnalysisModel, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the model parameter budget.")
    parser.add_argument("--model-format", type=int, choices=(7, 8), default=8)
    parser.add_argument("--shared-channels", type=int, default=256)
    parser.add_argument("--shared-blocks", type=int, default=30)
    parser.add_argument("--family-latent-width", type=int, default=768)
    parser.add_argument("--opponent-latent-width", type=int, default=1024)
    parser.add_argument("--policy-latent-width", type=int, default=1024)
    parser.add_argument("--opponent-blocks", type=int, default=24)
    parser.add_argument("--hidden-blocks", type=int, default=8)
    parser.add_argument("--value-blocks", type=int, default=6)
    parser.add_argument("--kyoku-blocks", type=int, default=6)
    parser.add_argument("--match-blocks", type=int, default=4)
    parser.add_argument("--policy-blocks", type=int, default=24)
    parser.add_argument("--task-width", type=int, default=512)
    parser.add_argument("--tile-width", type=int, default=128)
    parser.add_argument("--analysis-channels", type=int, default=192)
    parser.add_argument("--analysis-blocks", type=int, default=36)
    parser.add_argument("--analysis-latent-width", type=int, default=768)
    parser.add_argument("--state-width", type=int, default=768)
    parser.add_argument("--future-width", type=int, default=640)
    parser.add_argument("--policy-context-channels", type=int, default=144)
    parser.add_argument("--policy-context-blocks", type=int, default=6)
    parser.add_argument("--policy-context-width", type=int, default=384)
    parser.add_argument("--policy-width", type=int, default=1024)
    args = parser.parse_args()
    if args.model_format == 8:
        architecture: ModelArchitecture | StructuredModelArchitecture = (
            StructuredModelArchitecture(
                shared_channels=args.shared_channels,
                shared_blocks=args.shared_blocks,
                family_latent_width=args.family_latent_width,
                opponent_latent_width=args.opponent_latent_width,
                policy_latent_width=args.policy_latent_width,
                opponent_blocks=args.opponent_blocks,
                hidden_blocks=args.hidden_blocks,
                value_blocks=args.value_blocks,
                kyoku_blocks=args.kyoku_blocks,
                match_blocks=args.match_blocks,
                policy_blocks=args.policy_blocks,
                task_width=args.task_width,
                tile_width=args.tile_width,
                policy_context_channels=args.policy_context_channels,
                policy_context_blocks=args.policy_context_blocks,
                policy_context_width=args.policy_context_width,
                policy_width=args.policy_width,
            )
        )
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
    model = RiichiAnalysisModel(
        format_version=args.model_format, architecture=architecture
    )
    counts = count_parameters(model)
    print(
        json.dumps(
            {"architecture": architecture.to_dict(), "parameters": counts}, indent=2
        )
    )


if __name__ == "__main__":
    main()
