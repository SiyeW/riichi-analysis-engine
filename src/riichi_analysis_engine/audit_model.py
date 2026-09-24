from __future__ import annotations

import argparse
import json

from .architecture import (
    ModelArchitecture,
    SemanticModelArchitecture,
    StructuredModelArchitecture,
)
from .model import RiichiAnalysisModel, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the model parameter budget.")
    parser.add_argument(
        "--model-format", type=int, choices=(7, 8, 9, 10, 11, 12, 13), default=13
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
    parser.add_argument("--analysis-channels", type=int, default=192)
    parser.add_argument("--analysis-blocks", type=int, default=36)
    parser.add_argument("--analysis-latent-width", type=int, default=768)
    parser.add_argument("--state-width", type=int, default=768)
    parser.add_argument("--future-width", type=int, default=640)
    parser.add_argument("--policy-context-channels", type=int, default=144)
    parser.add_argument("--policy-context-blocks", type=int, default=6)
    parser.add_argument("--policy-context-width", type=int, default=384)
    parser.add_argument("--policy-width", type=int, default=1024)
    parser.add_argument("--semantic-backbone", choices=("cnn", "transformer"), default="cnn")
    parser.add_argument("--semantic-width", type=int, default=256)
    parser.add_argument("--semantic-stem-width", type=int, default=384)
    parser.add_argument("--semantic-event-width", type=int, default=192)
    parser.add_argument("--semantic-backbone-blocks", type=int, default=8)
    parser.add_argument("--semantic-event-blocks", type=int, default=4)
    parser.add_argument("--semantic-decoder-width", type=int, default=512)
    parser.add_argument("--semantic-attention-heads", type=int, default=8)
    parser.add_argument("--semantic-transformer-ff-multiplier", type=int, default=4)
    parser.add_argument("--semantic-transformer-tile-prior-blocks", type=int, default=0)
    parser.add_argument("--semantic-transformer-event-prior-blocks", type=int, default=0)
    parser.add_argument("--semantic-prior-version", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--semantic-design-version", type=int, choices=(1, 2),
        help="v13 architecture revision; defaults to 2 for new models",
    )
    args = parser.parse_args()
    if args.model_format in {12, 13}:
        design_version = args.semantic_design_version or (2 if args.model_format == 13 else 1)
        architecture = SemanticModelArchitecture(
            backbone=args.semantic_backbone,
            width=args.semantic_width,
            stem_width=args.semantic_stem_width,
            event_width=args.semantic_event_width,
            backbone_blocks=args.semantic_backbone_blocks,
            event_blocks=args.semantic_event_blocks,
            decoder_width=args.semantic_decoder_width,
            attention_heads=args.semantic_attention_heads,
            transformer_ff_multiplier=args.semantic_transformer_ff_multiplier,
            transformer_tile_prior_blocks=args.semantic_transformer_tile_prior_blocks,
            transformer_event_prior_blocks=args.semantic_transformer_event_prior_blocks,
            semantic_prior_version=args.semantic_prior_version,
            semantic_design_version=design_version,
        )
    elif args.model_format >= 8:
        family_latent_width = args.family_latent_width or (
            1024 if args.model_format == 11 else 768
        )
        task_width = args.task_width or (1024 if args.model_format == 11 else 512)
        architecture = (
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
