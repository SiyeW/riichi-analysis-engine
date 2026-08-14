from __future__ import annotations

import argparse
import json

from .model import RiichiAnalysisModel, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the model parameter budget.")
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=54)
    args = parser.parse_args()
    model = RiichiAnalysisModel(channels=args.channels, blocks=args.blocks)
    counts = count_parameters(model)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
