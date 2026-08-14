# Third-party notices

## Mortal and libriichi

The observation encoder, action space, rule-label calculations, and residual model family are based on [Mortal](https://github.com/Equim-chan/Mortal) and its `libriichi` component.

Mortal is licensed under the GNU Affero General Public License v3.0 or later. No Mortal or Akagi model weights are included in this repository.

The local `model.py`, gameplay loader, observation encoder, and constants were verified against Mortal commit `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2` on 2026-08-15.

The current local development build loads `libriichi` from a downloaded Mortal source tree. A public source release must include a reproducible way to build the exact `libriichi` binary it distributes.

## Training data

Training-data details are maintained outside the public repository.
