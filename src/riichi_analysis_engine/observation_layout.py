"""Stable ownership boundaries within libriichi observation v4.

The first range is state that remains meaningful for prediction heads.  The
tail contains the decision-only phase marker, legal-action features, and
single-player EV tables.  Keeping the split explicit prevents those candidate
specific features from consuming the shared analysis encoder's capacity.
"""

from __future__ import annotations

from .constants import MORTAL_OBS_CHANNELS, OBS_CHANNELS, RANK_FEATURE_CHANNELS


# Mortal v4's `at_kan_select` marker starts at channel 870.  The engine inserts
# explicit all-player rank features before the decision-only tail.
MORTAL_ANALYSIS_CHANNELS = 870
ANALYSIS_CHANNELS = MORTAL_ANALYSIS_CHANNELS + RANK_FEATURE_CHANNELS
POLICY_CONTEXT_START = ANALYSIS_CHANNELS
POLICY_CONTEXT_CHANNELS = MORTAL_OBS_CHANNELS - MORTAL_ANALYSIS_CHANNELS

if ANALYSIS_CHANNELS <= 0 or POLICY_CONTEXT_CHANNELS <= 0:
    raise RuntimeError("observation v4 split is invalid")
if ANALYSIS_CHANNELS + POLICY_CONTEXT_CHANNELS != OBS_CHANNELS:
    raise RuntimeError("observation v4 split does not cover every channel")
