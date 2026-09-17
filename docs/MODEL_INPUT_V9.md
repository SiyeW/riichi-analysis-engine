# Model input v9

Model format v9 separates hidden-state analysis from action selection. The two
inputs have different owners and must not be collapsed into an unnamed slice of
one observation tensor.

At the storage and model-call boundary, the two logical inputs are concatenated
into one versioned transport tensor to retain the existing packed-data path.
Only `compose_model_input` and `split_model_input` own that representation:
the first 1001 channels are the analysis observation and the remaining 156 are
the policy context. Model branches never consume the transport tensor directly.

## Analysis observation

`riichi-analysis-public-history-v1` is a `1001 x 34` hidden-information
observation. Its channel order and values are intentionally identical to
`mj_shanten_predictor_v2_3_tile_plane_v8`, the input used by the established
opponent shanten and deal-in model.

It contains the controlled player's visible hand, rule state, symmetric public
history for all four players, and an explicit cursor for every natural mjai
frame event. It does not contain the controlled player's computed waits,
furiten, shanten, legal actions, or single-player expected-value tables.

All analysis families consume this observation:

- opponent shanten, ron eligibility, and tile deal-in risk;
- concealed-hand and wall allocation;
- opponent dora and hand value;
- kyoku outcome and score accounts;
- final placement and score.

## Policy context

`riichi-analysis-policy-context-v1` contains only decision information that is
not already represented by the analysis observation. It is extracted from
libriichi observation v4 through named, tested ranges:

- owned and unseen dora summaries;
- the controlled player's waits, furiten, and shanten;
- kan-selection state, action candidates, and single-player efficiency tables.

The policy tower receives both the shared analysis representation and a small
encoder for this context. No other family may read policy context.

## Sample ownership

Analysis supervision and policy supervision have different natural sample
distributions. In the first v9 attribution run, the existing sample selection
and the existing `policy = -1` marker are deliberately preserved so that the
input is the only experimental change. A later one-view-per-event analysis
stream, if evidence warrants it, requires its own explicit format revision; it
must not be smuggled into this attribution run.

## Compatibility

The v9 observation schema, packed-data format, checkpoint format, and exported
weight metadata are versioned together. v8 packs and checkpoints are rejected;
they cannot be upgraded from their stored observation alone because they do not
contain the symmetric public-history timeline or explicit event cursor.

Training conversion and runtime inference must use the same public-history
encoder. Tests compare the v9 analysis observation element-for-element with the
established v2.3 encoder for all four perspectives and every supported frame
event.

## Attribution order

The first bounded v9 run changes the input contract while preserving the v8
network, targets, optimizer, and sample prefix as far as their contracts allow.
Only after that result may sampling, conditional deal-in supervision, or the
30-plus-24 residual split be changed. This keeps the cause of any recovery
measurable.
