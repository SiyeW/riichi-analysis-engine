# Model

The model keeps public-state analysis separate from decision context. For storage and model calls, the two inputs form a versioned `1157 × 34` tensor.

## Architecture

The 1,001-channel public-state input covers the controlled player's visible hand, the rules, symmetric public history for all four players, and the current event. Every analysis branch uses this input alone.

The 156-channel decision context is selected from `libriichi`'s Mortal v4 observation. It covers the controlled player's dora summary, shanten and furiten state, action phase, legal actions, and single-player efficiency tables, and is available only to the policy branch.

The default configuration is below. Each weight file records the architecture it was trained with.

| Component | Default structure | Parameters |
| --- | --- | ---: |
| Shared state encoder | 256 channels, 30 pre-activation residual blocks | 12,849,888 |
| Opponent analysis | 24 residual blocks, 1,024-dimensional output | 12,507,166 |
| Concealed tiles and wall | 8 residual blocks, 768-dimensional output | 4,516,400 |
| Dora and hand value | 6 residual blocks, 768-dimensional output | 4,563,532 |
| Kyoku analysis | 6 residual blocks, 768-dimensional output | 4,084,646 |
| Match analysis | 4 residual blocks, 768-dimensional output | 3,274,108 |
| Decision context | 144 channels, 6 residual blocks, 384-dimensional output | 1,266,134 |
| Action policy | 24 residual blocks, 1,024-dimensional output, 46 internal actions | 12,295,118 |
| Total |  | 55,356,992 |

Opponent analysis covers shanten, furiten or no yaku, and per-tile deal-in risk. Separate branches predict concealed tiles and the wall, dora and hand value, kyoku outcomes and score changes, and final placement and scores. The policy combines the same public state with its private decision context and learns recommendation strengths from recorded play.

Each supervised objective is balanced with a learned uncertainty weight during training. Checkpoints store those weights with the model configuration, and validation logs report their effective values.

## Labels

- Opponent shanten, furiten or no yaku, and per-tile deal-in labels are calculated from complete hidden state at each event.
- Concealed hands and the unrevealed wall use `0..4` classification labels. The three red fives use separate `0..1` labels.
- Dora and hand-value losses are applied only when the corresponding player eventually wins the kyoku. Dora uses `0..6` and `7+` classes together with a separate point estimate. Hand value uses 59 settlement values and excludes values that are impossible for the winner's dealer status. This includes 7,700 points for a non-dealer four-han, 30-fu ron, sanbaiman, double yakuman, and stacked yakuman.
- Kyoku outcomes cover draw, four tsumo results, and 28 ron results including double and triple ron. Win and deal-in probabilities for all four players are derived from these 33 outcomes.
- Final placement is learned as a joint distribution over all 24 placement permutations.
- Policy labels use Mortal v4's 46 internal actions and legal-action masks. The engine composes conditional internal actions into complete legal protocol candidates before returning them.

## Dataset split

The training split contains 8,944 games, selected as a stable filename-hash `1/20` sample of the 2025 logs. Validation uses the 246 games held out from the 2026 logs by the same MD5 bucket rule as Mortal's dataset preparation script.

Each public event uses one deterministic perspective to train the analysis outputs. When an action can be supervised, the acting player or explicit-pass perspective is also used to train the policy. Observations are compressed using two bitmaps and sparse `float16` values. Repeated dora indicators are counted repeatedly, while honba and deposits are excluded from hand-value labels.

## Running the test training job

Create the Python 3.11 training environment, then run `scripts/run-training-test.ps1`. Supply the 2025 log directory, the 2026 archive, and Mortal's Python module directory. The script reconverts both datasets, makes one pass over the globally shuffled training data, and exports inference weights.
