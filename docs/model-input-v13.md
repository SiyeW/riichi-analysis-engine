# Model input v13

Model format v13 replaces the opaque Mortal policy-context slice used by v9–v12
with a named, engine-owned rule-context contract. Existing datasets and weights
retain their original schema identifiers and are never reinterpreted as v13.

## Audit result

The former policy context mixed three different kinds of information:

- duplicated public facts, including dora summaries already represented by the
  semantic tile planes;
- exact position facts, including self shanten, waits, furiten and currently
  legal action classes;
- Mortal-specific decision tables and handcrafted single-player value features.

That boundary was unsuitable for a multi-task model. Exact facts required by
kyoku and match predictions were visible only to the policy decoder, while
other heads had to infer the same rules from sparse outcome labels.

## v13 boundary

The persisted v13 observation uses `riichi-analysis-model-input-v3` and
contains only two engine-owned components:

1. the audited public-analysis planes;
2. `riichi-analysis-rule-context-v2`, whose named features describe exact
   self-state and legal-action facts.

No Mortal observation channel is persisted or consumed by the v13 model.
`PlayerState` remains a replaceable rules backend during conversion and runtime:
its public getters calculate rule facts and its action mask validates labels,
but its observation tensor is not part of the v13 contract.

Per-tile rule features distinguish structural completion waits from ordinary
ron-yaku and tsumo-yaku legality. They also include effective draws, visible
remaining copies, legal discards, each discard's resulting shanten and ukeire,
whether it creates discard furiten, and kan candidates. Global rule features
cover exact shanten, separate discard/temporary/riichi furiten, riichi state,
interaction phase, and legal action classes. These features enter the shared
tile and global representations, so every prediction family can use them.

The exact shanten and ukeire kernel uses the same audited libriichi lookup-table
method as the established reference opponent model. Conversion and runtime call the
same implementation; neither path asks the network to approximate these rules.

Concrete candidate identity remains policy-specific. The policy decoder scores
the fixed action vocabulary and the authoritative action mask removes illegal
candidates. Conditional kan selection updates the rule context without
recomputing public history.

## Deliberate exclusions

- The model does not predict deterministic rule facts as auxiliary targets.
- Runtime post-processing does not manufacture a high win probability; the
  model receives the fact and learns how it changes future outcomes.
- Mortal's opaque single-player value tables are not copied into v13.
- The rule context contains no heuristic EV, danger judgement, or preferred
  discard. It supplies facts and leaves strategic consequences to the model.

Terminal `hora` and `ryukyoku` events remain labels rather than model inputs.
For the frame immediately before a win, analysis supervision is assigned to an
actual winner instead of a random seat; a multiple-ron frame selects one winner
deterministically so every frame still contributes exactly one analysis row.
Training policy and any broader sampling or loss weighting remain separate
decisions to make before v13 training.
