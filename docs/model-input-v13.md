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

The persisted v13 observation contains only two engine-owned components:

1. the audited public-analysis planes;
2. `riichi-analysis-rule-context-v1`, whose named features describe exact
   self-state and legal-action facts.

No Mortal observation channel is persisted or consumed by the v13 model.
`PlayerState` remains a replaceable rules backend during conversion and runtime:
its public getters calculate rule facts and its action mask validates labels,
but its observation tensor is not part of the v13 contract.

Per-tile rule features cover waits, legal discards, shanten-preserving and
shanten-lowering discards, and kan candidates. Global rule features cover exact
shanten, furiten and riichi state, interaction phase, and legal action classes.
These features enter the shared tile and global representations, so every
prediction family can use them.

Concrete candidate identity remains policy-specific. The policy decoder scores
the fixed action vocabulary and the authoritative action mask removes illegal
candidates. Conditional kan selection updates the rule context without
recomputing public history.

## Deliberate exclusions

- The model does not predict deterministic rule facts as auxiliary targets.
- Runtime post-processing does not manufacture a high win probability; the
  model receives the fact and learns how it changes future outcomes.
- Mortal's opaque single-player value tables are not copied into v13.
- General non-tenpai ukeire is not added through a second Python shanten
  implementation. It should be added only when the rules backend exposes one
  audited, efficient calculation shared by conversion and runtime.

Training policy, decisive-state sampling, and loss weighting are intentionally
outside this input-contract change and must be selected before v13 training.
