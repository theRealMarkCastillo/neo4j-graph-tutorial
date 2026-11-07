# Structural matching: entities can veto a wording match

Status: approved direction (2026-11-06), follow-up to
[structure over prose](2026-11-06-structure-over-prose-design.md) (D22). Plan in
[`../plans/2026-11-06-structural-matching.md`](../plans/2026-11-06-structural-matching.md).

## Problem

After the structure-over-prose work, three write-path decisions still read identity from
sentence wording or from the `kind` label:

1. **Duplicate detection** (`store._apply`). A new memory whose embedding is within
   `DUPLICATE_SCORE` of a current one (same dates) only reinforces that one and is never saved.
   The check also runs against memories stored earlier in the same extraction. Calibration's
   own NO pairs ("Maya lives in Boston" / "...in Seattle", "went running" / "went swimming")
   depend on the cut alone. If a new extraction model words things more alike, a different
   fact is merged away and lost, and no error is raised.
2. **The forget tombstone** (`store.forget`, `store._tombstones`). Within an episode, a new
   memory whose embedding is within `FORGOTTEN_SCORE` of a forgotten one is dropped. Its NO
   pairs ("Jordan moved to Denver" / "Alex is moving to Denver") are other true things said in
   the same conversation. The same drift drops them.
3. **The trip guard** (`store._apply`). A one-at-a-time relation (`lives_in`) is ignored when
   only a `kind == "status"` memory mentions its target. If the model files "I'm in Denver
   this week" as an `event` with an expiry, the guard misses it and the trip ends the real
   home. Pruning had exactly this flaw and was fixed by reading `valid_until` (D4 of the
   earlier spec).

## Principle

Merging two different things loses data. Splitting one thing into two only leaves a
near-copy. So structure gets a **veto**: wording similarity may decide a match only when the
graph agrees it could be the same thing. Where the graph can't tell, behaviour stays as it is
today.

## Design

### S1. The entity veto
An old memory (stored, stored earlier in this extraction, or a tombstone) can match a new
one only if **every entity the old one is ABOUT, other than the user, is also in the new
one's ABOUT**. The user node never counts.

- "Maya lives in Boston" {Maya, Boston} cannot match "Maya lives in Seattle" {Maya, Seattle}.
- "went running" {running} cannot match "went swimming" {swimming}. Activities are entities
  after D2 of the earlier spec.
- "Logan loves Minecraft" {Logan} still matches the re-read {Logan, Minecraft}, so a
  re-extraction still backfills entities onto the old memory (an existing test).
- Two memories about only the user ("likes coffee" / "likes tea") have nothing to veto with.
  Wording still decides there, as today. That is a stated limit, not a regression.
- If the extractor drops an entity it named before, the new memory is stored as a near-copy
  instead of reinforcing. This is accepted because it's the safe direction.

`kind` is **not** part of the veto. A label slip should not change identity (D22).

### S2. Tombstones carry entities
`forget()` records the forgotten memory's non-user ABOUT entity ids on the `Forgotten` node
(`about`). `_tombstones()` returns each vector with its id set, and S1 applies. A tombstone
with no `about` (made before this change) has an empty set, so it vetoes nothing, which is
today's behaviour.

### S3. "Temporary" for the trip guard is an expiry
The guard's temporary names come from memories with an expiry: `clock.parse_until(...)` is
not None. Lasting names come from memories without one. A status still gets its default
expiry in `clock.parse_until`. That is now the only place `kind` decides "temporary", and
it's the right place, since a status is short by definition.

## Out of scope (recorded)
- **Entity matching** (`resolve_entity`). It is already structural first (the extractor's
  `ref` id, then an exact name or alias). The vector fallback needs both `entity_match` and
  `name_match` to agree, and the one-person-role rule (one mother) merges by graph structure.
  The next step would be relation-aware matching. It is bigger and is not needed for the
  bugs above.
- **The sensitive-recall floor.** Recall is fuzzy by nature (D22 keeps similarity there).
  Recalling sensitive memories about entities the message names is a recall feature, not an
  identity fix.
- **"step" in `r.said`.** This needs a step-relation type in the ontology, which is an
  ontology change.
- **Scenario checks that match text.** Already listed as a deferred minor of the earlier
  plan.
