# Read-only re-extraction previews

## Intent and scope

Make `companion.reextract --dry-run` useful for deciding whether to backfill old
conversations. The approved scope is to show candidate text, entities, relations,
and dates; distinguish existing items from proposed changes using the store's
matching rules; and avoid database writes, including startup metadata writes.

The previous Mark preview returned one memory candidate without showing its text.
Its two proposed links already existed. Guest had no episodes. Counts alone do
not establish a reason to apply a backfill.

## Approaches

1. **Read-only inspection with shared matching helpers (recommended).** Separate
   entity lookup and memory duplicate selection from their write operations. Use
   those helpers in both the store and preview. This preserves the no-write
   requirement and avoids maintaining separate similarity rules. It requires a
   bounded refactor of matching code.
2. **Apply in a transaction, then roll back.** Reuses almost all write behavior,
   but still performs temporary writes and takes locks. Rejected because the
   requested dry run should avoid writes, not merely avoid committing them.
3. **Print the extraction without classification.** Cheap and read-only after
   fixing initialization, but leaves users to identify duplicates themselves.
   Rejected because distinguishing existing items is part of the requested scope.

The recommendation follows the requested no-write behavior. Implementation is
pending approval of this design; the user has not separately chosen an approach.

## Behavior

- Keep the current command and apply mode. Dry run never calls `apply`, initializes
  schema or ontology, creates users, or updates user embeddings. Use a Neo4j read
  transaction for preview reads so accidental write queries are rejected.
- An unknown user or user without episodes yields an explicit no-episodes result
  and creates nothing. A database missing required schema/indexes reports an
  actionable error rather than initializing itself during preview.
- For each eligible episode, display its id, date, and persona, followed by every
  candidate memory's text, kind, ABOUT references, and populated date fields.
- Display entity matches with existing ids or label them proposed new entities.
  Explicit `new` and `ref`, compatible kinds, aliases, and current similarity cuts
  retain their existing meaning. Name/kind/summary reconciliation proposals must
  not be presented as unchanged existing entities.
- Normalize and validate relation types through the ontology. Show existing
  active links, proposed additions, and requested endings separately. Include
  rejected types/ranges and temporary-trip guards rather than implying they will
  become stored links.
- Memory classification reuses tenant/persona visibility, expiry, date equality,
  entity-veto, tombstone, within-extraction repeat, and similarity rules. Show the
  matching id and score for reinforcement candidates; otherwise label the memory
  proposed new or skipped, with its reason. Show requested retirements, plan
  completions, and forgetting separately when present.
- Resolve references to proposed entities in memory without creating nodes. Mark
  unresolved references and classifications affected by them explicitly.
- Label the result a **current-graph preview**, not a guarantee of a later apply.
  Preview episodes do not mutate the graph for later episodes. A real apply may
  differ because earlier episodes changed graph state, another writer intervened,
  or a new model call produced different extraction. Do not claim that a displayed
  proposal has already been stored.

## Implementation boundaries

Add a read-only store construction path and query-only matching helpers; normal
chat/store initialization keeps its existing behavior. Preview formatting belongs
with re-extraction, not in the model prompt. No thresholds or ontology vocabulary
change. No live-data apply, automatic merging, or pushing is included.

Text output is sufficient for this change. Persisted executable plans, apply-from-
preview, a web UI, and a full in-memory simulation across episodes are out of scope.

## Verification

Use deterministic extractions for tests. Cover detailed text/date rendering,
existing/new entities, same-name distinctions, relation normalization and guards,
duplicate and entity-veto cases, unresolved references, and no-episode users.
Assert no schema, ontology, user, embedding-property, or memory writes occur in
dry run. Snapshot database nodes and relationships around a real-Neo4j test with
fake extraction; verify both dry-run state preservation and unchanged apply-mode
behavior. Run affected write-path tests and the full non-LLM suite.

Once implementation passes, rerun the already-approved Mark and guest dry runs
and review the richer output. Applying any real-user changes remains a separate
approval step.
