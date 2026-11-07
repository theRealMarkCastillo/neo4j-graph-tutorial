# Structure over prose: memory identity in the graph, not in sentence wording

Status: approved direction (2026-11-06), plan in
[`../plans/2026-11-06-structure-over-prose.md`](../plans/2026-11-06-structure-over-prose.md).

## Problem

The extractor turns each message into a free-text sentence plus a `kind` label. Later
decisions then compare sentences by embedding similarity against fixed thresholds, or switch
on `kind`. Those thresholds were calibrated on hand-written sentences, and nothing records
which extraction model produced the text they were tuned for. When the extraction model
changes how it words or labels memories, decisions drift silently, and the failure shows up
stages later.

Observed on 2026-11-06, after switching extraction from `google/gemma-4-31B-it` to
`gemini-3.8-flash` (scenario `situational`, check "three gym visits became one lasting habit"):

1. **Dates in text.** The new model writes `on 2026-11-23` into memory text (rule 2 asks for
   absolute dates in text), though `occurred_on` / `due_on` hold the date. With bge-small, a
   dated rock-climbing memory scored 0.825 against a dated gym memory: higher than gym pairs
   scored against each other (0.78–0.82).
2. **No graph identity.** All five gym memories were `ABOUT` only the user. Rule 10 keeps
   generic activities out of the graph ("those stay in the memory text only"), so habit
   detection had to compare sentences.
3. **`kind` as a switch.** One visit was filed as `plan`. Plans can't join a habit, so only
   two gym memories were eligible.
4. **Calibration doesn't match real inputs.** The 0.82 pattern cut was measured on undated,
   hand-written pairs. It already misclassified one of its own pairs.

## Principle

Where a decision is about *identity* (is this the same activity, is this the same day, is
this temporary), read it from graph structure and typed fields. Use prose similarity only
where meaning really is fuzzy (recall, relevance), and record which models a threshold was
measured with so drift fails loudly.

## Design

### D1. Dates live in fields; text is undated when a field holds the day
- Extraction rule 2 changes: a day that goes into `occurred_on`, `due_on` or `valid_until`
  is **not** written into `text`. Vague anchors with no field ("in the week before
  2026-11-20", "in 2025") stay in text, as now.
- The date is shown from the field wherever memory text is shown to the chat model or the
  user: `format_context` (semantic and graph-hop memories) and the web UI.
- No data migration: the real users have 2 dated texts, both vague anchors with no field.

### D2. Activities someone does are entities
- Rule 10 changes: an activity {user} or someone in their life *does* (the gym, running,
  yoga, therapy, piano lessons) is an entity of kind `activity`, named by its plain noun
  ("gym", "running"), even when unnamed. Every memory about doing it lists it in `about`.
  One-off generic things (a movie, pizza for dinner) still stay text-only.
- Re-mentions resolve through the existing machinery: the prompt's known-entity list and
  `ref` ids, then name/entity matching in `apply()`.

### D3. A habit is a graph count, not a cluster of similar sentences
- `consolidate()` finds habits as: an `activity` entity with occurrence memories `ABOUT` it
  on **at least 3 distinct days** within `PATTERN_WINDOW_DAYS` (60).
- Occurrence memories: `kind IN ['event', 'status', 'plan', 'feeling']`, current, not
  consolidated, not sensitive, not private, and not a `lapsed` plan. ("Occurrence" vs
  "lasting" is what `kind` means, so this gate stays.)
- The day of a memory: `coalesce(occurred_on, due_on, date(valid_until), date(created_at))`.
- The writer LLM (`find_pattern`) still writes the habit sentence or declines. The pattern
  memory is `ABOUT` the activity, and `apply()` also adds `user -practices-> activity`.
- Removed: vector clustering (`_clusters`), `PATTERN_SIMILARITY`, `Profile.pattern`,
  `Profile.cluster_instruction`, `embed_for_clustering`, and the `pattern` calibration check.
- Repeated feelings with no activity ("tired" three times) no longer roll up. Accepted:
  feelings decay, and the old clustering never had a test or scenario for this.

### D4. "Temporary" is a property
- Pruning deletes memories that have **expired** (`valid_until` in the past), are dormant
  and low-importance, instead of `kind IN ['status', 'feeling']`.
- The timeline lists any memory with `occurred_on` or `due_on`, not only events and plans.
- Decay half-lives stay per `kind`: how fast something fades is part of what it is.

### D5. Thresholds record the extraction model they were measured with
- `Profile` gains `extraction_model: str`, the model whose output the thresholds and scenario
  runs were measured on. Both profiles: `google/gemma-4-31B-it` until re-measured.
- Every scenario run includes a check, "thresholds measured with this extraction model",
  that fails when `config.LLM_EXTRACTION_MODEL` differs. The fix: re-run
  `companion.calibrate` and the scenarios on the new model, then update the field.

### D6. The extractor's output structure has its own tests
- LLM evals (`tests/test_11_extraction_evals.py`) check structure, not wording: a gym visit
  is about an `activity` entity named "gym"; a known day is in `occurred_on` and not in the
  text; a one-off dinner creates no entity. Drift shows up at the extractor.

## Out of scope (recorded as gaps)
- Structural duplicate detection (same entity + kind + day) in place of the 0.966 cosine.
- Calibration pairs sampled from real extraction runs instead of hand-written ones.

## Existing data
`uv run python -m companion.reextract --user <name>` re-reads stored conversations with the
new rules. Reinforcing an existing memory already merges its new `ABOUT` links, so activity
links are backfilled without a new migration tool. This writes user data: run it only when
the user says to.
