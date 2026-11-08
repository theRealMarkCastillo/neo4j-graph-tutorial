# Read-only Re-extraction Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce detailed current-graph previews using shared matching rules without writing to Neo4j.

**Architecture:** Give MemoryStore a guarded read-only view, extract read-only decisions from its existing write path, then consume those decisions in a separate preview module. Wire re-extraction dry runs to that module; preserve apply mode.

**Tech Stack:** Existing Python, Neo4j driver/Cypher, Pydantic, pytest; no dependencies added.

**Spec:** `docs/superpowers/specs/2026-11-07-reextract-preview-design.md` (approved 2026-11-07).

## Global Constraints

- No dry-run schema, ontology, user, vector-property, or memory writes, including temporary writes rolled back later.
- Share existing matching decisions; do not change thresholds, ontology vocabulary, or normal chat initialization.
- Preserve tenant/persona visibility, entity veto, dates, expiry, tombstones, and repeat checks.
- Print candidate text, kinds, ABOUT references, populated dates, match ids/scores, and reasons.
- Explicitly label current-graph limitations and unresolved references. Do not simulate cross-episode writes.
- No live-user apply, merge, or push is authorized for this feature. Mark/guest dry runs are already approved.
- Do not print `.env` values or commit real conversations/previews.

## Review Focus

1. Unknown users must not be created; empty users must not trigger model calls (Task 1/4).
2. READ routing alone is not a security boundary; accidental query and procedure writes must fail before execution (Task 1).
3. Same-name people, proposed entities, role merges, and name/kind updates must not be presented as unchanged matches (Task 2/3).
4. Reinforcement may modify metadata or retire other memories; show those proposals, not just an existing label (Task 3).
5. Episode dates, persona privacy, and unresolved references must not create false certainty about later apply behavior (Task 3/4).

---

### Task 1: Guarded read-only store views

**Files:** Modify `companion/store.py`; create `tests/test_reextract_preview.py`.

**Interfaces:** Add keyword-only `read_only: bool = False` to MemoryStore construction. Add `MemoryStore.read_view(driver, user_name: str, agent: str | None = None)` as a context manager yielding a read-only MemoryStore bound to one transaction. Normal callers keep their existing constructor behavior.

- [x] **Step 1: Write failing unit and database tests.**
  `test_read_view_skips_initialization` patches ensure_schema/ensure_ontology/embed to raise if called during construction; entering/exiting the view must not call them.
  `test_read_view_rejects_writes` attempts CREATE, SET, and a write procedure through `_q`, asserting an explicit read-only error and unchanged nodes/properties.
  `test_read_view_unknown_user_is_not_created` asserts `episodes(view) == []` and no User node before/after.
- [x] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k read_view`; verify failures are missing behavior, not infrastructure skips.
- [x] **Step 3: Implement the view.** Skip constructor initialization in read-only mode. Bind a read transaction to `_local.tx`. Before executing each query, inspect its `EXPLAIN` result and require query type `r`; reject all other query types and unavailable classification. Block `_write` in read-only mode. Read routing supplements this guard, rather than replacing it. Close/rollback the read transaction on every exit path.
- [x] **Step 4: Run the new tests and existing production transaction tests:** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_14_production.py`.
- [x] **Step 5: Commit** `Add guarded read-only memory store views`.

### Task 2: Share entity and memory decisions with the write path

**Files:** Modify `companion/store.py`; test `tests/test_reextract_preview.py` and existing write-path tests.

**Interfaces:** Extract `MemoryStore._find_entity(name, kind, aliases, summary, ref, new) -> tuple[str | None, np.ndarray]` returning a matched id and the incoming vector; `resolve_entity` calls it and retains mutation/reconciliation behavior. Extract `_entity_changes(current: dict, incoming: EntityOut) -> dict` for computed final name/kind/summary/aliases plus role-merge proposals. Extract `_memory_decision(mem: MemoryOut, vec, about_ids: set[str], episode_id: str | None, now: dt.datetime, stored: list[tuple[np.ndarray, set[str]]], tombstones: list[tuple[np.ndarray, set[str]]]) -> dict` returning `action` (`new`, `reinforce`, `skip`), `reason`, optional `id` and `score`; `_apply` consumes it before its existing writes.

- [x] **Step 1: Write failing parity tests.**
  `test_entity_match_preview_agrees_with_apply` covers ref, explicit new, alias, compatible kinds, and two same-name people.
  `test_entity_preview_reports_reconciliation` asserts placeholder rename, corrected name, kind upgrade, summary/alias changes, and role-merge proposals are reported.
  `test_memory_decision_agrees_with_apply` covers same/different dates, dropped ABOUT entity, stronger/extended reinforcement metadata, private visibility, expiry, episode tombstones, and same-extraction repeats. Patch similarity cuts where needed, following existing deterministic tests.
- [x] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k 'entity or memory_decision'`; verify missing-helper failures.
- [x] **Step 3: Extract helpers mechanically.** Keep entity selection ordering, raw-cosine versus index-score conventions, and query filtering before LIMIT unchanged. Keep empty candidate scores at negative infinity. Reconciliation computation may propose merges but never execute them. Preserve existing write logs and mutation order.
- [x] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_09_write_path.py tests/test_13_time.py tests/test_14_production.py tests/test_18_inspect.py tests/test_19_embeddings.py`.
- [x] **Step 5: Commit** `Share matching decisions between previews and writes`.

### Task 3: Classify and format detailed proposals

**Files:** Create `companion/preview.py`; modify `companion/store.py` only where shared relation guards are needed; test `tests/test_reextract_preview.py`.

**Interfaces:** Add `preview_extraction(store: MemoryStore, extraction: Extraction, episode_id: str, today: dt.date) -> list[dict]` and `format_preview(items: list[dict]) -> list[str]`. Each item has `category`, `action`, `text`, `reason`, optional `id`/`score`, and `details` (original candidate fields and proposed changes). Actions are `existing`, `new`, `update`, `reinforce`, `skip`, `reject`, `end`, or `unresolved`. Return ordinary JSON-compatible candidate details; never run apply.

- [x] **Step 1: Write failing deterministic tests.**
  `test_preview_includes_text_and_dates` asserts every memory text/kind/ABOUT and supplied date field appears.
  `test_preview_relations_use_ontology_and_trip_guard` covers synonyms/inverses, invalid group/range, existing links, endings, cardinality replacement, temporary-event guard, and unknown types proposed without writing RelType nodes.
  `test_preview_reports_memory_side_effects` covers reinforcement changes, supersedes/invalidate, completed/restated plans, and forgetting proposals.
  `test_preview_unresolved_and_proposed_entities` asserts synthetic proposed ids never leak as persisted ids; unresolved/merge-affected classifications are labeled rather than guessed.
- [x] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k preview`; verify failures reflect the missing module/functions.
- [x] **Step 3: Implement classification and formatting.** Resolve extraction-local references in memory, call shared matching helpers, and track earlier proposed memories for repeat checks. Extract a query-free relation eligibility helper from `_apply` for shared normalization/range/temporary decisions. Read current edges to identify existing/add/end/cardinality effects. Report uncertain effects from unperformed role merges, planned retirements, or unresolved references explicitly rather than claiming full transaction simulation. Expose all supplied retirements/completions/forgetting as proposals; mark unknown ids and restated-plan guards. Format stable indented text with clear reasons and existing ids.
- [x] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_09_write_path.py tests/test_12_ontology.py tests/test_13_time.py`.
- [x] **Step 5: Commit** `Render detailed current-graph extraction proposals`.

### Task 4: Wire dry runs, verify state preservation, document

**Files:** Modify `companion/reextract.py`, `README.md`, `lessons/20-growing-the-life-graph.md`; test `tests/test_reextract_preview.py`.

**Interfaces:** Keep `reextract(driver, user: str, dry_run: bool = False, last: int | None = None) -> list[str]` and the existing CLI. Dry runs enumerate episodes and call `episode_pass(..., apply=False, refresh=True)` inside guarded read views; normal mode retains apply behavior.

- [x] **Step 1: Add failing integration tests.**
  `test_dry_run_preserves_full_database_state` uses fake extraction and snapshots node/relationship properties and schema before/after, including user vectors and ontology; assert equality.
  `test_dry_run_uses_episode_date_and_persona` asserts each model input uses its original day/persona and hidden memories do not become visible.
  `test_dry_run_empty_user_skips_model` asserts explicit no-episodes text and no model invocation or created user.
  `test_dry_run_missing_index_is_actionable` asserts missing-index errors explain normal setup without issuing schema writes.
  `test_apply_mode_still_applies` asserts normal mode calls the existing write path and stores the fixture candidate.
- [x] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py`; verify the new behavior tests fail.
- [x] **Step 3: Wire CLI and docs.** Print episode id/date/persona and detailed items, a `current-graph preview` header, and the explicit no-cross-episode-simulation/no-guarantee caveat. Explain that normal setup is required for nonempty databases. Correct the old unconditional idempotence claim: matching limits, changing graph state, and fresh model calls can alter an apply result.
- [x] **Step 4: Run** `.venv/bin/python -m pytest -q`; require all non-LLM tests pass with database tests actually executed. Run `git diff --check`. Review the complete diff against the approved spec.
- [x] **Step 5: Run the already-approved real dry runs** for Mark and guest, save private previews outside the repository, and compare full user node/relationship property snapshots before/after. Review the candidate text and existing/new classifications without applying anything. Do not claim coverage of all live scenarios; none need to be rerun for deterministic preview formatting.
- [x] **Step 6: Commit** `Wire read-only re-extraction previews and document limits`.

## Self-review

All design requirements map to Tasks 1–4. Shared interfaces above are the only new cross-module contracts. The five Review Focus cases each have owning tests. The implementation deliberately reports uncertainty where current-graph inspection cannot reproduce sequential write effects; it must not silently substitute a rollback simulation or change matching thresholds.


## Execution record

Implemented locally on `codex/reextract-preview` (core commit `7bbdd73`). All four
completion checks passed. Final verification after review fixes: **314 passed,
49 LLM tests deselected**, with Neo4j tests executed and no database skips.

Independent review found three Important relation-preview discrepancies: overly
broad exclusivity endings, incoming rather than effective privacy, and missing
uncertainty for merge-dependent relations. All were fixed in one TDD pass.
Additional regressions cover persona-hidden explicit-id output, downstream-effect
disclosure, and JSON-compatible temporal metadata. The preview test file has
**53 passing cases**. No deferred minor findings remain.

Both approved live dry runs completed after the fixes. Full database node,
relationship, index and constraint snapshot hashes were unchanged. Private preview
output was kept outside the repository. No real-user apply, merge or push occurred.

### Rulings made during execution

- Ruling: Work in the existing codex/reextract-preview checkout — the user explicitly chose "Here" — isolation is branch-based; no other checkout will be touched.
- Ruling: Fix incomplete MemoryOut/EntityOut type imports as helpers are extracted; keep established query semantics and log strings — mechanical refactor — parity tests guard accidental drift.
- Ruling: Continue Task 2's pure tests/refactor while Task 1 database verification is blocked — keeps useful work moving without claiming completion — DB regressions could still require changes before either task is committed.
- Ruling: _entity_changes computes fields; role-merge lookup is the separate shared _role_merges helper because it requires a database — avoids querying from pure reconciliation tests — preview must call both.
- Ruling: Keep later memory classifications unresolved after pending merges/retirements rather than simulate writes — the spec explicitly permits current-graph uncertainty — this may be conservative but cannot promise a wrong exact apply.
- Ruling: Tasks 1–3 changes will be committed together after their database checks because infrastructure initially forced interleaved development — preserves verified coherent shared helpers — separate task history is less granular; ledger retains each task's evidence.
- Ruling: Commit Tasks 1–4 as one coherent implementation because their shared regression file exercises the CLI as well as helpers — avoids any committed revision with tests referring to uncommitted interfaces — less granular commit history; task test commands remain separate.
- Final: Ruling: exact multi-episode outcomes remain excluded — current-graph header explains earlier proposals are not applied and later model calls may differ — costs conservative rather than executable predictions.
- Final: Ruling: no provenance cascade simulation; show an explicit downstream relation/provenance warning on effective retirements/forgetting — full simulation violates bounded scope — downstream edge changes need a later apply review.
- Final: Ruling: preserve existing apply semantics for explicit retirement IDs and invalid-range role merges, but hide non-visible explicit-id metadata in the new preview — no unrelated write-path behavior changes — hidden-id effects are unresolved instead of exact.
- Final: Ruling: live extraction quality/state preservation verified separately on approved Mark/guest dry runs, not treated as a general model benchmark — no new model-quality claims — another extraction can differ.
- Final: Ruling: configured remote embeddings exercised through approved live previews; no provider matrix or full LLM scenario suite rerun for this read-only feature — provider code/thresholds unchanged — other provider/model combinations remain unverified.
