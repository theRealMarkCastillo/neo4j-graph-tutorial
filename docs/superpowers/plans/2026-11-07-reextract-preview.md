# Read-only Re-extraction Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

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

- [ ] **Step 1: Write failing unit and database tests.**
  `test_read_view_skips_initialization` patches ensure_schema/ensure_ontology/embed to raise if called during construction; entering/exiting the view must not call them.
  `test_read_view_rejects_writes` attempts CREATE, SET, and a write procedure through `_q`, asserting an explicit read-only error and unchanged nodes/properties.
  `test_read_view_unknown_user_is_not_created` asserts `episodes(view) == []` and no User node before/after.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k read_view`; verify failures are missing behavior, not infrastructure skips.
- [ ] **Step 3: Implement the view.** Skip constructor initialization in read-only mode. Bind a read transaction to `_local.tx`. Before executing each query, inspect its `EXPLAIN` result and require query type `r`; reject all other query types and unavailable classification. Block `_write` in read-only mode. Read routing supplements this guard, rather than replacing it. Close/rollback the read transaction on every exit path.
- [ ] **Step 4: Run the new tests and existing production transaction tests:** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_14_production.py`.
- [ ] **Step 5: Commit** `Add guarded read-only memory store views`.

### Task 2: Share entity and memory decisions with the write path

**Files:** Modify `companion/store.py`; test `tests/test_reextract_preview.py` and existing write-path tests.

**Interfaces:** Extract `MemoryStore._find_entity(name, kind, aliases, summary, ref, new) -> tuple[str | None, np.ndarray]` returning a matched id and the incoming vector; `resolve_entity` calls it and retains mutation/reconciliation behavior. Extract `_entity_changes(current: dict, incoming: EntityOut) -> dict` for computed final name/kind/summary/aliases plus role-merge proposals. Extract `_memory_decision(mem: MemoryOut, vec, about_ids: set[str], episode_id: str | None, now: dt.datetime, stored: list[tuple[np.ndarray, set[str]]], tombstones: list[tuple[np.ndarray, set[str]]]) -> dict` returning `action` (`new`, `reinforce`, `skip`), `reason`, optional `id` and `score`; `_apply` consumes it before its existing writes.

- [ ] **Step 1: Write failing parity tests.**
  `test_entity_match_preview_agrees_with_apply` covers ref, explicit new, alias, compatible kinds, and two same-name people.
  `test_entity_preview_reports_reconciliation` asserts placeholder rename, corrected name, kind upgrade, summary/alias changes, and role-merge proposals are reported.
  `test_memory_decision_agrees_with_apply` covers same/different dates, dropped ABOUT entity, stronger/extended reinforcement metadata, private visibility, expiry, episode tombstones, and same-extraction repeats. Patch similarity cuts where needed, following existing deterministic tests.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k 'entity or memory_decision'`; verify missing-helper failures.
- [ ] **Step 3: Extract helpers mechanically.** Keep entity selection ordering, raw-cosine versus index-score conventions, and query filtering before LIMIT unchanged. Keep empty candidate scores at negative infinity. Reconciliation computation may propose merges but never execute them. Preserve existing write logs and mutation order.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_09_write_path.py tests/test_13_time.py tests/test_14_production.py tests/test_18_inspect.py tests/test_19_embeddings.py`.
- [ ] **Step 5: Commit** `Share matching decisions between previews and writes`.

### Task 3: Classify and format detailed proposals

**Files:** Create `companion/preview.py`; modify `companion/store.py` only where shared relation guards are needed; test `tests/test_reextract_preview.py`.

**Interfaces:** Add `preview_extraction(store: MemoryStore, extraction: Extraction, episode_id: str, today: dt.date) -> list[dict]` and `format_preview(items: list[dict]) -> list[str]`. Each item has `category`, `action`, `text`, `reason`, optional `id`/`score`, and `details` (original candidate fields and proposed changes). Actions are `existing`, `new`, `update`, `reinforce`, `skip`, `reject`, `end`, or `unresolved`. Return ordinary JSON-compatible candidate details; never run apply.

- [ ] **Step 1: Write failing deterministic tests.**
  `test_preview_includes_text_and_dates` asserts every memory text/kind/ABOUT and supplied date field appears.
  `test_preview_relations_use_ontology_and_trip_guard` covers synonyms/inverses, invalid group/range, existing links, endings, cardinality replacement, temporary-event guard, and unknown types proposed without writing RelType nodes.
  `test_preview_reports_memory_side_effects` covers reinforcement changes, supersedes/invalidate, completed/restated plans, and forgetting proposals.
  `test_preview_unresolved_and_proposed_entities` asserts synthetic proposed ids never leak as persisted ids; unresolved/merge-affected classifications are labeled rather than guessed.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py -k preview`; verify failures reflect the missing module/functions.
- [ ] **Step 3: Implement classification and formatting.** Resolve extraction-local references in memory, call shared matching helpers, and track earlier proposed memories for repeat checks. Extract a query-free relation eligibility helper from `_apply` for shared normalization/range/temporary decisions. Read current edges to identify existing/add/end/cardinality effects. Report uncertain effects from unperformed role merges, planned retirements, or unresolved references explicitly rather than claiming full transaction simulation. Expose all supplied retirements/completions/forgetting as proposals; mark unknown ids and restated-plan guards. Format stable indented text with clear reasons and existing ids.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py tests/test_09_write_path.py tests/test_12_ontology.py tests/test_13_time.py`.
- [ ] **Step 5: Commit** `Render detailed current-graph extraction proposals`.

### Task 4: Wire dry runs, verify state preservation, document

**Files:** Modify `companion/reextract.py`, `README.md`, `lessons/20-growing-the-life-graph.md`; test `tests/test_reextract_preview.py`.

**Interfaces:** Keep `reextract(driver, user: str, dry_run: bool = False, last: int | None = None) -> list[str]` and the existing CLI. Dry runs enumerate episodes and call `episode_pass(..., apply=False, refresh=True)` inside guarded read views; normal mode retains apply behavior.

- [ ] **Step 1: Add failing integration tests.**
  `test_dry_run_preserves_full_database_state` uses fake extraction and snapshots node/relationship properties and schema before/after, including user vectors and ontology; assert equality.
  `test_dry_run_uses_episode_date_and_persona` asserts each model input uses its original day/persona and hidden memories do not become visible.
  `test_dry_run_empty_user_skips_model` asserts explicit no-episodes text and no model invocation or created user.
  `test_dry_run_missing_index_is_actionable` asserts missing-index errors explain normal setup without issuing schema writes.
  `test_apply_mode_still_applies` asserts normal mode calls the existing write path and stores the fixture candidate.
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_reextract_preview.py`; verify the new behavior tests fail.
- [ ] **Step 3: Wire CLI and docs.** Print episode id/date/persona and detailed items, a `current-graph preview` header, and the explicit no-cross-episode-simulation/no-guarantee caveat. Explain that normal setup is required for nonempty databases. Correct the old unconditional idempotence claim: matching limits, changing graph state, and fresh model calls can alter an apply result.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q`; require all non-LLM tests pass with database tests actually executed. Run `git diff --check`. Review the complete diff against the approved spec.
- [ ] **Step 5: Run the already-approved real dry runs** for Mark and guest, save private previews outside the repository, and compare full user node/relationship property snapshots before/after. Review the candidate text and existing/new classifications without applying anything. Do not claim coverage of all live scenarios; none need to be rerun for deterministic preview formatting.
- [ ] **Step 6: Commit** `Wire read-only re-extraction previews and document limits`.

## Self-review

All design requirements map to Tasks 1–4. Shared interfaces above are the only new cross-module contracts. The five Review Focus cases each have owning tests. The implementation deliberately reports uncertainty where current-graph inspection cannot reproduce sequential write effects; it must not silently substitute a rollback simulation or change matching thresholds.
