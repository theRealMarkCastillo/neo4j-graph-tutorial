# Structural Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wording similarity can no longer merge or drop memories about different entities, and the trip guard reads an expiry instead of `kind`.

**Architecture:** One rule, the *entity veto*, is added in front of the three wording matches in `Store._apply`: the stored duplicate, the repeat within one extraction, and the forget tombstone. An old memory may match only if all its non-user ABOUT entities are in the new memory's ABOUT. Tombstones gain an `about` list. The trip guard sorts names by `clock.parse_until(...)`, not by `kind`.

**Tech Stack:** Python 3, Neo4j 5.26 (Cypher pattern comprehensions), pytest. Use `uv run`.

**Spec:** [`docs/superpowers/specs/2026-11-06-structural-matching-design.md`](../specs/2026-11-06-structural-matching-design.md)

**Base:** branch off `structure-over-prose` (it needs activity entities and D22) after that branch merges, or on top of it if it hasn't merged yet.

## Global Constraints

- The user node never counts toward the veto (`NOT x:User` in Cypher; drop `self.user` in Python).
- `kind` is not part of the veto (spec S1).
- A tombstone with no `about` vetoes nothing (spec S2): old data behaves as today.
- No threshold values change. Tests that need a wording match set `store_module.DUPLICATE_SCORE` / `FORGOTTEN_SCORE` with `monkeypatch`, so they pass the same way on every embedding model.
- Commit messages are short and technical, with no attribution trailers.

## Review Focus

1. **The extractor dropped an entity on a re-read** (stored {Logan, Minecraft}, new {Logan}). Expected: stored as a near-copy, not reinforced. That's the safe direction. Pinned in Task 1.
2. **User-only memories** ("likes coffee" / "likes tea"). Expected: unchanged, wording decides. Pinned in Task 1 by the existing `test_duplicate_memory_is_reinforced_not_copied`.
3. **A veto hides the nearest memory but a farther, allowed one exists.** Expected: the allowed one still reinforces. The veto must be in the query's WHERE, before `ORDER BY ... LIMIT 1`. Pinned in Task 1.
4. **Old tombstones without `about`.** Expected: they still suppress. Pinned in Task 2 by the existing `test_forgotten_memory_is_not_re_extracted_in_the_same_episode`.
5. **A status trip with no explicit `valid_until`.** Expected: still a trip (the status gets its default expiry). Pinned in Task 3 by the existing `test_a_trip_does_not_end_the_real_home`.

---

### Task 1: The entity veto for duplicates and repeats

**Files:**
- Modify: `companion/store.py` (`_apply`, the memory loop at about lines 645–760)
- Test: `tests/test_09_write_path.py`

**Interfaces:**
- Produces: in `_apply`, `about_ids: set[str]` per memory: the non-user entity ids from `mem.about`, resolved with `self._lookup(n, known)` *before* the duplicate checks. `stored` becomes `list[tuple[np.ndarray, set[str]]]`. Task 2 reuses `about_ids`.

- [ ] **Step 1: Write the failing tests** (in `tests/test_09_write_path.py`)

```python
from companion import store as store_module

MAYA_BOSTON = dict(entities=[EntityOut(name="Maya", kind="person"), EntityOut(name="Boston", kind="place")])

def test_a_wording_match_about_other_entities_is_new(store, monkeypatch):
    monkeypatch.setattr(store_module, "DUPLICATE_SCORE", 0.0)  # wording says "same" for anything
    store.apply(Extraction(**MAYA_BOSTON, memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya", "Boston"])]), today=T)
    log = store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person"), EntityOut(name="Seattle", kind="place")],
                                 memories=[mem("Pytest's sister Maya lives in Seattle", about=["Maya", "Seattle"])]), today=T)
    assert len(store.memories()) == 2, log

def test_a_wording_match_about_the_same_entities_is_reinforced(store, monkeypatch):
    monkeypatch.setattr(store_module, "DUPLICATE_SCORE", 0.0)
    store.apply(Extraction(**MAYA_BOSTON, memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya", "Boston"])]), today=T)
    log = store.apply(Extraction(**MAYA_BOSTON, memories=[mem("Maya, Pytest's sister, lives in Boston", about=["Maya", "Boston"])]), today=T)
    assert any(line.startswith("reinforced") for line in log), log

def test_a_vetoed_nearest_does_not_hide_an_allowed_match(store, monkeypatch):
    monkeypatch.setattr(store_module, "DUPLICATE_SCORE", 0.0)
    store.apply(Extraction(**MAYA_BOSTON, memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya", "Boston"])]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                           memories=[mem("Pytest's sister Maya is a nurse", about=["Maya"])]), today=T)
    # nearest by wording is the Boston memory (vetoed: Boston isn't named); the nurse one is allowed
    log = store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                                 memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya"])]), today=T)
    [hit] = [line for line in log if line.startswith("reinforced")]
    assert "nurse" in hit, log

def test_dropping_an_entity_on_a_re_read_stores_a_copy(store, monkeypatch):
    monkeypatch.setattr(store_module, "DUPLICATE_SCORE", 0.0)
    store.apply(Extraction(**MAYA_BOSTON, memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya", "Boston"])]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Maya", kind="person")],
                           memories=[mem("Pytest's sister Maya lives in Boston", about=["Maya"])]), today=T)
    assert len(store.memories()) == 2

def test_repeats_about_other_entities_in_one_extraction_are_both_kept(store, monkeypatch):
    monkeypatch.setattr(store_module, "DUPLICATE_SCORE", 0.0)
    store.apply(Extraction(entities=[EntityOut(name="running", kind="activity"), EntityOut(name="swimming", kind="activity")],
                           memories=[mem("Pytest went running", "event", about=["running"]),
                                     mem("Pytest went swimming", "event", about=["swimming"])]), today=T)
    assert len(store.memories()) == 2
```

Note: `test_a_vetoed_nearest...` is a positive-control test, so it reinforces even with the cut at 0. With the veto, the Boston memory is excluded and the nurse memory (about {Maya} ⊆ {Maya}) is the only candidate.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/test_09_write_path.py -k "wording_match or vetoed_nearest or dropping_an_entity or repeats_about"`
Expected: `test_a_wording_match_about_other_entities_is_new`, `test_dropping_an_entity...` and `test_repeats_about_other_entities...` FAIL (one memory, reinforced or skipped). `test_..._same_entities_is_reinforced` PASSES (a control). `test_a_vetoed_nearest...` FAILS (it reinforces the Boston memory).

- [ ] **Step 3: Implement the veto in `companion/store.py` `_apply`**

- Before the tombstone check, compute `about_ids = {eid for n in mem.about if (eid := self._lookup(n, known)) and eid != self.user}`.
- Repeat check: `stored` holds `(vec, about_ids)`. A stored entry counts only if `its_ids <= about_ids`.
- Index query: pass `about=sorted(about_ids)` and add to the WHERE, before `RETURN ... ORDER BY ... LIMIT 1`:
  `AND all(x IN [(node)-[:ABOUT]->(e) WHERE NOT e:User | e.id] WHERE x IN $about)`
- The "new" log line: `[new: nothing similar with the same dates and entities]` when `nearest` is empty.
- The later `about` list for CREATE is unchanged (it still defaults to `["user"]`).

- [ ] **Step 4: Run the tests to verify they pass, plus the existing write-path tests**

Run: `uv run pytest -q tests/test_09_write_path.py tests/test_14_production.py tests/test_18_inspect.py`
Expected: all pass, including `test_a_repeated_memory_gains_the_entities_it_names` and `test_duplicate_memory_is_reinforced_not_copied`.

- [ ] **Step 5: Commit**

```bash
git add companion/store.py tests/test_09_write_path.py
git commit -m "Veto a duplicate match when the old memory is about other entities"
```

### Task 2: Tombstones carry entities

**Files:**
- Modify: `companion/store.py` (`forget`, `_tombstones`, the tombstone check in `_apply`)
- Test: `tests/test_09_write_path.py`

**Interfaces:**
- Consumes: `about_ids` from Task 1.
- Produces: `Forgotten.about: list[str]`. `_tombstones(episode_id) -> list[tuple[np.ndarray, set[str]]]`.

- [ ] **Step 1: Write the failing test**

```python
def test_a_tombstone_only_blocks_memories_about_its_entities(store, monkeypatch):
    monkeypatch.setattr(store_module, "FORGOTTEN_SCORE", 0.0)  # wording says "the forgotten one" for anything
    ep = store.start_episode(on=T)
    store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person"), EntityOut(name="Denver", kind="place")],
                           memories=[mem("Pytest's ex Jordan moved to Denver", about=["Jordan", "Denver"])]), ep, T)
    [m] = store.memories()
    assert store.forget(m["id"])
    log = store.apply(Extraction(entities=[EntityOut(name="Denver", kind="place")],
                                 memories=[mem("Pytest is moving to Denver next year", about=["Denver"])]), ep, T)
    assert [x["text"] for x in store.memories()] == ["Pytest is moving to Denver next year"], log
    log = store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person"), EntityOut(name="Denver", kind="place")],
                                 memories=[mem("Jordan, Pytest's ex, moved to Denver", about=["Jordan", "Denver"])]), ep, T)
    assert len(store.memories()) == 1, log
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/test_09_write_path.py -k tombstone_only`
Expected: FAIL at the first assertion (the Denver move was skipped as "a memory the user asked to forget").

- [ ] **Step 3: Implement**

- `forget()`: in the CREATE query, set `t.about = [(m)-[:ABOUT]->(e) WHERE NOT e:User | e.id]` before the DETACH DELETE.
- `_tombstones()`: return `(np.asarray(v), set(coalesce(about, [])))` pairs.
- `_apply`: count a tombstone only if `its_ids <= about_ids`.

- [ ] **Step 4: Run the tombstone tests**

Run: `uv run pytest -q tests/test_09_write_path.py tests/test_14_production.py tests/test_19_embeddings.py`
Expected: all pass, including `test_forgotten_memory_is_not_re_extracted_in_the_same_episode` (a user-only tombstone still blocks) and `test_a_tombstone_from_another_model_is_ignored`.

- [ ] **Step 5: Commit**

```bash
git add companion/store.py tests/test_09_write_path.py
git commit -m "Tombstones record their entities and only block memories about them"
```

### Task 3: The trip guard reads an expiry

**Files:**
- Modify: `companion/store.py` (the guard at about line 518 in `_apply`)
- Test: `tests/test_13_time.py`

- [ ] **Step 1: Write the failing test** (next to `test_a_trip_does_not_end_the_real_home`)

```python
def test_a_trip_filed_as_an_event_does_not_end_the_real_home(store):
    from companion.models import EntityOut, RelationOut
    store.apply(Extraction(entities=[EntityOut(name="Chicago", kind="place")],
                           relations=[RelationOut(source="user", target="Chicago", type="lives_in")]), today=D1)
    log = store.apply(Extraction(
        entities=[EntityOut(name="Denver", kind="place")],
        relations=[RelationOut(source="user", target="Denver", type="lives_in")],
        memories=[mem("Pytest is in Denver for work this week", "event", 3, about=["user", "Denver"],
                      valid_until="2026-11-22T23:59")]), today=D1)
    assert any(line.startswith("ignored") for line in log), log
    homes = store._q("MATCH (:User {id:$user})-[r:RELATED {type:'lives_in'}]->(e) WHERE r.until IS NULL RETURN e.name AS n")
    assert homes == [{"n": "Chicago"}]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/test_13_time.py -k "trip"`
Expected: the new test FAILS (Denver becomes home). `test_a_trip_does_not_end_the_real_home` passes.

- [ ] **Step 3: Implement**

Use the `now` already set at the top of `_apply`. `temporary` = names from memories where `clock.parse_until(m.valid_until, now, m.kind)` is not None. `lasting` = names from all other memories. Update the comment and log line to say "only a temporary memory mentions it".

- [ ] **Step 4: Run the time tests**

Run: `uv run pytest -q tests/test_13_time.py tests/test_09_write_path.py`
Expected: all pass. `test_a_trip_does_not_end_the_real_home` is known to be intermittent on the base (see the earlier ledger). If it fails, re-run it on its own before treating it as a regression.

- [ ] **Step 5: Commit**

```bash
git add companion/store.py tests/test_13_time.py
git commit -m "Trip guard: a temporary memory is one with an expiry, not a status"
```

### Task 4: Scenario checks test structure

**Files:**
- Modify: `companion/scenarios.py` (`Run`, `run_scenario`, the `personas` and `situational` checks)
- Test: `tests/test_19_embeddings.py` (or wherever scenario helpers are unit-tested; grep `run_scenario`)

- [ ] **Step 1: Write the failing unit test**: `Run` has `logs: list[str]`, and `run_scenario` appends every `memory_log` line and `close_episode` line to it. Test it with a fake app if one exists (grep `fake_llm` in `tests/conftest.py`). Otherwise assert on the `Run` dataclass field and on a small helper, `stored(run, activity) -> bool`, that finds a `+ [` log line with `(until` for a memory ABOUT that activity. Give the helper the memory ids from the log, then look them up with `include_retired`, or match the activity in the log if the ids are gone.
- [ ] **Step 2: Run it and see it fail.**
- [ ] **Step 3: Implement:** add `Run.logs`. Change "the gym was stored as a status that expires" to read from what was stored (logs), not from end state. Add "Chicago" to the race words in "Coach's replies use the race". Change "three gym visits became one lasting habit" to look for a current pattern (`evidence >= 3`) ABOUT the `gym` activity, not text that contains "gym".
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit** `git commit -m "Scenario checks read structure and write logs, not wording"`

### Task 5: Docs, and the full suite

**Files:**
- Modify: `lessons/09-memory-write-path.md` (the section 3 table row "New memory ≥ 0.975 similar"; the found-bugs list near the tombstone line): add the entity veto and why (merging loses data, a split only copies).
- Modify: `lessons/13-memory-over-time.md` (the "A trip almost moved the user" bullet): the guard reads an expiry, with the event case.
- Modify: `docs/design-decisions.md`: in D22, add the veto as the second application. Change the gap row **Structural dedupe** (line ~580) to say what's done (entity veto) and what is still open (user-only memories, where wording decides).
- Modify: `companion/calibrate.py` docstring: the `duplicate` and `forgotten` NO pairs that differ in entities are now vetoed by structure. Their cut matters for user-only memories.
- Modify: `docs/glossary.md`: add an "entity veto" row.

- [ ] **Step 1: Make the doc edits above.**
- [ ] **Step 2: Run the full fast suite**

Run: `uv run pytest -q`
Expected: all pass (the LLM tests are deselected by default).

- [ ] **Step 3: Run the LLM scenarios** (they cost money and vary run to run)

Run: `uv run pytest -q -m llm tests/test_11_scenarios.py tests/test_11_extraction_evals.py`
Expected: no new failures compared with the last run on the base branch.

- [ ] **Step 4: Commit**

```bash
git add lessons/09-memory-write-path.md lessons/13-memory-over-time.md docs/design-decisions.md docs/glossary.md companion/calibrate.py
git commit -m "Document the entity veto and the expiry-based trip guard"
```

## Implementation record (2026-11-06)

Implemented Tasks 1–5 on `codex/structural-matching`. Execution differences and validation:

- The `superpowers:executing-plans` skill and Opus model were unavailable. Followed this
  plan directly; an independent available-model review found no blocking correctness issues.
- Added seven database regression cases, observed them fail on the original store, then
  pass with the fixes. The plan's zero-threshold tests also exposed the original empty-list
  `max(..., default=0)` behaviour; empty candidate sets now score negative infinity.
- Added six unit cases for scenario log retention and expiry detection. Updated two trace
  assertions for the new explanation mentioning dates and entities.
- Full non-LLM suite: **246 passed, 49 deselected** against local Neo4j. Final focused rerun:
  **94 passed, 6 deselected**.
- Live scenario + extraction suite: **33 passed, 1 failed** (21m21s). All 14 scenarios passed.
  `test_forget_wins_over_boundary` returned both `forget_entities` and a boundary. The
  extractor, its models, and that test are unchanged from the base; this failure is in an unchanged extraction path, so it appears unrelated to this change.
  The live suite is not a passing release check. A three-trial focused rerun is recorded below.
- No re-extraction of real users, merge, or push. Only isolated test/eval users were used.
- Focused rerun with `EVAL_TRIALS=3`: **1 test passed** (all three trials, 3m32s).
  This supports an intermittent model-output failure; it does not erase the original result.
