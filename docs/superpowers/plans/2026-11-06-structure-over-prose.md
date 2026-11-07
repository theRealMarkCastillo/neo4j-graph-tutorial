# Structure over Prose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make habit detection, date handling and "temporary" decisions read graph structure and typed fields instead of sentence wording, so a change of extraction model can't silently break them.

**Architecture:** Extraction puts known days only in date fields and turns activities into `activity` entities. `consolidate()` finds habits by counting distinct days of memories `ABOUT` one activity, replacing vector clustering. Pruning and the timeline gate on fields. Embedding profiles record the extraction model they were measured with, and every scenario run checks it.

**Tech Stack:** Python 3.12, Neo4j 5.26 (Cypher), Pydantic, LangChain/LangGraph, pytest (`-m llm` for LLM evals), fastembed (bge-small) / remote Qwen3.

**Spec:** [`docs/superpowers/specs/2026-11-06-structure-over-prose-design.md`](../specs/2026-11-06-structure-over-prose-design.md) (decisions D1–D6 below refer to it)

## Global Constraints

- Run everything with `uv run`. Default tests: `uv run pytest -q` (Neo4j up, local embeddings). LLM evals: `uv run pytest -q -m llm`.
- Tests run on local bge-small unless `TEST_REMOTE_EMBEDDINGS=1` (`tests/conftest.py`): every threshold comes from `PROFILE`, never a literal.
- `PATTERN_WINDOW_DAYS = 60` and `PATTERN_MIN = 3` keep their values; `PATTERN_MIN` now counts **distinct days**.
- The activity relation type is `practices` (existing in `companion/ontology.py`).
- Commit messages: concise, no AI attribution trailers (user's global CLAUDE.md).
- Never run `companion.reextract` against real users (`user:mark`, `user:guest`) without the user's explicit go-ahead.
- Lesson prose follows the repo's style: short declarative sentences, a "why" for each choice, a regression story when a test came from a failure.

## Review Focus

1. **Same activity, different words** ("gym", "the gym", "workout"): the extractor must reuse the known entity via `ref`. Pinned by Task 2's eval `test_a_known_activity_is_reused`.
2. **Several mentions on one day** ("heading to the gym" + "back from the gym" the same afternoon) must not count as three days of habit. Pinned by Task 3's `test_one_day_of_gym_talk_is_not_a_habit`.
3. **A memory about two activities** ("gym then yoga") must count toward both. Pinned by Task 3's `test_a_memory_counts_for_each_activity`.
4. **Private or sensitive occurrences** must never feed a shared pattern (an existing rule the rewritten query must keep). Pinned by Task 3's `test_private_occurrences_dont_form_a_pattern`.
5. **A feeling with no `valid_until`** (extractor omitted it) must not be pruned by the new property gate. Pinned by Task 4's `test_prune_needs_an_expiry`.

---

### Task 0: Start from a clean tree

The uncommitted patch from the debugging session (date stripping before clustering, plans in clusters, calibration pairs) is superseded by this plan.

- [ ] **Step 1:** `git status --short` lists only `companion/embeddings.py`, `companion/store.py`, `companion/calibrate.py`, `tests/test_13_time.py` as modified, plus the untracked `docs/superpowers/`. If anything else is listed, stop and ask.
- [ ] **Step 2:** `git restore companion/embeddings.py companion/store.py companion/calibrate.py tests/test_13_time.py`
- [ ] **Step 3:** `git add docs/superpowers && git commit -m "Add structure-over-prose design and plan"`

---

### Task 1: Dates live in fields (D1)

**Files:**
- Modify: `companion/extract.py` (rule 2, rule 3's examples, the Examples block: dated texts lose the date when a field holds it)
- Modify: `companion/store.py` (`recall()` graph-hop memory map, `format_context`)
- Modify: `companion/static/index.html` (`SOURCES` "semantic" and "entities" renderers)
- Test: `tests/test_10_recall.py`, `tests/test_11_extraction_evals.py`

**Interfaces:**
- Produces: `format_context` shows a memory's day as ` (on YYYY-MM-DD)` from `occurred_on`, or ` (due YYYY-MM-DD)` from `due_on`, after the text. Graph-hop memories carry `occurred_on` and `due_on` keys.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_10_recall.py
def test_the_day_comes_from_the_field_not_the_text(store):
    store.apply(Extraction(memories=[m("Pytest went to the gym", "event", occurred_on="2026-11-03", importance=4)]), today=T)
    block = format_context(store.recall("did I go to the gym?", today=T, touch=False), T)
    assert "Pytest went to the gym (on 2026-11-03)" in block, block
```

```python
# tests/test_11_extraction_evals.py
def test_a_known_day_goes_in_the_field_not_the_text(llm_ready):
    def check():
        ex = extract("Just got back from the gym, my legs are dead.")
        dump = ex.model_dump_json(exclude_defaults=True)
        assert not any(re.search(r"\d{4}-\d\d-\d\d", m.text) for m in ex.memories), "date in text: " + dump
        assert any(m.occurred_on == WED.isoformat() for m in ex.memories), "day not in occurred_on: " + dump
    trials(check)
```

- [ ] **Step 2:** `uv run pytest -q tests/test_10_recall.py -k day_comes_from` → FAIL (no "(on 2026-11-03)"). `uv run pytest -q -m llm tests/test_11_extraction_evals.py -k known_day` → FAIL (date in text).
- [ ] **Step 3:** Rule 2 in `RULES`: keep "absolute dates, never relative", add: when the day goes into `occurred_on`, `due_on` or `valid_until`, don't repeat it in `text`; vague anchors with no field stay in text. Update every example in the Examples block that writes a known day into text. `test_a_move_supersedes_and_keeps_vague_dates_vague` must still hold.
- [ ] **Step 4:** In `recall()`, return `m.due_on AS due_on` from the semantic query (it already returns `occurred_on`) and add `.occurred_on, .due_on` to the graph-hop map projection `m {.id, .kind, .text, .importance, ...}`. In `format_context`, append the day to semantic and graph-hop lines (` (on …)` / ` (due …)`); follow-ups already show `due`. In `index.html`, show the same suffix in the "semantic" and "entities" renderers.
- [ ] **Step 5:** Both tests from Step 2 pass; `uv run pytest -q tests/test_10_recall.py tests/test_17_web.py` passes.
- [ ] **Step 6:** `git commit -am "Keep known days in date fields, not memory text"`

---

### Task 2: Activities someone does are entities (D2, D6)

**Files:**
- Modify: `companion/extract.py` (rule 10 and one example)
- Test: `tests/test_11_extraction_evals.py`

**Interfaces:**
- Produces: extraction output where a gym visit has an `EntityOut(name="gym", kind="activity")` (or `ref` to a known one) and the memory's `about` includes it.

- [ ] **Step 1: Write the failing evals**

```python
GYM = "e_gym: gym (activity) — Riley's gym workouts"  # the known-entity line format, as in the Jordan eval

@pytest.mark.parametrize("msg", ["Heading to the gym, talk later!",
                                 "Off to the gym again after work, then dinner.",
                                 "Just got back from the gym, my legs are dead."])
def test_a_gym_visit_is_about_the_gym_activity(llm_ready, msg):
    def check():
        ex = extract(msg)
        dump = ex.model_dump_json(exclude_defaults=True)
        assert any(e.kind == "activity" and "gym" in e.name.lower() for e in ex.entities), "no gym activity: " + dump
        assert any(any("gym" in a.lower() for a in m.about) for m in ex.memories), "memory not about gym: " + dump
    trials(check)

def test_a_known_activity_is_reused(llm_ready):
    def check():
        ex = extract("Hit a great workout at the gym this morning", entities=[GYM])
        dump = ex.model_dump_json(exclude_defaults=True)
        assert any("e_gym" in m.about for m in ex.memories) or any(e.ref == "e_gym" for e in ex.entities), dump
    trials(check)

def test_a_one_off_dinner_is_not_an_entity(llm_ready):
    def check():
        ex = extract("Had pizza for dinner tonight.")
        assert not ex.entities, ex.model_dump_json(exclude_defaults=True)
    trials(check)
```

- [ ] **Step 2:** `uv run pytest -q -m llm tests/test_11_extraction_evals.py -k "gym or activity or dinner"` → the gym cases FAIL (rule 10 keeps generic activities as text); the dinner case passes.
- [ ] **Step 3:** Rule 10: activities {user} or someone in their life *does* (gym, running, yoga, therapy, piano lessons) are `activity` entities named by the plain noun, even unnamed, and every memory about doing one lists it in `about`. One-off generic things (a movie, pizza for dinner) stay text-only. Change the "Off to the gym!" example to include `entities: gym (activity)` and `about [gym]`.
- [ ] **Step 4:** All evals in Step 1 pass; the whole file passes with `uv run pytest -q -m llm tests/test_11_extraction_evals.py`.
- [ ] **Step 5:** `git commit -am "Extract activities someone does as entities"`

---

### Task 3: A habit is a graph count (D3)

**Files:**
- Modify: `companion/store.py` (`consolidate()` step 2, `_store_pattern`; delete `_clusters`, `PATTERN_SIMILARITY`)
- Modify: `companion/embeddings.py` (delete `Profile.pattern`, `Profile.cluster_instruction`, `embed_for_clustering`; remove both from `PROFILES`)
- Modify: `companion/calibrate.py` (delete the `pattern` check and the `"cluster"` branch of `similarity`)
- Test: `tests/test_13_time.py` (replace the clustering tests), `tests/test_19_embeddings.py` (delete `test_clustering_uses_its_own_instruction`)

**Interfaces:**
- Consumes: `activity` entities and `ABOUT` links (Task 2); `writer(user_name, texts, today) -> MemoryOut | None` (unchanged).
- Produces: `MemoryStore._habits(now) -> list[dict]`, one dict per activity: `{"activity": <entity id>, "name": <str>, "members": [{"id", "text"}]}`. The pattern memory is `ABOUT` the activity, and `user -practices-> activity` exists afterwards.

- [ ] **Step 1: Write the failing tests** (in `tests/test_13_time.py`; replace `test_clustering_finds_the_group_an_early_pair_would_hide`, `test_clustering_does_not_chain` and the clustering parts of `test_repeated_statuses_roll_up_into_a_pattern`). Seed with `store.apply(Extraction(entities=[EntityOut(name="gym", kind="activity")], memories=[mem(text, kind, 2, about=["gym"])]), today=...)`.

| Test | Seed | Assert |
|---|---|---|
| `test_three_days_at_the_gym_make_a_habit` | status, plan (`due_on` set), event about `gym` on D1, D1+1, D1+3 | writer called once with all 3 texts; pattern `ABOUT` gym; `(:User)-[:RELATED {type:'practices'}]->(gym)` exists; `evidence == 3` |
| `test_one_day_of_gym_talk_is_not_a_habit` | 3 memories about gym, all on D1 | writer not called |
| `test_a_memory_counts_for_each_activity` | 3 days of "gym then yoga" about `[gym, yoga]` | writer called twice, once per activity |
| `test_wording_does_not_matter` | "Pytest is heading to the gym.", "Pytest plans to go to the gym after work and then have dinner", "Pytest went to the gym before work to keep up a workout streak" about gym on 3 days | writer called once (the regression from the gemini run) |
| `test_unlinked_memories_form_no_habit` | 3 gym statuses with no `about` | writer not called |
| `test_a_lapsed_plan_is_no_evidence` | 2 statuses + 1 plan about gym on 3 days, then `store._q("MATCH (m:Memory {user:$user, kind:'plan'}) SET m.status = 'lapsed'")` | writer not called |
| `test_private_occurrences_dont_form_a_pattern` | 3 days about gym, one `private=True` | writer not called |

Keep `test_writer_can_decline`, `test_consolidated_details_are_pruned_once_dormant` and the "running it again finds nothing new" assertion, re-seeded with `about=["gym"]`.

- [ ] **Step 2:** `uv run pytest -q tests/test_13_time.py` → the new tests FAIL (habits still come from clustering).
- [ ] **Step 3:** Implement `_habits(now)` as one Cypher query: `MATCH (a:Entity {user: $user, kind: 'activity'})<-[:ABOUT]-(m:Memory {user: $user})` with the occurrence filter from spec D3, the day `coalesce(m.occurred_on, m.due_on, date(m.valid_until), date(m.created_at))`, then group by `a` and keep groups with `count(DISTINCT day) >= $min`. Members are ordered by `created_at`. In `consolidate()`, loop over `_habits` instead of `_clusters`. In `_store_pattern`, write the pattern with `about=[activity id]` and `relations=[RelationOut(source="user", target=<activity id>, type="practices")]` through the same `apply()` call.
- [ ] **Step 4:** Delete the clustering code and profile fields listed under **Files**; `uv run pytest -q tests/test_13_time.py tests/test_19_embeddings.py` passes, and `grep -rn "_clusters\|PATTERN_SIMILARITY\|cluster_instruction\|embed_for_clustering" companion tests` returns nothing.
- [ ] **Step 5:** `git commit -am "Find habits by counting days per activity in the graph"`

---

### Task 4: "Temporary" is a property (D4)

**Files:**
- Modify: `companion/store.py` (`_prune`, `timeline`)
- Test: `tests/test_13_time.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_prune_needs_an_expiry(store):
    store.apply(Extraction(memories=[mem("Pytest felt grumpy", "feeling", 2)]), today=D1)  # no valid_until
    assert store.consolidate(D1 + dt.timedelta(days=61)) == []

def test_an_expired_event_is_pruned_like_a_status(store):
    store.apply(Extraction(memories=[mem("Pytest was stuck in traffic", "event", 2,
                                         valid_until=f"{D1.isoformat()}T18:00")]), today=D1)
    assert store.consolidate(D1 + dt.timedelta(days=400)) == ["pruned 1 dormant expired memories"]
```

Update `test_prune_deletes_only_dormant_trivia` to seed its feelings with `valid_until` (as the extractor does per rule 3) and expect the new log line `"pruned 1 dormant expired memories"`.

- [ ] **Step 2:** `uv run pytest -q tests/test_13_time.py -k "prune or expired"` → FAIL.
- [ ] **Step 3:** In `_prune`, replace `m.kind IN ['status', 'feeling']` with `m.valid_until IS NOT NULL AND m.valid_until <= $now`; the log line becomes `f"pruned {n} dormant expired memories"`. In `timeline`, drop `m.kind IN ['event', 'plan'] AND`.
- [ ] **Step 4:** `uv run pytest -q tests/test_13_time.py tests/test_18_inspect.py` passes.
- [ ] **Step 5:** `git commit -am "Prune and list the timeline by fields, not kind"`

---

### Task 5: Thresholds record their extraction model (D5)

**Files:**
- Modify: `companion/embeddings.py` (`Profile.extraction_model: str`; both profiles `"google/gemma-4-31B-it"`)
- Modify: `companion/scenarios.py` (`run_scenario` appends the check), `companion/calibrate.py` (print the field in the header)
- Test: `tests/test_19_embeddings.py`

**Interfaces:**
- Produces: `scenarios.threshold_check() -> tuple[str, bool]`, returning `("thresholds measured with this extraction model (<profile value> vs <config value>)", equal)`; `run_scenario` appends it to every result list.

- [ ] **Step 1: Write the failing test**

```python
def test_scenarios_check_the_extraction_model(monkeypatch):
    from companion import scenarios
    from companion.embeddings import PROFILE
    monkeypatch.setattr(scenarios.config, "LLM_EXTRACTION_MODEL", PROFILE.extraction_model)
    assert scenarios.threshold_check()[1] is True
    monkeypatch.setattr(scenarios.config, "LLM_EXTRACTION_MODEL", "some-other-model")
    desc, ok = scenarios.threshold_check()
    assert not ok and "some-other-model" in desc
```

- [ ] **Step 2:** `uv run pytest -q tests/test_19_embeddings.py -k extraction_model` → FAIL (`threshold_check` missing).
- [ ] **Step 3:** Add the field; add `import config` to `companion/scenarios.py` (it only imports names from config today) and `threshold_check()` reading `config.LLM_EXTRACTION_MODEL` at call time and the append in `run_scenario`; print `Measured with extraction model: …` in `calibrate.main()`.
- [ ] **Step 4:** `uv run pytest -q tests/test_19_embeddings.py` passes.
- [ ] **Step 5:** `git commit -am "Record which extraction model the thresholds were measured with"`

---

### Task 6: Docs

**Files:** `lessons/09-memory-write-path.md` (rules 2 and 10), `lessons/13-memory-over-time.md` (consolidation: graph count replaces clustering; keep the old clustering story as "what this replaced and why"), `lessons/19-choosing-an-embedding-model.md` (pattern row and `cluster_instruction` removed; `extraction_model` added), `docs/architecture.md` (the consolidation state diagram and flowchart), `docs/design-decisions.md` (new decision "Identity in structure, similarity only for fuzzy meaning" citing this incident; gaps table: add structural dedupe and calibration from real runs), `docs/glossary.md` (pattern row), `README.md` (how to re-measure after changing `LLM_EXTRACTION_MODEL`).

- [ ] **Step 1:** Make the edits. Every code name in the docs must exist: `grep -rn "PATTERN_SIMILARITY\|cluster_instruction\|embed_for_clustering\|complete linkage" docs lessons README.md` returns only the Lesson 13 history paragraph.
- [ ] **Step 2:** `git commit -am "Document structure-over-prose memory identity"`

---

### Task 7: Verify end to end, re-measure, hand over existing data

- [ ] **Step 1:** `uv run pytest -q` → all pass (skips only the 4 lesson-prerequisite skips).
- [ ] **Step 2:** `uv run pytest -q -m llm` (about 20 minutes, run in the background). Expected: every scenario's gym check passes; every scenario fails only `thresholds measured with this extraction model` while `.env` uses `gemini-3.8-flash`.
- [ ] **Step 3:** Re-measure on the current model: `EMBED_BASE_URL= uv run python -m companion.calibrate` and `uv run python -m companion.calibrate`. Update any threshold whose current value gets pairs wrong. Set `extraction_model` on both profiles to `config.LLM_EXTRACTION_MODEL`'s value, then re-run `uv run pytest -q -m llm`: everything passes. Commit `"Re-measure thresholds on <model>"`.
- [ ] **Step 4:** Ask the user before running `uv run python -m companion.reextract --user Mark --dry-run` (and then without `--dry-run`, and for `guest`) to backfill activity links on their real memories.
