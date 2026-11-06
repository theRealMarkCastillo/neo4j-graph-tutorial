# Lesson 16 — Personas: several companions, one memory

A user might talk to **Sage**, the warm companion, and **Coach**, the fitness coach. Both should
know the user signed up for a half marathon. Coach should *not* know what the user told Sage
"just between us". The same user, the same graph, different views.

This is a **visibility** problem. Every read path is a separate query, and each one could
forget the rule.

## The model

Every memory, relation and episode records **which persona** it came from:

| Property | On | Meaning |
|---|---|---|
| `agent` | Memory, RELATED, Episode | the persona that heard it |
| `private` | Memory, RELATED | only that persona may see it |

What's private:

| | Private? | Why |
|---|---|---|
| "Just between us, …" | yes | the user asked. The extractor sets `private` (prompt rule 4) |
| sensitive (health, finances…) | yes, automatically | the user told *this* persona; a coach bringing up a diagnosis is a breach |
| boundaries | **never** | "don't mention my weight", said to Sage, must protect with Coach too |
| everything else | no | facts about the user's life are why personas share a memory |
| episode summaries | per persona | each persona has its own diary: a summary can quote private things |

## One rule, pasted into every query

```python
def visible(v: str) -> str:
    return f"({v}.agent IS NULL OR {v}.agent = $agent OR NOT coalesce({v}.private, false))"
```

- `$agent` is always passed by `_q()`, like `$user` and `$now`.
- `agent IS NULL` keeps memories from before personas existed visible to all, so **old data
  still works without a migration**.

The rule goes into recall (core, semantic (the vector retriever), history, entity hop, follow-ups, yearly dates,
"recently"), the people list, inferred relatives, `memories()` and the timeline. It also goes
into the duplicate check, and three places that are easy to miss:

- **The extractor's prompt.** `extraction_candidates()` and `known_entities()` are shown to
  the LLM. If Coach's extractor saw Sage's private memory as a candidate, it could copy it into
  a new, shared memory.
- **Consolidation.** A pattern is shared. One built from private memories would repeat what
  they say.
- **Graph insights** (Lesson 15). The projection skips private memories and edges.

### Hiding a person, not just a memory

"I see Dr. Lee every Monday", said privately, creates a memory *and* an entity, Dr. Lee, the
therapist, with a `therapist` relation. Hiding the memory isn't enough: Coach's people list
would still say "Dr. Lee — therapist".

- A relation is **private** when only private memories in that extraction mention its target.
- `entity_visible(e)` hides an entity whose only links are someone else's private ones.
- **Once shared, stays shared.** If the user later tells Coach about Kai openly, the relation
  becomes shared for everyone: `r.private = $private AND coalesce(r.private, true)`.

## Using it

```bash
uv run python -m companion --persona Coach
```

`MemoryStore(driver, user, agent="Coach")`. The persona's voice comes from `ROLES` in
[`companion/graph.py`](../companion/graph.py):

```python
ROLES = {"Coach": "an upbeat fitness and habits coach who keeps things practical"}
```

`/memories` shows `(private to Sage)` on private memories, but only when Sage is the one
asking: Coach never sees the line at all.

## The scenario

```
-- session 1 (Sage)
Riley: I signed up for the Chicago half marathon in March!
   ✎ + [plan] Riley is signed up for the Chicago half marathon in March 2027
Riley: Just between us, I get really bad panic attacks before big races.
   ✎ + [fact] Riley experiences severe panic attacks before big races. (sensitive)
-- session 2 (Coach)
Coach: Hey Riley! Great to see you. What's on your mind today?
Riley: Can you help me plan my training for this week?
Coach: I'd love to! ... are you looking to push your distance a bit as we look toward
       that Chicago half marathon?
```

Coach used the shared plan without being told, and never brought up the panic attacks. The
greeting didn't refer to the earlier conversation either: Coach's diary is empty. All four
checks passed.

## Concepts and design choices

*Where this fits:* Step 11: who may see what when several personas share one memory. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [agent](../docs/glossary.md#agent) | sense 2: the `agent` property = which persona heard it | an LLM agent (sense 1) |
| private memory | visible only to the persona that heard it | sensitive (a stricter category) |
| visibility rule | Cypher predicates pasted into every read | database access control |
| [semantic recall](../docs/glossary.md#semantic) | sense 2: the vector retriever, filtered like every other | semantic memory |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| One shared store with per-persona visibility | personas know the same life; secrets stay private | every read path must apply the rule (mutation checks guard it) | a separate store per persona; database-level privileges | [D17](../docs/design-decisions.md#d17) |
| Hide derived data too (summaries, patterns, entities) | derived data repeats the private fact | harder to reason about | hide source memories only (leaks) | [D17](../docs/design-decisions.md#d17) |
| Boundaries are shared regardless of who heard them | safety beats privacy here | a persona learns a boundary exists | per-persona boundaries | [D17](../docs/design-decisions.md#d17) |

## Test it — [`tests/test_16_personas.py`](../tests/test_16_personas.py)

```bash
uv run python -m pytest tests/test_16_personas.py -v       # 9 tests, ~2 s, no LLM
uv run python -m pytest -m llm tests/test_16_personas.py   # "just between us" eval
uv run python -m companion.scenarios personas
```

**One test per read path.** The privacy rule lives in many queries, so a test that checks only
`recall()` would miss a leak through the people list or the extractor's prompt.

**Check that the tests can fail.** All nine passed on the first run, which proves little. So
I replaced `visible()` with `"true"` (no filtering) and ran them again: the four privacy
tests failed, as they should. The other five test different mechanisms (shared memories,
boundaries, the diary, patterns, insights). A test you've never seen fail might not be
testing anything.

The LLM eval checks that "just between us" and "don't tell anyone" set `private`, and that the
same sentence without them doesn't. It passed 3/3.

## Takeaways

- Sharing memory between agents is a **visibility rule** applied on every read path, including
  the ones that feed prompts, background jobs and analytics.
- Hide *derived* data too: entities, relations, summaries and patterns can all repeat a
  private memory.
- Safety rules (boundaries) are shared even when the conversation was private.
- Default new fields so old data keeps working (`agent IS NULL` = visible to all).
- Break the code on purpose and watch the tests fail.

## Next

[Lesson 17](17-web-ui.md) puts the chat, the life graph, the decay curves and these insights on one web page.

<!-- nav -->
---

**[Part 5: Running it for real](../README.md#part-5-running-it-for-real)** (lessons 14–16)

[← Lesson 15: Graph embeddings: who and what belong together?](15-graph-embeddings.md) · [All lessons](../README.md#lessons) · [Lesson 17: A web UI: chat, life graph, decay and insights on one page →](17-web-ui.md)
