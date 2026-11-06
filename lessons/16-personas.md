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

The rule goes into recall (core, semantic, history, entity hop, follow-ups, yearly dates,
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
.venv/bin/python -m companion --persona Coach
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

## Test it — [`tests/test_16_personas.py`](../tests/test_16_personas.py)

```bash
.venv/bin/python -m pytest tests/test_16_personas.py -v       # 9 tests, ~2 s, no LLM
.venv/bin/python -m pytest -m llm tests/test_16_personas.py   # "just between us" eval
.venv/bin/python -m companion.scenarios personas
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
