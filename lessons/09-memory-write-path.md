# Lesson 9 — The write path: from conversation to graph

After every exchange, the companion turns what was said into structured memory:

```
exchange ──► extract (LLM, structured output) ──► resolve entities ──► link relations
         ──► retire / complete old memories ──► dedupe ──► create memories (+ SUPERSEDES)
```

Code: [`companion/extract.py`](../companion/extract.py) (LLM),
[`companion/models.py`](../companion/models.py) (the output schema),
[`companion/store.py`](../companion/store.py) `MemoryStore.apply()`.

## 1. Extraction with structured output

The extractor gets the latest user message and reply, **today's date** (to resolve "next
Thursday"), the **known entities**, and the **existing memories it may update, with their ids**
(found by vector search on the user's message, plus every open plan). It returns an `Extraction`:

```python
class Extraction(BaseModel):
    entities: list[EntityOut]        # name, kind, aliases, summary
    relations: list[RelationOut]     # source ('user' or name), target, type, since
    memories: list[MemoryOut]        # text, kind, about, importance, occurred_on, due_on, supersedes
    invalidate: list[str]            # ids no longer true, no replacement
    completed_plans: list[str]       # ids of plans that happened / were cancelled
```

`llm.with_structured_output(Extraction, method="function_calling")` works on any endpoint
with tool calling; the code falls back to "reply with JSON matching this schema" for endpoints
without it. Extraction errors are caught so **memory problems never break the conversation**.

Key prompt rules (see `EXTRACT_SYSTEM`): one standalone third-person sentence per memory,
absolute dates, only what the user said, supersede instead of duplicating, list the protected
person in a boundary's `about`.

## 2. Entity resolution

`resolve_entity(name, kind, aliases, summary)`:

1. "user", "me" or the user's name → the `:User` node
2. exact match on name **or any alias** (case-insensitive)
3. vector match on `entity_vec`, same kind, score ≥ 0.93 ("Mom" ≈ "my mother")
4. otherwise create it

Then aliases are merged (`apoc.coll.toSet`) and the embedding refreshed. Names used in
memories/relations are looked up but **never auto-created**, which avoids junk nodes.

## 3. Dedupe, supersede, retire

| Situation | What happens |
|---|---|
| New memory ≥ 0.975 similar to a current one | **reinforce** the old one (`access_count`, `last_confirmed`, max importance) |
| Extractor lists `supersedes: [id]` | new memory created, old gets `invalidated_at`, `(new)-[:SUPERSEDES]->(old)` |
| `invalidate: [id]` | old gets `invalidated_at`, no replacement |
| `completed_plans: [id]` | plan `status = 'done'` (it stays on the timeline) |

### Tested without an LLM (hand-written extractions)

```
['linked user -sister-> Maya', 'linked Maya -lives_in-> Denver', 'linked user -pet-> Biscuit',
 '+ [fact] Testy lives in Portland', "+ [fact] Testy's sister Maya is a nurse in Denver",
 '+ [plan] Testy has a job interview at Acme on 2026-11-07', '+ [boundary] Testy prefers not to talk about their ex']

# second batch: the same Portland fact again, plus a move that supersedes it
['reinforced m_c4a8a5', '+ [fact] Testy moved to Seattle in October 2026  (replaces m_c4a8a5)']

history: [('Testy moved to Seattle in October 2026', retired=False), ('Testy lives in Portland', retired=True)]
```

## 4. Real extraction log (from the [demo](outputs/10-companion-demo.txt))

The `✎` lines are what `remember` wrote after each reply:

```
Alex: Hi! I'm Alex. I moved to Portland last month for a new job as a UX researcher at Lumen Health.
   ✎ linked user -lives_in-> Portland
   ✎ linked user -works_at-> Lumen Health
   ✎ + [fact] Alex moved to Portland in October 2026.
   ✎ + [fact] Alex works as a UX researcher at Lumen Health.

Alex: Honestly I'm a bit nervous — I'm giving my first big presentation to leadership next Thursday.
   ✎ + [plan] Alex is giving their first big presentation to leadership on 2026-11-12.

— a week later —
Alex: It went really well actually! My VP said it was the clearest research readout she's seen.
   ✎ plan done m_d8e5d4
   ✎ + [event] Alex received high praise from their VP for a research readout presentation on 2026-11-12 ...

Alex: Also, small update: I switched teams, I'm on the Research Ops team now instead of the patient app team.
   ✎ + [fact] Alex works on the Research Ops team at Lumen Health.  (replaces m_0c596b)
```

"Last month" became "October 2026" and "next Thursday" became `2026-11-12`, because the extractor
knows today's date and the prompt forbids relative dates.

> **Before that rule was added**, the first run stored "…presentation to leadership next
> Thursday" — true for one week, then wrong. Prompt rules are part of the data model.

## Inspect it in the Browser

```cypher
// The life graph
MATCH path = (:User {id:'user:alex (demo)'})-[:RELATED]->() RETURN path
```
```cypher
// Memories with what they're about, their versions and their source episode
MATCH p = (m:Memory {user:'user:alex (demo)'})-[:ABOUT|SUPERSEDES|FROM]->() RETURN p
```
```
ABOUT 12, FROM 10, SUPERSEDES 1   (2 episodes, 4 related entities + the user, 10 memories)
```
```cypher
// Episodes in order, with turn counts
MATCH (ep:Episode {user:'user:alex (demo)'})
RETURN ep.started_at, COUNT { (ep)-[:HAS_TURN]->() } AS turns, EXISTS { (ep)-[:NEXT]->() } AS has_next
ORDER BY ep.started_at
```
```
2026-11-05T12:00Z, 8, TRUE
2026-11-13T12:00Z, 6, FALSE
```
