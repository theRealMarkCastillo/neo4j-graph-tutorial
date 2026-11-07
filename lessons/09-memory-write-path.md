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

Key prompt rules (see `RULES` in [`companion/extract.py`](../companion/extract.py)): one standalone third-person sentence per memory,
absolute dates, only what the user said, supersede instead of duplicating, list the protected
person in a boundary's `about`.

Two rules put *identity* in structure rather than wording ([D22](../docs/design-decisions.md#d22)):
- **A known day goes in its field, not the text** (rule 2): "Riley went to the gym" with
  `occurred_on`, never "...on 2026-11-20". Prompts and the UI show the day from the field.
  Vague times with no field ("in the week before 2026-11-20") stay in the text.
- **Activities someone does are entities** (rule 10): "gym", "running", "yoga" become
  `activity` entities, and every memory about doing one is `ABOUT` it, whatever the
  sentence says. Lesson 13 counts habits from those links.

> Lesson 11 adds earlier turns as context, a second pass at the end of each conversation,
> and new fields: `ref`/`new` on entities, `ended` on relations, `annual_date` and
> `sensitive` on memories, and `forget_entities`.

## 2. Entity resolution

`resolve_entity(name, kind, aliases, summary)`:

1. "user", "me" or the user's name → the `:User` node
2. exact match on name **or any alias** (case-insensitive)
3. vector match on `entity_vec`, same kind, score ≥ 0.93 ("Mom" ≈ "my mother")
4. otherwise create it

Then aliases are merged (`apoc.coll.toSet`) and the embedding refreshed. Names used in
memories/relations are looked up but **never auto-created**, which avoids junk nodes.

**Placeholders get renamed.** Someone mentioned without a name gets a role name: "my son is
sick" makes `Mark's son`. When the name comes later ("his name is Logan"), the extractor
returns `Logan` with `ref` pointing at that node. The store then renames it: `Logan` becomes the
name and `Mark's son` an alias, on the **same node**, so its relations and memories stay
attached. It only renames when the old name is a role and the new one isn't. `is_role_name()`
strips "my " or "X's " and asks the ontology whether the rest is a relation type ("son",
"mom" → mother, "boss" → manager). A role or a nickname for someone already named ("Dad" or
"Bob" for Robert) only becomes an alias:

```
named Mark's son -> Logan (Mark's son kept as an alias)
```

**Who's who, over time.** One person can enter the graph several ways, so `resolve_entity`
also reconciles what it finds. Each case below came from a code review after the Logan bug,
and was reproduced before it was fixed:

| Situation | What happens | Log |
|---|---|---|
| A name is corrected ("it's Maia, not Maya") | the extractor sets `corrected=true`; the right name replaces the wrong one | `named Maya -> Maia (name corrected)` |
| A placeholder and a named node turn out to be one person | the extractor lists the role name as an alias; `merge_entities` moves relations and memories onto the named node and deletes the other | `merged Pytest's sister into Maya (the same person)` |
| "Mom" exists, then "my mother Linda" with no alias | someone has one mother (father, partner, spouse), so a role-named one is Linda. Never for sisters, sons or a stepmom | `merged Mom into Linda (one mother)` |
| Saved as a `thing`, later an `organization` | the kind upgrades (never downgrades), so `works_at` stops being rejected | `Acme is a organization (was thing)` |
| The extractor points "Jordan" at Sam | not kept as an alias: a different name is more likely a wrong `ref` than a nickname, and an alias would send every later "Jordan" to Sam. Roles ("Dad") and close names ("Bob" ~ "Robert") are kept | `kept Sam: 'Jordan' matched it by id but isn't one of its names` |

Merges only ever absorb a **role-named** node. Two nodes that both have real names stay apart,
because two Sams may really be two people (Lesson 11's `same_name` scenario).

## 3. Dedupe, supersede, retire

| Situation | What happens |
|---|---|
| New memory ≥ 0.975 similar to a current one, same dates, and every old non-user ABOUT entity is in the new ABOUT | **reinforce** the old one (`access_count`, `last_confirmed`, max importance) |
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

## Concepts and design choices

*Where this fits:* Steps 2, 3 and 6: extraction, entity resolution, dedupe and supersession. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| information extraction | turning the latest exchange into `Extraction` records | summarization |
| [schema](../docs/glossary.md#schema) | sense 2: the Pydantic schema the LLM must fill | the Neo4j schema |
| [entity](../docs/glossary.md#entity) | sense 2 → sense 1: a mention resolved to an `Entity` node | a database row |
| entity resolution | deciding a mention is a known entity | deduplication of *memories* |
| [dedupe by meaning](../docs/glossary.md#semantic) | sense 5: reworded copies count as the same memory | exact-text dedupe |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| A separate extractor LLM call after the reply | chat stays focused; facts said in passing are caught | one extra call per turn | a `save_memory` tool for the chat model (hot path) | [D6](../docs/design-decisions.md#d6) |
| Entity resolution as a cascade: id ref → name/alias → embedding + name check | cheap and certain checks first, fuzzy ones last | thresholds per embedding model; English role names | LLM judges every match; string matching only | [D7](../docs/design-decisions.md#d7) |
| Supersede and reinforce instead of overwrite or append | history survives, copies don't pile up | graph grows; reads filter current | overwrite (the LLM picks ADD/UPDATE/DELETE, as in Mem0 1.x); append-only and resolve conflicts at read time (Mem0 2.0) | [D8](../docs/design-decisions.md#d8) |

## Test it — [`tests/test_09_write_path.py`](../tests/test_09_write_path.py)

The LLM and the store meet at one typed object, `Extraction`. Hand-write the extraction a
model *would* return, and every storage rule becomes an exact, free test:

```python
store.apply(Extraction(memories=[mem("Pytest lives in Austin")]), today=T)
[old] = store.memories()
store.apply(Extraction(memories=[mem("Pytest lives in Chicago", supersedes=[old["id"]])]), today=T)
assert current(store) == {"Pytest lives in Chicago"}
```

The `store` fixture uses its own `pytest` user and wipes it before and after each test.

A code review of the write path produced regression tests. Each one failed before its fix:
- a correction that matched an existing memory was "reinforced" and **dropped its `supersedes`**;
- forgetting "Ann" deleted *"planning an anniversary trip"* (substring match, now whole words);
- short names ("Jo") and other entities' summaries ("Kai: Jo's brother") weren't redacted;
- `/forget <id>` didn't stick: the end-of-episode pass re-read the transcript and stored the
  same fact again. A tombstone (embedding and non-user ABOUT ids, no text) now blocks it for that episode,
  provided its entities are all named by the new memory;
- vector searches filtered by user *after* taking the top 5, so other users' memories could
  crowd out the real duplicate. Now they take 50, filter, then keep the top k.
- "his name is Logan" never reached the graph. The node stayed `Mark's son`: a known entity
  matched by `ref` kept its old name, and the new name wasn't even added as an alias. Found in a
  real chat; the three `placeholder`/`real_name` tests cover it.
- the review that followed found the five cases in the table above. Breaking each fix on purpose
  makes exactly its own test fail (a mutation check).

`apply()` runs several small transactions, not one. If something fails halfway, the graph is
left half-updated. That's acceptable for a tutorial, but production code would wrap each
memory in `session.execute_write`.

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

> **Later, in [Lesson 20](20-growing-the-life-graph.md):** the extractor only made entities for people, pets, places and organizations, so "Logan loves Minecraft" never became an edge. Lesson 20 widens what deserves a node, adds `evidence` to edges so they end when their memories do, and lets `thing` and `activity` match in entity resolution.

The **entity veto** also protects duplicate matching within an extraction: every old non-user
ABOUT entity must occur in the new ABOUT before wording can decide a match. A re-read may
add entities to a known memory. Dropping one stores a near-copy: merging different facts loses
data, whereas splitting a fact only copies it. User-only memories still depend on wording.

## Next

[Lesson 10](10-companion-recall-and-chat.md) builds the read path: recall those memories on every turn, and chat with a companion that uses them.

<!-- nav -->
---

**[Part 3: A companion that remembers](../README.md#part-3-a-companion-that-remembers)** (lessons 8–11)

[← Lesson 8: Designing memory for an AI assistant / companion](08-memory-design-for-companions.md) · [All lessons](../README.md#lessons) · [Lesson 10: Recall, the LangGraph companion, and the terminal chat →](10-companion-recall-and-chat.md)
