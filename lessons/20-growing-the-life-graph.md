# Lesson 20 — Growing the life graph past one hop

A user told the companion about their son over ten turns: Logan is 7, loves Lego and
Minecraft, builds inventions and gives guided tours of them, splashes in the rain in his
boots, watches Moana on Disney+ on his Samsung tablet. Afterwards the life graph in the web UI
showed one node, `Logan`, and a memory count. Nothing branched off him.

Nothing was lost. Neo4j held 13 memories `ABOUT` Logan. But the *graph* had stopped growing
at the first hop:

```
(Guest)-[:RELATED {type:'son'}]->(Logan)        ← the only edge
(Logan)<-[:ABOUT]-(:Memory {text:"Guest's son Logan likes Lego and Minecraft."})
                                                ← the rest is text inside memories
```

A life graph that only fans out from the user is just a contact list with notes attached.
This lesson makes it grow outward: Logan → Minecraft, Logan → his friend Max → Max's dog.
It also makes it stay correct as it grows and lets recall walk it. The changes land in three
places you already know: the extraction prompt (Lesson 9), the ontology (Lesson 12) and the
store's write, recall and upkeep code (Lessons 9, 10 and 13). In terms of the
[layers](../docs/glossary.md#layers), the graph grows in the facts layer (ABox), the ontology
layer (TBox) gets one new rule, and the memory layer is linked to the facts more tightly.

| Gap | Where | Fix |
|---|---|---|
| Liked things never became nodes | extraction prompt | Named things are entities; likes are relations |
| Recall only read edges to the user | recall query | Collect each matched entity's links, two steps out |
| "Minecraft" the thing ≠ "Minecraft game" the activity | entity resolution | Compatible kinds match each other |
| likes and dislikes both current | ontology | An `exclusive` rule |
| An edge outlives the memory it came from | write path | Edges carry `evidence` (their memories' ids) |
| Memories pointed at one entity | extraction prompt | `about` lists every entity a memory names |
| Nodes nothing points at | consolidation | Drop unconnected entities |
| Old conversations predate the new rules | new command | `companion.reextract` re-reads them |
| The UI drew entities only | web UI | Memories on hover; faint "mentioned with" edges |

## 1. Decide what deserves a node

The extractor did exactly what it was told. Rule 10 of the prompt said *people, pets, places
and organizations go in `entities`*, even though the schema allowed `activity` and `thing`
and the ontology already had `likes`, `dislikes` and `practices`. The model was never asked
to use them.

The fix is a sharper rule, not a looser one. Making everything a node ("the rain", "a movie",
"whiskey") would bury the people under clutter. The line we drew:

> Named things someone likes, dislikes, owns or does — a game, show, film, band, book, brand,
> product or hobby ("Minecraft", "Moana", "Lego", "chess") — are entities, kind thing or
> activity, each with its relation. Generic phrases ("playing in the rain", "a movie") stay
> in the memory text only.

(Later, the structure-over-prose change, [D22](../docs/design-decisions.md#d22), moved one
line: activities someone *does*, named or not ("gym", "running"), are now entities too, so
habits can be counted in the graph. One-off generic things still stay text-only.)

Rule 11 says who the source of the edge is: the person who has the preference.

> "my son Leo loves Minecraft" -> Leo -likes-> Minecraft, not user -likes-> Minecraft.

That one sentence is what turns a star around the user into a graph. The prompt also gets a
worked example showing both halves (Minecraft becomes a node; "the rain" doesn't), because
examples steer a model more reliably than rules do.

A live run against `google/gemma-4-31B-it` with Logan already known:

| Message | New entities | New relations |
|---|---|---|
| "He loves Lego and Minecraft" | Lego, Minecraft | Logan -likes-> Lego, Logan -likes-> Minecraft |
| "He likes watching Moana on Disney+ on his Samsung tablet" | Moana, Disney+, Samsung tablet | Logan -likes-> Moana |
| "He loves splashing in the rain" | — | — |

Disney+ and the tablet came out with no relation. Sections 5 and 7 deal with nodes like that.

### `about` names every entity

A memory's `about` list is how vector recall connects text back to the graph. "Logan likes
Moana" `ABOUT` Logan only is invisible from the Moana side. Rule 10 now says `about` lists
**every** entity the memory names, and the schema's field description repeats it (models read
field descriptions too).

## 2. Recall: walk the links, not just the spokes

Lesson 10's entity hop finds entities named or meant in the message and returns each one's
memories and its relation **to the user**:

```cypher
OPTIONAL MATCH (:User {id: $user})-[r:RELATED]->(e)
```

That's one hop from the user. "Does anyone here play Minecraft?" finds the Minecraft node, but
its relation to the user is empty, and the query never looks at `Logan -likes-> Minecraft`.

Now every matched entity also brings its links, in both directions and up to two steps
([`store.py`](../companion/store.py), `recall`):

```cypher
WITH e, relation, memories, COLLECT {
  MATCH p = (e)-[:RELATED*1..2]-(o:Entity)
  WHERE all(x IN nodes(p)[1..] WHERE NOT x:User AND <visible to this persona>
            AND NOT EXISTS { <a boundary protects x> })
    AND all(r2 IN relationships(p) WHERE r2.until IS NULL AND <visible>)
  UNWIND relationships(p) AS r2
  WITH DISTINCT r2, startNode(r2) AS s, endNode(r2) AS t
  RETURN {source: s.name, type: r2.type, target: t.name,
          source_rel: <s's relation to the user>, target_rel: <t's relation to the user>}
  LIMIT 15
} AS links
```

Four design choices are packed in there:

- **Undirected match.** `-[:RELATED*1..2]-` finds the edge from either end, so asking about
  Minecraft reaches Logan and asking about Logan reaches Minecraft.
- **Never through the user.** Every path would otherwise go `Logan → user → everyone`, and
  "two steps" would mean "the whole graph".
- **Every node on the path passes the same filters as the entity itself**: Lesson 11's
  boundaries and Lesson 16's persona visibility. A protected person can't leak in as the
  middle of a path. There's a test for exactly that.
- **Two steps, capped at 15 links.** "My son's friend's dog" needs two. Three starts pulling in
  strangers, and the cap keeps a well-connected node (a city everyone lives in) from flooding
  the prompt.

`COLLECT { }` is a subquery that returns a list per row, so the outer query keeps one row per
entity instead of multiplying rows by links.

### Reading edges as sentences

The prompt gets a new section:

```
How they connect:
- Logan (your son) likes Minecraft
- Max is Logan (your son)'s friend
- Biscuit is Max's pet
```

The edge convention from Lesson 12 is *"target is source's T"*, which reads well for roles
(`Logan -friend-> Max` = Max is Logan's friend) and badly for verbs (`Logan -likes-> Minecraft`
≠ "Minecraft is Logan's likes"). `_link_line` asks the ontology which one a type is: anything
under the `interests` group, or with an underscore (`lives_in`, `works_at`), reads straight
through; everything else gets the possessive form. The renderer reuses the ontology's
hierarchy instead of keeping its own list.

### What this covers without a query tool

Lesson 7's agent had a read-only Cypher tool. The companion doesn't: it's a fixed pipeline,
and every query it runs is a template that respects boundaries and personas. Links coming in
from the other end cover most "graph-wide" questions anyway:

- "Who in my life plays Minecraft?" → Minecraft matches → its incoming `likes` edges.
- "Who lives in Denver?" → Denver matches → every `lives_in` edge into it.
- "What do Logan and I both like?" → Logan's links, plus the user's own relations, which are
  already in the "People in their life" section.

If you later want counting ("how many friends does Logan have?"), that's when a tool with a
fixed set of graph queries earns its place.

## 3. Entity resolution across kinds

Section 1's run called Minecraft a `thing`; another run called it an `activity`. Exact names
already match across kinds (step 3 of `resolve_entity`), but the embedding match (step 4) was
filtered to the same kind, so "Minecraft game" as an activity would miss "Minecraft" as a
thing and create a second node.

```python
_KIND_GROUPS = [{"thing", "activity"}]   # the extractor can't tell these apart reliably

WHERE node.user = $user AND node.kind IN $kinds AND score >= $min
```

People and places stay strict: a person called "Jordan" and a place called "Jordan" really are
different nodes.

## 4. Ontology rule: `exclusive`

Lesson 12's rules were `symmetric`, `inverse`, `cardinality` and `range`. Growing the graph
into preferences needs one more:

```python
"likes":    {"parents": ["interests"], ..., "exclusive": ["dislikes"]},
"dislikes": {"parents": ["interests"], ..., "exclusive": ["likes"]},
```

When the write path links `Logan -dislikes-> Minecraft`, it ends any current
`Logan -likes-> Minecraft` (sets `until`, keeping it as history), the same way cardinality
`one` ends the old `lives_in`. The difference: cardinality is about *other targets* (one home
at a time), while `exclusive` is about *other types to the same target* (you can't like and
dislike the same thing). It's mirrored into `(:RelType {exclusive: [...]})` like the other
rules, so you can query it in the Browser.

The prompt also gains a line: a like that stops ("Ana's over Minecraft now") is an ended
relation, like a breakup.

## 5. Provenance: edges remember why they exist

Before this lesson, a memory and an edge saying the same thing were independent.
"Logan likes Minecraft" could be retired, superseded or forgotten, and `Logan -likes->
Minecraft` stayed current forever. With a one-hop graph that rarely mattered (relations to
the user mostly change through explicit events like moves and breakups). With preferences
and friends of friends it matters a lot.

Each edge now carries `evidence`: the ids of the memories that support it.

```cypher
MATCH (:Entity {id: $a})-[r:RELATED {type: $type}]->(:Entity {id: $b})
SET r.evidence = apoc.coll.toSet(coalesce(r.evidence, []) + $ids)
```

**Which memories count?** Only those `ABOUT` *both* ends of the edge. The first version used
"about the non-user end", and an existing test caught it straight away:

```python
# test_someone_elses_move_does_not_move_the_user
"Maya also lives in Boston"   about [Maya, Boston]
```

That memory is about Boston, so it became evidence for the *user's* `lives_in Boston`. When
"Maya moved to Denver" superseded it, the user's home ended. Requiring both ends fixes it.

**When a memory leaves** (`_retire_links`):

- replaced by a memory about the same two entities → the replacement inherits the support
  ("Logan likes Minecraft" → "Logan plays Minecraft every weekend": the edge stays);
- retired with no heir, or forgotten → removed from `evidence`; an edge left with **no**
  evidence ends.
- Edges from before this lesson have no `evidence` property and are never ended this way, and
  neither are edges the same extraction just asserted. Both are cases where "no evidence"
  means "we don't know", not "it's false".

This is the reification idea from Lesson 3 in miniature: a relationship property that points
back to its source, without turning every edge into a node.

## 6. Re-reading old conversations

Extraction runs when something is said. A better prompt helps from then on and does nothing
for the conversation where you described Logan. `companion.reextract` runs the end-of-episode
pass again over stored transcripts:

```bash
uv run python -m companion.reextract --user Guest --dry-run   # show, write nothing
uv run python -m companion.reextract --user Guest             # apply
uv run python -m companion.reextract --user Guest --last 3    # recent episodes only
```

The first dry run found **nothing**. The end-of-episode prompt (Lesson 11) says *don't repeat
memories that are already stored correctly*, and all of Logan's memories were. The model
obeyed and returned empty lists. Re-reading needs a different job description, so
`REFRESH_SYSTEM` says the rules changed, asks for every entity and relation the conversation
supports *even when the fact is already a memory*, and allows re-outputting an existing memory
with a longer `about`. The recorded preview from that version (it showed names and counts only):

```
ep_ffeecf: entities ['Lego', 'Minecraft'], links ['e_84c8fd -likes-> Lego', 'e_84c8fd -likes-> Minecraft']
ep_9f0e88: entities ['Samsung tablet', 'Disney+', 'Moana', 'Lego', 'Minecraft'], links [... 5 likes ...]
```

The write path usually reinforces memories it already has, and a reinforced memory also
gains `ABOUT` edges to any entity its new `about` names. This is not guaranteed idempotence:
a fresh extraction can word or link things differently, and earlier applied episodes can
change what later episodes match. Apply can also reconcile entities and retire old facts. Each
episode is read with its own date, so "next Friday" still means the Friday after *that*
conversation. This is the same idea as Lesson 19's `reembed`: when a derived artifact's
recipe changes, you need a backfill command, not just a new code path.

### Review a current-graph preview before applying

`--dry-run` now prints each episode's original date and persona, candidate memory text,
kind, ABOUT references and supplied date fields. It labels entities and links as existing,
new, or updates; memories as new, reinforcement, skipped, or unresolved, with match ids and
scores when available. Proposed entity reconciliation, link endings, retirements, plan
completions and forgetting remain proposals. No schema, ontology, user, vector-property or
memory writes occur, including startup metadata updates or temporary simulated writes.

Selection uses the same entity and duplicate helpers as the write path: explicit refs,
alias/name matching, embedding cuts, date equality, visibility and the entity veto.
A missing reference, or an earlier proposed merge/retirement that could affect a match,
is reported as unresolved. The preview does not pretend to execute those changes in memory.

Each episode is compared with the **current graph**, not the results of applying earlier
proposals. A later apply reruns the model, so it is not an execution of an approved saved
preview. Review changes again if the graph, extraction model, or rules have changed.

The database must already be set up for the active embedding profile. Missing vector
indexes produce an instruction to run normal setup; dry run does not create or repair them.
An unknown user or one without episodes gets an explicit no-episodes result and is not
created. Both dry run and apply send eligible stored conversations to the configured model.

## 7. Keeping the graph tidy

**Consolidation step 4: orphans** ([`store.py`](../companion/store.py), `_drop_orphans`).
An entity that no memory (current or retired) is about and that has no link, past or
present, is deleted. That's left over from a forgotten memory, or a name the extractor
listed without saying how it connects. Anything with history is kept: an ex-partner has
an ended edge, so they stay.

**The web UI** ([`web.py`](../companion/web.py), `/api/graph`) now:

- returns up to six memory texts per entity, shown on hover. Before, the graph showed only
  a count, which is why Logan looked empty. Sensitive memories are left out, and persona
  visibility applies as everywhere else;
- draws a faint dotted **"mentioned with"** edge between entities that a current memory
  names together but no relation links ("Logan watches Disney+"). It shows what the
  companion knows without pretending to be a typed relation.

## In the Browser

```cypher
// Everything two steps from Logan, never through the user
MATCH p = (:Entity {name: 'Logan'})-[:RELATED*1..2]-(o:Entity)
WHERE none(x IN nodes(p) WHERE x:User)
RETURN p
```

```cypher
// Each edge and the memories that justify it
MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE r.until IS NULL
OPTIONAL MATCH (m:Memory) WHERE m.id IN coalesce(r.evidence, [])
RETURN a.name, r.type, b.name, collect(m.text) AS because
```

```cypher
// Edges with no recorded evidence (from before provenance, or never backed by a memory)
MATCH (a:Entity)-[r:RELATED]->(b:Entity) WHERE r.until IS NULL AND r.evidence IS NULL
RETURN a.name, r.type, b.name
```

```cypher
// The exclusive rule, from the mirrored ontology
MATCH (t:RelType) WHERE size(t.exclusive) > 0 RETURN t.name, t.exclusive
```

## Concepts and design choices

- **Scope of extraction is a design decision, not a model limitation.** The model created
  the nodes as soon as the prompt asked. Write down what deserves a node and why, and give
  an example of what doesn't.
- **Edges need a source and a reason.** Who has the preference decides the source node;
  `evidence` records why the edge exists, so it can end when the reason does.
- **Bounded traversal.** Multi-hop recall needs three limits: depth (2), fan-out (15), and
  nodes it may not pass through (the user, protected entities, other personas' private
  links). Without them, "related" means "everything".
- **One rule, one place.** `exclusive` lives in the ontology next to cardinality. The write
  path reads it and doesn't know about likes or dislikes. `_link_line` asks the ontology's
  hierarchy how to phrase an edge.
- **"Unknown" isn't "false".** Edges without `evidence` are left alone. Treating missing
  provenance as "unsupported" would have ended every relation stored before this lesson.
- **Backfills are part of the change.** A new extraction rule ships with a way to apply it
  to the past, plus a dry run.

## Test it

Write path, [`tests/test_09_write_path.py`](../tests/test_09_write_path.py), no LLM:

| Test | Checks |
|---|---|
| `test_a_thing_and_an_activity_with_close_names_are_one_entity` | §3 compatible kinds |
| `test_dislikes_ends_likes_of_the_same_thing` | §4 `exclusive` |
| `test_a_link_ends_when_its_only_memory_is_retired` | §5 no heir → edge ends, other edges stay |
| `test_a_link_survives_when_its_memory_is_replaced_by_one_about_the_same_thing` | §5 heir inherits |
| `test_forgetting_a_memory_ends_the_link_it_alone_supported` | §5 via `forget()` |
| `test_a_repeated_memory_gains_the_entities_it_names` | §6 reinforce adds `ABOUT` |
| `test_consolidate_drops_entities_nothing_points_at` | §7 orphans |
| `test_someone_elses_move_does_not_move_the_user` (Lesson 11) | §5 evidence needs both ends |

Recall, [`tests/test_10_recall.py`](../tests/test_10_recall.py), and the UI,
[`tests/test_17_web.py`](../tests/test_17_web.py):

| Test | Checks |
|---|---|
| `test_entity_hop_follows_links_between_entities_from_either_end` | §2 from Logan and from Minecraft |
| `test_entity_hop_reaches_two_steps_out` | §2 son → friend → friend's pet |
| `test_a_protected_entity_is_not_reached_through_a_link` | §2 boundaries hold mid-path |
| `test_the_graph_shows_what_it_knows_and_links_co_mentioned_entities` | §7 hover texts, "mentioned with" |

```bash
uv run python -m pytest tests/test_09_write_path.py tests/test_10_recall.py tests/test_17_web.py -q
uv run python -m pytest -m llm tests/test_11_extraction_evals.py   # the prompt change, live
```

## Takeaways

- If the graph doesn't grow, check what the extractor was *asked* for before blaming it.
- Put the person who has the preference at the source of the edge. That's what makes a
  graph instead of a star.
- Multi-hop recall is a bounded, filtered walk, not a bigger `*` in a pattern.
- Ontology rules generalize: `exclusive` sits beside cardinality and the write path stays
  generic.
- Give edges provenance, so they can end when their memories do, and treat missing
  provenance as unknown.
- Every new rule needs a backfill and a dry run.

## Next

That's the end of the course. Ideas to keep going: a fixed-query graph tool for counting
questions; letting consolidation turn repeated "mentioned with" pairs into typed relations;
and scenario tests (Lesson 11) that run a whole conversation about a child's interests
and grade what recall says a week later.

<!-- nav -->
---

**[Part 7: Growing the graph](../README.md#part-7-growing-the-graph)** (lesson 20)

[← Lesson 19: Choosing an embedding model (and re-measuring everything that depends on it)](19-choosing-an-embedding-model.md) · [All lessons](../README.md#lessons)
