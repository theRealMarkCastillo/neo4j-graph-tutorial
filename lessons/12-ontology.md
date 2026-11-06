# Lesson 12 — An ontology for the life graph

Up to Lesson 11, relation types were whatever words the LLM chose: `girlfriend`, `partner`,
`student_of`, `likes`. Three problems followed:

- **Mismatched words.** The breakup in Lesson 11 ended "girlfriend", but the edge was stored as "partner".
- **Hand-written groups.** "Anyone in my family?" only worked for types someone had listed in a Python dict.
- **No rules.** Nothing knew that a person lives in one place at a time.

An **ontology** fixes this by describing what the relation types *mean*. (Terms are defined
in the [glossary](../docs/glossary.md). *Ontology* has several meanings; this lesson uses
[sense 2](../docs/glossary.md#ontology): a lightweight vocabulary with a hierarchy and rules, not
OWL. Why not OWL or free-form labels: [D9](../docs/design-decisions.md#d9).)

## Semantic memory, graph memory, ontology

| | Answers | Here |
|---|---|---|
| **Semantic memory** | *What kind of knowledge?* General facts, as opposed to episodic (events) | `fact`/`preference` memories, the life graph |
| **Graph memory** | *How is it stored?* Nodes and typed relationships you can traverse | Neo4j |
| **Ontology** | *What do the types mean?* Vocabulary, hierarchy, rules | `companion/ontology.py` → `(:RelType)` nodes |

So the companion is semantic + episodic memory, stored as a graph, searched with vectors,
and (now) typed by an ontology. Two senses of *semantic* meet in that sentence: semantic *memory*
is the kind of knowledge (sense 1), and "searched with vectors" is what products call semantic
*search* (sense 2). See [semantic](../docs/glossary.md#semantic).

## The two layers

An ontology separates what the *types* mean from the *facts* that use them. Description
logic calls the two layers the **TBox** (terminology) and the **ABox** (assertions); data
people say *ontology layer* and *instance layer*.

| Layer | Holds | Example | Stored as |
|---|---|---|---|
| **Ontology (TBox)** | types and rules | `sister` SUBTYPE_OF `sibling` SUBTYPE_OF `family`; `sibling` is symmetric; `lives_in` is one at a time; `pet` points at a `pet` | relation types: `(:RelType)` nodes. Entity kinds (`person`, `pet`, `place`…): code (`companion/models.py`) and each node's `kind` |
| **Facts (ABox)** | statements about individuals | `Riley -sister-> Maya`; Maya is a `person` | `(:Entity)` nodes and `RELATED` edges: the life graph |

Together they make the knowledge graph. They are two of the companion's seven layers: below
them are storage and the schema (the *shape* of data, not its meaning), above them the
memories, retrieval and the prompt. The glossary's [Layers](../docs/glossary.md#layers) lists
all seven, and says why this repo has no *semantic layer* in the BI sense.

Both live in Neo4j. Diagrams: [architecture §15–16](../docs/architecture.md#15-ontology--how-a-relation-is-stored).

## The ontology — [`companion/ontology.py`](../companion/ontology.py)

The Python dict is the source of truth. `ensure_ontology()` copies it into Neo4j on startup.
`/ontology` in the CLI prints it:

```
related
  family
    sibling  [symmetric]
      sister
      brother
    parent  [inverse=child]
      mother
      father
    child  [inverse=parent]
      daughter
      son
    ...
    niece_or_nephew  [inverse=aunt_or_uncle]
  romantic
    partner  [symmetric, one at a time]
      spouse  [symmetric, one at a time]  (also under family)
  friends ...
  work
    coworker  [symmetric]
    manager  [inverse=report]
    works_at  [range=organization]
  pets
    pet  [range=pet]
  places
    lives_in  [one at a time, range=place]
  ex  [symmetric]
```

What each part does:

| Part | Example | Used for |
|---|---|---|
| **Canonical names + synonyms** | `partner`: girlfriend, boyfriend, fiancé… | one name per meaning |
| **Hierarchy** (`SUBTYPE_OF`, a DAG) | `spouse` is under both `partner` and `family` | "anyone in my family", ending a relation at any level of detail |
| **Group words** | family: "family, relatives, siblings, parents, kids" | mapping a question to a branch of the hierarchy |
| **symmetric / inverse** | `sibling` symmetric; `parent` ↔ `child` | reading an edge from either end |
| **cardinality** | `lives_in`, `partner`: one at a time | a new one ends the old one |
| **range** | `pet` → a pet; `lives_in` → a place | rejecting impossible facts |
| **composition** | sibling ∘ child = niece_or_nephew | inference |

Direction convention, everywhere: `source -type-> target` means *target is source's type*.
`Riley -sister-> Maya` means Maya is Riley's sister.

## In the Browser

```cypher
// The whole hierarchy
MATCH p = (:RelType)-[:SUBTYPE_OF*]->(:RelType {name: 'related'}) RETURN p
```
```cypher
// What counts as "family"? (22 types)
MATCH (t:RelType)-[:SUBTYPE_OF*]->(:RelType {name: 'family'}) RETURN t.name ORDER BY t.name
```
```cypher
// Types the extractor used that aren't in the ontology yet, for review
MATCH (t:RelType {proposed: true}) RETURN t.name, t.closest, t.examples
```

## What happens to each relation

```
RelationOut(type="girlfriend")
  → normalize          girlfriend → partner (synonym) · daughter_of → flip · neighbor → new (proposed)
  → ended?             end edges of that type at any level of detail (boyfriend ended → partner ends)
  → range check        pet needs a pet
  → cardinality        lives_in is one at a time → end the old one
  → MERGE RELATED {type: 'partner', said: 'girlfriend'}
```

Deterministic test (no LLM — hand-written extractions through `MemoryStore.apply()`):

```
linked user -partner-> Kim (said 'girlfriend')
linked Leo -daughter-> Ana (said 'daughter_of')            ← "Ana daughter_of Leo", flipped
rejected user -pet-> Maya: pet needs a pet, Maya is a person
new relation type 'neighbor' proposed for the ontology (closest: coworker (0.759))
linked user -pet-> Rex (said 'dog')

ended user -partner-> Kim (said 'boyfriend')               ← stored as partner, ended as "boyfriend"
linked user -ex-> Kim
ended user -lives_in-> Austin (one lives_in at a time)     ← nobody said "I left Austin"
linked user -lives_in-> Denver (said 'moved_to')

inferred: [('Ana', 'niece_or_nephew', 'Leo')]
recall("Is anyone in my family a baby?") entity hop: [('Leo', 'brother'), ('Ana', 'niece_or_nephew, inferred via Leo')]
```

The extractor prompt now includes the canonical list (generated from the ontology) and the
direction convention, so most types arrive canonical. Normalization catches the rest.

## Inference

`infer_kin()` reads the user's active `RELATED` edges, also reads each edge backwards through
its inverse, and composes two hops:

```python
COMPOSE = {("sibling", "child"): "niece_or_nephew", ("parent", "sibling"): "aunt_or_uncle",
           ("partner", "parent"): "parent_in_law", ("aunt_or_uncle", "child"): "cousin", ...}
```

```
Riley -brother-> Leo -daughter-> Ana     ⇒  Ana is Riley's niece_or_nephew (via Leo)
Riley -brother-> Leo <-father- Ana       ⇒  same (father read backwards = child)
```

Inferred relations are **computed at recall time, not stored**: they show up in the people
list and the "family" entity hop marked `inferred via Leo`, and they disappear by themselves
if the underlying facts change. Storing derived facts ("materializing") is faster to query
but has to be kept in sync.

## What the experiments taught

### 1. Embedding similarity is bad at mapping relation words

The first version mapped unknown types to the nearest canonical type by embedding:

```
stepmom      → son        (closest label: 'stepson' 0.862)
step_brother → father     ('step father' 0.883)
lives_near   → lives_in   ('lives at' 0.896)      ← would end their real home (one at a time!)
grandmom     → grandchild ('grandchild' 0.804)
hubby        → spouse     ('husband' 0.88)         ← correct, but in the same score range
```

No threshold separates right from wrong: short words with shared letters (step-, grand-)
embed close together whatever they mean. Now:
- exact names and **synonym lists** do the mapping;
- **unknown types are kept as-is** and proposed for review;
- similarity is only a **hint** for the reviewer.

> An ontology is where *human-curated* meaning beats similarity. Let embeddings suggest; let
> the vocabulary decide.

### 2. An old entity-resolution bug: Seattle became Boston

The cardinality scenario failed even though the raw extraction was perfect:

```
raw extraction: entities: Seattle (place, "The city where Riley now lives")
                relations: user -lives_in-> e_98ae12 ended · user -lives_in-> Seattle
log:            (no relation lines at all)
```

`resolve_entity()`'s vector step (from Lesson 9) embeds `name (kind): summary`. The summary
"The city where Riley now lives" matched Boston's "The city where Riley lives" above 0.93,
so **Seattle was merged into Boston**. Every later step then looked fine and did nothing.

Fix: the vector step now also requires the **names** to be similar (name-only embeddings,
cosine ≥ 0.80):

```
Mom / my mother 0.873   NYC / New York 0.932   Rosa / Aunt Rosa 0.816    → can merge
Seattle / Boston 0.636  Austin / Chicago 0.606 Theo / Leo 0.562          → kept apart
```

A raw-extraction debug switch helped find it:

```bash
COMPANION_DEBUG_EXTRACT=1 uv run python -m companion.scenarios cardinality
```

Before finding the real cause, I added a backstop, `_sync_superseded_relations()`. When a
memory superseded one ABOUT an entity with a one-at-a-time edge, it moved the edge. A later
code review found it moved the *wrong* edges. "Maya moved to Denver" superseding "Maya
also lives in Boston" moved the **user's** home to Denver, because nothing checked whose home
it was. A regression test (`test_someone_elses_move_does_not_move_the_user`) reproduced it,
and the backstop was deleted. A guess that patches a symptom can create a new bug. Fix the
cause, and keep a test.

### 3. Sometimes the LLM does the inference itself

In the inference scenario the extractor stored `user -niece-> Ana` directly, so the ontology
had nothing left to infer. Both paths are fine. The test accepts either, and the
deterministic test above proves the inference path.

## New scenarios

| Scenario | Checks |
|---|---|
| **ontology_synonyms** | "boyfriend" stored as `partner`; the breakup ends it; the pet stays |
| **ontology_cardinality** | "Settling into my new place in Seattle" leaves exactly one active `lives_in` (Seattle); Boston kept as history |
| **ontology_inference** | "My brother Leo just had a baby girl, Ana" → Ana is the user's niece (stored or inferred); "newest member of my family?" → Ana |

Full run (Lesson 11's nine scenarios plus these three, real LLM calls), recorded in
[`outputs/12-scenarios.txt`](outputs/12-scenarios.txt):

```
  cross_turn_facts       4/4      ontology_synonyms      3/3
  correction             2/2      ontology_cardinality   3/3
  history                5/5      ontology_inference     4/4
  relationship_change    4/4
  same_name              4/4
  recurring_date         3/3
  forget_person          6/6
  family_multi_hop       3/3
  sensitive              4/4
  total                  45/45
```

Two things show up in the transcript. With the vocabulary in its prompt, the extractor wrote
`partner` directly for "boyfriend", so normalization had nothing to do. The deterministic test
covers that path. And the extractor again stored `user -niece-> Ana` itself, as in finding 3.
The Lesson 11 scenarios still pass with `RELATION_GROUPS` removed ("family" questions now go
through the hierarchy) and with the `_sync_superseded_relations` backstop deleted (below).

## Concepts and design choices

*Where this fits:* Step 4 (canonicalize relation types) and step 10 (infer new ones). The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [ontology](../docs/glossary.md#ontology) | sense 2: a lightweight vocabulary + hierarchy + rules, in Python and `(:RelType)` nodes | OWL (sense 1); a schema (sense 3) |
| TBox / ABox | the types and rules / the facts about individuals | schema vs data |
| [inference](../docs/glossary.md#inference) | sense 1: rules derive facts (brother's daughter → niece) | model inference (running an LLM); the LLM's own guessing |
| [semantic memory](../docs/glossary.md#semantic) | sense 1: the life graph is semantic memory, typed by the ontology | the Semantic Web (sense 3) |
| canonicalization | mapping relation *words* to one type | entity resolution (for *things*) |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| A lightweight ontology in code, mirrored into the graph | agreement on types without OWL tooling | someone maintains the vocabulary | free-form labels; OWL + reasoner; SHACL validation | [D9](../docs/design-decisions.md#d9) |
| Explicit synonyms, not embedding similarity, for relation words | embeddings mapped relation words badly (experiment 1) | every synonym is hand-listed | nearest-type by embedding (kept only as a review hint) | [D9](../docs/design-decisions.md#d9) |
| Unknown types kept and proposed, not rejected | new kinds of relationship do appear | a review step | reject unknown types; accept everything | [D9](../docs/design-decisions.md#d9) |
| LPG with our own ids, not RDF | properties on relationships, no URIs needed | no standard semantics to exchange | RDF/OWL via neosemantics | [D3](../docs/design-decisions.md#d3) |

## Test it — [`tests/test_12_ontology.py`](../tests/test_12_ontology.py)

The ontology is pure functions over a dict, the ideal shape for rules. Most tests need
neither Neo4j nor an LLM:

```python
@pytest.mark.parametrize("said,canon,how,flip", [
    ("girlfriend", "partner", "synonym", False),
    ("daughter_of", "daughter", "synonym", True),     # X_of flips direction
    ("landlord", "landlord", "new", False),           # unknown: proposed, never guessed
])
def test_normalize(said, canon, how, flip): ...
```

`test_infer_kin` is a table of tiny graphs → expected inferences. The store tests cover
synonyms, cardinality, range and specific-replaces-general. They also include the
Seattle/Boston regression, and a test that someone *else's* move doesn't move the user. That
test is why the `_sync_superseded_relations` backstop was deleted.

## Takeaways

- Semantic memory is *what* you know; graph memory is *how* it's stored; an ontology is *what the types mean*.
- Keep a **controlled vocabulary**: give the LLM the list, normalize with synonyms, and propose unknown types for review rather than guessing.
- Put rules where they're enforced deterministically: **range**, **cardinality**, **lineage** for endings.
- **Infer** at read time when facts change often; materialize only when queries need the speed.
- When a step "does nothing", print the raw LLM output before blaming the LLM.

> **Later, in [Lesson 20](20-growing-the-life-graph.md):** a fifth rule, `exclusive` (likes ends dislikes of the same thing), and the ontology's hierarchy used to phrase edges as sentences in the prompt.

## Next

[Lesson 13 — memory over time](13-memory-over-time.md): expiry, decay and consolidation.

<!-- nav -->
---

**[Part 4: A smarter memory](../README.md#part-4-a-smarter-memory)** (lessons 12–13)

[← Lesson 11: Harder scenarios, and testing memory with scenarios](11-harder-scenarios-and-evals.md) · [All lessons](../README.md#lessons) · [Lesson 13: Memory over time: expiry, decay and consolidation →](13-memory-over-time.md)
