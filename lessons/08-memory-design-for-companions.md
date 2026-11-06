# Lesson 8 — Designing memory for an AI assistant / companion

The goal: an assistant that knows the person's **life story** — people, places, history, plans,
preferences, how they're feeling — across many conversations, and updates that knowledge when
life changes. This lesson is the design; Lessons 9–10 build it.

## What a companion needs to remember

| Kind | Example | Lifetime | Why it matters |
|---|---|---|---|
| **fact** | "Alex's sister Maya is a nurse in Denver" | until it changes | biography |
| **preference** | "Alex likes quiet mornings and hates phone calls" | until it changes | personalization |
| **event** | "Alex's presentation was praised on 2026-11-12" | permanent, dated | life timeline |
| **plan** | "Maya visits for Thanksgiving on 2026-11-26" | open → done | proactive follow-ups |
| **feeling** | "Alex was nervous about the presentation" | fades within days | empathy, tone |
| **boundary** | "Don't bring up Alex's ex, Jordan" | until revoked | trust and safety |

Plus two things that aren't single memories:
- **The life graph**: who and what is in their life and how they relate (sister, pet, employer).
- **Episodes**: each conversation, with a summary and mood, linked in order — the "diary".

## Why graph + vectors (not just a vector store)

This is the project's central choice. [Design decisions D1–D2](../docs/design-decisions.md#d1)
compare it with Postgres + pgvector, a vector DB only, other graph databases, RDF, memory
frameworks (Mem0, Zep/Graphiti, Letta, Cognee, LangMem), cloud memory services and memory kept
as files. The kinds of memory in the table above come from
psychology: *semantic* memory here means general facts
([sense 1](../docs/glossary.md#semantic)), not vector search.

| Need | Vectors alone | Graph + vectors |
|---|---|---|
| "How's my sister?" → find Maya | ✅ fuzzy match | ✅ alias "my sister" → Maya, then hop |
| Everything about Maya | ❌ only text that *looks* similar | ✅ `(m)-[:ABOUT]->(Maya)` |
| "I moved to Seattle" replaces "lives in Portland" | ❌ both stay, contradicting | ✅ `SUPERSEDES` + `invalidated_at` |
| What was true in March? | ❌ | ✅ bi-temporal properties |
| Life timeline, open plans | ❌ | ✅ ordered by `occurred_on` / `due_on` |
| Never surface the ex | ❌ | ✅ exclude entities a boundary is `ABOUT` |
| Where did this memory come from? | ❌ | ✅ `(m)-[:FROM]->(Episode)` |

## The schema — [`companion/schema.py`](../companion/schema.py)

```
(:User:Entity {id, name, kind:'self'})
      │ RELATED {type: sister|pet|works_at|lives_in…, since}
      ▼
(:Entity {id, user, name, kind, aliases[], summary, embedding})   person|pet|place|organization|activity|thing
      ▲ ABOUT
      │
(:Memory {id, user, text, kind, importance 1-10, embedding,
          occurred_on, due_on, status,                 ← domain time: when it happened / is due
          created_at, invalidated_at,                  ← system time: when we learned / un-learned it
          last_accessed, access_count})                ← usage, for recency and reinforcement
      │ SUPERSEDES ──► (:Memory)  older version
      │ FROM ──────► (:Episode {id, user, started_at, ended_at, summary, mood, embedding})
                                  │ NEXT ──► (:Episode)
                                  │ HAS_TURN ──► (:Turn {idx, role, text, at})
```

Design choices, and where they come from in earlier lessons:

- **Memory is a node, not a relationship** (Lesson 3.2): a memory connects to *several*
  entities, has its own properties, and other things point at it (`SUPERSEDES`, `FROM`).
- **Entities are shared nodes** (Lesson 3.1): "who else is in Alex's life?" is a hop, and
  aliases ("my sister", "Mom") resolve to one node.
- **`MERGE` everything that has identity** (Lesson 2): the same Maya across 100 conversations.
- **Two time axes**: `occurred_on`/`due_on` say when something is true in the person's life;
  `created_at`/`invalidated_at` say when the system learned or retired it. Old facts are
  **retired, not deleted**, so history survives ("Alex used to live in Portland").
- **`user` property on every node** scopes all queries and vector searches to one person.
- **Three vector indexes** (Lesson 5): `memory_vec`, `entity_vec`, `episode_vec` (384-dim,
  local `bge-small` model, no API key). Lesson 19 adds a second set per embedding model, such
  as Qwen3-Embedding-8B with 4096 dims.

## Design principles for companion memory

1. **Write standalone memories.** "Alex's sister Maya is a nurse", never "she's a nurse" — each
   memory must make sense alone, years later. Same for dates: absolute, never "next Thursday".
2. **Update, don't accumulate.** New information supersedes old; duplicates reinforce.
3. **Importance and recency both matter.** Retrieval ranks by similarity, importance and how
   recently a memory was used; feelings decay in ~a week, facts over months.
4. **Proactivity comes from plans.** Open plans near their due date are always in context, so
   the companion asks "how did it go?" instead of waiting to be told.
5. **Boundaries are hard filters**, applied in the query (not just the prompt), and the
   boundary itself is always in context.
6. **The user owns their memory**: inspect (`/memories`, `/timeline`, `/people`), see history
   (`/history`), and **hard-delete** (`/forget`) — privacy beats provenance for explicit requests.
7. **Never invent.** The extractor records only what the user said or confirmed; the companion
   is told to ask rather than guess.

## Create the schema

```bash
uv run python -c "from neo4j import GraphDatabase; import config; from companion.schema import ensure_schema; \
d = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH); ensure_schema(d); print('ok')"
```

(`MemoryStore(...)` also runs it automatically.)

```cypher
SHOW INDEXES YIELD name, type WHERE name IN ['memory_vec','entity_vec','episode_vec','memory_user_kind']
```

## Concepts and design choices

*Where this fits:* The design of every step: what to remember, and how it's shaped in the graph. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [semantic](../docs/glossary.md#semantic) | sense 1: semantic *memory* = facts and preferences | semantic search (vector similarity) |
| episodic / prospective / procedural | events and conversations / plans / boundaries and the persona | the `Episode` node (one conversation, a unit of episodic memory) |
| [schema](../docs/glossary.md#schema) | sense 1: the graph's shape (labels, properties, indexes) | the ontology (meaning, Lesson 12) |
| bi-temporal | two time axes: when it's true vs when we learned it | versioning alone |
| [knowledge graph](../docs/glossary.md#knowledge-graph) | senses 1 and 4: the life graph, with validity times | the database |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Neo4j for graph and vectors together | biography is relational and changes over time | Cypher and graph ops skills | vector DB only; Postgres + pgvector; a managed memory service | [D1](../docs/design-decisions.md#d1) |
| Memory = sentence node `ABOUT` entities, plus a life graph | readable in the prompt, traversable in the graph | two representations to keep consistent | triples only; one profile document; raw transcripts | [D2](../docs/design-decisions.md#d2) |
| `user` on every node, one database | works on Community, one query for all users | a missed filter leaks data | a database per user; access control in the server | [D4](../docs/design-decisions.md#d4) |
| Retire, don't delete (except on request) | history and "why did it say that?" | every read filters current versions | overwrite in place; event sourcing | [D8](../docs/design-decisions.md#d8) |
| Boundaries as query filters | a prompt rule is a suggestion, a query filter is a guarantee | each read path must apply it | prompt-only rules; database-level security | [D17](../docs/design-decisions.md#d17) |

## Test it — [`tests/test_08_memory_schema.py`](../tests/test_08_memory_schema.py)

Schema setup must be idempotent, so the test runs `ensure_schema()` **twice** and then checks
that every constraint and vector index is `ONLINE`. A second test proves the id constraint
actually rejects a duplicate.

## Next

[Lesson 9](09-memory-write-path.md) builds the write path: an LLM turns each message into memories, entities and relations that land in this schema.

<!-- nav -->
---

[← Lesson 7: A LangGraph agent with graph tools](07-langgraph-agent-with-graph-tools.md) · [All lessons](../README.md#lessons) · [Lesson 9: The write path: from conversation to graph →](09-memory-write-path.md)
