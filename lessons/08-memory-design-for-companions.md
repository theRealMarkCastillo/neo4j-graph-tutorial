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
  local `bge-small` model, no API key).

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
.venv/bin/python -c "from neo4j import GraphDatabase; import config; from companion.schema import ensure_schema; \
d = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH); ensure_schema(d); print('ok')"
```

(`MemoryStore(...)` also runs it automatically.)

```cypher
SHOW INDEXES YIELD name, type WHERE name IN ['memory_vec','entity_vec','episode_vec','memory_user_kind']
```

## Test it — [`tests/test_08_memory_schema.py`](../tests/test_08_memory_schema.py)

Schema setup must be idempotent, so the test runs `ensure_schema()` **twice** and then checks
that every constraint and vector index is `ONLINE`. A second test proves the id constraint
actually rejects a duplicate.
