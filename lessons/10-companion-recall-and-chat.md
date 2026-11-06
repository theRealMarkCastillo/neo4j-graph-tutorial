# Lesson 10 — Recall, the LangGraph companion, and the terminal chat

Lesson 9 wrote memories into the graph. This lesson reads them back: on every message, recall
what matters, put it in the prompt, and reply. Then you can chat with the companion in a terminal.

## The read path: hybrid recall — `MemoryStore.recall()`

Every user message triggers one recall that combines five sources:

| Source | How | Why |
|---|---|---|
| **Core profile** | all boundaries + top 8 facts/preferences (importance ≥ 6) | always know who they are and what to avoid |
| **Semantic** (vector) | `memory_vec` top 50 → re-ranked, top 8 | what's relevant to *this* message |
| **Entity hop** | names/aliases in the message + `entity_vec`, then `(m)-[:ABOUT]->(e)` | "how's my sister?" pulls everything about Maya |
| **Follow-ups** | open plans due within ±14 days | proactive "how did it go?" |
| **Recent episodes** | last 3 summaries + mood | continuity between sessions |

> **Term:** *Semantic* in this table is the retriever's name: it finds memories by vector
> similarity ([semantic, sense 2](../docs/glossary.md#semantic)), and returns events and
> feelings as readily as facts. It isn't *semantic memory* (sense 1). The core profile is the
> closest thing to semantic memory here.

Semantic re-ranking (the "generative agents" recipe) mixes how well a memory matches the
message with how important and how recent it is. Lesson 13 replaced the original Cypher
expression with a tested Python function, [`decay.rank()`](../companion/decay.py):

```
rank = 0.6 × similarity + 0.4 × importance/10 × 0.5 ^ (age / half-life of its kind)
```

Memories that have faded below a threshold ("dormant") are left out entirely. See
[Lesson 13](13-memory-over-time.md) for the half-lives and why age counts from the last time
the user confirmed something, not the last time it was recalled.

Filters applied **in Cypher**, not left to the prompt:
- retired memories (`invalidated_at`) and finished plans (`status = 'done'`) are skipped
- entities a boundary is `ABOUT` never appear in the people list or the entity hop

Recalled memories get `last_accessed`/`access_count` updated, for analytics. (Since Lesson 13
this no longer keeps a memory "fresh": only the user confirming it again resets its decay.)

### Boundary filter test

```
memories: boundary "does not want to discuss their ex, Jordan" (about Jordan) + an event about Jordan
recall("I ran into Jordan today"):
  people:     ['Maya']                       ← Jordan excluded
  entity hop: []                             ← even though the message names Jordan
  core:       ['Boundary Test does not want to discuss their ex, Jordan']   ← rule stays visible
```

## The companion graph — [`companion/graph.py`](../companion/graph.py)

```
START ──► recall ──► respond ──► remember ──► END
```

> Lesson 11 extends this to `recall → respond → log → (worth remembering?) → remember`,
> adds an end-of-episode extraction pass, and fixes gaps found by scenario tests. The outputs
> below were recorded with the Lesson 10 version.

| Node | Does | Which memory |
|---|---|---|
| `recall` | `store.recall(last message)` → formatted memory block | long-term (Neo4j) |
| `respond` | persona + memory block + last 16 messages → LLM | short-term (checkpointer) |
| `remember` | log turns to the Episode, extract, `store.apply()` | writes long-term |

State extends `MessagesState` with `memory_context` and `memory_log`. The episode id is the
LangGraph `thread_id`; a `today` value can be passed in `config["configurable"]` (the demo uses
it to simulate a week passing).

Two extras make it feel like a companion rather than a chatbot:
- **`greeting()`** opens each session from memory (follow up on due plans, or pick up from last
  time), and `seed_thread()` puts it into the thread with `app.update_state(..., as_node="respond")`.
- **Episode summaries** at `/new` or `/quit` become the diary used for continuity.

## Terminal chat

```bash
uv run python -m companion                  # uses COMPANION_USER / COMPANION_NAME from config.py
uv run python -m companion --user Sam --debug   # --debug prints the memory block each turn
```

| Command | |
|---|---|
| `/memories [kind]` | current memories (ids, kind, importance, status) |
| `/all` | including retired ones |
| `/people` | the life graph |
| `/timeline` | events and plans by date |
| `/history <id>` | a memory and the versions it replaced |
| `/forget <id>` | hard-delete a memory |
| `/recall <text>` | show exactly what would be recalled for a message |
| `/debug` | toggle the memory block display |
| `/new` | save this episode (summary + mood) and start a new one |
| `/quit` | save and exit |

Smoke test (piped input):

```
Sage: Hi CLI Test, I'm Sage. I'd love to get to know you — how's your day going?
CLI Test: I just started learning the cello, my teacher is named Ines.
Sage: That's exciting! The cello has such a beautiful, rich sound. How are you finding it so far? ...
  ✎ linked user -student_of-> Ines
  ✎ + [fact] CLI Test started learning the cello on 2026-11-05.
CLI Test: /memories
  m_584be0  [fact] CLI Test started learning the cello on 2026-11-05.  ★5
CLI Test: /people
  Ines — student_of (person)
CLI Test: /quit
  ✎ episode saved: On 2026-11-05, CLI Test shared that they have begun learning the cello ... (mood: Excited and optimistic)
```

## The two-session demo — [`companion/demo.py`](../companion/demo.py)

```bash
uv run python -m companion.demo
```

It uses a separate user, `Alex (demo)`, and wipes it first. Full output:
[outputs/10-companion-demo.txt](outputs/10-companion-demo.txt). Highlights:

**Session 1 (2026-11-05)** — Alex introduces themselves: Portland, new UX research job, sister
Maya, beagle Biscuit, nervous about a presentation "next Thursday", asks not to bring up their ex.

**Session 2 (2026-11-13)** — the companion opens with:

```
Sage: Hey Alex! How did your big presentation to leadership go yesterday?
```

It remembered the plan, knew its date was 2026-11-12, and knew today is the 13th. Then:

```
Alex: Maya's visiting for Thanksgiving. Any ideas for things to do around Portland that work with a dog?
Sage: ... Since you're bringing Biscuit along, you could try Forest Park ...
```

It connected Maya, Portland and the dog without being told the dog's name in this session.

**Final state:**

```
/people
  Lumen Health — works_at (organization)
  Maya — sister (person)
  Biscuit — pet (pet)
  Portland — lives_in (place)

/timeline
  2026-11-12  [event] Alex received high praise from their VP for a research readout presentation on 2026-11-12 ...
  2026-11-12  [plan] Alex is giving their first big presentation to leadership on 2026-11-12. [done]
  2026-11-26  [plan] Alex's sister Maya is visiting Alex in Portland for Thanksgiving on 2026-11-26. [open]

Superseded facts
  now: Alex works on the Research Ops team at Lumen Health.
  was: Alex works as a UX researcher at Lumen Health.
```

**What the LLM sees** for `"what should we do when my sister visits?"`:

```
Core profile:
- Alex does not want the companion to bring up their ex, Jordan.
- Alex moved to Portland in October 2026.
- Alex's sister Maya is a nurse living in Denver.
- Alex and Maya are very close.

People and things in their life:
- Lumen Health (works_at)
- Maya (sister)
- Biscuit (pet)
- Portland (lives_in)

Open plans to follow up on (today is 2026-11-13):
- Alex's sister Maya is visiting Alex in Portland for Thanksgiving on 2026-11-26. (due 2026-11-26)

Relevant memories:
- [fact] Alex has a beagle named Biscuit.
- [event] Alex received high praise from their VP for a research readout presentation on 2026-11-12 ...
- [fact] Alex works on the Research Ops team at Lumen Health.

Recent conversations:
- 2026-11-13 (mood: Triumphant and excited): Alex shared the great news that their VP praised ...
- 2026-11-05 (mood: Excited but anxious): Alex introduced themselves, sharing that they recently moved ...
```

## Fixes the first demo run found

The first run (before these fixes) showed five problems. Each one is now handled in code or prompt:

| Problem seen | Fix |
|---|---|
| Plan stored as "…next Thursday" | extractor rule: absolute dates only |
| Core profile only had the boundary (threshold ★8) | core = boundaries + top 8 facts/preferences ★6+ |
| Both episodes dated with the real clock | `start_episode(on=...)` for simulated dates |
| Finished plan still in "Relevant memories" | skip `status = 'done'` in recall |
| The ex appeared in the people list | exclude entities a boundary is `ABOUT` |

## Concepts and design choices

*Where this fits:* Steps 8, 9 and 12: hybrid recall, ranking, and the memory block in the prompt. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [recall](../docs/glossary.md#recall) | sense 1: our `recall()`, run before every reply | the recall@k metric; MemGPT's *recall memory* |
| [semantic](../docs/glossary.md#semantic) | sense 2: the **Semantic** row is the vector retriever, and can return events and feelings too | semantic memory (facts only) |
| [core profile](../docs/glossary.md#profile) | sense 1: boundaries + top facts, always in the prompt | an embedding profile; LangMem's profile memory |
| [re-ranking](../docs/glossary.md#rerank-and-relevance) | sense 1: our formula, match × importance × recency | a cross-encoder reranker (Lesson 18) |
| [graph](../docs/glossary.md#graph) | sense 2 for `companion/graph.py` (LangGraph); sense 1 for recall's hops | each other |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Recall every turn, in code, from several retrievers | the companion notices plans and boundaries nobody asked about | every turn pays for recall, even "hi" | a `search_memory` tool the model calls when it wants | [D10](../docs/design-decisions.md#d10) |
| Vector + graph hop (hybrid sense 2) | "how's my sister?" reaches everything about Maya | no keyword retrieval | add BM25 and rank fusion | [D11](../docs/design-decisions.md#d11) |
| A transparent ranking formula | explainable and unit-testable | hand-picked weights | cross-encoder or learned ranking | [D12](../docs/design-decisions.md#d12) |
| Labelled plain-text sections in the prompt | section names tell the model *why* each item is there | wording changes behaviour; re-run scenarios | JSON blocks; an LLM-written summary | [D13](../docs/design-decisions.md#d13) |
| A fixed LangGraph workflow | predictable, testable memory behaviour | no adaptive retrieval mid-turn | a ReAct agent with memory tools | [D19](../docs/design-decisions.md#d19) |

## Test it — [`tests/test_10_recall.py`](../tests/test_10_recall.py)

- **Table-driven tests** for the small-talk gate: each bug becomes a new row.
- **Recall** against hand-written memories: semantic hits, boundaries always in core,
  sensitive memories kept out of core, plans in follow-ups.

Writing these found the worst bug in the repo. `NOT_PROTECTED` hides memories that share an
`ABOUT` entity with a boundary. But the user node is an `:Entity` too. A topic boundary
("don't ask about my job") is ABOUT the user, so it **hid every memory about the user**: plans,
facts, everything. The fix excludes `:User`.

The next test found a second bug in the same snippet. Its variable `e` collided with an
`e` already bound in the query it was pasted into, so inside the `EXISTS` "any entity"
silently became "this entity". A protected person then leaked through the entity hop and
through "No longer true".

> **Later, in [Lesson 20](20-growing-the-life-graph.md):** the entity hop here reads only each entity's relation *to the user*. Lesson 20 adds a bounded two-step walk over links between entities (Logan -likes-> Minecraft, found from either end) and a "How they connect" section in the prompt.

## Next

The first demo passed easy cases. [Lesson 11](11-harder-scenarios-and-evals.md) writes harder multi-session conversations and turns them into evals.

<!-- nav -->
---

**[Part 3: A companion that remembers](../README.md#part-3-a-companion-that-remembers)** (lessons 8–11)

[← Lesson 9: The write path: from conversation to graph](09-memory-write-path.md) · [All lessons](../README.md#lessons) · [Lesson 11: Harder scenarios, and testing memory with scenarios →](11-harder-scenarios-and-evals.md)
