# Lesson 11 — Harder scenarios, and testing memory with scenarios

Lesson 10's companion handles the happy path. Real life is messier: facts arrive in pieces,
people correct themselves, relationships end, two friends share a name, and people ask you to
forget things. This lesson:

1. changes **when and how** memories are extracted,
2. adds the graph and recall features the harder cases need,
3. builds a **scenario test suite** that drives the real companion through multi-session
   conversations and checks the graph and the replies.

```bash
.venv/bin/python -m companion.scenarios                 # all 9 scenarios (~5 min, real LLM calls)
.venv/bin/python -m companion.scenarios forget sam      # only matching scenarios
```

Full recorded run: [outputs/11-scenarios.txt](outputs/11-scenarios.txt) — **35/35 checks**
(gemma-4-31B-it on DeepInfra).

## 1. When does extraction run?

```
START → recall → respond → log ─(worth remembering?)─► remember → END
                                 └──────── no ─────────────────────┘
/new or /quit → close_episode(): end-of-episode pass → diary summary
```

| Pass | When | Sees | Catches |
|---|---|---|---|
| **per turn** (`remember` node) | after each reply, unless it's small talk | latest exchange **+ 6 earlier messages as context** | most facts, right away |
| **end of episode** (`close_episode`) | `/new`, `/quit`, end of a scripted session, if ≥ 2 user messages | the whole transcript | facts spread over turns, mistakes to fix |

### Should the extractor look further back than the last message?

Yes, but **as context, not as source**. The prompt says: record what's new in the LATEST
exchange; use earlier turns only to work out who "she" or "it" is.

| Option | Problem |
|---|---|
| Only the latest exchange (Lesson 10) | "She just got engaged!" — who is *she*? |
| Re-extract the whole window each turn | the same facts get re-extracted every turn: duplicates, cost, drift |
| **Latest exchange + earlier turns as context** ✅ | — |
| **+ an end-of-episode pass** ✅ | catches what only became clear later; fixes earlier mistakes |

```
Riley: My best friend from college, Priya, called me today.
   ✎ linked user -friend-> Priya
Riley: She just got engaged! The wedding is on June 12 next year.
   ✎ + [event] Riley's best friend Priya got engaged on 2026-11-02.     ← "she" resolved from context
   ✎ + [event] Priya's wedding is scheduled for 2027-06-12.
Riley: haha yeah
   (small talk — extraction skipped)
```

### Should the chat model decide when to save (a memory tool)?

Both styles are common:

| | Always-on extractor (this tutorial) | Chat model calls `save_memory` itself (MemGPT/LangMem "hot path") |
|---|---|---|
| Catches facts said in passing | ✅ | ❌ models under-save |
| Explicit "remember this" / "forget X" | ✅ (`forget_entities`) | ✅ and can confirm immediately |
| Cost | one extra call per turn | only when called |
| Reply quality | persona prompt stays focused | persona does two jobs |
| Small/open models | reliable | inconsistent tool use |

A good combination: the always-on extractor as the safety net, a **cheap conditional edge**
to skip turns with nothing to remember (`worth_remembering()`: "lol ok", "thanks!" — a regex,
no LLM call), and optionally a tool on the chat model for explicit requests.

## 2. The scenarios and what each needed

| Scenario | What had to change |
|---|---|
| **cross_turn_facts** | earlier turns as context; end-of-episode pass |
| **correction** ("actually it's Tuesday") | date-aware dedupe (below) |
| **history** ("where did I live before?") | graph walk over `SUPERSEDES`; "a change is an event AND a new fact" rule; anti-echo rule |
| **relationship_change** | `ended` relations (`r.until`); boundary filter on *all* recall paths; boundary tripwire |
| **same_name** (two Sams) | entity ids shown to the extractor; `ref` / `new` on entities |
| **recurring_date** | `annual_date: "MM-DD"` + "yearly dates coming up" in recall and greeting |
| **forget_person** | `forget_entities` → delete entity + memories, redact turns and summaries |
| **family_multi_hop** | relation **groups** ("family" → sister, aunt, cousin…) in the entity hop |
| **sensitive** | `sensitive` flag; never in core; recalled only as the best match |

### Same name, different person

The extractor now sees known entities **with ids and relations**:

```
e_3f1a2b: Sam (person, coworker) — coworker who takes credit for Riley's work
```

and returns `ref` (this is a known one) or `new: true` (same name, different person).

```
Riley: My cousin Sam is getting married in May!
   ✎ new entity Sam (e_4cb9a2), distinct from the known Sam
   ✎ linked user -cousin-> Sam
Riley: Sam from work did it again today, took credit for my slides in the team meeting.
   ✎ + [event] Riley's coworker Sam took credit for Riley's slides ...   ← ABOUT the coworker Sam
```

### Relationship changes

Relations get an `until` date instead of being deleted, so the graph keeps history:

```
Riley: Dana and I broke up last weekend.
   ✎ ended user -partner-> Dana
   ✎ plan done m_c6c16b                          ← the Lisbon trip
Riley: Please don't bring her up anymore, it hurts.
   ✎ + [boundary] Riley does not want the companion to bring up Dana because it is painful.
```

```cypher
MATCH (:User {id:'user:eval relationship_change'})-[r:RELATED]->(e) RETURN e.name, r.type, r.until
// Dana  partner  2026-11-16
// Dana  ex       null
```

**Bug found by this scenario:** Lesson 10's boundary filter hid the *person* (people list,
entity hop) but a *memory about them* — "trip to Lisbon with Dana" — could still come back
through vector search. Now every recall path applies:

```cypher
NOT EXISTS {
  MATCH (m)-[:ABOUT]->(:Entity)<-[:ABOUT]-(b:Memory {kind: 'boundary'})
  WHERE b.invalidated_at IS NULL AND b <> m }
```

**LLM miss found by this scenario:** in 2 of 3 runs the per-turn extractor spent its output on
the relationship change and never recorded the boundary. A stronger prompt rule wasn't
enough. Boundaries are safety-critical, so there's now a **regex tripwire**: if the message
looks like a boundary ("don't bring her up", "stop asking about…") and none was extracted,
extract once more with an explicit hint. After that: 3/3.

> Pattern: for the memories that matter most, don't rely on one LLM call. Pair it with a
> cheap deterministic check.

### History: "where did I live before Chicago?"

Vector search can't express *before*. Scores for that question:

```
0.797  Riley moved to Chicago in November 2026.        (current)
0.726  Riley lives in Austin.                          (retired)  ← lower than half the others
```

So history is a **graph walk**: take the best-matching current memories and follow
`SUPERSEDES` back to their old versions.

```cypher
MATCH (cur:Memory)-[:SUPERSEDES*1..3]->(old:Memory) WHERE cur.id IN $best_ids
RETURN old.text, date(old.invalidated_at) AS until, cur.text AS replaced_by
```

```
No longer true (history, not current):
- until 2026-11-20: Riley lives in Austin. → now: Riley lives in Chicago.

Riley: Remind me, where did I live before Chicago?
Sage:  You lived in Austin for about five years before making the move!
```

Two extraction problems surfaced here, each fixed:
- The move was stored only as an **event**, leaving "lives in Austin" current. New rule: a
  change is an event **and** a new fact that supersedes the old one.
- **Echo:** after the companion *answered* "you lived in Austin", the extractor stored that
  answer as a new memory. Fixes: a "don't echo recall" rule; the end-of-episode pass labels
  companion lines "context only — not a source"; and it only runs when there are ≥ 2 user
  messages (a single question has nothing cross-turn to find).

### Corrections and date-aware dedupe

```
Riley: I have a dentist appointment on Thursday.
   ✎ + [plan] Riley has a dentist appointment on 2026-11-12.
Riley: Oh wait, I just checked — it's actually on Tuesday the 17th.
   ✎ + [plan] Riley has a dentist appointment on 2026-11-17.  (replaces m_5f3070)
   ✎ (episode pass) reinforced m_682517
```

Before the fix, the episode pass wrote an *identical copy* that "replaced" the corrected plan,
because dedupe ignored anything listed in `supersedes`. But a real correction also reads
almost the same ("…on 2026-11-12" vs "…on 2026-11-17", similarity > 0.975). So a duplicate is
now: near-identical text **and the same dates**.

### Yearly dates

```
2026-11-02  Riley: My mom's birthday is December 2nd.
   ✎ + [fact] Riley's mother's birthday is on December 2nd. (yearly 12-02)
2026-11-28  Sage: Hi Riley! ... Your mom's birthday is coming up on December 2nd—did you
                  want to start brainstorming some plans for her?
```

`recall(today=2027-11-25)` lists it again for 2027-12-02: a yearly date never becomes "done".

### Forgetting a person

```
Riley: Actually, please forget everything about Jordan. I don't want you remembering him at all.
   ✎ forgot Jordan: 1 memories deleted, 3 turns and 0 summaries redacted
```

`forget_entity()` deletes the entity and every memory about **or naming** it, and replaces
the name in stored turns and summaries with `[forgotten]`. Also `/forget-entity <name>` in
the CLI. Limits worth knowing: the in-session LangGraph checkpointer still holds this
conversation's messages until the session ends, and a memory that refers to the person
without naming them ("my ex texted me") is not caught.

### "Anyone in my family…?"

The entity hop now maps group words to relation types:

```python
RELATION_GROUPS = {"family": {"mother", "sister", "brother", "aunt", "cousin", ...},
                   "friends": {...}, "work": {"coworker", "boss", ...}, "pets": {...}}
```

```
Riley: Is anyone in my family in healthcare? I have a question about mixing two medications.
Sage:  Your brother Leo is a paramedic and your aunt Rosa is a pharmacist, so you definitely
       have some healthcare experts in the family! ...
```

### Sensitive memories

The extractor flags health, diagnoses, medication, sexuality, religion, finances, legal trouble
and trauma as `sensitive`. They are never in the core profile, and the semantic search includes
them only when they are **(near) the best match** for the message.

Why relative, not a fixed score? With `bge-small` every score falls in a narrow band:

```
"I'm struggling to focus, my meds don't help"   ADHD memory 0.781  ← top match
"What should I do this weekend?"                ADHD memory 0.691  (top: hiking 0.768)
```

A fixed cutoff that works for one query fails another; "is it the top hit?" works for both.

## 3. The scenario harness — [`companion/scenarios.py`](../companion/scenarios.py)

```python
Scenario(
    "same_name", "Two different people named Sam stay separate",
    [(D(2026, 11, 2),  ["My coworker Sam keeps taking credit for my work. So annoying."]),
     (D(2026, 11, 9),  ["My cousin Sam is getting married in May!"]),
     (D(2026, 11, 16), ["Sam from work did it again today, took credit for my slides ..."])],
    [("two Sam entities", lambda r: r.q("MATCH (e:Entity {user:$user, name:'Sam'}) RETURN count(e) AS n")[0]["n"] == 2),
     ("the slides incident is about coworker Sam", lambda r: bool(about_entity(r, "Sam", "slides", "coworker"))),
     ...]),
```

- Each scenario gets its own user (`eval <name>`), wiped first, with **simulated dates**.
- Checks mix **graph assertions** (Cypher) and **behavior** (what the companion said).
- LLM output varies: run several times before trusting a pass rate. Every bug in this
  lesson came from reading transcripts, including ones where all checks passed (the
  duplicate dentist plan).

```
=== Summary
  cross_turn_facts       4/4
  correction             2/2
  history                5/5
  relationship_change    4/4
  same_name              4/4
  recurring_date         3/3
  forget_person          6/6
  family_multi_hop       3/3
  sensitive              4/4
  total                  35/35
```

## Takeaways

- Extract per turn **with context**, then do a second pass per conversation.
- Gate the expensive step with a cheap edge; protect critical memories with a deterministic tripwire.
- Use the graph for what embeddings can't do: *before/after* (`SUPERSEDES`), *who* (ids for
  same-name people), *groups* (relation types), *what's protected* (boundaries).
- Small embedding models compress scores: prefer relative thresholds.
- Test memory with multi-session scenarios that check both the graph and the replies, and read the transcripts.
