# Lesson 13 — Memory over time: expiry, decay and consolidation

"I'm heading to the gym" used to become a permanent memory:

```
[event] Riley went to the gym on 2026-11-18.     importance 2, never expires
```

A year later, recall could still bring it up. Before this lesson the only sense of time was:
- a small recency nudge in ranking;
- a 14-day window for plan follow-ups;
- facts ending when something superseded them.

Nothing ever dropped out because of age.

The fix is not one TTL. There are three different questions about time:

| Question | Mechanism | "At the gym" | "Felt tired" | "Lives in Chicago" |
|---|---|---|---|---|
| Is it still **true**? | **validity**: `valid_until` | until ~14:00 | until midnight | until superseded |
| Is it still **worth bringing up**? | **decay**: strength over time | a day | a couple of weeks | always (but may go *stale*) |
| Should it still be **stored this way**? | **consolidation** | 3 visits → "works out regularly" | pruned once dormant | kept |

## 1. One clock — [`companion/clock.py`](../companion/clock.py)

Everything time-dependent asks one function what "now" is:
- in a live chat, the real time;
- in the demo, the scenarios and the tests, noon on the simulated day.

Nothing calls `datetime.now()` directly, so a test can store a memory on day 1 and recall it
on day 40 in the same second, with no sleeping and no mocking of datetime.

The prompts now say *"It is now 2026-11-18 12:00 (Wednesday)"*. The extractor needs the time
of day to work out "about two hours from now".

## 2. Validity: `valid_until`

There's a new memory kind, `status`: a temporary situation ("at the gym", "at the airport",
"in Denver for the week"). The extractor gives it a `valid_until`. A feeling about today gets
the end of today. Lasting things get none.

```
"Off to the gym, talk later!"         -> [status] Riley is heading to the gym (until 2026-11-18 14:00)
"Morning! I slept terribly though."   -> [feeling] Riley slept terribly on 2026-11-19 (until 2026-11-19 23:59)
"I live in Chicago now."              -> [fact] Riley lives in Chicago (no end)
```

`clock.parse_until()` makes it safe:
- a date means the end of that day;
- a naive time is the user's local time;
- a status *always* ends: if the time is missing, unreadable or already past, it lasts 3 hours.

**Every recall path filters on it,** next to the old retirement check:

```cypher
m.invalidated_at IS NULL AND (m.valid_until IS NULL OR m.valid_until > $now)
```

That's the bi-temporal model from Lesson 8, now with both axes:

| | When it was true (valid time) | When we knew it (transaction time) |
|---|---|---|
| start | `occurred_on` (events) | `created_at` |
| end | **`valid_until`** (new) | `invalidated_at` (superseded or retired) |

`memories(today=...)` is an **as-of query**: it also requires `created_at <= now`. A status
stored on day 3 was never "current" on day 1, even though its `valid_until` is later.

**Expired isn't erased.** A status that ended in the last 24 hours goes into a new context section:

```
Recently, no longer true (you could ask how it went):
- until 2026-11-18 14:00: Riley is heading to the gym
```

The next morning's greeting, from the scenario run: *"Hi Riley! It's good to see you. How did
your gym session go yesterday?"*

Boundaries can expire too: "don't talk about work **today**" gets a `valid_until`, and the
next day the topic is open again.

### Why not a database TTL?

Neo4j has no built-in TTL. APOC has one (`apoc.ttl.enabled=true` plus `apoc.ttl.expireIn`),
which **deletes** nodes on a schedule. Filtering at read time is better here:
- the memory stays as history ("how was the gym on Tuesday?");
- the "Recently" follow-up needs expired statuses for another day;
- consolidation needs the repeats to find a pattern.

Use hard TTL deletion where *not keeping* data is the point (privacy, data minimization).
Pruning (section 4) is this repo's version of that, and it's selective.

## 3. Decay — [`companion/decay.py`](../companion/decay.py)

Validity says whether something is *true*. Decay says whether it's still *worth bringing up*:

```
strength = importance/10 × 0.5 ^ (age / half-life of its kind)
```

| Kind | Half-life | Importance 2 is dormant after | Importance 6 |
|---|---|---|---|
| status | 1 day | 2 days | ~4 days |
| feeling | 7 days | 14 days | ~25 days |
| event | 90 days | 180 days | ~320 days |
| fact, preference, boundary, plan | — | never | never |

- **Dormant** (strength < 0.05): kept in the graph, left out of recall. Milestones
  (importance ≥ 8) never go dormant.
- **Age counts from the last *confirmation*.** When the user says it again, `apply()`
  reinforces it and the clock resets, like spaced repetition. *Recall* doesn't reset it;
  otherwise anything recalled once would keep itself alive.
- **Ranking** is `0.6 × match + 0.4 × strength`. Strength breaks ties; it doesn't filter: an
  old memory that matches the question well still beats a new vague one.
- **Facts don't decay; they go stale.** "Works at Acme" isn't less true after a year, but it
  might have changed. Facts not confirmed in over a year get a marker, and the persona checks
  casually instead of asserting: *(not confirmed in over a year: may be out of date, check
  before relying on it)*.

The ranking used to be a Cypher expression. It's now Python: Cypher fetches the candidates,
`decay.rank()` orders them. **Rules you want to unit-test belong in plain functions.**
`test_strength_halves_every_half_life` is three lines.

## 4. Consolidation — `MemoryStore.consolidate()`

A background job runs after every conversation (`close_episode`) and from `/consolidate`:

1. **Lapse.** Open plans more than 14 days past due become `lapsed`, so the extractor stops
   seeing them as candidates.
2. **Roll up.** An activity (an `activity` entity such as "gym") with occurrence memories
   (events, statuses, plans, feelings from the last 60 days) `ABOUT` it on **3 or more
   distinct days** is a habit, as long as at least one of them happened (three plans from one
   message are not a habit). Its memories go to a small LLM step,
   `extract.find_pattern()`, which writes one lasting memory or declines. The pattern is
   `ABOUT` the activity, the user gets a `practices` link to it, each member gets a
   `SUMMARIZES` edge and a `consolidated_at` stamp, and the pattern's `evidence` grows by
   each memory it newly summarizes (pruning a member later doesn't lower it). A memory about
   two activities counts for each one's habit, even when the second forms later. Once an activity has a pattern, later days attach to it
   as more evidence, without asking the LLM for a new sentence.
3. **Prune.** Delete dormant memories with importance ≤ 3 that have **expired** (`valid_until`
   has passed), but only once they're rolled up or older than the 60-day window. Events stay
   as history even when dormant: they have no `valid_until`.

```
✎ + [status] Riley is at the gym before work (until 2026-11-23 15:00)
✎ (upkeep) pattern from 3 memories: + [fact] Riley goes to the gym regularly.
✎ (upkeep) pruned 1 dormant expired memories
```

This is the episodic → semantic step from the [glossary](../docs/glossary.md#semantic)
(*semantic memory*, sense 1: general knowledge, not vector search): many episodes become one
piece of general knowledge.

**A habit is counted in the graph, not guessed from wording.** `_habits()` is one Cypher
query: activity ← `ABOUT` ← memory, grouped by activity, keeping groups whose memories fall on
3+ distinct days. A memory's day is when it happened (`occurred_on`, or a plan's `due_on`
once that day has come), else the day it was said. So:

- several mentions on one day ("heading to the gym", then "back from the gym") count once;
- a plan still in the future isn't a day yet: "gym Mon, Wed, Fri next week" is one message,
  not a habit;
- `valid_until` is when a status *ends*, not another day it happened;
- a memory about two activities ("gym, then yoga") counts for both.

No similarity threshold is involved. Whether three sentences are "about the gym" was decided
once, at extraction, when each was linked to the `gym` entity (Lesson 9, rule 10).

**What this replaced, and why.** The first version found habits by *clustering* sentence
embeddings: complete linkage at a measured cut (0.82 for bge-small), so a cluster couldn't
chain gym → climbing → hiking. It worked for the sentences it was measured on. Then the
extraction model changed (`gemma-4-31B-it` → `gemini-3.8-flash`) and the same scenario failed:
the new model wrote longer sentences with dates in them ("…went to the gym before work on
2026-11-23 to keep up a workout streak"). No pair of gym memories reached 0.82 any more, and a
dated rock-climbing memory scored *closer* to a dated gym memory (0.825) than the gym memories
did to each other: the shared date format made unrelated sentences look alike. One visit was
also filed as a `plan`, a kind the clustering never looked at.

Lowering the cut would have passed the test and broken again on the next model. The fix
moved the decision out of the wording: dates live in their fields (Lesson 9, rule 2), the
activity is an entity, and a habit is a count. The full analysis is in
[design decisions](../docs/design-decisions.md#d22).

## What the tests and evals found

- **"places" was used as a relation type.** Asking about "I'm at the airport waiting for my
  flight to Denver" produced `user -places-> Denver`, a *group heading* from the ontology list.
  Group names are now rejected in `normalize()`, and the prompt says they're headings.
- **A trip almost moved the user.** "I'm in Denver for the week" came back with
  `user -lives_in-> Denver`. Because `lives_in` is one-at-a-time (Lesson 12), that would end
  their real home. Adding a prompt *rule* didn't change the output; a prompt *example* did. A
  store guard now also ignores a one-at-a-time relation that only a status mentions.
  Again: let code back up the prompt where a mistake is costly.
- **Pruning ate the pattern.** The first version pruned a minor status after ~2 days, before
  three visits could pile up over a week. It also deleted pattern members, so the scenario's
  "evidence ≥ 3" check failed. Pruning now waits for consolidation or the window, and the
  pattern keeps an `evidence` count.
- **The as-of filter hid the scenarios' memories.** Scenario dates (late November 2026) were
  after the real today, so a default `created_at <= now` made their memories invisible. The
  as-of cut now applies only when a `today` is given.
- **A closed episode whose summary failed was left open**, never linked by `NEXT`. That was a
  code-review leftover, fixed here with a test.

## Concepts and design choices

*Where this fits:* Steps 9 and 13: decay in ranking, validity at read time, consolidation offline. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| valid time / transaction time | `valid_until`, `occurred_on` / `created_at`, `invalidated_at` | each other (the bi-temporal split) |
| [forget](../docs/glossary.md#forget) | senses 3 and 4: decay ranks lower; pruning deletes trivia | hard delete (sense 1) |
| [semantic](../docs/glossary.md#semantic) | sense 1: consolidation turns repeated *episodic* memories into one *semantic* habit | vector search |
| consolidation | offline: patterns, lapsed plans, pruning | summarizing one conversation |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Three mechanisms for three questions about time | true? worth mentioning? stored this way? | three things to tune | one TTL that deletes; recency-only ranking | [D14](../docs/design-decisions.md#d14) |
| Filter expiry at read time instead of a database TTL | history, follow-ups and consolidation still need expired items | every read filters | APOC TTL (right when deletion is the goal) | [D14](../docs/design-decisions.md#d14) |
| An injectable clock | time-travel is a parameter, tests are deterministic | every function takes `today`/`now` | wall-clock time everywhere | [D20](../docs/design-decisions.md#d20) |
| A habit is memories ABOUT one activity on 3+ days, counted in Cypher | doesn't depend on how the extractor words or dates sentences | the extractor must link each memory to the activity | clustering sentence embeddings (the first version); LLM-only grouping; reflection | [D22](../docs/design-decisions.md#d22) |

## Test it — [`tests/test_13_time.py`](../tests/test_13_time.py)

```bash
uv run python -m pytest tests/test_13_time.py -v                # 25 tests, ~2 s, no LLM
EVAL_TRIALS=3 uv run python -m pytest -m llm tests/test_13_time.py
uv run python -m companion.scenarios situational
```

- **Unit:**
  - `parse_until` as a table (date → end of day, missing → 3 hours, past → default);
  - the decay curve and dormancy;
  - "a strong old match still outranks a weak new one".
- **Store, time-travelling with `today=`:**
  - a status is current at noon, *recent* the next day and gone two days later;
  - `memories()` as of a date;
  - a one-day boundary expires;
  - an old minor feeling goes dormant, unless it was confirmed again;
  - an old fact is marked stale;
  - the roll-up, using a **fake pattern writer**: 3 days make a habit, one day of gym talk
    doesn't, future plans aren't days yet, plans alone aren't a habit, a known habit gathers evidence
    without a rewrite and keeps it when members are pruned, a shared memory still counts for
    a later habit;
  - pruning only expired, dormant trivia;
  - plans lapse;
  - a trip doesn't end the real home.
- **LLM evals:**
  - "off to the gym" is a status ending within 6 hours;
  - "tired today" ends today;
  - a week's trip ends within the week and adds no `lives_in`;
  - facts never expire;
  - the pattern writer finds a habit and declines coincidences.

  Every one passed 3 runs out of 3.
- **Scenario `situational`, over five days:**
  1. "Heading to the gym" → a status that expires.
  2. The next day nothing says they're at the gym, and the greeting asks how it went.
  3. "Back from the gym, legs are dead".
  4. "Off to the gym again after work".
  5. "Gym again before work" → a lasting habit with evidence ≥ 3.

`consolidate()` takes the pattern writer as an argument (`writer=find_pattern` in
production, a lambda in tests). **Passing the LLM in** is what lets the counting, linking
and pruning be tested exactly and for free, while the LLM part gets its own eval.

### Results

Full run: [outputs/13-scenarios.txt](outputs/13-scenarios.txt).

| Suite | Result |
|---|---|
| Fast tests (`pytest -q`, no LLM) | 129 passed |
| Lesson 13 LLM evals, 3 runs each | 6/6 |
| Extraction evals (Lesson 11), `EVAL_TRIALS=3` | 10/10 |
| Scenarios, all 13 | 49/49 (after two fixes, below) |

The habit check failed twice, for two different reasons. (This was the clustering version;
[above](#4-consolidation--memorystoreconsolidate) is why it was later replaced.)

**1. The input didn't contain three visits.** The third gym message was "just got back from
the gym, my legs are dead". The extractor (reasonably) stored it as a *feeling*: "Riley's legs
feel exhausted after a workout". It has no "gym" and is only 0.73 similar to the gym statuses.
Lowering the threshold would have "fixed" it and let unrelated memories chain together.
Instead the scenario got a clearer extra visit ("off to the gym again after work").

**2. Then the clustering lost a group that was there.** On the next run there were four gym
statuses and still no pattern. The first-fit clustering went through them in time order:

```
heading to the gym   + just got back   0.86  -> pair A
going after work     vs just got back  0.80  -> can't join A, starts pair B
at the gym before    vs just got back  0.81  -> can't join A, joins B
```

That makes two pairs and no group of three. Yet heading / after work / before work are 0.85,
0.90 and 0.85 to each other. Finding the largest group first fixed it, and a regression test kept these exact sentences
(until the graph count replaced clustering altogether). The full run then gave 49/49, and the gym habit was the only pattern formed in
any scenario:

```
✎ (upkeep) pattern from 3 memories: + [fact] Riley goes to the gym regularly.
```

When a scenario fails, first ask whether the *input* really contained what the check expects.
If it did, the next suspect is an order-dependent shortcut in the code. A real LLM run varies a
little each time, which is how it found a case the fixed fake data in the unit tests never did.

## Takeaways

- Time in memory is three questions: **true?** (validity), **worth mentioning?** (decay),
  **stored this way?** (consolidation). One TTL can't answer all three.
- Make the clock an input. Then time-travel is a parameter, and time-dependent tests are
  deterministic.
- Filter at read time and keep history; delete only on purpose (privacy, trivia), and only
  after consolidation has had its chance.
- Facts don't fade, they go stale: check them instead of dropping them.
- Measure thresholds on real examples. Better still, decide identity ("is this the gym?")
  once, as a link in the graph, and count links instead of comparing sentences.

> **Later, in [Lesson 20](20-growing-the-life-graph.md):** consolidation gets a fourth step that drops entities with no memory and no link left, and edges end when every memory that supported them is retired.

## Next

[Lesson 14](14-production-hardening.md) makes the write path safe for failures and concurrency, and moves extraction to the background.

<!-- nav -->
---

**[Part 4: A smarter memory](../README.md#part-4-a-smarter-memory)** (lessons 12–13)

[← Lesson 12: An ontology for the life graph](12-ontology.md) · [All lessons](../README.md#lessons) · [Lesson 14: Production hardening: transactions, locks and background extraction →](14-production-hardening.md)
