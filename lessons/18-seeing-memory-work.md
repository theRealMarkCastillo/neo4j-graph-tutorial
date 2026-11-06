# Lesson 18: watching memory work, one turn at a time

Lesson 17's page shows what memory **holds**: the graph, the decay curves and the related pairs. Each turn
also does two things you can't see from the result:

- a **read**: recall picks a few memories, and they become text in the prompt;
- a **write**: extraction proposes memories, and the store keeps, merges or drops each one.

This lesson makes both visible. Click any reply in the chat and the right-hand side shows
what memory did for that reply.

```bash
.venv/bin/python -m companion.web        # http://127.0.0.1:8000, then click a reply
```

| Tab | Shows | Teaches |
|---|---|---|
| **This turn** | timings, everything recall found grouped by retriever, the extraction's decisions with reasons | the read path (Lesson 10) and the write path (Lesson 9) |
| **Prompt** | the exact system prompt, with the memory block highlighted, plus token counts | the model is stateless: "memory" is text rebuilt every turn |
| **Life graph** | what this turn's recall touched, highlighted; a **Seen by** switch for the other persona | graph hops; private vs. shared (Lesson 16) |
| **Strength** | a **Look ahead** slider, 0–365 days | decay, dormancy and expiry (Lesson 13) |
| **Try recall** | recall for any text, on any date, as either persona, without chatting | how recall changes with wording, time and persona |
| **Diary** | episodes (turns, memories taken from each, summary) and consolidated patterns | episodes (Lesson 11) and consolidation (Lesson 13) |

## How the trace is collected — [`companion/graph.py`](../companion/graph.py)

`build_companion(store, background, trace=[])` takes an optional list. Each node adds its part
to the current turn's entry as it runs:

```
recall    -> trace.append({message, recall: <everything store.recall returned>, recall_ms})
respond   -> system_prompt, window (messages sent), respond_ms, usage (tokens)
route     -> extraction = {status: "skipped", reason: "small talk"}         (gate said no)
remember  -> extraction = {status: "queued"}
  job     -> status "running", waited_ms, candidates, entities
          -> status "done", extracted (raw LLM output), log, llm_ms, write_ms
```

- **No new logic.** The trace records what the nodes already compute. Recall isn't run a
  second time to display it, so what the page shows is exactly what the model received.
- **The background job fills in its part later.** The page shows "Extracting in the background…"
  until the worker finishes, then redraws. That's Lesson 14's queue, visible.
- **Without `trace`, nothing changes.** The terminal chat and the scenarios don't pass one.

The API exposes it as `GET /api/trace?n=` (default: the latest turn). It also returns
`highlight`, the entities recall touched: the ones it hopped to, plus whatever its memories are
`ABOUT`.

## The read: recall, by retriever

Recall isn't one search. It's nine retrievers, each answering a different question, and the
**This turn** tab lists them separately with what each found:

| Retriever | Finds | Why it exists |
|---|---|---|
| Core profile | every boundary, top facts and preferences | must always be known, whatever the message |
| Vector matches | nearest memories, re-ranked | what the message is *about* |
| Graph hop | entities named or meant, and their memories | "Maya" → everything about Maya, even with no word overlap |
| Open plans | due within ±14 days | so it follows up without being asked |
| Yearly dates | birthdays in the next two weeks | same |
| Recently ended | statuses that expired in the last day | "how was the gym?" |
| History | older versions (`SUPERSEDES`) | "where did I live before?" |
| People | everyone related to the user | who's who |
| Past conversations | the last diary summaries | continuity |

Plus one list of what was **left out**: matches that faded below the dormant threshold. Seeing
a memory match and still be dropped is the clearest demonstration of decay.

From a real run (DeepInfra gemma, `What should I bring Maya as a gift?`):

```
VECTOR MATCHES 2
[event] Trace Demo's sister Maya got a job at Acme in Denver.   match 0.757 → rank 0.694
[plan]  Trace Demo is visiting Maya in Denver on 2026-11-13.     match 0.816 → rank 0.69
GRAPH HOP 1
Maya  sister
OPEN PLANS 1
Trace Demo is visiting Maya in Denver on 2026-11-13.   due 2026-11-13
```

The plan matched better (0.816) but ranked lower: rank combines match with importance and
strength, and the event was more important. The same plan comes back through three retrievers.
`format_context` removes the repeats before it goes in the prompt (open the **Prompt** tab to
see them deduplicated).

## What the trace found: recall that can't say "nothing"

Typing **dogs** into *Try recall* for a user who never mentioned dogs still returned two
"vector matches" about his son. Vector search returns its *nearest* memories, related or not
(Lesson 15's kNN had the same problem). A cut-off on the vector score doesn't work with
`bge-small`. The scores overlap:

| Message | Related? | Best vector match |
|---|---|---|
| what do you remember about my kid? | yes | 0.740 |
| dogs | no | 0.741 |
| tell me a joke | no | 0.778 |
| work was stressful | no | 0.796 |

Two ideas that didn't help, measured on the same 30 messages: bge's query instruction prefix,
and subtracting how well the message matches unrelated people's lives. Both still overlapped.

**A reranker separates topical matches.** A cross-encoder (`Xenova/ms-marco-MiniLM-L-6-v2`,
80 MB, local through fastembed, about 20 ms a message) reads the message and the memory
*together* and scores relevance:

| | Best relevance (logit) |
|---|---|
| every unrelated message | **≤ −9.7** (most about −11) |
| direct questions ("where do I live?", "how's Biscuit?") | **≥ −7.6** |
| "dogs" for Alex, who has a beagle | −5.4 |

**But it can't make a leap.** The first version dropped everything below −8.5. Then the
`sensitive` scenario failed: "I'm struggling to focus at work, my meds don't seem to help"
scored **−10.9** against "diagnosed with ADHD and takes medication", the same as noise. So did
"I'm nervous about Thursday" (the presentation) and "my puppy chewed my shoes" (the beagle).
Four rerankers up to 1 GB missed the same cases, and so did combining both scores:

| Message | Memory | Vector | Reranker | Related? |
|---|---|---|---|---|
| my meds don't seem to help | ADHD + medication | 0.758 | −10.9 | yes |
| I'm nervous about Thursday | Thursday presentation | 0.751 | −11.3 | yes |
| dogs (Mark has no dog) | son is sick | 0.742 | −11.0 | no |
| work was stressful | son is sick | 0.797 | −11.2 | no |

Small local models match topics; they don't reason. So recall **labels instead of dropping**.
A match below `RELEVANCE_MIN = −8.5` is marked `weak` and sorted after the strong ones. In the
trace it's tagged **loosely related** with its score. Typing "dogs" still shows the son's
memories, now clearly marked as weak.

**The label stays out of the prompt.** The first labelling version put weak matches under
their own heading in the prompt: *"Loosely related (maybe not about this message)"*. Then the
`personas` scenario failed. "Can you help me plan my training for this week?" scored −11.2
against "signed up for the Chicago half marathon" (so did "how many miles should I run?"). The
label told Coach to ignore its most useful memory, and Coach stopped mentioning the race. A
label is a claim. When the scorer is often wrong, putting its verdict in the prompt spreads
the mistake. So the prompt gets one *Relevant memories* list, strong matches first. The chat
model can make the leap the reranker can't.

The better fix is a stronger embedding model, measured on these same messages. That means
re-embedding everything and re-measuring every threshold in this course, so it's its own change.

Two more things the tests caught:

- **Don't rewrite the message.** Memories are third person ("Alex lives in Portland") and
  messages first person ("where do I live?"), so swapping "I" for the name seemed obvious. It
  rescued some related misses but made 11 of 16 unrelated messages look relevant: "Alex's car
  broke down" scores high against "Alex lives in Portland" on the name alone.
- **A current status is never weak.** "What am I doing?" scored −8.9 against "Pytest is at the
  gym", and Lesson 13's test caught it. A current status is context for any message, like
  *Recently ended*.

People, Core profile, Open plans and Past conversations aren't scored at all: they're the
background a friend always has, whatever you just said.

## The write: decisions with reasons — [`companion/store.py`](../companion/store.py)

Before this lesson the log said *what* happened (`reinforced m_ab12`). Now each decision says
**why**, with the number it was decided on. For example:

```
+ [plan] Trace Demo is visiting Maya in Denver on 2026-11-13.  [new: nothing similar with the same dates]
+ [fact] Pytest likes hiking  [new: closest known 0.712 < 0.975]
reinforced m_3f2a1c "Pytest plays the cello" (similarity 1.000 ≥ 0.975)
skipped a repeat within this extraction: ... (cosine 0.981 ≥ 0.975)
+ [fact] ... (sensitive) (private to Sage)
```

- The duplicate lookup now returns the **closest** memory and its score, not just "a match
  above 0.975". The decision is the same; the log just shows how close it was.
- "With the same dates" matters: "dentist on the 12th" and "dentist on the 17th" read almost
  the same, so dates must match for something to count as a duplicate (Lesson 9).
- The two in-transaction checks (a repeat inside one extraction, a just-forgotten memory)
  compare **raw cosine** in Python, while the index reports a 0–1 score. The log says which
  one it is, so the numbers aren't compared against the wrong scale.

The tab also lists the **known memories shown to the extractor**: the candidates, with
their ids, so it can say "retire m_ab12" or "this supersedes m_cd34" instead of writing a
near-copy. If a duplicate slips through, check here first: was the old memory a candidate?
Under **Raw extraction** is the LLM's JSON before the store's rules applied.

These reasons are in `MemoryStore` itself, so the terminal chat's ✎ lines show them too.

## Timings: what you wait for, and what you don't

```
recall 42 ms │ reply 3526 ms │
             │               │ queued 31 ms │ extract 8719 ms │ write 22 ms
```

Measured with the hosted gemma model: recall is tens of milliseconds, the reply 3–8 seconds,
and extraction **8–10 seconds**. That's longer than the reply. Without Lesson 14's background
worker, every message would wait for both, about 12–18 seconds. The header also shows the
worker's queue (`memory worker: 1 job running/queued`).

## The other views

**Prompt.** The memory block is usually a few hundred characters of a ~1,900-character prompt,
and the API reports its tokens (590 in, 85 out above). Recent messages (at most 16) are sent
alongside: that's short-term memory, kept by the LangGraph checkpointer.

**Seen by.** The graph draws as the selected persona sees it. In your own view, a dashed
outline marks entities the other persona can't see, and dashed edges are private relations.
Switch to the other persona and they disappear. The privacy rules are the same `visible()`
filters from Lesson 16; the page just runs them as the other persona.

**Look ahead.** The slider recomputes strength, dormancy and expiry N days from now. Nothing
is written. A feeling of importance 3 goes dormant after about 18 days (7-day half-life), an
event of importance 6 after about 11 months (90 days), and facts, preferences and plans never
fade. A situational memory ("tired today") shows as *expired*. The open plan for the 13th is
still listed a month later. It only lapses when the end-of-episode upkeep runs
(`consolidate`), so the slider shows the graph as it is now, not after upkeep has run.

**Try recall** runs `store.recall(..., touch=False)`. A normal recall increments
`access_count` on what it returns. A preview mustn't, or looking at memory would change it.
`test_the_recall_preview_only_looks` checks that, and it fails if `touch=False` is removed
(checked).

## Test it — [`tests/test_18_inspect.py`](../tests/test_18_inspect.py)

```bash
.venv/bin/python -m pytest tests/test_18_inspect.py -v    # 8 tests, ~4 s, no LLM
```

A trace that is wrong teaches the wrong thing, so it gets tests like any other feature:

- a turn records the recall, the prompt (with `<memories>`), timings and decisions with reasons;
- the second mention is `reinforced` with its similarity, the old memory was a candidate, and
  Miso is highlighted;
- small talk shows *skipped* and why;
- (in [`test_10_recall.py`](../tests/test_10_recall.py)) "do you like pizza?" gets only
  weak matches; "my dog hates storms" keeps the dog on top; "my meds don't help" still reaches
  the ADHD memory in the prompt, though the reranker scored it weak;
- the recall preview doesn't change `access_count`;
- looking 90 days ahead halves an event's strength and expires "tired today", without writing anything;
- the persona lens hides Sage's private Dr. Lee from Coach;
- the diary lists the open episode with its turns and memories;
- the reasons come from `MemoryStore`, so they're in the terminal too.

## Takeaways

- Show the **read** as well as the write. Most "why did it say that?" questions are answered
  by what recall returned, not by what was stored.
- Record the trace **where the work happens** (the graph nodes) instead of recomputing it for
  display. A second recall could return something different from what the model saw.
- Every decision should carry its **reason and its number**. Thresholds stop being magic once
  you can see a 0.971 that just missed.
- Inspection must be **read-only**: a preview that bumps access counts changes what it measures.
- Vector search can't say "nothing found". Measure related vs unrelated scores before picking a
  cut-off. When the signals overlap, **don't drop, and don't tell the model either**: hiding a
  related memory, or calling it unrelated, costs more than showing a weak one to a model that
  can judge it. Show the score where a person reads it (the trace).

## Next

That's the end of the course. A good place to keep going: the reasons and timings in the trace are
what you'd send to tracing tools like LangSmith or OpenTelemetry in production.
