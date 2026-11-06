# Lesson 14 — Production hardening: transactions, locks and background extraction

Lessons 9–13 ran one thing at a time: one message, one extraction, one write, then the reply.
A real deployment breaks both of those assumptions:

- **Things fail half-way.** The LLM returns, entities and relations get written, then the
  memory step hits a network error. Before this lesson, every Cypher statement committed on its
  own, so the graph kept "sister Maya" with no memory behind it.
- **Things happen at once.** Extraction takes one to several seconds, so it should run in the
  background while the companion answers. Then two writes for the same user can overlap.

This lesson fixes both, and measures the bug before fixing it.

## 1. All or nothing: one transaction per `apply()`

`MemoryStore._write(work)` runs `work()` inside one **managed write transaction**:

```python
def apply(self, ex, episode_id=None, today=None):
    return self._write(lambda: self._apply(ex, episode_id, today))
```

```python
with self.driver.session() as session:
    return session.execute_write(unit)   # unit(tx) runs work() with every query on tx
```

- **Commit or roll back as a whole.** If anything raises, nothing from this extraction is
  stored. A retry then starts from a clean graph instead of adding a second copy of the half
  that got through.
- **Automatic retries.** `execute_write` retries *transient* errors (a deadlock, a cluster
  leader change) by running `work` again. So `work` must not do anything outside the graph
  that can't be repeated, such as sending a message.
- **Nested calls join.** `consolidate()` calls `apply()` to store a pattern. Inside a
  transaction, `_write` just runs the inner work in the same transaction.
- `forget_entity()` is transactional too. A forget that stops half-way would leave the name
  in some turns and not others.

How do all the existing queries join the transaction without rewriting them? `_q()`, the one
function every query goes through, checks for an open transaction:

```python
tx = getattr(self._local, "tx", None)
if tx is not None:
    return tx.run(query, **params).data()       # inside _write(): join it
records, _, _ = self.driver.execute_query(...)  # otherwise: its own auto-commit
```

`self._local` is a `threading.local()`. That detail matters in section 3.

### What the test does

`test_a_failure_half_way_rolls_back_everything` makes the step between relations and memories
raise. Afterwards there is no entity, no relation and no memory. Then it runs the same
extraction again, which stores everything once.

**Injecting a failure at a precise point** (`monkeypatch.setattr(store, "_tombstones", boom)`)
is the standard way to test rollback. Waiting for a real network error doesn't work.

### Two side effects of transactions

- **The vector index doesn't see your own uncommitted writes.** Measured: after creating a
  memory inside a transaction, `db.index.vector.queryNodes` in the same transaction finds 0
  matches, and 1 after commit. So `apply()` now also compares each new memory with the ones it
  just stored, or an extraction listing the same fact twice would store it twice.
- **Not everything is in the transaction.** `ontology.propose_type()` records a new relation
  type through its own connection, so a rolled-back extraction can still leave a *proposed*
  type behind. That's acceptable here (it's a suggestion for a human to review), but every
  "all or nothing" claim needs a list of what's outside it.

## 2. One writer per user: the lock

The duplicate check in `apply()` is *check, then act*: look for a similar memory, and create
one if there is none. Two writers running at the same moment both check, both find nothing
(neither has committed), and both create.

I measured it: four threads apply the same fact at the same instant, five trials.

| | Copies stored per trial |
|---|---|
| Transaction, no lock | 4, 4, 4, 4, 4 |
| Transaction + user lock | 1, 1, 1, 1, 1 |

A transaction alone doesn't help. Neo4j's default isolation is *read committed*: you never see
another transaction's uncommitted writes, but nothing stops two transactions making the same
decision.

The fix is one line at the start of every write transaction:

```cypher
MATCH (u:User {id: $user}) SET u.write_lock = datetime()
```

Writing to a node takes Neo4j's **write lock** on it until commit. A second writer for the same
user blocks on this statement. When it continues, the first one has committed, so its
duplicate check finds the memory and reinforces it. The test's log shows exactly that:
`+`, `reinforced`, `reinforced`, `reinforced`.

- **Different users never wait for each other**, because each has their own node.
- **It works across processes.** Two web-server workers, or a nightly consolidation job, all
  go through the same database lock. A Python lock would only cover one process.
- **Keep the lock short.** `consolidate()` calls the LLM *before* opening its transaction,
  and stores the pattern and its links in a short one afterwards. Holding a user's lock during
  a multi-second LLM call would stall every other write for that user.

### A bug the transaction work found

`consolidate()` used to find the pattern it had just stored by its exact text, to link the
evidence to it. When `apply()` *reinforced* an older pattern with slightly different wording
instead of creating one, no node had that text, so the `SUMMARIZES` links and the evidence
count were silently skipped. `apply()` now records the ids it wrote (`written_ids`) and the
links use those. The regression test is
`test_a_pattern_links_its_evidence_even_when_it_reinforces_an_older_one`.

## 3. Background extraction — [`companion/worker.py`](../companion/worker.py)

The reply doesn't need the extraction, so the companion shouldn't wait for it:

```
before:  message -> recall -> reply -> log -> extract (LLM) -> write -> prompt again
after:   message -> recall -> reply -> log -> queue the job -> prompt again
                                                  └─ worker: extract (LLM) -> write
```

`Background` is a `ThreadPoolExecutor` with **one** worker thread:

- **Jobs for a user run in order.** Job 2's prompt includes the memories job 1 wrote ("Miso
  is already known"), so they can't overlap. The extraction candidates are read *inside*
  the job, not when it's queued, so each job sees everything before it.
- `submit(job)` returns at once. `drain()` returns log lines from finished jobs without
  blocking. `wait()` blocks until the queue is empty.
- **Errors are collected, not lost.** An exception in a background thread vanishes unless
  someone collects it, so the worker turns it into a log line:
  `memory update failed: ...`.

The queue and the lock do different jobs. The queue gives **order and speed**. The lock gives
**correctness**, even when something outside this queue writes too.

### Where the conversation has to wait

| When | Why |
|---|---|
| Before the end-of-episode pass (`/new`, `/quit`) | it re-reads the episode and must see every memory |
| Before `/memories`, `/people`, `/recall`... | showing the graph without queued updates would confuse |
| Never before a reply | that's the point |

Turn on `--sync` to go back to inline extraction (Lessons 10–13 behaviour). The scenario runner
and the demo use sync mode, because their checks read the graph straight after each message.

### Thread safety

Two threads now use the store at once: the chat thread recalls while the worker writes.

- **The open transaction is per thread** (`threading.local`). If it were a plain attribute,
  the chat thread's `recall()` would run its queries *inside the worker's transaction*. They
  would then commit or roll back with someone else's work, from a thread that isn't allowed to
  use that transaction. `test_a_reader_never_joins_another_threads_transaction` holds a write
  open on one thread and reads from another: the reader sees 0 memories, then 1 after commit.
- **The Neo4j driver is thread-safe; sessions and transactions are not.** Each `execute_query`
  and each `_write` opens its own session.
- **A forget in the background** used to scrub the live conversation in the same step. Now
  the job finishes after the reply, so the next turn's `recall` applies the redaction
  (`test_a_background_forget_is_scrubbed_from_the_conversation_next_turn`).

### A smoke run

```
BG Smoke: I just adopted a cat named Miso!
Sage: That's wonderful news! Miso is such a cute name. Is Miso a kitten or an adult? ...
BG Smoke: She is a tabby and loves boxes.
Sage: Classic tabby behavior! ...
BG Smoke: /memories
  ✎ linked user -pet-> Miso
  ✎ + [event] BG Smoke adopted a cat named Miso on 2026-11-05.
  ✎ + [fact] BG Smoke's cat Miso is a tabby.
  ✎ + [preference] BG Smoke's cat Miso loves boxes.
```

Both replies came back without waiting. The memory lines arrived afterwards, and the second
job already knew about Miso from the first.

## Concepts and design choices

*Where this fits:* Step 6 made safe: atomic writes, no races, extraction off the reply's critical path. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| transaction | one `apply()` commits or rolls back as a unit | a session (a connection) |
| write lock | Neo4j's lock on the user node, held until commit | a Python lock (one process only) |
| background worker | a second thread running extraction jobs in order | async/await |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| One transaction per extraction | no half-applied contradictions | slow work (LLM, embeddings) must happen outside it | a transaction per statement | [D16](../docs/design-decisions.md#d16) |
| Serialize writers per user with a lock on the user node | removes the check-then-act duplicate race | writes for one user wait for each other | a queue per user; optimistic retries; content-hash constraints | [D16](../docs/design-decisions.md#d16) |
| Background extraction thread | the reply doesn't wait for extraction | a memory may land after the next turn starts | extract inline; a job queue service (Celery, SQS) | [D6](../docs/design-decisions.md#d6) |

## Test it — [`tests/test_14_production.py`](../tests/test_14_production.py)

```bash
.venv/bin/python -m pytest tests/test_14_production.py -v     # 10 tests, ~2 s, no LLM
```

Concurrency bugs don't show up when you run one thing at a time, so each test **forces the
interleaving** it's about:

| Test | How it forces it |
|---|---|
| rollback | `monkeypatch` makes a step half-way through raise |
| two writers | `threading.Barrier(4)` releases four threads at the same instant |
| reader vs. open write | `threading.Event`s hold the write open while the other thread reads |
| reply doesn't wait | a fake extractor blocks on an `Event` until the test has checked the reply |
| jobs in order | each job records start and end; any overlap would interleave them |

The last two use a **fake chat model** (`FakeListChatModel` from LangChain) and a fake
extractor. The test is about timing, so a real LLM would only add cost and variation.

## Takeaways

- Wrap each logical change in **one transaction**, and test it by making a step in the middle fail.
- A transaction doesn't stop **check-then-act races**. Lock something both writers must
  touch (here, the user node), and measure the race before and after.
- Hold locks for milliseconds: do slow work (LLM calls, embeddings when possible) outside
  the transaction.
- Background work needs a **queue for order**, a **place for errors**, and clear **points
  where callers wait**.
- Per-thread state for anything tied to a transaction: shared state across threads is how a
  read ends up inside someone else's write.

## Next

[Lesson 15](15-graph-embeddings.md) uses Graph Data Science on the life graph to find who belongs together.

<!-- nav -->
---

[← Lesson 13: Memory over time: expiry, decay and consolidation](13-memory-over-time.md) · [All lessons](../README.md#lessons) · [Lesson 15: Graph embeddings: who and what belong together? →](15-graph-embeddings.md)
