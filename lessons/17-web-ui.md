# Lesson 17 — A web UI: chat, life graph, decay and insights on one page

The terminal chat shows what the companion *says*. A page can also show what it *knows*, and
how that changes as you talk: the life graph growing, memories fading, the graph embeddings
grouping people.

```bash
.venv/bin/python -m companion.web                      # http://127.0.0.1:8000
.venv/bin/python -m companion.web --today 2026-11-20   # simulate a date
```

Pick a name and a persona (Sage or Coach), then Start.

## The pieces

```
browser (companion/static/index.html, d3)
   │  fetch JSON
   ▼
FastAPI (companion/web.py) ── one Chat: store + LangGraph app + Background worker + episode
   │                                          │
   ▼                                          ▼
MemoryStore (Lessons 9–16)            the worker writes memories (Lesson 14)
```

| Endpoint | Does | From lesson |
|---|---|---|
| `POST /api/session` | start a conversation (`user`, `persona`) and return the greeting | 10, 16 |
| `POST /api/chat` | one message → the reply. Memory extraction is queued, not awaited | 10, 14 |
| `GET /api/log` | ✎ lines from finished background jobs, and whether more are running | 14 |
| `GET /api/memories` | current memories with age, half-life and strength | 13 |
| `GET /api/graph` | entities and current relations this persona may see | 8, 16 |
| `GET /api/related` | FastRP or node-similarity pairs | 15 |
| `POST /api/end` | wait for the worker, run the end-of-episode pass, save the diary | 11, 14 |

The API adds **no new memory logic**. Everything goes through the same `MemoryStore`,
`build_companion` and `insights` code as the terminal chat. So the privacy rules from Lesson 16
apply to the page automatically, and a test checks that they do.

### Background extraction, seen from a browser

`/api/chat` returns as soon as the reply exists. The page then **polls** `/api/log` every
1.2 seconds while the worker is busy, prints the ✎ lines when they arrive, and redraws the
graph. Polling is the simplest thing that works for one user. Server-sent events or a
WebSocket would push the lines instead, and are worth it once there are many clients.

### The decay chart

The server sends each memory's `importance`, `age_days` and `half_life`. The page draws
`strength(t) = importance/10 × 0.5^((age + t) / half-life)` for the next 30 days, the formula in
`companion/decay.py` repeated in JavaScript, with a dashed line at the dormant threshold.
Facts and preferences are flat, grey lines: they don't fade (they go stale instead).

**Duplicated logic is a risk.** If `decay.py` changes, the chart will quietly disagree with
recall. The API also returns each memory's `strength` *now*, computed in Python, so the table
under the chart is always right, and a mismatch with the curve's starting point is visible.

## What I checked in a browser

```
Web Demo: I just adopted a beagle named Biscuit! My sister Maya is coming to Portland next
          week to meet him.
Sage: A beagle! That is such wonderful news. Biscuit is an adorable name ...
  ✎ linked user -pet-> Biscuit
  ✎ linked user -sister-> Maya
  ✎ linked user -lives_in-> Portland
  ✎ + [event] Web Demo adopted a beagle named Biscuit on 2026-11-05.
  ✎ + [plan] Web Demo's sister Maya is visiting Portland in the week of 2026-11-09 to meet Biscuit.
  ✎ + [fact] Web Demo lives in Portland.
```

- The reply appeared first. The ✎ lines arrived a few seconds later, and the graph redrew with
  Maya, Biscuit and Portland around the user.
- The strength tab showed the event fading slowly (90-day half-life), with the fact and the
  plan flat. "+30d" was cut off at the right edge, so the chart's margin was widened.
- Ending the session saved the diary summary.

## Concepts and design choices

*Where this fits:* All steps, seen from outside: a thin API over the same store. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [graph](../docs/glossary.md#graph) | sense 1 drawn as sense 3: the life graph as a picture | the LangGraph |
| fake (test double) | a working stand-in LLM for API tests | a mock that checks calls |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| A thin FastAPI app reusing `MemoryStore` and the companion graph | every rule (privacy, boundaries, decay) applies unchanged | a Python server to run | a separate API layer re-implementing rules | [D17](../docs/design-decisions.md#d17) |
| One HTML page, no build step | readable source for learners | not a production front end | React/Vue SPA | [D19](../docs/design-decisions.md#d19) |
| Local only, no auth | a teaching tool on your machine | not safe on a network (the lesson lists what's missing) | auth, TLS, rate limits | [D17](../docs/design-decisions.md#d17) |

## Test it — [`tests/test_17_web.py`](../tests/test_17_web.py)

```bash
.venv/bin/python -m pytest tests/test_17_web.py -v    # 5 tests, ~2 s, no LLM
```

FastAPI's `TestClient` calls the app **in-process**: no server, no port, no browser. The chat
model, the extractor and the diary summarizer are faked, so the tests check the **wiring**:

- a whole conversation: session → chat → background log → memories (the strength of a new
  event is exactly importance/10) → graph → related → end;
- chat before a session is 409, an empty message is 422, an unknown method is 422;
- the page is served and pins its d3 version;
- a Coach session never sees Sage's sensitive memory through the API.

The LLM itself is already covered by the evals in Lessons 5, 7, 11, 13 and 16. Testing it
again through HTTP would cost money and add nothing.

### The whole suite after Lessons 14–17

Transactions and the persona filter touched every write and read query, so everything was
rerun:

| Suite | Result |
|---|---|
| Fast tests (`pytest -q`, no LLM) | 159 passed |
| All LLM evals (`pytest -m llm`, ~15 min) | 39 passed |
| Scenarios, all 14 ([outputs/17-scenarios.txt](outputs/17-scenarios.txt)) | 53/53 |

## Before putting it on a network

This is a local tutorial app: one conversation at a time, bound to `127.0.0.1`, no login.
Anyone who can reach it can read the memories. A real deployment needs at least:

- **authentication**, with the user taken from the login, never from the request body;
- **one `Chat` per user session**, not one global one;
- **rate limits** on `/api/chat` (every call costs LLM tokens);
- **several server workers**, which Lesson 14's user lock already makes safe for writes.

## Takeaways

- A UI over memory shows its state: the graph, what's fading, what's grouped. That's useful for
  users and for debugging.
- Keep the API thin. Reuse the same store and graph code so every rule (privacy, boundaries,
  decay) applies everywhere without being re-implemented.
- Test the wiring with fakes and `TestClient`. Leave the model to its evals.
- Logic duplicated in the frontend (the decay curve) needs a check against the backend.

## Next

[Lesson 18](18-seeing-memory-work.md) shows what memory does on each turn: what recall found, the exact prompt, and why each write happened.
