# Neo4j graph tutorial

Hands-on lessons on graph databases with Neo4j — from Cypher basics to graph algorithms,
vector search, GraphRAG, LangGraph agents, and a graph + vector biographical memory for an
AI companion — with every query's recorded output.

## Quick start

```bash
cp .env.example .env              # set NEO4J_PASSWORD; LLM_* for lessons 5 and 7–20
docker compose up -d --build      # Neo4j 5 + APOC + Graph Data Science
uv sync                           # Python 3.12 + the exact tested versions, into .venv
```

Needs [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or
`brew install uv`). Every command below runs through `uv run`, which keeps `.venv` in step with
`uv.lock`. Without uv: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`,
then use `.venv/bin/python` wherever these docs say `uv run python`.

- Neo4j Browser: http://localhost:7474 (user `neo4j`, password from `.env`)
- Fresh database? Load the starting graph — see [Lesson 0](lessons/00-setup.md).

## For architects of agent memory

Lessons 8–20 build a companion that remembers a person's life, on Neo4j. Three documents
explain the design rather than the code:

- **[Design decisions and tradeoffs](docs/design-decisions.md)**: 21 decisions (why Neo4j, why
  memories are sentence nodes, why recall runs every turn, why forgetting deletes...). Each
  covers what it costs, the alternatives (Postgres + pgvector, a vector DB, other graph
  databases, RDF/OWL, Mem0, Zep/Graphiti, Letta, Cognee, LangMem, Neo4j's own agent-memory
  package, cloud memory services, memory as files, GraphRAG), when to revisit it, and the gaps
  this design doesn't cover.
- **[The pipeline as concepts](docs/design-decisions.md#the-pipeline-as-concepts)**: each step
  of a chat turn mapped to its concept and the field it comes from (psychology, NLP,
  databases, information retrieval, knowledge representation).
- **[Layers](docs/glossary.md#layers)**: the companion's seven logical layers (storage, schema,
  ontology/TBox, facts/ABox, memory, retrieval, context), with the standard term for each and
  what *semantic layer* and *memory layer* mean elsewhere. Diagram:
  [architecture §24](docs/architecture.md#24-layers-of-the-companion).
- **[Glossary](docs/glossary.md)**: terms that mean different things in different fields.
  *[Semantic](docs/glossary.md#semantic)*, *[ontology](docs/glossary.md#ontology)*,
  *[recall](docs/glossary.md#recall)*, *[profile](docs/glossary.md#profile)* and others have
  one section each, sense by sense. A second table compares
  [academic, practitioner and marketing usage](docs/glossary.md#academic-industry-and-marketing-usage)
  of terms like *context engineering*, *context graph*, *memory layer*, *semantic layer* and
  *GraphRAG*.

Every lesson has a **Concepts and design choices** section that does the same at lesson
level: which sense each term has there, and the choices made, with costs and alternatives.

## Lessons

### Part 1: Neo4j fundamentals

Run Neo4j, read and write with Cypher, model for performance, run graph algorithms from Python.

| # | Lesson | Covers |
|---|---|---|
| 0 | [Setup](lessons/00-setup.md) | Docker, seed data, Python env, reset |
| 1 | [Graph basics and reading](lessons/01-graph-basics-and-reading.md) | Nodes/relationships, `MATCH`, filtering, aggregation, multi-hop, shortest path, `WITH` |
| 2 | [Writing data](lessons/02-writing-data.md) | `CREATE`, `MERGE`, `SET`/`REMOVE`, constraints, indexes, `UNWIND`, `DETACH DELETE` |
| 3 | [Modeling and performance](lessons/03-modeling-and-performance.md) | Property vs node, reifying relationships, `PROFILE`, parameters, supernodes |
| 4 | [Graph algorithms and Python](lessons/04-graph-algorithms-and-python.md) | GDS projections, PageRank, Louvain, similarity, write-back, Python driver |

### Part 2: Graphs meet AI

Vectors and GraphRAG, where graphs help agents, and a LangGraph agent with graph tools.

| # | Lesson | Covers |
|---|---|---|
| 5 | [Vectors and GraphRAG](lessons/05-vectors-and-graphrag.md) | Vector indexes, local embeddings, hybrid vector + graph retrieval |
| 6 | [Graphs in AI and agents](lessons/06-graphs-in-ai-and-agents.md) | Agent memory, graph tools, when embeddings help |
| 7 | [LangGraph agent with graph tools](lessons/07-langgraph-agent-with-graph-tools.md) | `StateGraph`, `ToolNode`, checkpointer, read-only Cypher tool |

### Part 3: A companion that remembers

Design a person's memory, write it from conversation, recall it every turn, and test it with scenarios.

| # | Lesson | Covers |
|---|---|---|
| 8 | [Memory design for companions](lessons/08-memory-design-for-companions.md) | Memory kinds, life graph, episodes, bi-temporal facts, boundaries |
| 9 | [The memory write path](lessons/09-memory-write-path.md) | Structured extraction, entity resolution, dedupe, supersede, plans |
| 10 | [Recall and the companion chat](lessons/10-companion-recall-and-chat.md) | Hybrid recall, LangGraph companion, terminal CLI, two-session demo |
| 11 | [Harder scenarios and evals](lessons/11-harder-scenarios-and-evals.md) | Context-window + end-of-episode extraction, same-name people, ended relations, history, forgetting, sensitive memories, scenario tests |

### Part 4: A smarter memory

Give relations meaning with an ontology, and let memories expire, fade and consolidate.

| # | Lesson | Covers |
|---|---|---|
| 12 | [An ontology for the life graph](lessons/12-ontology.md) | Relation vocabulary, synonyms, hierarchy, symmetric/inverse/cardinality/range rules, kinship inference, `(:RelType)` nodes; terms in the [glossary](docs/glossary.md) |
| 13 | [Memory over time](lessons/13-memory-over-time.md) | Expiry (`valid_until`, status memories), decay and staleness, consolidation into habits, pruning, one testable clock |

### Part 5: Running it for real

Transactions and background extraction, what graph embeddings add, and several personas sharing one memory.

| # | Lesson | Covers |
|---|---|---|
| 14 | [Production hardening](lessons/14-production-hardening.md) | One transaction per write, a user-node lock against duplicate races, background extraction, thread safety |
| 15 | [Graph embeddings](lessons/15-graph-embeddings.md) | FastRP + kNN vs. node similarity on the life graph, Cypher projections, seeds and cut-offs |
| 16 | [Personas](lessons/16-personas.md) | Several companions sharing one memory: private vs. shared memories, hiding derived data, per-persona diaries |

### Part 6: Seeing and tuning it

A web UI, a per-turn trace of what memory did, and choosing an embedding model by measuring.

| # | Lesson | Covers |
|---|---|---|
| 17 | [A web UI](lessons/17-web-ui.md) | FastAPI + one HTML page: chat, life graph, decay curves, related entities; API tests with a fake LLM |
| 18 | [Watching memory work](lessons/18-seeing-memory-work.md) | Per-turn trace: what recall found by retriever, the exact prompt, why each write happened, timings, persona lens, look-ahead clock, diary |
| 19 | [Choosing an embedding model](lessons/19-choosing-an-embedding-model.md) | Compare models on your own data, query instructions, re-measure every threshold per model, vectors versioned by model, re-embed |

### Part 7: Growing the graph

Let the life graph expand past the user: what deserves a node, bounded multi-hop recall, edge provenance.

| # | Lesson | Covers |
|---|---|---|
| 20 | [Growing the life graph past one hop](lessons/20-growing-the-life-graph.md) | What deserves a node, edges from the person who has the preference, bounded two-hop recall, an `exclusive` ontology rule, edge provenance, orphan cleanup, re-extracting old conversations |

**Architecture:** component, LangGraph, sequence, data-model and lifecycle diagrams for the
agent and the companion are in [docs/architecture.md](docs/architecture.md).

## LLM settings (lessons 5 and 7–20)

Any OpenAI-compatible endpoint with tool calling works (OpenAI, DeepInfra, OpenRouter,
Ollama, LM Studio, vLLM...). Set in `.env`:

| Key | |
|---|---|
| `LLM_BASE_URL` | e.g. `https://api.openai.com/v1`, `http://localhost:11434/v1` |
| `LLM_MODEL` | chat model name |
| `LLM_API_KEY` | key (any value for local servers) |
| `LLM_EXTRACTION_MODEL` | optional, model for memory extraction (defaults to `LLM_MODEL`) |
| `LLM_TEMPERATURE` | chat temperature |
| `COMPANION_USER`, `COMPANION_NAME` | default user and companion names |

**Embeddings: local to learn, remote to use.** Lessons 5–7 and the default test run always use
a small local model (`BAAI/bge-small-en-v1.5` via fastembed): no key, offline, the same numbers
every run, and the numbers recorded in Lessons 8–18. For real use of the companion we
recommend the larger remote model: it separates related from unrelated memories far better
(AUC 0.994 vs 0.854) and fixes misses the small model can't. Lesson 19 has the full table of
what runs where. It takes three more keys:

| Key | |
|---|---|
| `EMBED_BASE_URL` | an OpenAI-compatible `/embeddings` endpoint; empty = local bge-small |
| `EMBED_MODEL` | e.g. `Qwen/Qwen3-Embedding-8B`; needs a measured profile in `companion/embeddings.py` |
| `EMBED_API_KEY` | optional, defaults to `LLM_API_KEY` |

After switching, run `uv run python -m companion.reembed`. The default test run uses the
local model; the [release check](#release-check-on-the-model-you-ship) runs everything on yours. The reranker that labels unrelated recall matches also runs locally (`RERANK_MODEL`,
default `Xenova/ms-marco-MiniLM-L-6-v2`, Lesson 18; not used when the embedding profile sets
`weak_below`, Lesson 19). Local models are downloaded once and loaded once per process.

## Files

| Path | What |
|---|---|
| `Dockerfile`, `docker-compose.yml` | Neo4j 5.26 with APOC + GDS, data in named volumes |
| `.env.example` → `.env` | Neo4j credentials and LLM keys (`.env` is git-ignored) |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencies, the exact tested versions, the Python version (`uv sync`) |
| `requirements.txt` | The same pinned versions for pip, generated from `uv.lock` |
| `config.py` | Loads `.env` for the scripts |
| `observability.py` | Optional MLflow tracing and eval logging (see *Tests and evals*) |
| `seed/00_seed.cypher` | Starting graph for Lesson 1 |
| `team_graph.py` | Lesson 4 — Python driver patterns |
| `embed.py` | Lesson 5 — local embeddings + vector index |
| `graphrag.py` | Lesson 5 — GraphRAG retrieval (+ LLM answer via the OpenAI SDK if `LLM_*` is set) |
| `team_agent.py` | Lesson 7 — LangGraph agent with graph tools |
| `companion/` | Lessons 8–10 — memory schema, store, extraction, LangGraph companion, CLI, demo |
| `companion/scenarios.py` | Lessons 11–12 — multi-session scenario tests for memory |
| `companion/ontology.py` | Lesson 12 — relation-type ontology: vocabulary, rules, inference |
| `companion/clock.py`, `companion/decay.py` | Lesson 13 — one clock; the forgetting curve |
| `companion/worker.py` | Lesson 14 — background extraction worker |
| `companion/insights.py` | Lesson 15 — graph embeddings (FastRP + kNN) and node similarity |
| `companion/web.py`, `companion/static/` | Lessons 17–18 — web UI (FastAPI + one HTML page) and the per-turn memory trace |
| `companion/embeddings.py`, `calibrate.py`, `reembed.py` | Lesson 19 — local or remote embeddings, per-model thresholds, measuring them, backfilling vectors |
| `companion/reextract.py` | Lesson 20 — re-read stored conversations under the current extraction rules |
| `companion/trace_to_case.py` | Turn a traced extraction (MLflow) into a new eval case (see *Tests and evals*) |
| `docs/architecture.md` | Architecture diagrams (Mermaid) for lessons 7–17 (the *how*; design-decisions.md is the *why*) |
| `tests/`, `pytest.ini` | Per-lesson tests and LLM evals (see *Tests and evals*) |
| `docs/glossary.md` | Glossary of overlapping memory / graph / ontology terms, with a section per overloaded word |
| `docs/design-decisions.md` | Design decisions and tradeoffs, alternatives, gaps, comparison with other memory systems |
| `lessons/outputs/` | Recorded runs for lessons 7, 10, 11, 12, 13 and 17 |

## Tests and evals

Every lesson has tests in [`tests/`](tests), named after it (`test_05_vectors_and_graphrag.py`).

```bash
uv run python -m pytest                    # unit + database tests: seconds, free (default)
uv run python -m pytest -m "not db and not llm"   # unit tests only, no Neo4j needed
uv run python -m pytest -m llm             # LLM evals: real calls, cost money
EVAL_TRIALS=3 uv run python -m pytest -m llm tests/test_11_extraction_evals.py   # pass rates
```

A `-m` on the command line *replaces* the default `-m "not llm"` in `pytest.ini`, so spell
out `and not llm` when you add your own filter. Database tests never leave data behind: writes run in rolled-back transactions, and memory
tests use their own `pytest` user. A test for a lesson you haven't done yet is skipped with a
reason. [Lesson 0](lessons/00-setup.md#test-it) explains the three kinds of test, and each
lesson ends with a *Test it* section on what its tests teach.

### Traces and eval history in MLflow (optional)

The Lesson 18 trace shows what memory did on one turn. For more detail, such as every LLM call
with its prompt, tokens and latency, plus eval runs you can compare over time, turn on local
MLflow tracing. It's off unless `MLFLOW_TRACKING_URI` is set:

```bash
uv sync --group observability                                    # once: installs mlflow
uv run --group observability mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
```

Then, in another terminal, add `MLFLOW_TRACKING_URI=http://localhost:5001` to `.env` (or
prefix one command with it), and run as usual with `--group observability`:

```bash
uv run --group observability python -m companion                 # every turn is a trace
uv run --group observability python -m pytest -m llm             # one MLflow run per eval session
```

Open http://localhost:5001, experiment `companion`:
- **Traces**: a chat turn is one trace (`recall` → `respond` → `remember`, the LLM calls, and
  the Neo4j spans `store.recall`, `store.extraction_candidates`, `store.apply`), grouped by
  session. Background extraction is its own trace in the same session. An eval case is one
  trace holding every LLM call its trials made.
- **Runs**: each `pytest -m llm` session and each `python -m companion.scenarios` run is one
  MLflow run. It logs the models, `EVAL_TRIALS`, a hash of the prompt files, the git commit,
  `passed`/`failed`/`pass_rate`, and an `evals.json` table of every case with its trace id.
- **Verdicts on traces**: each case's pass/fail is attached to its trace as feedback (a pytest
  case gets `passed`; a scenario gets one entry per check, with the check's wording). Filter
  to the failures and read the exact prompts that produced them.

pytest and the scenario checks still decide pass or fail; MLflow records the verdicts and keeps
the history. [`observability.py`](observability.py) does the setup and logging; the pytest hooks
are at the end of [`tests/conftest.py`](tests/conftest.py).

**From a bad chat turn to a new eval case.** When a traced chat stores something wrong, open
that turn's `remember` trace in the UI, copy its id, and run:

```bash
uv run --group observability python -m companion.trace_to_case tr-...            # print the case
uv run --group observability python -m companion.trace_to_case tr-... --append   # add it to the evals
```

It writes a case for [`tests/test_11_extraction_evals.py`](tests/test_11_extraction_evals.py)
that replays the extraction with the exact inputs the trace recorded, and shows what the model
returned in a comment. Write the check for the rule that broke (it should fail), fix the prompt,
rerun `pytest -m llm`, and compare the two runs in MLflow.

### Release check: on the model you ship

The default run pins the local embedding model, so it stays fast, free and offline. Before you
rely on a change, run the tests, the LLM evals and the scenarios on the model in your `.env`:

```bash
TEST_REMOTE_EMBEDDINGS=1 uv run python -m pytest              # ~5 min: every embedding is remote
TEST_REMOTE_EMBEDDINGS=1 uv run python -m pytest -m llm       # LLM evals, costs money
uv run python -m companion.scenarios                          # scripted conversations (use .env)
uv run python -m companion.calibrate                          # are the thresholds still right?
```

Tests that check a threshold read it from the active model's profile. Tests that are about one
model (bge-small's cross-encoder labels, a calibration pair a model is known to miss) say so
and check the right thing for each. A model's known misses are listed in
[`tests/test_19_embeddings.py`](tests/test_19_embeddings.py) (`KNOWN_MISSES`): anything else
failing is a regression.

## Common commands

```bash
docker compose ps                     # status / health
docker compose logs -f neo4j          # logs
docker compose stop                   # stop, keep data
docker compose down -v                # stop and DELETE all data
uv sync                               # after a pull: install the locked versions
uv lock --upgrade                     # newer versions; then run the tests and commit uv.lock
uv run python team_agent.py           # Lesson 7 agent (interactive)
uv run python -m companion            # companion chat (/help for commands; --persona Coach, --sync)
uv run python -m companion.web        # Lessons 17-18 web UI on http://127.0.0.1:8000 (click a reply to see its trace)
uv run python -m companion.demo       # scripted two-session companion demo
uv run python -m companion.scenarios  # Lesson 11–16 scenario tests (real LLM calls)
uv run python -m companion.calibrate  # Lesson 19: measure thresholds for the embedding model in .env
uv run python -m companion.reembed    # Lesson 19: backfill vectors after switching models
uv run python -m companion.reextract --user Mark --dry-run  # Lesson 20: re-read old conversations
uv run python graphrag.py "Who should own our new Rust performance work?"
```
