# Neo4j graph tutorial

Hands-on lessons on graph databases with Neo4j — from Cypher basics to graph algorithms,
vector search, GraphRAG, LangGraph agents, and a graph + vector biographical memory for an
AI companion — with every query's recorded output.

## Quick start

```bash
cp .env.example .env              # set NEO4J_PASSWORD; LLM_* for lessons 5 and 7–18
docker compose up -d --build      # Neo4j 5 + APOC + Graph Data Science
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

- Neo4j Browser: http://localhost:7474 (user `neo4j`, password from `.env`)
- Fresh database? Load the starting graph — see [Lesson 0](lessons/00-setup.md).

## Lessons

| # | Lesson | Covers |
|---|---|---|
| 0 | [Setup](lessons/00-setup.md) | Docker, seed data, Python env, reset |
| 1 | [Graph basics and reading](lessons/01-graph-basics-and-reading.md) | Nodes/relationships, `MATCH`, filtering, aggregation, multi-hop, shortest path, `WITH` |
| 2 | [Writing data](lessons/02-writing-data.md) | `CREATE`, `MERGE`, `SET`/`REMOVE`, constraints, indexes, `UNWIND`, `DETACH DELETE` |
| 3 | [Modeling and performance](lessons/03-modeling-and-performance.md) | Property vs node, reifying relationships, `PROFILE`, parameters, supernodes |
| 4 | [Graph algorithms and Python](lessons/04-graph-algorithms-and-python.md) | GDS projections, PageRank, Louvain, similarity, write-back, Python driver |
| 5 | [Vectors and GraphRAG](lessons/05-vectors-and-graphrag.md) | Vector indexes, local embeddings, hybrid vector + graph retrieval |
| 6 | [Graphs in AI and agents](lessons/06-graphs-in-ai-and-agents.md) | Agent memory, graph tools, when embeddings help |
| 7 | [LangGraph agent with graph tools](lessons/07-langgraph-agent-with-graph-tools.md) | `StateGraph`, `ToolNode`, checkpointer, read-only Cypher tool |
| 8 | [Memory design for companions](lessons/08-memory-design-for-companions.md) | Memory kinds, life graph, episodes, bi-temporal facts, boundaries |
| 9 | [The memory write path](lessons/09-memory-write-path.md) | Structured extraction, entity resolution, dedupe, supersede, plans |
| 10 | [Recall and the companion chat](lessons/10-companion-recall-and-chat.md) | Hybrid recall, LangGraph companion, terminal CLI, two-session demo |
| 11 | [Harder scenarios and evals](lessons/11-harder-scenarios-and-evals.md) | Context-window + end-of-episode extraction, same-name people, ended relations, history, forgetting, sensitive memories, scenario tests |
| 12 | [An ontology for the life graph](lessons/12-ontology.md) | Relation vocabulary, synonyms, hierarchy, symmetric/inverse/cardinality/range rules, kinship inference, `(:RelType)` nodes; terms in the [glossary](docs/glossary.md) |
| 13 | [Memory over time](lessons/13-memory-over-time.md) | Expiry (`valid_until`, status memories), decay and staleness, consolidation into habits, pruning, one testable clock |
| 14 | [Production hardening](lessons/14-production-hardening.md) | One transaction per write, a user-node lock against duplicate races, background extraction, thread safety |
| 15 | [Graph embeddings](lessons/15-graph-embeddings.md) | FastRP + kNN vs. node similarity on the life graph, Cypher projections, seeds and cut-offs |
| 16 | [Personas](lessons/16-personas.md) | Several companions sharing one memory: private vs. shared memories, hiding derived data, per-persona diaries |
| 17 | [A web UI](lessons/17-web-ui.md) | FastAPI + one HTML page: chat, life graph, decay curves, related entities; API tests with a fake LLM |
| 18 | [Watching memory work](lessons/18-seeing-memory-work.md) | Per-turn trace: what recall found by retriever, the exact prompt, why each write happened, timings, persona lens, look-ahead clock, diary |

**Architecture:** component, LangGraph, sequence, data-model and lifecycle diagrams for the
agent and the companion are in [docs/architecture.md](docs/architecture.md).

## LLM settings (lessons 5 and 7–18)

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

Embeddings run locally (`BAAI/bge-small-en-v1.5` via fastembed), no key needed.

## Files

| Path | What |
|---|---|
| `Dockerfile`, `docker-compose.yml` | Neo4j 5.26 with APOC + GDS, data in named volumes |
| `.env.example` → `.env` | Neo4j credentials and LLM keys (`.env` is git-ignored) |
| `config.py` | Loads `.env` for the scripts |
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
| `docs/architecture.md` | Architecture diagrams (Mermaid) for lessons 7–17 |
| `tests/`, `pytest.ini` | Per-lesson tests and LLM evals (see *Tests and evals*) |
| `docs/glossary.md` | Glossary of overlapping memory / graph / ontology terms |
| `lessons/outputs/` | Recorded runs for lessons 7, 10, 11, 12, 13 and 17 |

## Tests and evals

Every lesson has tests in [`tests/`](tests), named after it (`test_05_vectors_and_graphrag.py`).

```bash
.venv/bin/python -m pytest                    # unit + database tests: seconds, free (default)
.venv/bin/python -m pytest -m "not db and not llm"   # unit tests only, no Neo4j needed
.venv/bin/python -m pytest -m llm             # LLM evals: real calls, cost money
EVAL_TRIALS=3 .venv/bin/python -m pytest -m llm tests/test_11_extraction_evals.py   # pass rates
```

A `-m` on the command line *replaces* the default `-m "not llm"` in `pytest.ini`, so spell
out `and not llm` when you add your own filter. Database tests never leave data behind: writes run in rolled-back transactions, and memory
tests use their own `pytest` user. A test for a lesson you haven't done yet is skipped with a
reason. [Lesson 0](lessons/00-setup.md#test-it) explains the three kinds of test, and each
lesson ends with a *Test it* section on what its tests teach.

## Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f neo4j      # logs
docker compose stop               # stop, keep data
docker compose down -v            # stop and DELETE all data
.venv/bin/python team_agent.py    # Lesson 7 agent (interactive)
.venv/bin/python -m companion     # companion chat (/help for commands; --persona Coach, --sync)
.venv/bin/python -m companion.web # Lessons 17-18 web UI on http://127.0.0.1:8000 (click a reply to see its trace)
.venv/bin/python -m companion.demo   # scripted two-session companion demo
.venv/bin/python -m companion.scenarios   # Lesson 11–16 scenario tests (real LLM calls)
.venv/bin/python graphrag.py "Who should own our new Rust performance work?"
```
