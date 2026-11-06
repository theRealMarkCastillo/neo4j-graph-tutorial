# Neo4j graph tutorial

Hands-on lessons on graph databases with Neo4j — from Cypher basics to graph algorithms,
vector search, GraphRAG, LangGraph agents, and a graph + vector biographical memory for an
AI companion — with every query's recorded output.

## Quick start

```bash
cp .env.example .env              # set NEO4J_PASSWORD; LLM_* for lessons 7–10
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

## LLM settings (lessons 7–10)

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
| `graphrag.py` | Lesson 5 — GraphRAG retrieval (+ Claude answer if a key is set) |
| `team_agent.py` | Lesson 7 — LangGraph agent with graph tools |
| `companion/` | Lessons 8–10 — memory schema, store, extraction, LangGraph companion, CLI, demo |
| `companion/scenarios.py` | Lesson 11 — multi-session scenario tests for memory |
| `lessons/outputs/` | Recorded runs for lessons 7, 10 and 11 |

## Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f neo4j      # logs
docker compose stop               # stop, keep data
docker compose down -v            # stop and DELETE all data
.venv/bin/python team_agent.py    # Lesson 7 agent (interactive)
.venv/bin/python -m companion     # Lesson 10 companion chat (/help for commands)
.venv/bin/python -m companion.demo   # scripted two-session companion demo
.venv/bin/python -m companion.scenarios   # Lesson 11 scenario tests (real LLM calls)
.venv/bin/python graphrag.py "Who should own our new Rust performance work?"
```
