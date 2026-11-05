# Neo4j graph tutorial

Hands-on lessons on graph databases with Neo4j — from Cypher basics to graph algorithms,
vector search and GraphRAG — with every query's recorded output.

## Quick start

```bash
cp .env.example .env              # set NEO4J_PASSWORD; ANTHROPIC_API_KEY is optional
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

## Common commands

```bash
docker compose ps                 # status / health
docker compose logs -f neo4j      # logs
docker compose stop               # stop, keep data
docker compose down -v            # stop and DELETE all data
.venv/bin/python graphrag.py "Who should own our new Rust performance work?"
```
