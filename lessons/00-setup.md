# Lesson 0 — Setup

## Start Neo4j

```bash
cp .env.example .env      # then set NEO4J_PASSWORD (and ANTHROPIC_API_KEY if you want LLM answers)
docker compose up -d --build
docker compose ps         # wait for "healthy" (first start downloads APOC + GDS, ~1 min)
```

- **Neo4j Browser:** http://localhost:7474 — log in as `neo4j` with the password from `.env`
- **Bolt URI (drivers):** `neo4j://localhost:7687`

`NEO4J_AUTH` only applies the first time the database is created. If you change the
password in `.env` later, also run `ALTER CURRENT USER SET PASSWORD FROM 'old' TO 'new'`
in the Browser (or reset with `docker compose down -v`, which **deletes all data**).

## Load the starting graph (fresh database only)

```bash
source .env
docker compose exec neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" -f /seed/00_seed.cypher
```

Check it:

```cypher
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n
```

```
label, n
"Person", 5
"Skill", 4
"Project", 3
```

## Python environment (Lessons 4–5)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All scripts read connection settings and keys from `.env` via [`config.py`](../config.py).

## Starting over

```bash
docker compose down -v      # deletes the database volume
docker compose up -d
# then reload seed/00_seed.cypher
```

> The lessons build on each other: later outputs assume you ran the earlier lessons' write
> queries in order. Outputs recorded here are from the original run on 2026-11-05.
