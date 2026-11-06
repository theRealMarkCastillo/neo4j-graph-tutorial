# Lesson 0 — Setup

## Start Neo4j

```bash
cp .env.example .env      # then set NEO4J_PASSWORD (and LLM_* if you want LLM answers)
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

## Test it

Every lesson from here on has tests in [`tests/`](../tests). Run them early and often:

```bash
.venv/bin/python -m pytest                 # fast tests: unit + database, a few seconds, free
.venv/bin/python -m pytest tests/test_00_setup.py -v
.venv/bin/python -m pytest -m llm          # LLM evals: real calls, cost money, vary run to run
```

Three kinds of test, cheapest first. Most of the suite is the first two.

| Kind | Needs | Example | Runs |
|---|---|---|---|
| **unit** | nothing | `normalize("girlfriend") == "partner"` | always |
| **db** | Neo4j up | a write in a rolled-back transaction | by default; skipped if Neo4j is down |
| **llm** (eval) | `LLM_*` in `.env` | "a move supersedes the old home" | only with `-m llm` |

[`tests/test_00_setup.py`](../tests/test_00_setup.py) is a *smoke test*: Neo4j answers, APOC and
GDS are installed, the seed is loaded. The cheapest test catches the most common problem.
A test for a lesson you haven't done yet is **skipped with a reason** (`-ra` shows it), not failed.

## Starting over

```bash
docker compose down -v      # deletes the database volume
docker compose up -d
# then reload seed/00_seed.cypher
```

> The lessons build on each other: later outputs assume you ran the earlier lessons' write
> queries in order. Outputs recorded here are from the original run on 2026-11-05.
