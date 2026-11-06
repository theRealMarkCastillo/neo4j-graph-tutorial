# Lesson 1 — Graph basics and reading data with Cypher

## Concepts

A relational database stores **tables** and connects them at query time with JOINs.
A graph database stores the **connections themselves** as data:

- **Nodes** — things. They have **labels** (`:Person`, `:Project`) — think "type".
- **Relationships** — connections. They have a **type** (`:WORKS_ON`) and a **direction**.
- **Properties** — key/value pairs on nodes *and* relationships (`{name:'Alice'}`, `{since:2019}`).

Following a relationship is a pointer hop, so "friends of friends of friends" stays fast where
SQL would need another JOIN per hop.

### Cypher draws patterns in ASCII

```
(alice:Person)-[:KNOWS]->(bob:Person)
   node        relationship     node
```

### The sample graph

People `KNOWS` each other, `Dev` `MANAGES` Alice and Cara, people `HAS_SKILL {level}`,
people `WORKS_ON` projects, projects `REQUIRES` skills.

---

## Getting edges to show in the Browser

`MATCH (n) RETURN n` only returns nodes, so the Browser shows dots with no lines. A query
returns exactly what you ask for — return relationships or paths to see edges:

```cypher
MATCH p=()-[]->() RETURN p
```

```
Nodes 12, Relationships 20
HAS_SKILL 6, KNOWS 3, MANAGES 2, REQUIRES 4, WORKS_ON 5
```

Equivalent: `MATCH (a)-[r]->(b) RETURN a, r, b`

---

## Exercise: who works on "Mobile App"?

```cypher
MATCH (p:Person)-[:WORKS_ON]->(:Project {name:'Mobile App'})
RETURN p.name
```

```
p.name
"Bob"
"Cara"
```

---

## Ten queries from basic to advanced

### 1 — Filtering
```cypher
MATCH (p:Person) WHERE p.role = 'Engineer'
RETURN p.name ORDER BY p.name
```
```
"Alice"
"Cara"
```
`WHERE` works like SQL. `{role:'Engineer'}` inside the pattern does the same thing.

### 2 — Filtering on relationship properties
```cypher
MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill)
WHERE r.level >= 4
RETURN p.name, s.name, r.level ORDER BY r.level DESC
```
```
p.name, s.name, r.level
"Alice", "Python", 5
"Bob", "UX", 5
"Eve", "ML", 5
"Cara", "TypeScript", 4
"Eve", "Python", 4
```
Alice's TypeScript (level 3) is filtered out — the condition is on the *edge*.

### 3 — Aggregation (no GROUP BY)
```cypher
MATCH (p:Person)-[:WORKS_ON]->(pr:Project)
RETURN pr.name, count(p) AS people, collect(p.name) AS team
```
```
pr.name, people, team
"Search Revamp", 2, ["Alice", "Eve"]
"Mobile App", 2, ["Cara", "Bob"]
"Legacy Migration", 1, ["Alice"]
```
Grouping is implicit: every non-aggregated column is a grouping key. `collect()` builds lists.

### 4 — Finding what's missing
```cypher
MATCH (p:Person) WHERE NOT (p)-[:WORKS_ON]->()
RETURN p.name
```
```
"Dev"
```
A pattern inside `WHERE` is an existence check.

### 5 — Multi-hop (friend of a friend)
```cypher
MATCH (:Person {name:'Alice'})-[:KNOWS]->()-[:KNOWS]->(fof)
RETURN fof.name
```
```
"Eve"
```
Alice → Cara → Eve. One more hop is just more pattern, not another JOIN.

### 6 — Variable-length paths
```cypher
MATCH (:Person {name:'Alice'})-[:KNOWS*1..3]->(x)
RETURN DISTINCT x.name
```
```
"Cara"
"Bob"
"Eve"
```
`*1..3` = follow KNOWS 1 to 3 times. In SQL this would need a recursive CTE.

### 7 — Shortest path across any relationship type
```cypher
MATCH path = shortestPath(
  (:Person {name:'Bob'})-[*]-(:Person {name:'Eve'}))
RETURN [n IN nodes(path) | n.name] AS route, length(path) AS hops
```
```
route, hops
["Bob", "Alice", "Search Revamp", "Eve"], 3
```
No arrowhead (`-[*]-`) ignores direction, so the route can pass *through a project*.
The Browser warns about the unbounded `[*]` — prefer `[*..6]` on real data.

### 8 — Recommendation (who could join a project?)
```cypher
MATCH (pr:Project {name:'Mobile App'})-[:REQUIRES]->(s)<-[:HAS_SKILL]-(p:Person)
WHERE NOT (p)-[:WORKS_ON]->(pr)
RETURN p.name, s.name
```
```
p.name, s.name
"Alice", "TypeScript"
```
Arrows meet in the middle: "a skill the project needs **and** a person has". Cara already
works on it, so she's excluded.

### 9 — Pipelining with `WITH` (single points of failure)
```cypher
MATCH (pr:Project {status:'active'})-[:REQUIRES]->(s:Skill)
OPTIONAL MATCH (s)<-[:HAS_SKILL]-(p:Person)
WITH pr, s, count(p) AS experts, collect(p.name) AS who
WHERE experts < 2
RETURN pr.name, s.name, experts, who
```
```
pr.name, s.name, experts, who
"Search Revamp", "ML", 1, ["Eve"]
"Mobile App", "UX", 1, ["Bob"]
```
`OPTIONAL MATCH` ≈ LEFT JOIN. `WITH` pipes results to the next stage, so you can filter on
an aggregate (like SQL's `HAVING`).

### 10 — Chained traversal
```cypher
MATCH (m:Person)-[:MANAGES]->(e)-[:WORKS_ON]->(pr)
RETURN m.name, collect(DISTINCT pr.name) AS projects
```
```
m.name, projects
"Dev", ["Legacy Migration", "Search Revamp", "Mobile App"]
```
As a picture, return `path = (m)-[:MANAGES]->(e)-[:WORKS_ON]->(pr)`: 6 nodes, 5 relationships.

---

## Concepts and design choices

*Where this fits:* Reading the graph: the traversal that recall (step 8) relies on. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [graph](../docs/glossary.md#graph) | sense 1: the Neo4j data graph of nodes and relationships | a LangGraph state graph (Lesson 7) |
| traversal / hop | following stored relationships from node to node | a SQL join (computed at query time from keys) |
| Cypher | Neo4j's pattern-matching query language; GQL (ISO, 2024) is close to it | SPARQL (RDF), Gremlin (TinkerPop) |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| A property graph queried with Cypher patterns | multi-hop and variable-length questions read like the picture you'd draw | a second query language next to SQL | Postgres recursive CTEs; Apache AGE (openCypher on Postgres); SPARQL over RDF | [D1](../docs/design-decisions.md#d1) |
| Variable-length paths (`*1..3`) for open-ended questions | no need to know the depth in advance | unbounded patterns can explode on dense graphs: always cap the length | fixed-depth queries; precomputed relationships | [D11](../docs/design-decisions.md#d11) |

## Test it — [`tests/test_01_reading.py`](../tests/test_01_reading.py)

Each test pins one query's answer on the seed graph:

```python
def test_exercise_who_works_on_mobile_app(driver):
    team = names(driver, "MATCH (p:Person)-[:WORKS_ON]->(:Project {name:'Mobile App'}) RETURN p.name AS name")
    assert {"Bob", "Cara"} <= team
```

Note the `<=` (subset): later lessons add people, so assert what *this* lesson established
rather than exact totals that will change. Brittle tests that break for good reasons teach
people to ignore failures.

```bash
uv run python -m pytest tests/test_01_reading.py -v
```

## Takeaways

- **Return nodes, relationships or paths** to get a picture; return values to get a table.
- **Patterns are the query.** Arrows can point either way, or meet in the middle `->(s)<-`.
- **Hops are cheap.** `*1..3` and `shortestPath` replace recursive SQL.
- **`WITH`** joins query stages into a pipeline; use it to filter on aggregates.

## Next

[Lesson 2](02-writing-data.md) changes the graph: `CREATE`, `MERGE`, `SET` and `DELETE`, and the constraints that keep it clean.

<!-- nav -->
---

**[Part 1: Neo4j fundamentals](../README.md#part-1-neo4j-fundamentals)** (lessons 0–4)

[← Lesson 0: Setup](00-setup.md) · [All lessons](../README.md#lessons) · [Lesson 2: Writing data →](02-writing-data.md)
