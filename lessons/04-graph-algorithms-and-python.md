# Lesson 4 — Graph algorithms (GDS) and Python

Needs the Graph Data Science plugin — already enabled by the [Dockerfile](../Dockerfile).
Check: `RETURN gds.version()` → `"2.13.14"` in the original run.

## 4.1 — Project an in-memory graph

GDS algorithms run on an in-memory **projection** containing only the nodes and relationships
you pick.

```cypher
CALL gds.graph.project('team', ['Person','Project','Skill'], {
  KNOWS:     {orientation:'UNDIRECTED'},
  MANAGES:   {orientation:'UNDIRECTED'},
  WORKS_ON:  {orientation:'UNDIRECTED'},
  HAS_SKILL: {orientation:'UNDIRECTED'},
  REQUIRES:  {orientation:'UNDIRECTED'}
}) YIELD graphName, nodeCount, relationshipCount
RETURN graphName, nodeCount, relationshipCount
```
```
graphName, nodeCount, relationshipCount
"team", 16, 50
```
50 = 25 relationships stored in both directions (`UNDIRECTED`). Role/Assignment nodes are left out.

## 4.2 — PageRank: who matters most?

```cypher
CALL gds.pageRank.stream('team') YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
RETURN labels(n)[0] AS type, n.name AS name, round(score, 3) AS pagerank
ORDER BY pagerank DESC
```
```
type, name, pagerank
"Person", "Alice", 2.143
"Person", "Cara", 1.322
"Skill", "Python", 1.131
"Project", "Mobile App", 1.124
"Project", "Search Revamp", 1.108
"Person", "Eve", 1.106
"Person", "Finn", 0.91
"Person", "Bob", 0.881
"Person", "Dev", 0.855
"Skill", "TypeScript", 0.836
"Skill", "Rust", 0.694
...
```
High score = connected to other well-connected nodes. Alice is the hub of this team.

## 4.3 — Louvain: find communities

```cypher
CALL gds.louvain.stream('team') YIELD nodeId, communityId
WITH communityId, gds.util.asNode(nodeId) AS n
RETURN communityId, collect(labels(n)[0] + ': ' + n.name) AS members
ORDER BY size(members) DESC
```
```
communityId, members
10, ["Person: Alice", "Person: Cara", "Person: Dev", "Skill: TypeScript", "Project: Legacy Migration", "Person: Finn"]
7,  ["Person: Eve", "Skill: Python", "Skill: ML", "Project: Search Revamp"]
1,  ["Person: Bob", "Skill: UX", "Project: Mobile App"]
14, ["Skill: Rust", "Person: Gia"]
15, ["Skill: Go"]
```
Structure only — no names or labels used — yet it finds a core team, a Search/ML pod, a design
pod, the Rust newcomers, and an orphaned `Go` (Hal was deleted in Lesson 2).
Community ids vary between runs.

## 4.4 — Node similarity: who's alike?

```cypher
CALL gds.nodeSimilarity.stream('team', {
  relationshipTypes:['HAS_SKILL'], nodeLabels:['Person','Skill'], topK:1
}) YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS a, gds.util.asNode(node2) AS b, similarity
WHERE a:Person
RETURN a.name AS person, b.name AS most_similar, round(similarity, 2) AS jaccard
ORDER BY jaccard DESC
```
```
person, most_similar, jaccard
"Alice", "Cara", 0.5
"Cara", "Alice", 0.5
"Finn", "Gia", 0.5
"Gia", "Finn", 0.5
"Eve", "Alice", 0.33
```
Jaccard = shared skills ÷ total distinct skills. The core of "people like you…" recommendations.

## 4.5 — Write results back

```cypher
CALL gds.pageRank.write('team', {writeProperty:'pagerank'}) YIELD nodePropertiesWritten AS pr
CALL gds.louvain.write('team', {writeProperty:'community'}) YIELD communityCount, nodePropertiesWritten AS lv
RETURN pr, lv, communityCount
```
```
pr, lv, communityCount
16, 16, 5
```
```cypher
MATCH (p:Person) WHERE p.pagerank > 1
RETURN p.name, round(p.pagerank, 2) AS pagerank, p.community ORDER BY pagerank DESC
```
```
p.name, pagerank, p.community
"Alice", 2.14, 10
"Cara", 1.32, 10
"Eve", 1.11, 7
```
Modes: `.stream` (return), `.stats` (summary), `.mutate` (projection only), `.write` (database).

## 4.6 — From Python: [`team_graph.py`](../team_graph.py)

```bash
uv run python team_graph.py
```
```
Active projects:
  Search Revamp: Eve, Alice
  Mobile App: Bob, Cara
  (took 41 ms)

Wrote: Bob -[:HAS_SKILL {level:2}]-> TypeScript

Alice's network (up to 3 hops), by PageRank:
  Cara   pagerank=1.32
  Eve    pagerank=1.11
  Bob    pagerank=0.88
```
Three driver patterns:
1. `driver.execute_query(...)` for one-offs (handles session + retries).
2. `session.execute_write(fn)` / `execute_read(fn)` — retried on transient errors.
3. `$params` everywhere, never f-strings in Cypher.

**Gotcha:** a projection is a snapshot — it won't see Bob's new skill. Refresh with
`CALL gds.graph.drop('team')` and project again.

---

## Concepts and design choices

*Where this fits:* Analytics over the graph, used again for the life graph in Lesson 15. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| [projection](../docs/glossary.md#graph) | sense 4: an in-memory copy of part of the graph for GDS | a Cypher map projection `m {.id, .text}` |
| [node similarity](../docs/glossary.md#similarity-and-score) | sense 4: Jaccard over shared neighbours | cosine similarity of embeddings |
| [memory](../docs/glossary.md#memory) | sense 5: RAM used by GDS projections | the companion's memories |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| Run algorithms inside Neo4j with GDS | no data export, results written back as properties | projections use RAM and must be dropped | export to NetworkX / igraph / Spark GraphFrames | [D1](../docs/design-decisions.md#d1) |
| Stream results while exploring, write back only what queries need | nothing stale stored by accident | write-back must be re-run when the graph changes | always write back; compute on every request | [D14](../docs/design-decisions.md#d14) |

## Test it — [`tests/test_04_algorithms_and_python.py`](../tests/test_04_algorithms_and_python.py)

- **Design for tests.** `add_skill(tx, ...)` takes a transaction, so the test hands it a
  rolled-back one. [`team_graph.py`](../team_graph.py) keeps its demo under
  `if __name__ == "__main__"`, so importing it writes nothing.
- **Test algorithms on a graph you can reason about by hand.** In a 5-node star, PageRank
  *must* rank the hub first. `test_pagerank_on_a_tiny_known_graph` builds one with
  `gds.graph.project.cypher` and drops it afterwards.

## Cheat sheet

| Step | Cypher |
|---|---|
| Load a subgraph into memory | `CALL gds.graph.project('name', labels, relTypes)` |
| Influence / importance | `gds.pageRank` (also `gds.betweenness`, `gds.degree`) |
| Clusters | `gds.louvain` (also `gds.wcc`, `gds.labelPropagation`) |
| "Similar to" | `gds.nodeSimilarity`, `gds.knn` |
| Paths | `gds.shortestPath.dijkstra`, `gds.allShortestPaths` |
| Clean up | `CALL gds.graph.drop('name')` |

## Next

[Lesson 5](05-vectors-and-graphrag.md) adds meaning: vectors on nodes, and GraphRAG that starts from a vector match and follows the graph from there.

<!-- nav -->
---

**[Part 1: Neo4j fundamentals](../README.md#part-1-neo4j-fundamentals)** (lessons 0–4)

[← Lesson 3: Modeling and performance](03-modeling-and-performance.md) · [All lessons](../README.md#lessons) · [Lesson 5: Vector search and GraphRAG →](05-vectors-and-graphrag.md)
