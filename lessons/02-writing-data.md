# Lesson 2 — Writing data

Run these in order after Lesson 1. Each step's output is what the Browser status bar showed.

### 2.1 — CREATE a node
```cypher
CREATE (f:Person {name:'Finn', role:'Engineer'}) RETURN f
```
```
Created 1 node, set 2 properties, added 1 label
```
No schema to declare first — you don't define a `Person` table, you just create one.

### 2.2 — Connect existing nodes (MATCH, then CREATE)
```cypher
MATCH (f:Person {name:'Finn'}), (a:Person {name:'Alice'}), (d:Person {name:'Dev'})
CREATE (f)-[k:KNOWS {since:2026}]->(a), (d)-[m:MANAGES]->(f)
RETURN f, k, a, d, m
```
```
Created 2 relationships, set 1 property
(info: "This query builds a cartesian product between disconnected patterns")
```
The info note is expected here — each comma-separated pattern matches exactly one node.

**Gotcha:** `CREATE (:Person {name:'Alice'})-[:KNOWS]->(...)` *without* the `MATCH` creates a
**second Alice**.

### 2.3 — MERGE = "find it, or create it if missing"
```cypher
MERGE (:Skill {name:'Rust'}) MERGE (:Skill {name:'Rust'}) MERGE (:Skill {name:'Rust'})
WITH 1 AS ignore
MATCH (r:Skill {name:'Rust'}) RETURN count(r) AS rust_nodes
```
```
rust_nodes
1
Created 1 node, set 1 property, added 1 label
```
Safe to repeat (idempotent). With `CREATE` you'd have 3 Rust nodes.

> Side lesson: my first try used `MATCH (all:Skill ...)` and failed with a syntax error —
> `all` is a built-in function (`all(x IN list WHERE ...)`) and can't be a variable name.

### 2.4 — MERGE with ON CREATE / ON MATCH (+ UNWIND)
```cypher
UNWIND ['Finn','Finn','Gia'] AS who
MERGE (p:Person {name: who})
  ON CREATE SET p.created = datetime(), p.visits = 1
  ON MATCH  SET p.visits = coalesce(p.visits, 0) + 1
RETURN p.name, p.visits, p.created
```
```
p.name, p.visits, p.created
"Finn", 2, null
"Finn", 2, null
"Gia", 1, 2026-11-05T23:01:00.557000000Z
Created 1 node, set 5 properties, added 1 label
```
- Gia was new → `ON CREATE`. Finn existed → `ON MATCH` twice (0→1→2).
- Both Finn rows show **2** because they're the *same node*, shown in its final state.
- `UNWIND` turns a list into rows — the standard batch-load pattern.

### 2.5 — SET / REMOVE properties and labels
```cypher
MATCH (p:Person {name:'Finn'})
SET p.role = 'Senior Engineer', p:Mentor
REMOVE p.visits
RETURN p.name, labels(p), properties(p)
```
```
p.name, labels(p), properties(p)
"Finn", ["Person", "Mentor"], {role:"Senior Engineer", name:"Finn"}
Set 2 properties, added 1 label
```
A node can have any number of labels — handy as tags or roles. `SET p.x = null` also removes.

### 2.6 — Constraints
```cypher
CREATE CONSTRAINT person_name IF NOT EXISTS
FOR (p:Person) REQUIRE p.name IS UNIQUE
```
```
Added 1 constraint
```
```cypher
CREATE (:Person {name:'Alice', role:'Impostor'})
```
```
Neo.ClientError.Schema.ConstraintValidationFailed
Node(0) already exists with label `Person` and property `name` = 'Alice'
```
A uniqueness constraint also creates an index, and makes `MERGE` safe under concurrency.

### 2.7 — Indexes
```cypher
CREATE INDEX project_status IF NOT EXISTS FOR (p:Project) ON (p.status)
```
```cypher
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, owningConstraint
```
```
name, type, labelsOrTypes, properties, owningConstraint
"index_343aff4e", "LOOKUP", null, null, null
"index_f7700477", "LOOKUP", null, null, null
"person_name", "RANGE", ["Person"], ["name"], "person_name"
"project_status", "RANGE", ["Project"], ["status"], null
```
Index the properties a query **starts from**. The hops after that don't need indexes.

### 2.8 — Bulk load: UNWIND + MERGE
```cypher
UNWIND [{name:'Gia', skill:'Rust', level:4}, {name:'Gia', skill:'Python', level:3},
        {name:'Finn', skill:'Rust', level:2}, {name:'Hal', skill:'Go', level:5}] AS row
MERGE (p:Person {name: row.name})
MERGE (s:Skill {name: row.skill})
MERGE (p)-[h:HAS_SKILL]->(s)
SET h.level = row.level
RETURN p, h, s
```
```
Created 2 nodes, created 4 relationships, set 6 properties, added 2 labels
```
Only Hal and Go were new. Re-runnable without duplicates.

### 2.9 — DELETE vs DETACH DELETE
```cypher
MATCH (p:Person {name:'Hal'}) DELETE p
```
```
Neo.ClientError.Schema.ConstraintValidationFailed
Cannot delete node<15>, because it still has relationships. To delete this node, you must first delete its relationships.
```
```cypher
MATCH (p:Person {name:'Hal'}) DETACH DELETE p
```
```
Deleted 1 node, deleted 1 relationship
```
The `Go` skill survives — deleting Hal doesn't touch his neighbors.

⚠️ `MATCH (n) DETACH DELETE n` wipes the entire database.

---

## Concepts and design choices

*Where this fits:* Writing the graph: what the memory write path (steps 3, 6, 7) is built from. The whole pipeline, step by step, is in [design decisions → the pipeline as concepts](../docs/design-decisions.md#the-pipeline-as-concepts).

**Terms with a specific meaning here** (every sense is in the [glossary](../docs/glossary.md#words-with-several-meanings)):

| Term | Means here | Not to be confused with |
|---|---|---|
| MERGE | find-or-create: the basis of identity (one Maya across 100 conversations) | an UPSERT on a key; MERGE matches the whole pattern |
| [schema](../docs/glossary.md#schema) | sense 1: constraints and indexes | a Pydantic schema (Lesson 9) or an ontology (Lesson 12) |
| [index](../docs/glossary.md#index) | sense 1: a lookup index on properties | a vector index (Lesson 5) |

**Design choices** (with alternatives in [design-decisions.md](../docs/design-decisions.md)):

| We chose | Because | It costs | Instead, you could | |
|---|---|---|---|---|
| MERGE on anything with identity, backed by a uniqueness constraint | idempotent writes; reruns don't duplicate | MERGE without a constraint can still race and create two nodes | CREATE plus application-side checks; content-hash keys | [D7](../docs/design-decisions.md#d7) |
| Batch with UNWIND | one round trip for many rows | one large transaction to retry if it fails | per-row writes; `neo4j-admin database import` for bulk loads | [D16](../docs/design-decisions.md#d16) |

## Test it — [`tests/test_02_writing.py`](../tests/test_02_writing.py)

Testing writes against a real database without leaving anything behind: run them in a
transaction and **roll it back** (the `tx` fixture in [`tests/conftest.py`](../tests/conftest.py)).

```python
def test_merge_is_idempotent(tx):
    tx.run("MERGE (:Skill {name:'Zig'}) MERGE (:Skill {name:'Zig'}) MERGE (:Skill {name:'Zig'})")
    assert count(tx, "MATCH (s:Skill {name:'Zig'}) RETURN count(s)") == 1
```

Writing the tests turned up a surprise. In Neo4j 5, a plain `DELETE` on a node that still
has relationships does **not** fail when the statement runs. It fails when the transaction
*commits*. `test_delete_needs_detach_when_node_has_relationships` pins that down.

## Cheat sheet

| Goal | Cypher |
|---|---|
| Add a node | `CREATE (n:Label {k: v})` |
| Link existing nodes | `MATCH (a), (b) ... CREATE (a)-[:REL]->(b)` |
| Find it, or create it if missing | `MERGE (n:Label {key: v})` |
| Different action for new vs existing | `MERGE ... ON CREATE SET ... ON MATCH SET ...` |
| Change properties or labels | `SET n.k = v`, `SET n:Extra`, `REMOVE n.k`, `REMOVE n:Extra` |
| Enforce uniqueness | `CREATE CONSTRAINT ... REQUIRE n.k IS UNIQUE` |
| Speed up lookups | `CREATE INDEX ... FOR (n:Label) ON (n.k)` |
| Bulk load | `UNWIND $rows AS row MERGE ...` |
| Delete a node | `DELETE` (fails if it has relationships) or `DETACH DELETE` |
