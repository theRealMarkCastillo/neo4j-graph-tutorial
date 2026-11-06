# Lesson 3 — Modeling and performance

## 3.1 — Turn a property into a node

```cypher
MATCH (p:Person) WHERE p.role IS NOT NULL
MERGE (r:Role {name: p.role})
MERGE (p)-[:HAS_ROLE]->(r)
RETURN p, r
```
```
Created 5 nodes, created 6 relationships, set 5 properties, added 5 labels
```
Returning only `p, r` shows no edges (the Lesson 1 lesson again!). Since it's a `MERGE`,
just query the paths:

```cypher
MATCH path = (:Person)-[:HAS_ROLE]->(:Role) RETURN path
```
```
Nodes 11 (Person 6, Role 5), Relationships 6 (HAS_ROLE)
```
Alice and Cara now share **one** `Engineer` node.

**Why?** As a property, `role` can only be filtered on. As a node, a role can be
*traversed*: `(me)-[:HAS_ROLE]->(r)<-[:HAS_ROLE]-(peer)`, and you can attach data to it.

> **Rule of thumb:** if you'll ask "who else shares this value?" or want to hang data off it,
> make it a **node**. If it just describes one thing (name, date, score), keep a **property**.

## 3.2 — Turn a relationship into a node

A relationship connects exactly two nodes. Alice's work on Search Revamp involves three
(Alice, the project, her role on it) plus its own data (60% allocation, start date):

```cypher
MATCH (a:Person {name:'Alice'}), (pr:Project {name:'Search Revamp'})
MERGE (asg:Assignment {id:'alice-search'})
SET asg.allocation = 0.6, asg.start = date('2026-01-15')
MERGE (a)-[:ASSIGNED]->(asg)
MERGE (asg)-[:ON]->(pr)
MERGE (r:Role {name:'Tech Lead'})
MERGE (asg)-[:AS_ROLE]->(r)
WITH asg MATCH path = (asg)-[]-() RETURN path
```
```
Created 2 nodes, created 3 relationships, set 4 properties, added 2 labels
Alice -ASSIGNED-> Assignment(0.6) -ON-> Search Revamp
                              \--AS_ROLE-> Tech Lead
```
Use this when a relationship needs to point at a third thing, or other things need to point
at *it*. Common cases: orders, transactions, employment stints, meetings.

## 3.3 — Reading execution plans with PROFILE

`EXPLAIN` shows the plan without running the query; `PROFILE` runs it and shows real numbers.
Read plans **bottom-up**.

### a) No label → scan every node
```cypher
PROFILE MATCH (n {name:'Alice'}) RETURN n
```
```
+ProduceResults   rows 1   db hits 3
+Filter           rows 1   db hits 23   n.name = $autostring_0
+AllNodesScan     rows 23  db hits 24
Total database accesses: 50
```

### b) With a label → index seek
```cypher
PROFILE MATCH (n:Person {name:'Alice'}) RETURN n
```
```
+ProduceResults          rows 1   db hits 2
+NodeUniqueIndexSeek     rows 1   db hits 2   UNIQUE n:Person(name) WHERE name = $autostring_0
Total database accesses: 4
```
**50 → 4** from adding `:Person`. On millions of nodes that's milliseconds vs minutes.
**Always label the node a query starts from.**

### c) Traversal = Expand, no joins
```cypher
PROFILE MATCH (:Person {name:'Alice'})-[:KNOWS]->(friend)-[:HAS_SKILL]->(s:Skill)
RETURN friend.name, s.name
```
```
+ProduceResults         rows 2   db hits 0
+Projection             rows 2   db hits 4
+Filter                 rows 2   db hits 2    s:Skill
+Expand(All)            rows 2   db hits 12   (friend)-[:HAS_SKILL]->(s)
+Expand(All)            rows 2   db hits 11   (anon)-[:KNOWS]->(friend)
+NodeUniqueIndexSeek    rows 1   db hits 2    UNIQUE :Person(name)
```
Index once to find the start, then **Expand** follows pointers. Cost grows with what you
touch, not database size — **index-free adjacency**.

("Estimated Rows 0" just means planner statistics hadn't caught up on a tiny new database.)

## 3.4 — Parameters

```cypher
:param who => 'Cara'
```
```cypher
MATCH path = (:Person {name: $who})-[:HAS_SKILL|WORKS_ON|HAS_ROLE]->() RETURN path
```
```
Cara -HAS_SKILL-> TypeScript, Cara -WORKS_ON-> Mobile App, Cara -HAS_ROLE-> Engineer
```
`|` = "any of these types". In app code always use `$params`: no Cypher injection, and the
plan is compiled once and cached.

## Supernodes (concept only)

A node with millions of relationships (e.g. `:Country {name:'USA'}`) makes any traversal
*through* it explode. Fixes: use a property instead, split the node (per year/region), or use
more specific relationship types so queries only expand the slice they need.

---

## Test it — [`tests/test_03_modeling.py`](../tests/test_03_modeling.py)

Tests can check *how* a query runs, not just what it returns. `EXPLAIN` gives the plan
without running the query, so a test can fail when someone drops the label that made the
index usable:

```python
with_label = plan_ops(driver, "MATCH (n:Person {name:'Alice'}) RETURN n")
assert any("IndexSeek" in op for op in with_label)
assert "AllNodesScan" in plan_ops(driver, "MATCH (n {name:'Alice'}) RETURN n")
```

`test_parameters_not_string_formatting` passes an injection-shaped name as a parameter
and shows it's treated as plain data.

## Cheat sheet

| If the thing... | Make it a... |
|---|---|
| Describes one entity | **Property** |
| Is shared ("who else has X?") | **Node** (`:Role`, `:City`, `:Tag`) |
| Connects two things, maybe with a little data | **Relationship** with properties |
| Connects 3+ things, or is referenced itself | **Node** in the middle (`:Assignment`, `:Order`) |
| Groups/tags nodes cheaply | **Extra label** (`:Mentor`) |

Performance: avoid **AllNodesScan** / **NodeByLabelScan + Filter**; aim for **IndexSeek**;
index start points; use `$parameters`.
