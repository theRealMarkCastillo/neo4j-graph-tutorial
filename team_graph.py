"""Lesson 4.6 — talking to Neo4j from Python with the official driver."""
from neo4j import GraphDatabase

from config import NEO4J_AUTH as AUTH, NEO4J_URI as URI


def add_skill(tx, person: str, skill: str, level: int):
    # MERGE keeps it idempotent; $params avoid injection and reuse the cached plan.
    tx.run(
        """
        MATCH (p:Person {name: $person})
        MERGE (s:Skill {name: $skill})
        MERGE (p)-[h:HAS_SKILL]->(s)
        SET h.level = $level
        """,
        person=person, skill=skill, level=level,
    )


def friends_of_friends(tx, name: str):
    result = tx.run(
        """
        MATCH (:Person {name: $name})-[:KNOWS*1..3]->(x:Person)
        RETURN DISTINCT x.name AS name, x.pagerank AS pagerank
        ORDER BY pagerank DESC
        """,
        name=name,
    )
    return [record.data() for record in result]


with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()

    # 1) Quick one-off query: execute_query handles session + retries for you.
    records, summary, _ = driver.execute_query(
        "MATCH (p:Person)-[:WORKS_ON]->(pr:Project {status: $status}) "
        "RETURN pr.name AS project, collect(p.name) AS team",
        status="active",
        database_="neo4j",
    )
    print("Active projects:")
    for r in records:
        print(f"  {r['project']}: {', '.join(r['team'])}")
    print(f"  (took {summary.result_available_after} ms)\n")

    # 2) Managed transactions: retried automatically on transient errors.
    with driver.session(database="neo4j") as session:
        session.execute_write(add_skill, "Bob", "TypeScript", 2)
        print("Wrote: Bob -[:HAS_SKILL {level:2}]-> TypeScript\n")

        print("Alice's network (up to 3 hops), by PageRank:")
        for row in session.execute_read(friends_of_friends, "Alice"):
            pr = row["pagerank"]
            print(f"  {row['name']:<6} pagerank={pr:.2f}" if pr else f"  {row['name']}")
