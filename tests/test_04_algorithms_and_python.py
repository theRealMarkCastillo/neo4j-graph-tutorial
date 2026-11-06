"""Lesson 4 — GDS algorithms and the Python driver.

Driver code is easiest to test when each query lives in a function that takes a transaction
(`add_skill(tx, ...)`): the test hands it a rolled-back transaction instead of a real one.
That's also why team_graph.py keeps its demo under `if __name__ == "__main__"`.
"""
from conftest import require, rows
from team_graph import add_skill, friends_of_friends


def level(tx, person, skill):
    rec = tx.run("MATCH (:Person {name:$p})-[h:HAS_SKILL]->(:Skill {name:$s}) RETURN h.level AS l",
                 p=person, s=skill).single()
    return rec and rec["l"]


def test_add_skill_writes_and_updates_level(tx):
    add_skill(tx, "Bob", "Haskell", 2)
    assert level(tx, "Bob", "Haskell") == 2
    add_skill(tx, "Bob", "Haskell", 4)  # MERGE: updates, doesn't duplicate
    assert level(tx, "Bob", "Haskell") == 4
    n = tx.run("MATCH (:Person {name:'Bob'})-[h:HAS_SKILL]->(:Skill {name:'Haskell'}) RETURN count(h)").single()[0]
    assert n == 1


def test_add_skill_for_unknown_person_writes_nothing(tx):
    add_skill(tx, "Nobody", "Haskell", 3)  # MATCH finds no person, so the MERGEs never run
    assert tx.run("MATCH (s:Skill {name:'Haskell'}) RETURN count(s)").single()[0] == 0


def test_friends_of_friends(tx):
    names = [r["name"] for r in friends_of_friends(tx, "Alice")]
    assert {"Bob", "Cara", "Eve"} <= set(names)
    assert len(names) == len(set(names)), "DISTINCT should remove repeats"


def test_pagerank_written_back(driver):
    require(driver, "MATCH (p:Person) WHERE p.pagerank IS NOT NULL RETURN count(p) > 0",
            "write PageRank back (Lesson 4.5) first")
    missing = rows(driver, "MATCH (p:Person) WHERE p.pagerank IS NULL RETURN p.name AS name")
    assert missing == [], "every Person should have a pagerank"


def test_pagerank_on_a_tiny_known_graph(driver):
    # Algorithms are easiest to test on a graph small enough to reason about by hand:
    # in a star, every spoke points at the hub, so the hub must rank first.
    graph = "pytest_star"
    driver.execute_query("CALL gds.graph.drop($g, false)", g=graph)
    try:
        driver.execute_query("""
            CALL gds.graph.project.cypher($g,
              'UNWIND range(0, 4) AS id RETURN id',
              'UNWIND range(1, 4) AS id RETURN id AS source, 0 AS target')
            """, g=graph)
        ranked = rows(driver, "CALL gds.pageRank.stream($g) YIELD nodeId, score "
                              "RETURN nodeId ORDER BY score DESC", g=graph)
        assert ranked[0]["nodeId"] == 0
    finally:
        driver.execute_query("CALL gds.graph.drop($g, false)", g=graph)
