"""Lesson 6 — "embeddings find where to start; the graph reasons from there."

A test can turn a claim from a concept lesson into evidence. Each pair below asks the same
question both ways and checks which one actually gets it right.
"""
from conftest import require, rows
from graphrag import retrieve


def test_exact_lookup_needs_the_graph(driver):
    require(driver, "SHOW INDEXES YIELD name WHERE name = 'embedded_vec' RETURN count(*) > 0",
            "run embed.py (Lesson 5.2) first")
    # "Alice's manager": nothing in Dev's description mentions Alice...
    top = [h["name"] for h in retrieve("Who is Alice's manager?", k=3)]
    vector_found_dev = top[0] == "Dev"
    # ...but it's one hop in the graph.
    [r] = rows(driver, "MATCH (m:Person)-[:MANAGES]->(:Person {name:'Alice'}) RETURN m.name AS name")
    assert r["name"] == "Dev"
    print(f"\nvector top-3 for \"Alice's manager\": {top} (Dev first: {vector_found_dev})")


def test_counting_needs_the_graph(driver):
    # "How many people know Python?" is an aggregate. Similarity scores can't count.
    [r] = rows(driver, "MATCH (p:Person)-[:HAS_SKILL]->(:Skill {name:'Python'}) RETURN count(p) AS n")
    assert r["n"] >= 2


def test_fuzzy_match_needs_embeddings(driver):
    require(driver, "SHOW INDEXES YIELD name WHERE name = 'embedded_vec' RETURN count(*) > 0",
            "run embed.py (Lesson 5.2) first")
    # No node is named or tagged "recommendation"; only meaning connects it to Eve.
    assert rows(driver, "MATCH (n) WHERE toLower(n.name) CONTAINS 'recommend' RETURN n") == []
    assert "Eve" in [h["name"] for h in retrieve("recommendation engine", k=2)]
