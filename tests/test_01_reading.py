"""Lesson 1 — reading with Cypher. Each test pins down one query's answer on the seed graph.

Assert on what the lesson established ("Bob and Cara work on Mobile App"), using subsets
(<=) rather than exact equality, so data added in later lessons doesn't break these tests.
"""
from conftest import rows


def names(driver, query, **params):
    return {r["name"] for r in rows(driver, query, **params)}


def test_exercise_who_works_on_mobile_app(driver):
    team = names(driver, "MATCH (p:Person)-[:WORKS_ON]->(:Project {name:'Mobile App'}) RETURN p.name AS name")
    assert {"Bob", "Cara"} <= team


def test_filter_on_relationship_property(driver):
    strong = names(driver, """
        MATCH (p:Person)-[r:HAS_SKILL]->(:Skill {name:'Python'}) WHERE r.level >= 4
        RETURN p.name AS name""")
    assert {"Alice", "Eve"} <= strong


def test_missing_pattern_finds_managers_without_projects(driver):
    idle = names(driver, "MATCH (p:Person) WHERE NOT (p)-[:WORKS_ON]->() RETURN p.name AS name")
    assert "Dev" in idle and "Alice" not in idle


def test_friend_of_a_friend(driver):
    fof = names(driver, "MATCH (:Person {name:'Alice'})-[:KNOWS]->()-[:KNOWS]->(f) RETURN f.name AS name")
    assert "Eve" in fof  # Alice -> Cara -> Eve


def test_variable_length_includes_every_hop(driver):
    reach = names(driver, "MATCH (:Person {name:'Alice'})-[:KNOWS*1..3]->(x) RETURN DISTINCT x.name AS name")
    assert {"Bob", "Cara", "Eve"} <= reach


def test_shortest_path_length(driver):
    [r] = rows(driver, """
        MATCH path = shortestPath((:Person {name:'Bob'})-[*]-(:Person {name:'Eve'}))
        RETURN length(path) AS hops""")
    assert r["hops"] <= 3


def test_aggregation_counts_people_per_project(driver):
    counts = {r["project"]: r["n"] for r in rows(driver, """
        MATCH (p:Person)-[:WORKS_ON]->(pr:Project) RETURN pr.name AS project, count(p) AS n""")}
    assert counts["Search Revamp"] >= 2 and counts["Mobile App"] >= 2
