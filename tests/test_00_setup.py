"""Lesson 0 — is the environment ready? Smoke tests: the cheapest tests that catch the most."""
from conftest import rows


def test_neo4j_answers(driver):
    assert rows(driver, "RETURN 1 AS one") == [{"one": 1}]


def test_plugins_installed(driver):
    # Lesson 4 needs GDS, later lessons use APOC; the Dockerfile installs both.
    assert rows(driver, "RETURN apoc.version() AS v")[0]["v"]
    assert rows(driver, "RETURN gds.version() AS v")[0]["v"]


def test_seed_loaded(driver):
    people = {r["name"] for r in rows(driver, "MATCH (p:Person) RETURN p.name AS name")}
    assert {"Alice", "Bob", "Cara", "Dev", "Eve"} <= people, "load seed/00_seed.cypher (Lesson 0)"
