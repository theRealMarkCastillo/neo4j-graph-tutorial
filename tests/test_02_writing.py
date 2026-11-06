"""Lesson 2 — writing data. Every test runs in the `tx` fixture, which is rolled back:
the pattern for testing writes against a real database without leaving anything behind."""
import pytest
from neo4j.exceptions import ConstraintError, Neo4jError

from conftest import require


def count(tx, query, **params) -> int:
    return tx.run(query, **params).single()[0]


def test_create_then_read_back_in_same_transaction(tx):
    tx.run("CREATE (:Person {name:'Testy', role:'Tester'})")
    assert count(tx, "MATCH (p:Person {name:'Testy'}) RETURN count(p)") == 1


def test_rolled_back_writes_are_gone(driver):
    with driver.session() as s:
        t = s.begin_transaction()
        t.run("CREATE (:Person {name:'Ghost'})")
        t.rollback()
    assert driver.execute_query("MATCH (p:Person {name:'Ghost'}) RETURN count(p) AS n").records[0]["n"] == 0


def test_merge_is_idempotent(tx):
    tx.run("MERGE (:Skill {name:'Zig'}) MERGE (:Skill {name:'Zig'}) MERGE (:Skill {name:'Zig'})")
    assert count(tx, "MATCH (s:Skill {name:'Zig'}) RETURN count(s)") == 1


def test_merge_on_create_vs_on_match(tx):
    q = """MERGE (p:Person {name:'Ivy'})
           ON CREATE SET p.visits = 1 ON MATCH SET p.visits = p.visits + 1 RETURN p.visits"""
    assert count(tx, q) == 1
    assert count(tx, q) == 2


def test_unique_constraint_blocks_duplicates(driver, tx):
    require(driver, "SHOW CONSTRAINTS YIELD name WHERE name = 'person_name' RETURN count(*) > 0",
            "create the person_name constraint (Lesson 2.6) first")
    with pytest.raises(ConstraintError):
        tx.run("CREATE (:Person {name:'Alice', role:'Impostor'})").consume()


def test_delete_needs_detach_when_node_has_relationships(driver):
    # Surprise worth a test: plain DELETE doesn't fail when the statement runs, it fails
    # when the transaction COMMITS (Neo4j checks for dangling relationships at commit).
    with driver.session() as s:
        t = s.begin_transaction()
        t.run("CREATE (:Person {name:'Hal'})-[:KNOWS]->(:Person {name:'Hal2'})")
        t.run("MATCH (p:Person {name:'Hal'}) DELETE p").consume()  # no error yet
        with pytest.raises(Neo4jError, match="still has relationships"):
            t.commit()  # fails, so nothing was written

    with driver.session() as s:
        t = s.begin_transaction()
        t.run("CREATE (:Person {name:'Hal'})-[:KNOWS]->(:Person {name:'Hal2'})")
        t.run("MATCH (p:Person {name:'Hal'}) DETACH DELETE p").consume()
        assert count(t, "MATCH (p:Person {name:'Hal'}) RETURN count(p)") == 0
        t.rollback()
    assert count_db(driver, "MATCH (p:Person) WHERE p.name STARTS WITH 'Hal' RETURN count(p)") == 0


def count_db(driver, query):
    return driver.execute_query(query).records[0][0]
