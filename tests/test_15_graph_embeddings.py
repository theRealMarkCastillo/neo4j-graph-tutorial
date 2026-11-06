"""Lesson 15 — graph embeddings (FastRP + kNN) vs. node similarity on the life graph.

    .venv/bin/python -m pytest tests/test_15_graph_embeddings.py -v

The fixture graph is small and hand-made so the right answer is obvious to a human:
a climbing crowd (Ana, Ben, Cy, Boulder Gym) and a work crowd (Dee, Eli, Acme).
No memory mentions Cy together with Ana or Ben: Cy is only linked to the gym.
"""
import datetime as dt

import pytest

from companion import insights
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut

D1 = dt.date(2026, 11, 18)


def ent(name, kind="person"):
    return EntityOut(name=name, kind=kind)


def fact(text, *about, kind="fact"):
    return MemoryOut(text=text, kind=kind, importance=5, about=list(about))


@pytest.fixture
def life(store):
    store.apply(Extraction(
        entities=[ent("Ana"), ent("Ben"), ent("Cy"), ent("Boulder Gym", "place"),
                  ent("Dee"), ent("Eli"), ent("Acme", "organization")],
        relations=[RelationOut(source="user", target="Ana", type="friend"),
                   RelationOut(source="user", target="Acme", type="works_at"),
                   RelationOut(source="Dee", target="Acme", type="works_at")],
        memories=[fact("Ana and Ben climb with Pytest at Boulder Gym on Tuesdays", "user", "Ana", "Ben", "Boulder Gym"),
                  fact("Ben is training for a bouldering competition", "Ben", "Boulder Gym"),
                  fact("Cy sets the routes at Boulder Gym", "Cy", "Boulder Gym"),
                  fact("Dee and Eli are on Pytest's team at Acme", "user", "Dee", "Eli", "Acme"),
                  fact("Eli leads the Acme quarterly planning", "Eli", "Acme")]), today=D1)
    return store


def names(rows):
    return {r["other"] for r in rows}


def test_fastrp_finds_cys_crowd_two_hops_away(life):
    assert names(insights.related(life, "Cy", method="fastrp")) == {"Boulder Gym", "Ana", "Ben"}


def test_node_similarity_only_sees_shared_neighbours(life):
    """Jaccard compares direct neighbours. Cy's only memory is about the gym, so Ana and Ben
    (no shared memory with Cy) score 0: exact and explainable, but it can't see the crowd."""
    assert names(insights.related(life, "Cy", method="jaccard")) == {"Boulder Gym"}


def test_the_two_crowds_stay_apart(life):
    work = {"Dee", "Eli", "Acme"}
    for person in ("Dee", "Eli"):
        assert names(insights.related(life, person)) <= work


def test_results_repeat_with_a_fixed_seed(life):
    """FastRP starts from random vectors and kNN samples: without randomSeed + concurrency 1,
    the scores (and sometimes the order) change from run to run."""
    assert insights.related(life, "Cy") == insights.related(life, "Cy")


def test_a_boundary_protected_person_never_appears(life):
    """An insight must not route around a boundary: protected entities are not projected."""
    life.apply(Extraction(memories=[fact("Pytest does not want to talk about Ben", "Ben", kind="boundary")]), today=D1)
    everything = insights.related(life, k=50)
    assert "Ben" not in {r["name"] for r in everything} | names(everything)


def test_the_projection_is_always_dropped(life, driver):
    """GDS graphs live in memory until dropped. A leak per call would eventually exhaust the heap."""
    insights.related(life, "Cy")
    with pytest.raises(ValueError):
        insights.related(life, "Cy", method="nope")
    left = driver.execute_query("CALL gds.graph.list() YIELD graphName WHERE graphName STARTS WITH 'life_' RETURN count(*) AS n")
    assert left.records[0]["n"] == 0
