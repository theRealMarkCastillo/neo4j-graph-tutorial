"""Lesson 12 — the ontology. Most of it is pure functions over a dict, so most tests need
neither Neo4j nor an LLM: the ideal shape for code that encodes rules."""
import datetime as dt

import pytest

from companion import ontology as o
from companion.models import EntityOut, Extraction, RelationOut


@pytest.mark.parametrize("said,canon,how,flip", [
    ("sister", "sister", "exact", False),
    ("girlfriend", "partner", "synonym", False),
    ("Hubby", "spouse", "synonym", False),
    ("sisters", "sister", "exact", False),          # plural stripped
    ("daughter_of", "daughter", "synonym", True),   # X_of flips direction
    ("lives near", "lives_near", "new", False),     # unknown: kept, never guessed
    ("stepmom", "mother", "synonym", False),        # a listed synonym (embeddings had said "son")
    ("landlord", "landlord", "new", False),         # unknown: proposed for review
    ("places", "places", "group", False),           # a heading, not a type: rejected
    ("pets", "pet", "exact", False),                # ...but a plural type still works
])
def test_normalize(said, canon, how, flip):
    assert o.normalize(said) == (canon, how, flip)


def test_hierarchy():
    assert {"sibling", "family", "related"} <= o.ancestors("sister")
    assert {"partner", "family"} <= o.ancestors("spouse")   # a DAG: two parents
    assert "sister" in o.descendants("family")


def test_rules():
    assert o.rule("lives_in", "cardinality") == "one"
    assert o.inverse("parent") == "child"
    assert o.range_ok("pet", "pet") and not o.range_ok("pet", "person")
    assert o.range_ok("sister", "person")


def test_group_words_map_questions_to_branches():
    assert "family" in o.groups_in("anyone in my family?")
    assert "work" in o.groups_in("how are my coworkers")


@pytest.mark.parametrize("edges,expected", [
    ([{"src": "u", "dst": "leo", "type": "brother"}, {"src": "leo", "dst": "ana", "type": "daughter"}],
     {"ana": ("niece_or_nephew", "leo")}),
    # the same fact written from Ana's side ("Ana's father is Leo") reads backwards via the inverse
    ([{"src": "u", "dst": "leo", "type": "brother"}, {"src": "ana", "dst": "leo", "type": "father"}],
     {"ana": ("niece_or_nephew", "leo")}),
    ([{"src": "u", "dst": "mom", "type": "mother"}, {"src": "mom", "dst": "rosa", "type": "sister"}],
     {"rosa": ("aunt_or_uncle", "mom")}),
    # already directly related: nothing to infer
    ([{"src": "u", "dst": "leo", "type": "brother"}, {"src": "leo", "dst": "ana", "type": "daughter"},
      {"src": "u", "dst": "ana", "type": "niece"}], {}),
])
def test_infer_kin(edges, expected):
    got = {i["id"]: (i["relation"], i["via"]) for i in o.infer_kin(edges, "u")}
    assert got == expected


def test_prompt_vocabulary_lists_canonical_types():
    v = o.prompt_vocabulary()
    assert "partner" in v and "lives_in" in v


# --- through the store (Neo4j, no LLM) ---------------------------------------------------

T = dt.date(2026, 11, 5)
E = lambda n, k="person": EntityOut(name=n, kind=k)  # noqa: E731
R = lambda a, b, t, **kw: RelationOut(source=a, target=b, type=t, **kw)  # noqa: E731


def active(store):
    return {(r["t"], r["n"]) for r in store._q(
        "MATCH (:User {id:$user})-[r:RELATED]->(e) WHERE r.until IS NULL RETURN r.type AS t, e.name AS n")}


def test_synonym_stored_canonically_and_ended_by_another_synonym(store):
    store.apply(Extraction(entities=[E("Kim")], relations=[R("user", "Kim", "girlfriend")]), today=T)
    assert ("partner", "Kim") in active(store)
    store.apply(Extraction(relations=[R("user", "Kim", "boyfriend", ended=True)]), today=T)
    assert ("partner", "Kim") not in active(store)


def test_cardinality_one_ends_the_previous_home(store):
    store.apply(Extraction(entities=[E("Austin", "place")], relations=[R("user", "Austin", "lives_in")]), today=T)
    log = store.apply(Extraction(entities=[E("Denver", "place")], relations=[R("user", "Denver", "moved_to")]), today=T)
    assert {t for t in active(store) if t[0] == "lives_in"} == {("lives_in", "Denver")}
    assert any("one lives_in at a time" in line for line in log)


def test_group_names_are_not_relations(store):
    # Found by asking about "I'm at the airport waiting for my flight to Denver" (Lesson 13).
    log = store.apply(Extraction(entities=[E("Denver", "place")], relations=[R("user", "Denver", "places")]), today=T)
    assert any("is a group" in line for line in log)
    assert not active(store)


def test_range_check_rejects_impossible_facts(store):
    log = store.apply(Extraction(entities=[E("Maya")], relations=[R("user", "Maya", "pet")]), today=T)
    assert any(line.startswith("rejected") for line in log)
    assert ("pet", "Maya") not in active(store)


def test_seattle_is_not_merged_into_boston(store):
    # Regression for the entity-resolution bug found in Lesson 12 (similar summaries).
    store.apply(Extraction(entities=[EntityOut(name="Boston", kind="place", summary="The city where Pytest lives")]), today=T)
    store.apply(Extraction(entities=[EntityOut(name="Seattle", kind="place", summary="The city where Pytest now lives")]), today=T)
    names = {r["n"] for r in store._q("MATCH (e:Entity {user:$user, kind:'place'}) RETURN e.name AS n")}
    assert names == {"Boston", "Seattle"}


def test_more_specific_type_replaces_the_general_one(store):
    store.apply(Extraction(entities=[E("Dana")], relations=[R("user", "Dana", "partner")]), today=T)
    store.apply(Extraction(relations=[R("user", "Dana", "wife")]), today=T)
    assert [t for t in active(store) if t[1] == "Dana"] == [("spouse", "Dana")]
