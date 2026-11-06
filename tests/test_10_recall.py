"""Lesson 10 — recall and chat. Deterministic parts first (the small-talk gate, recall
queries, the prompt block); how the LLM *uses* the memories is graded in test_11."""
import datetime as dt

import pytest

from companion.graph import worth_remembering
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.store import format_context

T = dt.date(2026, 11, 5)


@pytest.mark.parametrize("text,expected", [
    ("ok", False), ("haha yeah", False), ("Thanks!", False), ("good night", False),
    ("ok so my sister is moving to Denver", True),
    ("My dog Rex is sick", True),
    ("no", False),
])
def test_small_talk_gate(text, expected):
    # Table-driven tests make a regex's behaviour readable, and every bug becomes a new row.
    assert worth_remembering(text) is expected


def test_yes_to_a_question_is_worth_remembering():
    # Prompt review: "yep" answering "Is the interview Friday?" confirms a fact.
    assert worth_remembering("yep", "Is your interview on Friday?")
    assert not worth_remembering("yep", "That sounds fun.")
    # ...but laughter after a question is still small talk (scenario cross_turn_facts caught this)
    assert not worth_remembering("haha yeah", "Are you excited?")


def m(text, kind="fact", **kw):
    return MemoryOut(text=text, kind=kind, importance=kw.pop("importance", 7), **kw)


@pytest.fixture
def life(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Maya", kind="person"), EntityOut(name="Rex", kind="pet")],
        relations=[RelationOut(source="user", target="Maya", type="sister"),
                   RelationOut(source="user", target="Rex", type="pet")],
        memories=[m("Pytest's sister Maya is a nurse in Denver", about=["Maya"]),
                  m("Pytest's dog Rex is afraid of thunderstorms", about=["Rex"]),
                  m("Pytest doesn't want to talk about their job", "boundary", importance=9),
                  m("Pytest is being treated for anxiety", sensitive=True, importance=8),
                  m("Pytest has a job interview at Acme", "plan", due_on="2026-11-06")],
    ), today=T)
    return store


def test_semantic_recall_finds_the_right_memory(life):
    r = life.recall("how is your sister doing?", today=T)
    texts = [x["text"] for x in r["semantic"]] + [x["text"] for e in r["entities"] for x in e["memories"]]
    assert "Pytest's sister Maya is a nurse in Denver" in texts


def test_boundaries_are_always_in_core(life):
    r = life.recall("what should I cook tonight?", today=T)
    assert any("doesn't want to talk about their job" in x["text"] for x in r["core"])


def test_sensitive_memories_stay_out_of_core(life):
    r = life.recall("what should I cook tonight?", today=T)
    assert not any("anxiety" in x["text"] for x in r["core"])


def test_followups_include_upcoming_plans(life):
    # Regression: the "job" boundary is ABOUT the user, and used to hide every memory
    # about the user, this plan included.
    r = life.recall("hi", today=T)
    assert any("interview" in x["text"] for x in r["followups"])


def test_person_boundary_hides_that_person_only(life):
    life.apply(Extraction(memories=[m("Don't bring up Maya", "boundary", about=["Maya"], importance=9)]), today=T)
    r = life.recall("how is your sister doing?", today=T)
    texts = [x["text"] for x in r["semantic"]] + [x["text"] for e in r["entities"] for x in e["memories"]]
    assert not any("nurse" in t for t in texts)
    assert any("interview" in x["text"] for x in r["followups"])


def test_prompt_block_has_labelled_sections(life):
    block = format_context(life.recall("how is Rex?", today=T), today=T)
    assert "Core profile:" in block and "People and things in their life:" in block
    assert "Rex (pet)" in block


def test_empty_memory_says_so():
    empty = {k: [] for k in ["core", "semantic", "entities", "people", "followups", "upcoming",
                             "history", "episodes", "recent"]}
    assert "No memories yet" in format_context(empty)


def test_protected_person_does_not_leak_through_history_or_entity_hop(store):
    store.apply(Extraction(
        entities=[EntityOut(name="Dana", kind="person"), EntityOut(name="Maya", kind="person")],
        memories=[m("Pytest's partner is Dana", about=["user", "Dana"]),
                  m("Maya introduced Pytest to Dana", "event", about=["Maya", "Dana"])]), today=T)
    old = next(x for x in store.memories() if "partner" in x["text"])
    store.apply(Extraction(memories=[
        m("Pytest is single now", supersedes=[old["id"]]),
        m("Don't bring up Dana", "boundary", about=["Dana"], importance=9)]), today=T)
    block = format_context(store.recall("am I single? how is Maya?", today=T), today=T)
    assert "Dana" not in block.replace("Don't bring up Dana", "")
