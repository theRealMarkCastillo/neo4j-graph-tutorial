"""Lesson 16 — several personas sharing one user's memory.

    .venv/bin/python -m pytest tests/test_16_personas.py -v          # store rules, no LLM
    .venv/bin/python -m pytest -m llm tests/test_16_personas.py      # does the extractor mark "just between us"?

Two stores for the same user: `sage` and `coach`. What the user tells one persona privately
must never reach the other through ANY read path: recall, the people list, the extractor's
prompt, the diary, inspection, patterns or graph insights. Each path gets its own test,
because each one is a separate query that could forget the rule.
"""
import datetime as dt

import pytest

from companion import insights
from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.store import MemoryStore, format_context

D1 = dt.date(2026, 11, 18)


def mem(text, kind="fact", about=("user",), **kw):
    return MemoryOut(text=text, kind=kind, importance=kw.pop("importance", 7), about=list(about), **kw)


@pytest.fixture
def sage(store):
    store.agent = "Sage"
    return store


@pytest.fixture
def coach(driver, sage):
    return MemoryStore(driver, "pytest", agent="Coach")


def texts(rows):
    return " ".join(r["text"] for r in rows)


def test_shared_memories_reach_every_persona(sage, coach):
    sage.apply(Extraction(memories=[mem("Pytest signed up for the Chicago half marathon in March")]), today=D1)
    assert "half marathon" in texts(coach.memories())
    assert "half marathon" in format_context(coach.recall("training for my race", today=D1), D1)


def test_a_private_memory_stays_with_the_persona_that_heard_it(sage, coach):
    sage.apply(Extraction(memories=[mem("Pytest gets nervous before every race", kind="feeling",
                                        private=True)]), today=D1)
    assert "nervous" in texts(sage.memories())
    assert "nervous" not in texts(coach.memories())
    assert "nervous" not in format_context(coach.recall("how do I feel before races?", today=D1), D1)
    assert "nervous" not in texts(coach.extraction_candidates("nervous before races", today=D1))


def test_sensitive_is_private_by_default(sage, coach):
    sage.apply(Extraction(memories=[mem("Pytest has panic attacks before races", sensitive=True)]), today=D1)
    assert [m["private"] for m in sage.memories()] == [True]
    assert coach.memories() == []


def test_boundaries_protect_with_every_persona(sage, coach):
    """A boundary set with Sage must hold with Coach too, even if the user asked privately."""
    sage.apply(Extraction(memories=[mem("Pytest does not want to talk about their weight", kind="boundary",
                                        importance=9, private=True)]), today=D1)
    assert "weight" in texts(coach.recall("anything", today=D1)["core"])


def test_someone_only_mentioned_privately_is_hidden_everywhere(sage, coach):
    """The people list, the extractor's entity list and the entity hop would each give away
    "Dr. Lee, the therapist" even if the memory itself is hidden."""
    sage.apply(Extraction(entities=[EntityOut(name="Dr. Lee", kind="person", summary="Pytest's therapist")],
                          relations=[RelationOut(source="user", target="Dr. Lee", type="therapist")],
                          memories=[mem("Pytest sees Dr. Lee every Monday", about=("user", "Dr. Lee"),
                                        private=True)]), today=D1)
    assert "Dr. Lee" in [p["name"] for p in sage.recall("people", today=D1)["people"]]
    found = coach.recall("I'm seeing Dr. Lee today", today=D1)
    assert "Dr. Lee" not in [p["name"] for p in found["people"]]
    assert "Dr. Lee" not in [e["name"] for e in found["entities"]]
    assert not any("Dr. Lee" in e for e in coach.known_entities())


def test_once_shared_a_relation_stays_shared(sage, coach):
    """Told privately to Sage, later said openly to Coach: now everyone may know."""
    sage.apply(Extraction(entities=[EntityOut(name="Kai", kind="person")],
                          relations=[RelationOut(source="user", target="Kai", type="partner")],
                          memories=[mem("Pytest is dating Kai", about=("user", "Kai"), private=True)]), today=D1)
    assert "Kai" not in [p["name"] for p in coach.recall("people", today=D1)["people"]]
    coach.apply(Extraction(relations=[RelationOut(source="user", target="Kai", type="partner")],
                           memories=[mem("Pytest's partner Kai is a chef", about=("user", "Kai"))]), today=D1)
    assert "Kai" in [p["name"] for p in coach.recall("people", today=D1)["people"]]
    assert "Kai" in [p["name"] for p in sage.recall("people", today=D1)["people"]]


def test_each_persona_keeps_its_own_diary(sage, coach):
    """Episode summaries can quote private things, so they aren't shared."""
    ep = sage.start_episode(on=D1)
    sage.log_turn(ep, "pytest", "something private")
    sage.end_episode(ep, "Pytest talked to Sage about race nerves.", "anxious")
    assert sage.recall("hello", today=D1)["episodes"]
    assert coach.recall("hello", today=D1)["episodes"] == []


def test_patterns_are_never_built_from_private_memories(sage):
    """A pattern is shared. Built from private memories, it would leak what they say."""
    for i, text in enumerate(["Pytest is at the therapist", "Pytest went to the therapist after work",
                              "Pytest is heading to the therapist again"]):
        sage.apply(Extraction(memories=[mem(text, kind="status", importance=2, private=True)]),
                   today=D1 + dt.timedelta(days=i))
    called = []
    sage.consolidate(D1 + dt.timedelta(days=3), writer=lambda *a: called.append(a), prune=False)
    assert called == []


def test_graph_insights_respect_privacy(sage, coach):
    sage.apply(Extraction(entities=[EntityOut(name="Ana", kind="person"), EntityOut(name="Dr. Lee", kind="person")],
                          memories=[mem("Pytest and Ana both see Dr. Lee", about=("user", "Ana", "Dr. Lee"),
                                        private=True),
                                    mem("Ana is Pytest's running partner", about=("user", "Ana"))]), today=D1)
    pairs = insights.related(coach, k=50)
    assert "Dr. Lee" not in {r["name"] for r in pairs} | {r["other"] for r in pairs}


@pytest.mark.llm
@pytest.mark.parametrize("message, private", [
    ("Just between us, I'm thinking about quitting my job.", True),
    ("Don't tell anyone, but I failed my driving test again.", True),
    ("I'm thinking about quitting my job.", False),
])
def test_extractor_marks_just_between_us(llm_ready, message, private):
    from companion.extract import extract_memories
    ex = extract_memories("Riley", message, "Thanks for telling me.", candidates=[], entities=[], today=D1)
    assert ex.memories, "nothing extracted"
    assert any(m.private for m in ex.memories) == private
