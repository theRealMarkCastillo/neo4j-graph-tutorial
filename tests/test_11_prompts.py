"""Lesson 11 — prompts are code too: the deterministic parts get unit tests.

What a prompt *says* can be checked without calling the model: the rendered text, the
tripwire regex, how transcripts are labelled. What the model *does* with it is in
test_11_extraction_evals.py (opt in with -m llm)."""
import datetime as dt

import pytest

from companion.extract import BOUNDARY_HINT, TURN_SYSTEM, _system, line


@pytest.mark.parametrize("text,hit", [
    ("Please don't bring her up again", True),
    ("I don't want to talk about my dad", True),
    ("I'd rather not talk about work", True),
    ("Let's not get into my divorce", True),
    ("Leave my ex out of it", True),
    ("I love talking about my dog", False),
])
def test_boundary_tripwire(text, hit):
    assert bool(BOUNDARY_HINT.search(text)) is hit


def test_companion_lines_are_marked_as_non_sources():
    assert line("Riley", "Riley", "I moved") == "Riley: I moved"
    assert "not a source" in line("Riley", "Sage", "You still live in Austin, right?")


def test_rendered_prompt_has_everything_filled_in():
    text = _system(TURN_SYSTEM, "Riley", [{"id": "m_1", "kind": "plan", "status": "open", "text": "Dentist"}],
                   ["e_1: Maya (person, sister)"], dt.date(2026, 11, 20))
    assert "It is now 2026-11-20 12:00 (Friday)" in text  # the clock, with a time (Lesson 13)
    assert "m_1: [plan/open] Dentist" in text
    assert "e_1: Maya" in text
    assert "lives_in" in text            # the ontology vocabulary
    assert "{" not in text.replace("{user}", "")  # no unfilled placeholders


@pytest.mark.parametrize("phase", ["turn", "retry", "episode", "refresh"])
def test_forget_wins_over_a_conflicting_boundary(monkeypatch, phase):
    from companion import extract as module
    from companion.models import Extraction, MemoryOut

    # The exact contradictory output from the failed live eval.
    conflict = Extraction(
        memories=[MemoryOut(text="Riley does not want Jordan to ever be mentioned again.",
                            kind="boundary", about=["e_9"], importance=9)],
        forget_entities=["e_9"],
    )
    outputs = iter([Extraction(forget_entities=["e_9"]), conflict] if phase == "retry" else [conflict])
    monkeypatch.setattr(module, "_run", lambda *args: next(outputs))
    known = ["e_9: Jordan (person, ex) — Riley's ex"]
    if phase in ("episode", "refresh"):
        ex = module.extract_episode("Riley", "Forget Jordan, and don't ever mention him again.", [], known,
                                    refresh=phase == "refresh")
    else:
        ex = module.extract_memories("Riley", "Forget Jordan, and don't ever mention him again.",
                                     "Okay.", [], known)
    assert ex.forget_entities == ["e_9"]
    assert ex.memories == []


def test_forgetting_one_person_keeps_a_boundary_about_another(monkeypatch):
    from companion import extract as module
    from companion.models import Extraction, MemoryOut

    output = Extraction(forget_entities=["e_9"], memories=[
        MemoryOut(text="Do not mention Jordan", kind="boundary", about=["Jordan"], importance=9),
        MemoryOut(text="Do not mention Dana", kind="boundary", about=["e_10"], importance=9),
    ])
    monkeypatch.setattr(module, "_run", lambda *args: output)
    ex = module.extract_memories("Riley", "Forget Jordan, but don't mention Dana.", "Okay.", [],
                                 ["e_9: Jordan (person, ex)", "e_10: Dana (person, coworker)"])
    assert [m.about for m in ex.memories] == [["e_10"]]


def test_boundary_retry_preserves_its_forget_instruction(monkeypatch):
    from companion import extract as module
    from companion.models import Extraction, MemoryOut

    replies = iter([Extraction(), Extraction(forget_entities=["e_9"], memories=[
        MemoryOut(text="Do not mention Jordan", kind="boundary", about=["e_9"], importance=9),
        MemoryOut(text="Do not mention Dana", kind="boundary", about=["e_10"], importance=9),
    ])])
    monkeypatch.setattr(module, "_run", lambda *args: next(replies))
    ex = module.extract_memories("Riley", "Forget Jordan, and don't mention Dana.", "Okay.", [],
                                 ["e_9: Jordan (person, ex)", "e_10: Dana (person)"])
    assert ex.forget_entities == ["e_9"]
    assert [m.about for m in ex.memories] == [["e_10"]]


def test_forgetting_one_id_does_not_remove_a_same_named_person(monkeypatch):
    from companion import extract as module
    from companion.models import Extraction, MemoryOut

    output = Extraction(forget_entities=["e_9"], memories=[
        MemoryOut(text="Do not mention the other Jordan", kind="boundary", about=["e_10"], importance=9),
    ])
    monkeypatch.setattr(module, "_run", lambda *args: output)
    ex = module.extract_memories("Riley", "Forget my ex Jordan.", "Okay.", [],
                                 ["e_9: Jordan (person, ex)", "e_10: Jordan (person, coworker)"])
    assert ex.memories == output.memories


@pytest.mark.parametrize("forgotten,about,entities,known,keep", [
    ("JORDAN", "jordan", [], [], False),
    ("Ann", "anniversary", [], [], True),
    ("Jordan", "e_9", [], ["e_9: Jordan (person, ex)"], False),
    ("e_9", "my ex", [{"name": "Jordan", "kind": "person", "ref": "e_9", "aliases": ["my ex"]}], [], False),
    ("e_9", "Jordan", [], ["e_9: Jordan (person)", "e_10: Jordan (person)"], True),
    ("Jordan", "Dana", [], [], True),
])
def test_forget_matching_uses_explicit_unambiguous_identity(forgotten, about, entities, known, keep):
    from companion.extract import _forget_wins
    from companion.models import EntityOut, Extraction, MemoryOut

    memory = MemoryOut(text="A remembered fact", kind="fact", about=[about], importance=5)
    ex = Extraction(forget_entities=[forgotten], memories=[memory],
                    entities=[EntityOut(**entity) for entity in entities])
    result = _forget_wins(ex, known)
    assert result.memories == ([memory] if keep else [])
    assert ex.memories == [memory]  # normalization does not mutate raw model output
