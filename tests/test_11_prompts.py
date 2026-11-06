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
