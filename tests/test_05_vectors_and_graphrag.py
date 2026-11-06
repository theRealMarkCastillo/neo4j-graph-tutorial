"""Lesson 5 — vector search and GraphRAG.

Retrieval is the part of RAG you can measure without an LLM: a small set of questions, each
labelled with the node that SHOULD come back, scored as hit@k. Run it whenever you change the
embedding model, the descriptions or k, and compare the number, not your impression.
"""
import pytest

from conftest import require
from graphrag import build_context, retrieve

# question -> any of these names in the top 3 counts as a hit
RETRIEVAL_CASES = [
    ("Who could help us build a recommendation engine?", {"Eve", "Search Revamp"}),
    ("Who knows Rust and low-latency systems?", {"Finn", "Gia"}),
    ("Who cares about accessibility and user research?", {"Bob"}),
    ("Who handles hiring and career growth?", {"Dev"}),
    ("Which project moved billing to the cloud?", {"Legacy Migration"}),
    ("Who builds React Native components?", {"Cara", "Mobile App"}),
    ("Who mentors juniors on testing?", {"Alice"}),
]
HIT_RATE_MIN = 6 / 7  # the bar: lower it only on purpose, and say why


@pytest.fixture
def embedded(driver):
    require(driver, "SHOW INDEXES YIELD name WHERE name = 'embedded_vec' RETURN count(*) > 0",
            "run embed.py (Lesson 5.2) first")


def test_build_context_formats_facts():
    # Unit test, no database: the prompt-building step is plain string work.
    hits = [{"type": "Person", "name": "Eve", "score": 0.84, "description": "Data scientist.",
             "facts": ["Eve HAS_SKILL ML (level 5)"]}]
    ctx = build_context(hits)
    assert ctx.startswith("[Person] Eve (match 0.84)")
    assert "  - Eve HAS_SKILL ML (level 5)" in ctx


def test_retrieval_returns_scored_hits_with_facts(embedded):
    hits = retrieve("search ranking", k=3)
    assert len(hits) == 3
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert all(h["facts"] for h in hits), "each hit should bring graph facts along"


def test_retrieval_hit_rate(embedded):
    results = []
    for question, expected in RETRIEVAL_CASES:
        top = [h["name"] for h in retrieve(question, k=3)]
        results.append((question, expected, top, bool(expected & set(top))))
    report = "\n".join(f"{'HIT ' if ok else 'MISS'} {q!r} -> {top} (want {sorted(e)})"
                       for q, e, top, ok in results)
    rate = sum(ok for *_, ok in results) / len(results)
    print(f"\nhit@3 = {rate:.2f}\n{report}")
    assert rate >= HIT_RATE_MIN, report


# --- LLM evals: real calls, graded by code, opt in with `pytest -m llm` -----------------

@pytest.mark.llm
def test_answer_uses_the_context(embedded, llm_ready):
    from graphrag import answer

    q = "Who could help us build a recommendation engine?"
    text = answer(q, build_context(retrieve(q)))
    assert text and "Eve" in text


@pytest.mark.llm
def test_answer_admits_when_graph_has_nothing(embedded, llm_ready):
    # Regression: an earlier prompt ("say so instead of guessing") made the model refuse
    # even for the recommendation question above; this pair keeps both behaviors honest.
    from graphrag import answer

    q = "Who on the team speaks Japanese?"
    text = answer(q, build_context(retrieve(q)))
    assert text
    named = {"Alice", "Bob", "Cara", "Dev", "Eve", "Finn", "Gia"} & set(text.replace(",", " ").split())
    assert not named, f"invented a Japanese speaker: {text}"
