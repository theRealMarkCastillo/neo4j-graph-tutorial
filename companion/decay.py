"""Lesson 13 — decay: how relevant a memory still is, as it ages.

Validity (clock.py, `valid_until`) answers "is it still TRUE?". Decay answers "is it still
WORTH BRINGING UP?". They are different: "Riley felt tired" was true on 2026-11-18 forever,
but a month later nobody should mention it.

strength = importance/10 x 0.5 ** (age / half-life of its kind)

- age counts from when it was stored or last CONFIRMED by the user (saying it again resets
  the clock, like spaced repetition). Being recalled does not count: otherwise anything
  recalled once would keep itself alive.
- facts, preferences, boundaries and plans don't decay. A fact doesn't get less true with age;
  it may get STALE, and the companion should check it rather than drop it (see `is_stale`).
- below DORMANT_BELOW a memory is dormant: kept in the graph, left out of recall.
  Milestones (importance >= MILESTONE) never go dormant.

Pure functions, so the curve has unit tests (tests/test_13_time.py).
"""
from __future__ import annotations

HALF_LIFE_DAYS: dict[str, float | None] = {
    "status": 1,       # "at the gym": already expired via valid_until; decays in a day after
    "feeling": 7,      # "tired", "excited about the trip"
    "event": 90,       # "went to a concert"
    "fact": None, "preference": None, "boundary": None, "plan": None,
}
DORMANT_BELOW = 0.05
MILESTONE = 8          # importance at which an event is a life milestone: never dormant
STALE_DAYS = 365       # a fact not confirmed for this long is worth checking
WEIGHT_MATCH, WEIGHT_STRENGTH = 0.6, 0.4


def strength(importance: int, kind: str, age_days: float) -> float:
    half_life = HALF_LIFE_DAYS.get(kind)
    base = importance / 10
    if not half_life or age_days <= 0:
        return base
    return base * 0.5 ** (age_days / half_life)


def is_dormant(importance: int, kind: str, age_days: float) -> bool:
    return importance < MILESTONE and strength(importance, kind, age_days) < DORMANT_BELOW


def rank(score: float, importance: int, kind: str, age_days: float) -> float:
    """Recall order: how well it matches the message, plus how strong it still is."""
    return WEIGHT_MATCH * score + WEIGHT_STRENGTH * strength(importance, kind, age_days)


def is_stale(kind: str, age_days: float) -> bool:
    return kind == "fact" and age_days > STALE_DAYS


def days_until_dormant(importance: int, kind: str) -> float | None:
    """For the lesson's table: how long until a memory stops being recalled."""
    half_life = HALF_LIFE_DAYS.get(kind)
    if not half_life or importance >= MILESTONE:
        return None
    import math
    ratio = DORMANT_BELOW / (importance / 10)
    return 0.0 if ratio >= 1 else half_life * math.log(ratio, 0.5)
