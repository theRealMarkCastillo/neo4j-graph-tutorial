"""Lesson 13 — one clock for the whole companion.

Every time-dependent step (the prompts, writes, recall filters, decay) asks this module what
"now" is, given the conversation's `today`. In a live chat `today` is the real date and now
is the real time; in the demo, scenarios and tests `today` is simulated and now is noon
that day. Nothing calls datetime.now() directly, so tests can travel in time.
"""
from __future__ import annotations

import datetime as dt

# A status the extractor forgot to give an end time ("I'm at the gym") lasts this long.
DEFAULT_STATUS_HOURS = 3


def now_for(today: dt.date) -> dt.datetime:
    """The current moment for a conversation happening on `today` (timezone-aware, local)."""
    if today == dt.date.today():
        return dt.datetime.now().astimezone()
    return dt.datetime(today.year, today.month, today.day, 12).astimezone()


def describe(now: dt.datetime) -> str:
    """How the prompts show the clock: '2026-11-18 12:00 (Wednesday)'."""
    return f"{now:%Y-%m-%d %H:%M} ({now:%A})"


def parse_until(value: str | None, now: dt.datetime, kind: str) -> dt.datetime | None:
    """Turn the extractor's valid_until into a datetime.

    '2026-11-18T14:00' -> that time (local); '2026-11-18' -> the end of that day.
    A status always gets an end: if it's missing, unreadable or already past, it lasts
    DEFAULT_STATUS_HOURS from now. Other kinds without a usable end never expire."""
    until = None
    if value:
        try:
            until = dt.datetime.fromisoformat(value.strip())
            if len(value.strip()) <= 10:  # a date: true until the end of that day
                until = until.replace(hour=23, minute=59)
            if until.tzinfo is None:
                until = until.astimezone()  # naive times are the user's local time
        except ValueError:
            until = None
    if until is not None and until <= now:
        until = None
    if until is None and kind == "status":
        until = now + dt.timedelta(hours=DEFAULT_STATUS_HOURS)
    return until
