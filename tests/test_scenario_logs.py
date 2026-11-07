"""Scenario checks retain write-time evidence even after a memory is pruned."""
from types import SimpleNamespace

import pytest

from companion import scenarios


@pytest.mark.parametrize("verbose", [False, True])
def test_run_records_turn_and_close_logs(monkeypatch, verbose):
    class Store:
        agent = "Sage"

        def __init__(self, *args):
            pass

        def wipe(self):
            pass

        def start_episode(self, **kwargs):
            return "episode"

    class App:
        def stream(self, *args, **kwargs):
            yield {"remember": {"memory_log": ["+ [status] Heading to gym (until 2026-11-18 15:00)"]}}

    monkeypatch.setattr(scenarios, "MemoryStore", Store)
    monkeypatch.setattr(scenarios, "build_companion", lambda store: App())
    monkeypatch.setattr(scenarios, "greeting", lambda *args: "Hello")
    monkeypatch.setattr(scenarios, "seed_thread", lambda *args: None)
    monkeypatch.setattr(scenarios, "close_episode", lambda *args: ["pruned expired memory"])
    monkeypatch.setattr(scenarios, "threshold_check", lambda: ("model", True))
    observed = []

    def check(run):
        observed.extend(run.logs)
        return scenarios.stored_temporary(run, "gym")

    sc = scenarios.Scenario("logs", "", [(scenarios.D(2026, 11, 18), ["Gym!"])], [("expiry", check)])
    assert all(ok for _, ok in scenarios.run_scenario(None, sc, verbose=verbose))
    assert observed == ["+ [status] Heading to gym (until 2026-11-18 15:00)", "pruned expired memory"]


@pytest.mark.parametrize("line,expected", [
    ("+ [event] Gym trip (until 2026-11-18 15:00)", True),
    ("+ [fact] Gym is a habit", False),
    ("reinforced gym (until 2026-11-18 15:00)", False),
    ("+ [status] At work (until 2026-11-18 15:00)", False),
])
def test_stored_temporary_requires_new_activity_and_expiry(line, expected):
    assert scenarios.stored_temporary(SimpleNamespace(logs=[line]), "gym") is expected
