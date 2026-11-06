"""Lesson 11 — the multi-session scenarios as pytest cases (real LLM, ~1 minute each).

    pytest -m llm tests/test_11_scenarios.py -k forget     # one scenario
    .venv/bin/python -m companion.scenarios                 # same, with full transcripts
"""
import pytest

from companion.scenarios import SCENARIOS, run_scenario


@pytest.mark.llm
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario(driver, llm_ready, scenario):
    results = run_scenario(driver, scenario, verbose=False)
    failed = [desc for desc, ok in results if not ok]
    assert not failed, f"{scenario.name}: failed checks: {failed}"
