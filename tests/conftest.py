"""Shared fixtures. Three kinds of test live in this folder:

    unit       pure Python, no database, no LLM            always run, milliseconds
    db         real Neo4j, writes rolled back or isolated  run by default, skipped if Neo4j is down
    llm        real LLM calls, graded by code              opt in:  pytest -m llm

Rules the fixtures enforce:
- A test never leaves data behind: Cypher writes go through `tx` (rolled back), memory tests
  use their own `pytest` user, wiped before and after.
- A test whose lesson you haven't done yet is SKIPPED with a reason, not failed.
"""
import os
import sys
from pathlib import Path

import pytest

# The tests run on the local embedding model by default (Lesson 19): offline, fast, free. Its
# vectors live apart from a remote model's, so this is safe even when your chats use
# EMBED_BASE_URL. TEST_REMOTE_EMBEDDINGS=1 runs them on the model in .env (the README's release
# check): thresholds come from the active profile, so the same tests hold on either model.
if not os.getenv("TEST_REMOTE_EMBEDDINGS"):
    os.environ["EMBED_BASE_URL"] = ""  # set before config loads .env, which won't override it

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import config, companion, ...

import config  # noqa: E402


@pytest.fixture(scope="session")
def driver():
    from neo4j import GraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    drv = GraphDatabase.driver(config.NEO4J_URI, auth=config.NEO4J_AUTH)
    try:
        drv.verify_connectivity()
    except (ServiceUnavailable, AuthError, OSError) as e:
        pytest.skip(f"Neo4j not reachable ({type(e).__name__}): run `docker compose up -d`")
    yield drv
    drv.close()


@pytest.fixture
def tx(driver):
    """A transaction that is ALWAYS rolled back: write freely, nothing persists."""
    with driver.session(database="neo4j") as session:
        t = session.begin_transaction()
        try:
            yield t
        finally:
            if t.closed() is False:
                t.rollback()


def rows(driver, query: str, **params) -> list[dict]:
    records, _, _ = driver.execute_query(query, **params)
    return [r.data() for r in records]


def require(driver, query: str, reason: str, **params) -> None:
    """Skip the test unless `query` returns a truthy first value (a lesson prerequisite)."""
    found = rows(driver, query, **params)
    if not found or not next(iter(found[0].values())):
        pytest.skip(reason)


@pytest.fixture
def store(driver):
    """A MemoryStore for a throwaway user, wiped before and after the test."""
    from companion.store import MemoryStore

    s = MemoryStore(driver, "pytest")
    s.wipe()
    yield s
    s.wipe()
    s._q("MATCH (u:User {id: $user}) DETACH DELETE u")


@pytest.fixture(scope="session")
def llm_ready():
    if not config.llm_configured():
        pytest.skip("LLM_MODEL is not set in .env")


@pytest.hookimpl(tryfirst=True)  # tag before `-m` filtering runs
def pytest_collection_modifyitems(items):
    """Tag every test that touches Neo4j with `db`: `pytest -m "not db and not llm"` = unit tests."""
    for item in items:
        if {"driver", "tx", "store"} & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.db)


def trials(check):
    """Run an eval case EVAL_TRIALS times (default 1); fail with every failing output."""
    import os
    n = int(os.environ.get("EVAL_TRIALS", "1"))
    failures = []
    for _ in range(n):
        try:
            check()
        except AssertionError as e:
            failures.append(str(e))
    assert not failures, f"{len(failures)}/{n} trials failed:\n" + "\n---\n".join(failures)
