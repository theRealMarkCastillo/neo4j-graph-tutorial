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


# ---------------------------------------------------------------- MLflow eval logging (optional)
# With MLFLOW_TRACKING_URI set (observability.py), `pytest -m llm` becomes one MLflow run:
#   params   the models, EVAL_TRIALS and a hash of the prompts, so runs can be compared;
#   metrics  passed / failed / pass_rate;
#   evals.json  one row per case: outcome, duration, the failure, and its trace id;
#   traces   each case is one trace holding every LLM call its trials made.
# The pass/fail verdict is still pytest's; MLflow is where you compare runs and read the prompts.
import observability  # noqa: E402

_eval_rows: list[dict] = []
_trace_key = pytest.StashKey[str]()


def _start_eval_run():
    import hashlib
    import subprocess

    import mlflow

    root = Path(__file__).resolve().parents[1]
    prompts = hashlib.sha256()
    for f in ("companion/extract.py", "companion/graph.py"):  # where the prompts live
        prompts.update((root / f).read_bytes())
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True).stdout.strip()
    mlflow.start_run(run_name=f"evals {sha}{'+dirty' if dirty else ''}")
    mlflow.log_params({
        "llm_model": config.LLM_MODEL, "extraction_model": config.LLM_EXTRACTION_MODEL,
        "embed_model": config.EMBED_MODEL, "eval_trials": os.environ.get("EVAL_TRIALS", "1"),
        "prompts_sha": prompts.hexdigest()[:12], "git_sha": sha, "git_dirty": bool(dirty),
    })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Each LLM eval case runs inside one span, so all its trials' LLM calls form one trace."""
    if not (observability.enabled() and item.get_closest_marker("llm")):
        yield
        return
    import mlflow

    if mlflow.active_run() is None:
        _start_eval_run()
    with observability.span(item.nodeid.split("::")[-1], "CHAIN",
                            inputs={"test": item.nodeid}) as sp:
        observability.tag_trace(eval_case=item.nodeid)
        item.stash[_trace_key] = sp.trace_id
        outcome = yield
        if outcome.excinfo:
            sp.set_outputs({"failed": str(outcome.excinfo[1])[:4000]})
            sp.set_status("ERROR")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not (observability.enabled() and item.get_closest_marker("llm")):
        return
    if report.when == "call" or (report.when == "setup" and report.skipped):
        _eval_rows.append({
            "case": item.nodeid, "outcome": report.outcome,
            "seconds": round(report.duration, 2),
            "failure": str(report.longrepr)[-4000:] if report.failed else "",
            "trace_id": item.stash.get(_trace_key, ""),
        })


def pytest_sessionfinish(session):
    if not (observability.enabled() and _eval_rows):
        return
    import mlflow

    if mlflow.active_run() is None:  # every case was skipped before it ran
        return
    passed = sum(r["outcome"] == "passed" for r in _eval_rows)
    failed = sum(r["outcome"] == "failed" for r in _eval_rows)
    mlflow.log_metrics({"passed": passed, "failed": failed,
                        "pass_rate": passed / (passed + failed) if passed + failed else 0.0})
    mlflow.log_table({k: [r[k] for r in _eval_rows] for k in _eval_rows[0]}, "evals.json")
    mlflow.end_run("FAILED" if failed else "FINISHED")
