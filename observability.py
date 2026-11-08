"""Optional MLflow tracing and eval logging. Off unless MLFLOW_TRACKING_URI is set in .env.

    uv sync --group observability                         # once: installs mlflow
    uv run --group observability mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
    # .env: MLFLOW_TRACKING_URI=http://localhost:5001, then
    uv run --group observability python -m companion

What you get once it's on:
- every LangGraph node and LLM call, with its prompt, output, tokens and latency
  (mlflow.langchain.autolog() — all LLM calls here go through ChatOpenAI);
- the Neo4j work autolog can't see: recall, extraction candidates and the write path,
  as spans made by `traced` below;
- `pytest -m llm` evals logged as one MLflow run per session (tests/conftest.py).

Everything here is a no-op when tracing is off or mlflow isn't installed, so the course
runs exactly as before without it.
"""
from __future__ import annotations

import contextlib
import functools

_mlflow = None  # the mlflow module once setup() turned tracing on


def setup(uri: str | None, experiment: str = "companion") -> bool:
    """Turn tracing on if `uri` (config.MLFLOW_TRACKING_URI) is set and mlflow is installed.
    Called once by config.py; safe to call again."""
    global _mlflow
    if _mlflow is not None:
        return True
    if not uri:
        return False
    try:
        import mlflow
    except ImportError:
        print("MLFLOW_TRACKING_URI is set but mlflow isn't installed: uv sync --group observability")
        return False
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    mlflow.langchain.autolog()
    _mlflow = mlflow
    return True


def enabled() -> bool:
    return _mlflow is not None


def traced(name: str | None = None, span_type: str = "UNKNOWN"):
    """Decorator: record a call as a span (inputs = arguments, output = return value).

    Decided per call, so it works on functions defined before setup() ran.
    span_type is an mlflow SpanType value: "RETRIEVER", "CHAIN", "TOOL", ..."""
    def wrap(fn):
        traced_fn = None

        @functools.wraps(fn)
        def call(*args, **kwargs):
            nonlocal traced_fn
            if _mlflow is None:
                return fn(*args, **kwargs)
            if traced_fn is None:
                traced_fn = _mlflow.trace(fn, name=name or fn.__name__, span_type=span_type)
            return traced_fn(*args, **kwargs)
        return call
    return wrap


@contextlib.contextmanager
def span(name: str, span_type: str = "CHAIN", inputs: dict | None = None):
    """A span around a block (a new trace if none is active). Yields the span, or None when off."""
    if _mlflow is None:
        yield None
        return
    with _mlflow.start_span(name=name, span_type=span_type) as s:
        if inputs:
            s.set_inputs(inputs)
        yield s


def tag_trace(user: str | None = None, session: str | None = None, **tags: str) -> None:
    """Label the active trace so the UI can group a conversation's turns. No-op outside a trace."""
    if _mlflow is None or _mlflow.get_current_active_span() is None:
        return
    with contextlib.suppress(Exception):  # tagging must never break a chat turn
        _mlflow.update_current_trace(user=user, session_id=session,
                                     tags={k: str(v) for k, v in tags.items()} or None)


# ------------------------------------------------------------------ eval runs (Lesson 11's evals)
# pytest -m llm (tests/conftest.py) and companion.scenarios both log the same way:
#   one MLflow run per session: the models, a hash of the prompt files, the git commit;
#   each case's verdict as feedback ON its trace, so the UI shows pass/fail next to the prompts;
#   passed / failed / pass_rate, and a table of every case.
# The verdict always comes from the code checks; MLflow only records it.

PROMPT_FILES = ("companion/extract.py", "companion/graph.py")  # where the prompts live


def start_eval_run(kind: str, **params) -> None:
    """Start the run every case's trace is linked to. Does nothing when tracing is off."""
    if _mlflow is None or _mlflow.active_run() is not None:
        return
    import hashlib
    import subprocess
    from pathlib import Path

    import config

    root = Path(__file__).resolve().parent
    prompts = hashlib.sha256()
    for f in PROMPT_FILES:
        prompts.update((root / f).read_bytes())
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()  # noqa: E731
    sha, dirty = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    _mlflow.start_run(run_name=f"{kind} {sha}{'+dirty' if dirty else ''}", tags={"eval_kind": kind})
    _mlflow.log_params({
        "llm_model": config.LLM_MODEL, "extraction_model": config.LLM_EXTRACTION_MODEL,
        "embed_model": config.EMBED_MODEL, "prompts_sha": prompts.hexdigest()[:12],
        "git_sha": sha, "git_dirty": dirty, **params,
    })


def record_verdict(trace_id: str | None, check: str, passed: bool, why: str = "",
                   source: str = "eval") -> None:
    """Attach a code check's verdict to a trace (the UI's assessments column)."""
    if _mlflow is None or not trace_id:
        return
    from mlflow.entities import AssessmentSource, AssessmentSourceType

    try:
        # Traces upload in the background: the trace must reach the server before its verdict.
        _mlflow.flush_trace_async_logging()
        _mlflow.log_feedback(trace_id=trace_id, name=check, value=passed, rationale=why or None,
                             source=AssessmentSource(AssessmentSourceType.CODE, source))
    except Exception as e:  # logging a verdict must never fail the eval
        print(f"  (MLflow: couldn't record the verdict on {trace_id}: {type(e).__name__}: {e})")


def finish_eval_run(rows: list[dict]) -> None:
    """Log totals and the case table, then end the run. rows: dicts with at least `outcome`."""
    if _mlflow is None or _mlflow.active_run() is None:
        return
    if rows:
        passed = sum(r["outcome"] == "passed" for r in rows)
        failed = sum(r["outcome"] == "failed" for r in rows)
        _mlflow.log_metrics({"passed": passed, "failed": failed,
                             "pass_rate": passed / (passed + failed) if passed + failed else 0.0})
        _mlflow.log_table({k: [r.get(k, "") for r in rows] for k in rows[0]}, "evals.json")
    _mlflow.end_run("FAILED" if any(r["outcome"] == "failed" for r in rows) else "FINISHED")
