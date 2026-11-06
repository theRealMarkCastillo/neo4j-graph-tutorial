"""Optional MLflow tracing and eval logging. Off unless MLFLOW_TRACKING_URI is set.

    uv sync --group observability                         # once: installs mlflow
    uv run --group observability mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
    MLFLOW_TRACKING_URI=http://localhost:5001 uv run --group observability python -m companion

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
import os

_mlflow = None  # the mlflow module once setup() turned tracing on


def setup() -> bool:
    """Turn tracing on if MLFLOW_TRACKING_URI is set and mlflow is installed. Safe to call twice."""
    global _mlflow
    if _mlflow is not None:
        return True
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        return False
    try:
        import mlflow
    except ImportError:
        print("MLFLOW_TRACKING_URI is set but mlflow isn't installed: uv sync --group observability")
        return False
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "companion"))
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
