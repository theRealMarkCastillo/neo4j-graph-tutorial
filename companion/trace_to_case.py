"""Turn a traced extraction into a new eval case (Lesson 11's loop, with MLflow traces).

    uv run --group observability python -m companion.trace_to_case <trace id>            # print it
    uv run --group observability python -m companion.trace_to_case <trace id> --append   # add it

The round trip:
1. Chat with tracing on (MLFLOW_TRACKING_URI in config.py) and spot a wrong memory.
2. In the MLflow UI, open that turn's `remember` trace and copy its id (tr-...).
3. Run this. It writes a pytest case that replays the extraction with the exact inputs the
   trace recorded (message, reply, candidates, known entities, earlier turns, date), and shows
   what the model returned in a comment.
4. Replace the TODO with a check for the rule that was broken. The case should FAIL now.
5. Fix the prompt, then run `pytest -m llm` again and compare the runs in MLflow.

The case calls extract_memories() directly, like the other cases in
tests/test_11_extraction_evals.py: no database, so it's cheap to run many times.
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import observability

EVAL_FILE = Path(__file__).resolve().parents[1] / "tests" / "test_11_extraction_evals.py"


def _compact(value):
    """Drop empty fields from the recorded output, the way model_dump(exclude_defaults) would."""
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if v not in (None, [], {}, "", False, 0)}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _literal(value, indent: int) -> str:
    """A Python literal for `value`, wrapped to fit the case's indentation."""
    import pprint

    text = pprint.pformat(value, width=96 - indent, sort_dicts=False)
    return textwrap.indent(text, " " * indent).lstrip()


def case_from_span(trace_id: str, inputs: dict, outputs, name: str | None = None) -> str:
    msg = inputs["user_msg"]
    slug = name or "_".join(re.findall(r"[a-z0-9]+", msg.lower())[:6]) or "case"
    earlier = [tuple(pair) for pair in inputs.get("earlier") or []]
    args = [repr(inputs["user"]), repr(msg), repr(inputs.get("assistant_msg", ""))]
    for key, value in (("candidates", inputs.get("candidates") or []),
                       ("entities", inputs.get("entities") or []), ("earlier", earlier)):
        if value:
            args.append(f"{key}={_literal(value, 12)}")
    if inputs.get("today"):
        args.append(f"today=dt.date.fromisoformat({inputs['today']!r})")
    if inputs.get("companion"):
        args.append(f"companion={inputs['companion']!r}")
    recorded = textwrap.indent(_literal(_compact(outputs), 0), "        #   ")
    call = ",\n            ".join(args)
    return f'''

def test_from_trace_{slug}(llm_ready):
    """From MLflow trace {trace_id}. TODO: the rule this case pins, in one line."""
    def check():
        ex = extract_memories(
            {call},
        )
        dump = ex.model_dump_json(exclude_defaults=True)
        # What the model returned when the trace was recorded:
{recorded}
        assert False, "TODO: check the rule that was broken. " + dump
    trials(check)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trace_id")
    parser.add_argument("--name", help="test name suffix (default: from the message)")
    parser.add_argument("--append", action="store_true", help=f"append to {EVAL_FILE.name}")
    args = parser.parse_args()

    import config  # noqa: F401  (turns tracing on from config.MLFLOW_TRACKING_URI)

    if not observability.enabled():
        sys.exit("Set MLFLOW_TRACKING_URI in config.py (and run `uv sync --group observability`) first.")
    import mlflow

    trace = mlflow.get_trace(args.trace_id)
    if trace is None:
        sys.exit(f"No trace {args.trace_id} on {mlflow.get_tracking_uri()}")
    spans = [s for s in trace.data.spans if s.name == "extract_memories"]
    if not spans:
        sys.exit("That trace has no extract_memories span. With background extraction it's in "
                 "the turn's separate `remember` trace (same session): copy that trace's id. "
                 "(Small talk skips extraction, so its turns have none.)")
    cases = "".join(case_from_span(args.trace_id, s.inputs, s.outputs, args.name) for s in spans)
    if args.append:
        with EVAL_FILE.open("a") as f:
            f.write(cases)
        print(f"Appended {len(spans)} case(s) to {EVAL_FILE}: fill in the TODOs.")
    else:
        print(cases)


if __name__ == "__main__":
    main()
