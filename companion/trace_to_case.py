"""Turn a traced extraction into a new eval case (Lesson 11's loop, with MLflow traces).

    uv run --group observability python -m companion.trace_to_case <trace id>            # print it
    uv run --group observability python -m companion.trace_to_case <trace id> --append   # add it

The round trip:
1. Chat with tracing on (MLFLOW_TRACKING_URI, see observability.py) and spot a wrong memory.
2. In the MLflow UI, open that turn's `remember` trace (or the trace holding the end-of-episode
   `extract_episode` span) and copy its id (tr-...).
3. Run this. It writes a pytest case that replays the extraction with the exact inputs the
   trace recorded (message, reply, candidates, known entities, earlier turns, date), and shows
   what the model returned in a comment.
4. Replace the TODO with a check for the rule that was broken. The case should FAIL now.
5. Fix the prompt, then run `pytest -m llm` again and compare the runs in MLflow.

The case calls extract_memories() (a turn) or extract_episode() (the end-of-episode pass over
the whole transcript) directly, like the other cases in tests/test_11_extraction_evals.py: no
database, so it's cheap to run many times.
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


def _slug(text: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", text.lower())[:6]) or "case"


def _memories_call(inputs: dict) -> tuple[str, list[str], str]:
    """extract_memories(): one turn, with the earlier turns it was shown."""
    msg = inputs["user_msg"]
    earlier = [tuple(pair) for pair in inputs.get("earlier") or []]
    args = [repr(inputs["user"]), repr(msg), repr(inputs.get("assistant_msg", ""))]
    for key, value in (("candidates", inputs.get("candidates") or []),
                       ("entities", inputs.get("entities") or []), ("earlier", earlier)):
        if value:
            args.append(f"{key}={_literal(value, 12)}")
    return "extract_memories", args, msg


def _episode_call(inputs: dict) -> tuple[str, list[str], str]:
    """extract_episode(): the end-of-episode pass over the whole transcript."""
    transcript = inputs["transcript"]
    args = [repr(inputs["user"]), _literal(transcript, 12),
            _literal(inputs.get("candidates") or [], 12), _literal(inputs.get("entities") or [], 12)]
    if inputs.get("refresh"):
        args.append("refresh=True")
    # Name it after the first user line, not the "User:" prefix.
    first = transcript.split("\n", 1)[0].split(":", 1)[-1]
    return "extract_episode", args, "episode " + first


SPAN_CALLS = {"extract_memories": _memories_call, "extract_episode": _episode_call}


def _unique(slug: str, taken: set[str]) -> str:
    """`slug`, numbered if a test of that name exists: a second `def` would silently replace it."""
    n, candidate = 1, slug
    while candidate in taken:
        n += 1
        candidate = f"{slug}_{n}"
    taken.add(candidate)
    return candidate


def case_from_span(trace_id: str, span_name: str, inputs: dict, outputs,
                   name: str | None = None, taken: set[str] | None = None) -> str:
    func, args, label = SPAN_CALLS[span_name](inputs)
    slug = _unique(name or _slug(label), taken if taken is not None else set())
    if inputs.get("today"):
        args.append(f"today=dt.date.fromisoformat({inputs['today']!r})")
    if inputs.get("companion"):
        args.append(f"companion={inputs['companion']!r}")
    recorded = textwrap.indent(_literal(_compact(outputs), 0), "        #   ")
    call = ",\n            ".join(args)
    return f'''

def test_from_trace_{slug}(llm_ready):
    """From MLflow trace {trace_id} ({span_name}). TODO: the rule this case pins, in one line."""
    def check():
        ex = {func}(
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

    import config  # noqa: F401  (turns tracing on from MLFLOW_TRACKING_URI in .env)

    if not observability.enabled():
        sys.exit("Set MLFLOW_TRACKING_URI in .env (and run `uv sync --group observability`) first.")
    import mlflow

    trace = mlflow.get_trace(args.trace_id)
    if trace is None:
        sys.exit(f"No trace {args.trace_id} on {mlflow.get_tracking_uri()}")
    spans = [s for s in trace.data.spans if s.name in SPAN_CALLS]
    if not spans:
        sys.exit("That trace has no extract_memories or extract_episode span. With background "
                 "extraction it's in the turn's separate `remember` trace (same session): copy "
                 "that trace's id. (Small talk skips extraction, so its turns have none.)")
    # Test names already in the eval file, so a new case never shadows one.
    taken = set(re.findall(r"^def test_from_trace_(\w+)\(", EVAL_FILE.read_text(), re.M))
    cases = "".join(case_from_span(args.trace_id, s.name, s.inputs, s.outputs, args.name, taken)
                    for s in spans)
    if args.append:
        with EVAL_FILE.open("a") as f:
            f.write(cases)
        print(f"Appended {len(spans)} case(s) to {EVAL_FILE}: fill in the TODOs.")
    else:
        print(cases)


if __name__ == "__main__":
    main()
