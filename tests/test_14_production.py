"""Lesson 14 — production hardening: transactions, one writer per user, background extraction.

    .venv/bin/python -m pytest tests/test_14_production.py -v

Concurrency bugs don't show up when you run one thing at a time, so these tests force the
interleavings on purpose: a failure half-way through a write, two writers at the same moment,
a reader while a write is still open.
"""
import datetime as dt
import threading
import time

import pytest

from companion.models import EntityOut, Extraction, MemoryOut, RelationOut
from companion.worker import Background

D1 = dt.date(2026, 11, 18)


def fact(text: str, about=("user",)) -> MemoryOut:
    return MemoryOut(text=text, kind="fact", importance=6, about=list(about))


def count(store, label: str = "Memory") -> int:
    return store._q(f"MATCH (n:{label} {{user: $user}}) WHERE NOT n:User RETURN count(n) AS n")[0]["n"]


# ---------------------------------------------------------------- all or nothing

def test_a_failure_half_way_rolls_back_everything(store, monkeypatch):
    """Entities and relations are written before memories. Make the memory step fail: the
    entity and the relation must disappear too, or the graph says "sister Maya" with no memory
    behind it, and a retry would add a second copy of whatever did get written."""
    def boom(*_):
        raise RuntimeError("simulated crash between the relation and the memory")
    monkeypatch.setattr(store, "_tombstones", boom)
    ex = Extraction(entities=[EntityOut(name="Maya", kind="person")],
                    relations=[RelationOut(source="user", target="Maya", type="sister")],
                    memories=[fact("Pytest's sister Maya lives in Denver", ["user", "Maya"])])
    with pytest.raises(RuntimeError):
        store.apply(ex, today=D1)
    assert count(store, "Entity") == 0
    assert count(store, "Memory") == 0
    assert store._q("MATCH (:User {id: $user})-[r:RELATED]->() RETURN count(r) AS n")[0]["n"] == 0

    monkeypatch.undo()  # the same extraction, retried, now works and is stored once
    store.apply(ex, today=D1)
    assert (count(store, "Entity"), count(store, "Memory")) == (1, 1)


# ---------------------------------------------------------------- one writer per user

def test_two_writers_at_once_store_one_memory(store):
    """Four threads apply the same fact at the same instant. Without the user lock, each one
    checks for a duplicate, finds none (the others haven't committed) and creates its own copy.
    With it, they take turns: one creates, three reinforce."""
    ex = Extraction(memories=[fact("Pytest is allergic to peanuts")])
    start, logs = threading.Barrier(4), []

    def writer():
        start.wait()
        logs.extend(store.apply(ex, today=D1))

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert count(store) == 1
    assert sorted(line.split()[0] for line in logs) == ["+", "reinforced", "reinforced", "reinforced"]


def test_a_reader_never_joins_another_threads_transaction(store):
    """The open transaction is kept per thread. While a write is in progress on one thread,
    a read on another runs on its own, and can't see the uncommitted memory."""
    inside, release, seen = threading.Event(), threading.Event(), []

    def slow_write():
        def work():
            store._q("CREATE (:Memory {id: 'm_open', user: $user, text: 'not committed yet'})")
            inside.set()
            release.wait(5)
        store._write(work)

    t = threading.Thread(target=slow_write)
    t.start()
    inside.wait(5)
    seen.append(count(store))  # main thread, while the write is still open
    release.set()
    t.join()
    assert seen == [0]
    assert count(store) == 1


def test_repeats_inside_one_extraction_are_stored_once(store):
    """Inside one transaction, the vector index doesn't see memories created earlier in the
    same transaction, so apply() also compares against what it just stored."""
    log = store.apply(Extraction(memories=[fact("Pytest lives in Chicago"),
                                           fact("Pytest lives in Chicago")]), today=D1)
    assert count(store) == 1
    assert any(line.startswith("skipped a repeat") for line in log)


def test_a_pattern_links_its_evidence_even_when_it_reinforces_an_older_one(store):
    """Bug found while adding transactions: consolidate() found the pattern again by its TEXT.
    When apply() reinforced an existing pattern with slightly different wording, nothing
    matched, so the links and the evidence count were silently skipped. Now it uses ids."""
    store.apply(Extraction(memories=[MemoryOut(text="Pytest goes to the gym regularly.",
                                               kind="preference", importance=6)]), today=D1)
    for i, text in enumerate(["Pytest is at the gym", "Pytest went to the gym after work",
                              "Pytest is heading to the gym for leg day"]):
        store.apply(Extraction(memories=[MemoryOut(text=text, kind="status", importance=2)]),
                    today=D1 + dt.timedelta(days=i + 1))
    writer = lambda *_: MemoryOut(text="Pytest goes to the gym regularly!", kind="preference", importance=6)
    log = store.consolidate(D1 + dt.timedelta(days=4), writer=writer, prune=False)
    assert any("reinforced" in line for line in log)
    evidence = store._q("MATCH (p:Memory {user: $user, kind: 'preference'}) RETURN p.evidence AS n")
    assert evidence == [{"n": 3}]


# ---------------------------------------------------------------- the background worker (no database)

def test_jobs_run_in_order_one_at_a_time():
    """One worker thread per user: job 2 reads what job 1 wrote, so they must not overlap."""
    bg, order = Background(), []

    def job(n):
        def run():
            order.append(f"start {n}")
            time.sleep(0.01)
            order.append(f"end {n}")
            return [f"job {n}"]
        return run

    for n in range(3):
        bg.submit(job(n))
    assert bg.wait(5) == ["job 0", "job 1", "job 2"]
    assert order == ["start 0", "end 0", "start 1", "end 1", "start 2", "end 2"]
    bg.close()


def test_submit_returns_immediately_and_drain_collects_finished_logs():
    bg, go = Background(), threading.Event()
    bg.submit(lambda: (go.wait(5), ["done"])[1])
    t = time.perf_counter()
    assert bg.drain() == []          # nothing finished yet, and drain() doesn't block
    assert time.perf_counter() - t < 0.1
    go.set()
    assert bg.wait(5) == ["done"]
    assert bg.drain() == []          # each line is handed out once
    bg.close()


def test_a_failing_job_is_reported_not_lost():
    """An exception in a background thread vanishes unless someone collects it."""
    bg = Background()
    bg.submit(lambda: 1 / 0)
    bg.submit(lambda: ["the next job still runs"])
    lines = bg.wait(5)
    assert lines[0].startswith("memory update failed: ZeroDivisionError")
    assert lines[1] == "the next job still runs"
    bg.close()


# ---------------------------------------------------------------- the companion with a worker (fake LLM)

@pytest.fixture
def fake_llm(monkeypatch):
    """Swap the chat model for a canned one: these tests are about timing, not wording."""
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    import companion.graph as g
    monkeypatch.setattr(g, "chat_model", lambda: FakeListChatModel(responses=["Nice!"]))
    return g


def chat(app, text: str, episode_id: str) -> dict:
    from langchain_core.messages import HumanMessage
    return app.invoke({"messages": [HumanMessage(text)]}, {"configurable": {"thread_id": episode_id, "today": D1}})


def test_the_reply_does_not_wait_for_memory(store, fake_llm, monkeypatch):
    gate = threading.Event()

    def slow_extract(*_, **__):  # stands in for a multi-second LLM call
        gate.wait(60)  # released below; generous so a slow remote embedding endpoint can't end it early
        return Extraction(memories=[fact("Pytest started learning the cello")])

    monkeypatch.setattr(fake_llm, "extract_memories", slow_extract)
    bg = Background()
    app = fake_llm.build_companion(store, bg)
    episode_id = store.start_episode(on=D1)

    out = chat(app, "I started learning the cello!", episode_id)
    assert out["messages"][-1].content == "Nice!"  # answered...
    assert count(store) == 0                         # ...while the memory is still being written
    gate.set()
    assert bg.wait(60)[0].startswith("+ [fact] Pytest started learning the cello")  # remote embeddings can take seconds
    assert count(store) == 1
    bg.close()


def test_a_background_forget_is_scrubbed_from_the_conversation_next_turn(store, fake_llm, monkeypatch):
    """In sync mode `remember` redacts the thread right away. In the background the job ends
    after the turn, so the next turn's `recall` applies the redaction instead."""
    store.apply(Extraction(entities=[EntityOut(name="Jordan", kind="person")],
                           memories=[fact("Pytest's ex is Jordan", ["user", "Jordan"])]), today=D1)
    monkeypatch.setattr(fake_llm, "extract_memories",
                        lambda *_, **__: Extraction(forget_entities=["Jordan"]))
    bg = Background()
    app = fake_llm.build_companion(store, bg)
    episode_id = store.start_episode(on=D1)

    chat(app, "Please forget everything about Jordan.", episode_id)
    assert any(line.startswith("forgot Jordan") for line in bg.wait(60))
    out = chat(app, "ok thanks", episode_id)
    texts = " ".join(m.content for m in out["messages"])
    assert "Jordan" not in texts and "[forgotten]" in texts
    bg.close()
