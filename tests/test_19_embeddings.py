"""Lesson 19 — choosing an embedding model: profiles, query instructions, re-embedding.

    uv run python -m pytest tests/test_19_embeddings.py -v

These run on the local model by default (conftest.py sets that up), and a fake endpoint
stands in for the remote one, so they check the wiring without an API key. With
TEST_REMOTE_EMBEDDINGS=1 they run on the model in config.py instead, and the calibration test
checks THAT model's thresholds (the release check in the README).
"""
import numpy as np
import pytest

import config
from companion import calibrate, embeddings
from companion.embeddings import PROFILES, embed, profile_for
from companion.models import Extraction, MemoryOut
from companion.schema import vector_indexes

QWEN = PROFILES["Qwen/Qwen3-Embedding-8B"]


def test_the_default_model_keeps_the_names_lessons_8_to_18_use():
    local = profile_for("BAAI/bge-small-en-v1.5")
    assert (local.prop, local.index("memory_vec")) == ("embedding", "memory_vec")
    # conftest: the local model unless TEST_REMOTE_EMBEDDINGS is set; either way, the profile
    # in use is the one measured for the configured model
    assert embeddings.PROFILE == (profile_for(config.EMBED_MODEL) if embeddings.remote() else local)


def test_another_model_gets_its_own_property_and_indexes():
    assert QWEN.prop == "embedding_qwen3_embedding_8b"
    stmts = " ".join(vector_indexes(QWEN))
    assert "memory_vec_qwen3_embedding_8b" in stmts and "n.embedding_qwen3_embedding_8b" in stmts
    assert "`vector.dimensions`: 4096" in stmts  # an index has ONE size: 384 and 4096 can't share


def test_a_model_without_measured_thresholds_is_refused():
    with pytest.raises(SystemExit, match="calibrate"):
        profile_for("some/new-model")


@pytest.fixture
def fake_remote(monkeypatch):
    """The Qwen profile with a fake endpoint that records what it was sent."""
    sent = []

    def compute(texts):
        sent.append(list(texts))
        return np.array([[len(t), 1.0, 0.0] for t in texts], dtype=np.float32)

    monkeypatch.setattr(embeddings, "PROFILE", QWEN)
    monkeypatch.setattr(embeddings, "_compute", compute)
    monkeypatch.setattr(embeddings, "_cache", type(embeddings._cache)())
    return sent


def test_a_query_gets_the_instruction_and_stored_text_does_not(fake_remote):
    embed(["Logan has a fever"])
    embed(["how is my kid?"], query=True)
    stored, query = fake_remote
    assert stored == ["Logan has a fever"]
    assert query[0].startswith("Instruct: ") and query[0].endswith("how is my kid?")


def test_vectors_are_unit_length_and_cached(fake_remote):
    a = embed(["Mom", "Mom", "mother"])
    assert fake_remote == [["Mom", "mother"]]  # one call, each text once
    assert embed(["mother"]) == [a[2]] and len(fake_remote) == 1  # from the cache
    assert embed(["mother"], query=True) and len(fake_remote) == 2  # a query is a different vector
    assert all(abs(np.linalg.norm(v) - 1) < 1e-6 for v in a)


def test_clustering_uses_its_own_instruction(fake_remote):
    embeddings.embed_for_clustering(["Riley is at the gym"])
    assert fake_remote[0][0].startswith("Instruct: Identify the activity")


# Pairs a model is KNOWN to get wrong, measured and accepted (Lesson 19). Qwen's name scores
# overlap (yes 0.76-0.97, no 0.63-0.85): no single cut gets every pair, and 0.84 keeps
# different people apart at the cost of a few nicknames, which the extractor usually lists
# as aliases anyway. Anything NOT listed here failing is a regression.
KNOWN_MISSES = {
    "Qwen/Qwen3-Embedding-8B": {"name_match": {("Bob", "Robert"), ("Liz", "Elizabeth"), ("Ohio", "Iowa")}},
}


def test_the_active_models_thresholds_hold_on_the_calibration_pairs():
    """The pairs calibrate.py measures a new model on also pin down the current one. These
    three thresholds decide alone; entity matching and sensitive recall have a second check
    (names, the margin to the best match), so a single cut isn't expected to be perfect."""
    p = embeddings.PROFILE
    for name, scale, _, side, pairs in calibrate.CHECKS:
        if name in ("duplicate", "forgotten", "name_match"):
            wrong = {(a, b) for ok, a, b in pairs
                     if (calibrate.similarity(a, b, side, scale) >= getattr(p, name)) != ok}
            assert wrong == KNOWN_MISSES.get(p.model, {}).get(name, set()), f"{p.model} {name}: {wrong}"


def test_best_cut_is_the_middle_of_the_gap():
    assert calibrate.best_cut([0.9, 0.95], [0.5, 0.7]) == pytest.approx(0.8)


def test_reembed_fills_in_missing_vectors_and_recall_finds_them_again(store, driver):
    from companion.reembed import reembed

    store.apply(Extraction(memories=[MemoryOut(text="Pytest's son Logan has a fever", kind="event", importance=6)]))
    store._q(f"MATCH (m:Memory {{user: $user}}) REMOVE m.{embeddings.PROFILE.prop}")  # as if made by another model
    assert store.recall("how is Logan feeling?")["semantic"] == []

    counts = reembed(driver, user="pytest")
    assert counts["Memory"] == 1 and counts["User"] == 0  # only what was missing
    assert [m["text"] for m in store.recall("how is Logan feeling?")["semantic"]] == ["Pytest's son Logan has a fever"]


def test_a_tombstone_from_another_model_is_ignored(store):
    store._q("CREATE (:Forgotten {user: $user, episode: 'ep_x'})")  # no vector for this model
    assert store._tombstones("ep_x") == []


def test_a_model_that_separates_well_labels_weak_matches_by_its_own_score(store, monkeypatch):
    """Qwen3-8B sets weak_below: its vector score decides, and no cross-encoder runs."""
    import dataclasses

    from companion import store as store_module

    # the model's own cut if it has one (Qwen: 0.656); on bge-small, a cut between its scores
    # for this pair (about 0.85 and 0.7). Rex is a real match on either model.
    cut = store_module.PROFILE.weak_below or 0.8
    monkeypatch.setattr(store_module, "PROFILE", dataclasses.replace(store_module.PROFILE, weak_below=cut))
    monkeypatch.setattr(store_module, "relevance", lambda *a: pytest.fail("the cross-encoder ran"))
    store.apply(Extraction(memories=[MemoryOut(text="Pytest's dog Rex is afraid of thunderstorms", kind="fact", importance=6),
                                     MemoryOut(text="Pytest works as an accountant", kind="fact", importance=6)]))
    r = store.recall("my dog hates storms", touch=False)
    assert r["relevance_by"] == "vector score" and r["relevance_min"] == cut
    assert {m["text"]: m["weak"] for m in r["semantic"]} == {
        m["text"]: m["score"] < cut for m in r["semantic"]}
    assert not r["semantic"][0]["weak"] and r["semantic"][-1]["weak"]
