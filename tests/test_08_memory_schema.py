"""Lesson 8 — the memory schema. Schema setup must be idempotent: running it twice is a test."""
from companion.schema import ensure_schema
from conftest import rows


def test_schema_is_idempotent_and_complete(driver):
    ensure_schema(driver)
    ensure_schema(driver)  # IF NOT EXISTS everywhere: the second run must not fail
    constraints = {r["name"] for r in rows(driver, "SHOW CONSTRAINTS YIELD name RETURN name")}
    assert {"entity_id", "memory_id", "episode_id"} <= constraints
    indexes = {r["name"]: r["state"] for r in rows(driver, "SHOW INDEXES YIELD name, state RETURN name, state")}
    for name in ["memory_vec", "entity_vec", "episode_vec", "memory_user_kind", "memory_text"]:
        assert indexes.get(name) == "ONLINE", name


def test_memory_ids_are_unique(store):
    from neo4j.exceptions import ConstraintError
    import pytest

    store._q("CREATE (:Memory {id: 'm_dup', user: $user})")
    with pytest.raises(ConstraintError):
        store._q("CREATE (:Memory {id: 'm_dup', user: $user})")
