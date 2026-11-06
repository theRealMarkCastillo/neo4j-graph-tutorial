"""Lesson 3 — modeling and performance. Tests can check *how* a query runs, not just what it
returns: EXPLAIN gives the plan without running the query, so you can assert an index is used."""
from conftest import require, rows


def operators(plan) -> set[str]:
    """All operator names in an EXPLAIN plan tree."""
    ops = {plan["operatorType"].split("@")[0]}
    for child in plan.get("children", []):
        ops |= operators(child)
    return ops


def plan_ops(driver, query) -> set[str]:
    _, summary, _ = driver.execute_query("EXPLAIN " + query)
    return operators(summary.plan)


def test_label_lets_the_planner_use_an_index(driver):
    require(driver, "SHOW INDEXES YIELD labelsOrTypes, properties "
                    "WHERE labelsOrTypes = ['Person'] AND properties = ['name'] RETURN count(*) > 0",
            "create the person_name constraint (Lesson 2.6), which adds an index")
    with_label = plan_ops(driver, "MATCH (n:Person {name:'Alice'}) RETURN n")
    assert any("IndexSeek" in op for op in with_label), with_label


def test_no_label_scans_every_node(driver):
    assert "AllNodesScan" in plan_ops(driver, "MATCH (n {name:'Alice'}) RETURN n")


def test_role_property_became_role_nodes(driver):
    require(driver, "MATCH (r:Role) RETURN count(r) > 0", "run the Role refactor (Lesson 3.1) first")
    missing = rows(driver, """
        MATCH (p:Person) WHERE p.role IS NOT NULL AND NOT (p)-[:HAS_ROLE]->(:Role {name: p.role})
        RETURN p.name AS name""")
    assert missing == []


def test_parameters_not_string_formatting(driver):
    # A name with a quote would break (or inject into) a string-built query; a parameter is safe.
    tricky = "O'Brien'}) DETACH DELETE (n) //"
    assert rows(driver, "MATCH (p:Person {name: $name}) RETURN p", name=tricky) == []
