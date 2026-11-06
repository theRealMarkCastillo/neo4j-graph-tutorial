"""Lesson 12 — a small ontology for the companion's life graph.

The memory graph stores FACTS ("Alex -sister-> Maya"). The ontology stores what the relation
TYPES mean, so the system can normalize and check them, and infer new facts:

- vocabulary   one canonical name per relation, plus synonyms  ("girlfriend" -> partner)
- hierarchy    SUBTYPE_OF, a DAG                               (sister -> sibling -> family)
- rules        symmetric, inverse, cardinality, range,         (one lives_in at a time; pet -> a pet;
               exclusive                                        likes ends dislikes of the same thing)
- inference    kinship composition                             (sibling's child -> niece_or_nephew)

Edge direction convention everywhere:  (source)-[:RELATED {type: T}]->(target)
means "target is source's T".  user -sister-> Maya  =  Maya is the user's sister.

The Python dict below is the source of truth. ensure_ontology() mirrors it into Neo4j as
(:RelType)-[:SUBTYPE_OF]->(:RelType) so it can be queried and viewed in the Browser.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

PERSON = ["person"]

# name: parents (SUBTYPE_OF), synonyms, and rules. Nodes with `words` are GROUPS: the words
# people use for the whole category ("anyone in my family...").
ONTOLOGY: dict[str, dict] = {
    "related": {"parents": []},

    # ---- groups
    "family":   {"parents": ["related"], "words": r"famil(y|ies)|relatives?|siblings?|parents|kids|children"},
    "romantic": {"parents": ["related"], "words": r"partners?|dating|love life|relationships?"},
    "friends":  {"parents": ["related"], "words": r"friends?|buddies"},
    "work":     {"parents": ["related"], "words": r"co-?workers?|colleagues?|work|team|job|office"},
    "pets":     {"parents": ["related"], "words": r"pets?|animals?"},
    "places":   {"parents": ["related"], "words": r"places?|cities|towns|where i('ve)? lived"},
    "interests": {"parents": ["related"], "words": r"hobbies|interests?"},

    # ---- family (base kin types first; specific ones are subtypes)
    "sibling":  {"parents": ["family"], "symmetric": True, "range": PERSON,
                 "synonyms": ["step_sibling", "half_sibling"]},
    "sister":   {"parents": ["sibling"], "synonyms": ["sis", "stepsister", "step_sister", "half_sister", "older_sister", "younger_sister", "big_sister", "little_sister"], "range": PERSON},
    "brother":  {"parents": ["sibling"], "synonyms": ["bro", "stepbrother", "step_brother", "half_brother", "older_brother", "younger_brother", "big_brother", "little_brother"], "range": PERSON},
    "parent":   {"parents": ["family"], "inverse": "child", "range": PERSON},
    "mother":   {"parents": ["parent"], "synonyms": ["mom", "mum", "mama", "mommy", "stepmother", "step_mother", "stepmom", "step_mom"], "range": PERSON},
    "father":   {"parents": ["parent"], "synonyms": ["dad", "papa", "daddy", "stepfather", "step_father", "stepdad", "step_dad"], "range": PERSON},
    "child":    {"parents": ["family"], "inverse": "parent", "synonyms": ["kid", "baby"], "range": PERSON},
    "daughter": {"parents": ["child"], "synonyms": ["stepdaughter"], "range": PERSON},
    "son":      {"parents": ["child"], "synonyms": ["stepson"], "range": PERSON},
    "grandparent": {"parents": ["family"], "inverse": "grandchild", "synonyms": ["grandmother", "grandma", "grandmom", "grandfather", "grandpa", "granddad", "granny", "nana"], "range": PERSON},
    "grandchild":  {"parents": ["family"], "inverse": "grandparent", "synonyms": ["granddaughter", "grandson"], "range": PERSON},
    "aunt_or_uncle":   {"parents": ["family"], "inverse": "niece_or_nephew", "range": PERSON},
    "aunt":     {"parents": ["aunt_or_uncle"], "synonyms": ["auntie"], "range": PERSON},
    "uncle":    {"parents": ["aunt_or_uncle"], "range": PERSON},
    "niece_or_nephew": {"parents": ["family"], "inverse": "aunt_or_uncle", "range": PERSON},
    "niece":    {"parents": ["niece_or_nephew"], "range": PERSON},
    "nephew":   {"parents": ["niece_or_nephew"], "range": PERSON},
    "cousin":   {"parents": ["family"], "symmetric": True, "range": PERSON},
    "in_law":   {"parents": ["family"], "range": PERSON},
    "sibling_in_law": {"parents": ["in_law"], "symmetric": True, "synonyms": ["sister_in_law", "brother_in_law"], "range": PERSON},
    "parent_in_law":  {"parents": ["in_law"], "synonyms": ["mother_in_law", "father_in_law"], "range": PERSON},

    # ---- romantic (spouse is BOTH romantic and family: the hierarchy is a DAG, not a tree)
    "partner":  {"parents": ["romantic"], "symmetric": True, "cardinality": "one",
                 "synonyms": ["girlfriend", "boyfriend", "gf", "bf", "significant_other", "dating", "fiance", "fiancee", "fiancé", "fiancée"], "range": PERSON},
    "spouse":   {"parents": ["partner", "family"], "symmetric": True, "cardinality": "one",
                 "synonyms": ["wife", "husband", "married_to", "married", "hubby", "wifey"], "range": PERSON},
    "ex":       {"parents": ["related"], "symmetric": True,
                 "synonyms": ["ex_partner", "ex_girlfriend", "ex_boyfriend", "ex_wife", "ex_husband", "former_partner", "ex_spouse"], "range": PERSON},

    # ---- friends
    "friend":      {"parents": ["friends"], "symmetric": True, "synonyms": ["buddy", "pal", "mate"], "range": PERSON},
    "best_friend": {"parents": ["friend"], "synonyms": ["bff", "bestie", "close_friend", "closest_friend"], "range": PERSON},
    "roommate":    {"parents": ["friends"], "symmetric": True, "synonyms": ["housemate", "flatmate"], "range": PERSON},

    # ---- work
    "coworker": {"parents": ["work"], "symmetric": True, "synonyms": ["colleague", "teammate", "work_friend"], "range": PERSON},
    "manager":  {"parents": ["work"], "inverse": "report", "synonyms": ["boss", "supervisor", "lead"], "range": PERSON},
    "report":   {"parents": ["work"], "inverse": "manager", "synonyms": ["direct_report", "employee"], "range": PERSON},
    "mentor":   {"parents": ["work"], "synonyms": ["teacher", "coach", "tutor", "student_of"], "range": PERSON},
    "works_at": {"parents": ["work"], "synonyms": ["employer", "employed_by", "job_at", "works_for"], "range": ["organization"]},
    "studies_at": {"parents": ["related"], "synonyms": ["attends", "student_at", "goes_to_school_at"], "range": ["organization", "place"]},

    # ---- pets
    "pet":      {"parents": ["pets"], "synonyms": ["owns_pet", "has_pet", "dog", "cat"], "range": ["pet"]},

    # ---- places (a person lives in ONE place at a time)
    "lives_in": {"parents": ["places"], "cardinality": "one", "synonyms": ["resides_in", "home", "lives_at", "moved_to"], "range": ["place"]},
    "from":     {"parents": ["places"], "synonyms": ["hometown", "grew_up_in", "born_in", "is_from"], "range": ["place"]},

    # ---- interests
    "likes":    {"parents": ["interests"], "synonyms": ["enjoys", "loves", "fan_of", "favorite"],
                 "exclusive": ["dislikes"]},
    "dislikes": {"parents": ["interests"], "synonyms": ["hates", "avoids"], "exclusive": ["likes"]},
    "practices": {"parents": ["interests"], "synonyms": ["plays", "does", "hobby", "learning", "learns"]},
}

# Kinship composition: (base of hop 1, base of hop 2) -> inferred relation.
# "my sibling's child is my niece_or_nephew".
COMPOSE = {
    ("sibling", "child"): "niece_or_nephew",
    ("sibling", "partner"): "sibling_in_law",
    ("sibling", "spouse"): "sibling_in_law",
    ("partner", "sibling"): "sibling_in_law",
    ("spouse", "sibling"): "sibling_in_law",
    ("partner", "parent"): "parent_in_law",
    ("spouse", "parent"): "parent_in_law",
    ("parent", "sibling"): "aunt_or_uncle",
    ("parent", "parent"): "grandparent",
    ("child", "child"): "grandchild",
    ("aunt_or_uncle", "child"): "cousin",
    ("sibling", "sibling"): "sibling",
}
BASES = ["sibling", "parent", "child", "spouse", "partner", "grandparent", "grandchild",
         "aunt_or_uncle", "niece_or_nephew", "cousin"]

# Embedding similarity is NOT used to map unknown types automatically: for short relation
# words it confuses look-alikes ("stepmom" ~ "stepson" 0.86, "lives_near" ~ "lives_in" 0.90)
# as often as it finds real synonyms ("hubby" ~ "husband" 0.88). It is only a review hint.


def _key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.strip().lower()).strip("_")


@lru_cache(maxsize=1)
def _synonyms() -> dict[str, str]:
    out = {}
    for name, spec in ONTOLOGY.items():
        if is_group(name):
            continue
        out[name] = name
        for s in spec.get("synonyms", []):
            out[_key(s)] = name
    return out


@lru_cache(maxsize=None)
def ancestors(name: str) -> frozenset[str]:
    """The type and everything above it: sister -> {sister, sibling, family, related}."""
    seen = {name}
    for p in ONTOLOGY.get(name, {}).get("parents", ["related"] if name != "related" else []):
        seen |= ancestors(p)
    return frozenset(seen)


def descendants(name: str) -> set[str]:
    return {t for t in ONTOLOGY if name in ancestors(t)}


def lineage(name: str) -> set[str]:
    """Types that mean the same relation at another level of detail (for ending edges)."""
    return (set(ancestors(name)) | descendants(name)) - {"related", *groups()}


def groups() -> dict[str, str]:
    return {n: s["words"] for n, s in ONTOLOGY.items() if "words" in s}


def groups_in(text: str) -> list[str]:
    return [g for g, words in groups().items() if re.search(rf"\b({words})\b", text, re.I)]


def types_in_group(group: str) -> list[str]:
    return sorted(descendants(group))


def base(name: str) -> str | None:
    """Most specific kinship base: daughter -> child, spouse -> spouse, aunt -> aunt_or_uncle."""
    anc = ancestors(name)
    for b in BASES:
        if b in anc:
            return b
    return None


def inverse(name: str) -> str | None:
    """How the target sees the source: inverse(daughter) = parent, inverse(sister) = sibling."""
    b = base(name) or name
    spec = ONTOLOGY.get(b, {})
    if spec.get("symmetric"):
        return b
    return spec.get("inverse")


def rule(name: str, key: str):
    return ONTOLOGY.get(name, {}).get(key)


@lru_cache(maxsize=1)
def _label_vectors():
    from companion.store import embed  # local import: store imports this module
    labels = [(k.replace("_", " "), v) for k, v in _synonyms().items() if "words" not in ONTOLOGY.get(v, {})]
    vecs = np.array(embed([l for l, _ in labels]))
    return labels, vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def normalize(t: str) -> tuple[str, str, bool]:
    """Map whatever the LLM wrote to a canonical type. Returns (type, how, flip):
    how  = 'exact' | 'synonym' | 'new' (unknown: kept as-is and proposed for review)
           | 'group' (a category like "places" or "family": not a relation, rejected)
    flip = True for "X_of" forms, which point the other way:
           "Ana -daughter_of-> Leo" is stored as "Leo -daughter-> Ana"."""
    k = _key(t)
    syn = _synonyms()
    for cand in (k, k[:-1] if k.endswith("s") else None):  # plural: "teammates", "pets"
        if cand in ONTOLOGY and not is_group(cand):
            return cand, "exact", False
        if cand and cand in syn:
            return syn[cand], "synonym", False
    if k.endswith("_of") and k[:-3] in syn:
        return syn[k[:-3]], "synonym", True
    if is_group(k):  # "user -places-> Denver": the LLM used a heading from the prompt list
        return k, "group", False
    return k, "new", False


def is_group(name: str) -> bool:
    return name == "related" or "words" in ONTOLOGY.get(name, {})


def closest(t: str) -> tuple[str, float]:
    """Nearest canonical type by embedding — a hint for a human reviewing proposed types."""
    from companion.store import embed
    labels, vecs = _label_vectors()
    v = np.array(embed([_key(t).replace("_", " ")])[0])
    sims = vecs @ (v / np.linalg.norm(v))
    i = int(sims.argmax())
    return labels[i][1], round(float(sims[i]), 3)


def range_ok(name: str, target_kind: str | None) -> bool:
    allowed = rule(name, "range")
    return not allowed or target_kind in allowed or target_kind is None


def prompt_vocabulary() -> str:
    """Compact list of canonical relation types for the extractor prompt."""
    lines = []
    for g in groups():
        members = [t for t in types_in_group(g) if t != g and "words" not in ONTOLOGY[t]]
        lines.append(f"  {g}: {', '.join(members)}")
    other = [t for t in ONTOLOGY if t != "related" and "words" not in ONTOLOGY[t]
             and not any(g in ancestors(t) for g in groups())]
    if other:
        lines.append(f"  other: {', '.join(other)}")
    return "\n".join(lines)


def infer_kin(edges: list[dict], user_id: str) -> list[dict]:
    """Two-hop kinship inference over the user's active RELATED edges.

    edges: {src, dst, type} using entity ids. Each edge is also read backwards through its
    inverse, so "Ana -father-> Leo" counts the same as "Leo -daughter-> Ana"."""
    out_edges: dict[str, list[tuple[str, str]]] = {}
    for e in edges:
        b = base(e["type"])
        if b:
            out_edges.setdefault(e["src"], []).append((e["dst"], b))
        inv = inverse(e["type"])
        if inv:
            out_edges.setdefault(e["dst"], []).append((e["src"], base(inv) or inv))
    direct = {dst for dst, _ in out_edges.get(user_id, [])}
    found = {}
    for mid, b1 in out_edges.get(user_id, []):
        for far, b2 in out_edges.get(mid, []):
            rel = COMPOSE.get((b1, b2))
            if rel and far != user_id and far not in direct and far not in found:
                found[far] = {"id": far, "relation": rel, "via": mid}
    return list(found.values())


def tree(root: str = "related", depth: int = 0) -> list[str]:
    """Indented view of the hierarchy, with each type's rules."""
    spec = ONTOLOGY.get(root, {})
    flags = [k for k in ("symmetric",) if spec.get(k)]
    flags += [f"inverse={spec['inverse']}"] if spec.get("inverse") else []
    flags += ["one at a time"] if spec.get("cardinality") == "one" else []
    flags += [f"range={'/'.join(spec['range'])}"] if spec.get("range") and spec["range"] != PERSON else []
    also = [p for p in spec.get("parents", [])[1:]]
    line = "  " * depth + root + (f"  [{', '.join(flags)}]" if flags else "") + (f"  (also under {', '.join(also)})" if also else "")
    out = [line]
    for child, cs in ONTOLOGY.items():
        if (cs.get("parents") or [None])[0] == root:
            out += tree(child, depth + 1)
    return out


def ensure_ontology(driver) -> None:
    """Mirror the ontology into Neo4j: (:RelType {name, ...})-[:SUBTYPE_OF]->(:RelType)."""
    rows = [{"name": n, "parents": s.get("parents", []), "synonyms": s.get("synonyms", []),
             "symmetric": bool(s.get("symmetric")), "inverse": s.get("inverse"),
             "cardinality": s.get("cardinality", "many"), "range": s.get("range", []),
             "exclusive": s.get("exclusive", []),
             "group_words": s.get("words")} for n, s in ONTOLOGY.items()]
    driver.execute_query("CREATE CONSTRAINT reltype_name IF NOT EXISTS FOR (t:RelType) REQUIRE t.name IS UNIQUE")
    driver.execute_query(
        """
        UNWIND $rows AS row
        MERGE (t:RelType {name: row.name})
        SET t.synonyms = row.synonyms, t.symmetric = row.symmetric, t.inverse = row.inverse,
            t.cardinality = row.cardinality, t.range = row.range, t.exclusive = row.exclusive, t.group_words = row.group_words,
            t.proposed = false
        WITH t, row
        OPTIONAL MATCH (t)-[old:SUBTYPE_OF]->() DELETE old
        WITH DISTINCT t, row
        UNWIND row.parents AS p
        MERGE (pt:RelType {name: p})
        MERGE (t)-[:SUBTYPE_OF]->(pt)
        """,
        rows=rows,
    )


def propose_type(driver, name: str, example: str) -> str:
    """An unknown relation type: keep it, but record it (with a hint) for human review."""
    hint, score = closest(name)
    driver.execute_query(
        """
        MERGE (t:RelType {name: $name})
        ON CREATE SET t.proposed = true, t.examples = []
        SET t.examples = (coalesce(t.examples, []) + $example)[-5..],
            t.closest = $hint, t.closest_score = $score
        WITH t MATCH (r:RelType {name: 'related'}) MERGE (t)-[:SUBTYPE_OF]->(r)
        """,
        name=name, example=example, hint=hint, score=score,
    )
    return f"{hint} ({score})"
