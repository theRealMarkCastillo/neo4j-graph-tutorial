"""Read-only, current-graph extraction proposals. No simulated or temporary writes.

Matching is shared with MemoryStore. Unperformed merges/retirements and unresolved
references are reported explicitly rather than pretending to reproduce a write transaction.
"""
from __future__ import annotations

import datetime as dt
import json

from companion import clock, ontology
from companion.embeddings import embed
from companion.models import Extraction
from companion.store import MemoryStore, _to_date, is_role_name, one_person_roles, temporary_relation


def _item(category, action, text, reason, *, id=None, score=None, details=None):
    return dict(category=category, action=action, text=text, reason=reason,
                id=id, score=score, details=details or {})


def _fields(model):
    return model.model_dump(mode="json", exclude_none=True)


def preview_extraction(store: MemoryStore, extraction: Extraction, episode_id: str,
                       today: dt.date) -> list[dict]:
    if not store._read_only:
        raise ValueError("preview_extraction requires a read-only store view")
    ex, now = extraction, clock.now_for(today)
    items, known, entity_info = [], {}, {}
    pending_merges = []

    def entity(eid):
        if eid not in entity_info:
            rows = store._q("MATCH (e:Entity {id:$id, user:$user}) RETURN properties(e) AS e", id=eid)
            entity_info[eid] = rows[0]["e"] if rows else {}
        return entity_info[eid]

    def lookup(name):
        return store._lookup(name, known)

    def display(eid):
        return entity(eid).get("name", "proposed entity") if eid and eid.startswith("proposal:") else eid

    for index, ent in enumerate(ex.entities):
        if store._is_user(ent.name):
            eid = store.user
        else:
            eid, _ = store._find_entity(ent.name, ent.kind, ent.aliases, ent.summary, ent.ref, ent.new)
        # Exact references to an entity proposed earlier in this extraction need no node.
        if not eid and not ent.new and not ent.ref:
            eid = next((known[n.lower()] for n in [ent.name, *ent.aliases]
                        if n.lower() in known and known[n.lower()].startswith("proposal:")), None)
        details = _fields(ent)
        if not eid:
            eid = f"proposal:{index}"
            entity_info[eid] = dict(name=ent.name, kind=ent.kind, aliases=ent.aliases, summary=ent.summary)
            items.append(_item("entity", "new", ent.name, "no current entity match", details=details))
        elif eid.startswith("proposal:"):
            items.append(_item("entity", "new", ent.name, "references an earlier proposed entity", details=details))
        elif eid == store.user:
            items.append(_item("entity", "existing", ent.name, "user reference", id=eid, details=details))
        else:
            cur = entity(eid)
            cur = {**cur, "aliases": cur.get("aliases", []), "summary": cur.get("summary") or ""}
            changes = store._entity_changes(cur, ent)
            changed = {k: changes[k] for k in ("name", "kind", "summary", "aliases") if changes[k] != cur.get(k)}
            merges = store._role_merges(eid, ent.aliases)
            if merges:
                changed["merge_entities"] = merges
                pending_merges.extend(merges)
            details.update(changes=changed, before={k: cur.get(k) for k in ("name", "kind", "summary", "aliases")})
            items.append(_item("entity", "update" if changed else "existing", ent.name,
                               "matched current entity" + ("; reconciliation proposed" if changed else ""),
                               id=eid, details=details))
            entity_info[eid] = {**cur, **{k: changes[k] for k in ("name", "kind", "summary", "aliases")}}
        for n in [ent.name, *ent.aliases, *([ent.ref] if ent.ref else [])]:
            if n.lower() in known and known[n.lower()] != eid:
                items.append(_item("reference", "unresolved", n, "names multiple entities; apply uses the last declaration"))
            known[n.lower()] = eid

    # Resolve both ends and normalize first, so reasserted ended relations can be kept.
    resolved = []
    for rel in ex.relations:
        a, b = lookup(rel.source), lookup(rel.target)
        canon, how, flip = ontology.normalize(rel.type)
        a, b = (b, a) if flip else (a, b)
        resolved.append((rel, a, b, canon, "flipped" if flip else how))
    linking = {(a, b, canon) for rel, a, b, canon, how in resolved
               if a and b and a != b and not rel.ended and how != "group"
               and not temporary_relation(ex, rel, canon, now)}
    private_names = {n.lower() for m in ex.memories if store._private(m) for n in m.about}
    shared_names = {n.lower() for m in ex.memories if not store._private(m) for n in m.about}

    for rel, a, b, canon, how in resolved:
        source, target = (rel.target, rel.source) if how == "flipped" else (rel.source, rel.target)
        text = f"{source} -{canon}-> {target}"
        details = {**_fields(rel), "canonical_type": canon, "flipped": how == "flipped"}
        if not a or not b:
            items.append(_item("relation", "unresolved", text, "unresolved source or target", details=details))
            continue
        if a == b:
            items.append(_item("relation", "skip", text, "self relation is ignored", details=details))
            continue
        if how == "group":
            items.append(_item("relation", "reject", text, "ontology group is not a relation type", details=details))
            continue
        if temporary_relation(ex, rel, canon, now):
            items.append(_item("relation", "skip", text, "only a temporary memory mentions it (trip, not move)", details=details))
            continue
        if rel.ended and (a, b, canon) in linking:
            items.append(_item("relation", "skip", text, "same extraction reasserts the link", details=details))
            continue
        if not rel.ended and not ontology.range_ok(canon, entity(b).get("kind")):
            items.append(_item("relation", "reject", text, "ontology range does not allow target kind", details=details))
            continue
        types = sorted(ontology.lineage(canon) | {canon}) if rel.ended else [canon]
        edges = store._q("MATCH (:Entity {id:$a, user:$user})-[r:RELATED]->(:Entity {id:$b}) "
                         "WHERE r.until IS NULL AND r.type IN $types RETURN properties(r) AS r", a=a, b=b, types=types)
        details["existing_links"] = [e["r"] for e in edges]
        details["private"] = rel.target.lower() in private_names and rel.target.lower() not in shared_names
        if rel.ended:
            items.append(_item("relation", "end" if edges else "skip", text,
                               "end current links" if edges else "no current link to end", details=details))
            continue
        if a == store.user and canon in one_person_roles() and not is_role_name(entity(b).get("name", "")):
            roles = store._q("MATCH (:User {id:$user})-[r:RELATED]->(e:Entity) "
                             "WHERE e.id <> $b AND r.until IS NULL AND r.type IN $types "
                             "AND NOT toLower(coalesce(r.said,'')) CONTAINS 'step' RETURN e.id AS id, e.name AS name",
                             b=b, types=sorted(ontology.lineage(canon) | {canon}))
            roles = [r for r in roles if is_role_name(r["name"]) and "step" not in rel.type.lower()]
            if roles:
                details["merge_entities"] = roles
                pending_merges.extend(roles)
        reason = "active link already exists; would refresh link metadata" if edges else "proposed link"
        if how == "new":
            reason += "; unknown type would be proposed to ontology"
        items.append(_item("relation", "existing" if edges else "new", text, reason, details=details))
        # The same cardinality and exclusive-type rules used by apply end other links.
        replacement_types = sorted(ontology.lineage(canon) | {canon}) if ontology.rule(canon, "cardinality") == "one" else []
        exclusive = ontology.rule(canon, "exclusive") or []
        for old in store._q("MATCH (:Entity {id:$a, user:$user})-[r:RELATED]->(e:Entity) WHERE r.until IS NULL "
                            "AND (r.type IN $exclusive OR (r.type IN $types AND (e.id <> $b OR r.type <> $type))) "
                            "RETURN e.id AS id, e.name AS name, r.type AS type", a=a, b=b, type=canon,
                            types=replacement_types, exclusive=exclusive):
            items.append(_item("relation", "end", f"{rel.source} -{old['type']}-> {old['name']}",
                               "cardinality/exclusivity replacement", id=old["id"]))

    def memory(mid):
        rows = store._q("MATCH (m:Memory {user:$user, id:$id}) RETURN properties(m) AS m", id=mid)
        return rows[0]["m"] if rows else None

    pending_retirements = set(ex.invalidate)
    for mid in ex.invalidate:
        cur = memory(mid)
        items.append(_item("retirement", "end" if cur and cur.get("invalidated_at") is None else "skip",
                           cur["text"] if cur else mid, "requested invalidation" if cur else "unknown memory id", id=mid))
    for mid in ex.completed_plans:
        cur = memory(mid)
        found = cur and cur.get("kind") == "plan" and cur.get("status") == "open"
        restated = found and (any(mid in m.supersedes for m in ex.memories) or any(
            m.kind == "plan" and _to_date(m.due_on) is not None and _to_date(m.due_on) == cur.get("due_on")
            and m.text.strip().lower() == cur["text"].strip().lower() for m in ex.memories))
        items.append(_item("completion", "end" if found and not restated else "skip", cur["text"] if cur else mid,
                           "same extraction re-states plan; keep open" if restated else
                           "would mark plan done" if found else "no open plan with this id", id=mid))

    stored, tombstones = [], store._tombstones(episode_id)
    vectors = embed([m.text for m in ex.memories]) if ex.memories else []
    for mem, vec in zip(ex.memories, vectors):
        details = _fields(mem)
        resolved_about = [(n, lookup(n)) for n in mem.about]
        about_ids = {eid for _, eid in resolved_about if eid and eid != store.user}
        unresolved = [n for n, eid in resolved_about if not eid]
        decision = store._memory_decision(mem, vec, about_ids, episode_id, now, stored, tombstones)
        action, reason = decision["action"], decision["reason"]
        if unresolved or pending_merges or pending_retirements:
            details["current_graph_match"] = {k: decision[k] for k in ("action", "id", "score") if k in decision}
            details["unresolved_about"] = unresolved
            action, reason = "unresolved", "unresolved references or earlier merge/retirement proposals; write outcome not simulated"
        if action == "new":
            stored.append((vec, about_ids))
        if action == "reinforce":
            cur = memory(decision["id"])
            old_until, until = cur.get("valid_until"), clock.parse_until(mem.valid_until, now, mem.kind)
            old_native = old_until.to_native() if hasattr(old_until, "to_native") else old_until
            changes = {"importance": max(cur["importance"], mem.importance),
                       "access_count": cur.get("access_count", 0) + 1, "last_confirmed": now.isoformat(),
                       "valid_until": max(old_native, until).isoformat() if old_native and until else
                           old_native.isoformat() if old_native else None,
                       "about_incoming": [display(eid) for _, eid in resolved_about if eid]}
            details["changes"] = changes
            details["before"] = {k: cur.get(k) for k in ("text", "importance", "valid_until", "access_count")}
        items.append(_item("memory", action, mem.text, reason.strip(),
                           id=decision.get("id") if action == "reinforce" else None,
                           score=decision.get("score"), details=details))
        for mid in mem.supersedes:
            cur = memory(mid)
            effective = action in ("new", "reinforce") and mid != decision.get("id") and cur and cur.get("invalidated_at") is None
            items.append(_item("retirement", "end" if effective else "unresolved" if action == "unresolved" else "skip",
                               cur["text"] if cur else mid,
                               "replaced by proposed memory" if effective else "no effective replacement established", id=mid))
            if effective:
                pending_retirements.add(mid)
    for name in ex.forget_entities:
        eid = lookup(name)
        items.append(_item("forgetting", "end" if eid and eid != store.user else "skip", name,
                           "would delete entity and matching memories, redact turns and summaries; not applied"
                           if eid and eid != store.user else "no forgettable entity resolved",
                           id=eid if eid and not eid.startswith("proposal:") else None))
    return items


def format_preview(items: list[dict]) -> list[str]:
    lines = []
    for item in items:
        match = f" id={item['id']}" if item.get("id") else ""
        score = f" score={item['score']:.3f}" if item.get("score") is not None else ""
        lines.append(f"  [{item['category']}:{item['action']}]{match}{score} {item['text']}")
        lines.append(f"    reason: {item['reason']}")
        if item.get("details"):
            lines.append("    " + json.dumps(item["details"], ensure_ascii=False, default=str, sort_keys=True))
    return lines
