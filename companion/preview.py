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
from companion.store import MemoryStore, _to_date, is_role_name, one_person_roles, temporary_relation, visible


def _item(category, action, text, reason, *, id=None, score=None, details=None):
    return dict(category=category, action=action, text=text, reason=reason,
                id=id, score=score, details=details or {})


def _fields(model):
    return model.model_dump(mode="json", exclude_none=True)


def preview_extraction(store: MemoryStore, extraction: Extraction, episode_id: str,
                       today: dt.date, *, refresh: bool = False) -> list[dict]:
    with store.refresh_scope(refresh):
        return _preview_extraction(store, extraction, episode_id, today, refresh=refresh)


def _preview_extraction(store: MemoryStore, extraction: Extraction, episode_id: str,
                        today: dt.date, *, refresh: bool) -> list[dict]:
    if not store._read_only:
        raise ValueError("preview_extraction requires a read-only store view")
    ex, now = extraction, clock.now_for(today)
    if not any(getattr(ex, field) for field in type(ex).model_fields):
        return []
    policy = store.refresh_privacy(ex, episode_id) if refresh else {"private": False}
    protected = policy["private"]
    items, known, entity_info = [], {}, {}
    if refresh:
        items.append(_item("privacy", "existing", "refresh privacy floor", policy["reason"],
                           details={"private": protected, "rule": "private to source persona; no shared reconciliation" if protected else "shared source evidence"}))
    pending_merges = []
    merge_affected = set()

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
            details["private"] = protected
            items.append(_item("entity", "new", ent.name,
                               f"new entity private to {store.agent}" if protected else "no current entity match", details=details))
        elif eid.startswith("proposal:"):
            items.append(_item("entity", "new", ent.name, "references an earlier proposed entity", details=details))
        elif eid == store.user:
            items.append(_item("entity", "existing", ent.name, "user reference", id=eid, details=details))
        else:
            cur = entity(eid)
            cur = {**cur, "aliases": cur.get("aliases", []), "summary": cur.get("summary") or ""}
            changes = store._entity_changes(cur, ent)
            changed = {k: changes[k] for k in ("name", "kind", "summary", "aliases") if changes[k] != cur.get(k)}
            if protected:
                details.update(suppressed_changes=changed, changes={},
                               before={k: cur.get(k) for k in ("name", "kind", "summary", "aliases")})
                items.append(_item("entity", "existing", ent.name,
                                   "existing entity metadata retained during private refresh", id=eid, details=details))
                entity_info[eid] = cur
                for n in [ent.name, *ent.aliases, *([ent.ref] if ent.ref else [])]:
                    known[n.lower()] = eid
                continue
            merges = store._role_merges(eid, ent.aliases)
            if merges:
                changed["merge_entities"] = merges
                pending_merges.extend(merges)
                merge_affected.update([eid, *(m["id"] for m in merges)])
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
        details = {**_fields(rel), "canonical_type": canon, "flipped": how == "flipped",
                   "source_match": display(a), "target_match": display(b)}
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
        if protected and (guard := store.refresh_relation_guard(rel, a, b, canon, how)):
            existing = store._q("MATCH (:Entity {id:$a, user:$user})-[r:RELATED {type:$type}]->(:Entity {id:$b}) "
                                "RETURN properties(r) AS r", a=a, b=b, type=canon)
            old_private = all(bool(r["r"].get("private")) for r in existing) if existing else None
            details.update(refresh_protection=True, private=old_private, private_before=old_private,
                           private_after=old_private, incoming_private=True)
            items.append(_item("relation", guard["action"], text, guard["reason"], details=details))
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
        incoming_private = protected or (rel.target.lower() in private_names and rel.target.lower() not in shared_names)
        before_private = all(bool(e["r"].get("private")) for e in edges) if edges else None
        after_private = before_private if refresh and edges else (
            all(incoming_private and e["r"].get("private", True) for e in edges) if edges else incoming_private)
        details.update(incoming_private=incoming_private, private_before=before_private,
                       private_after=after_private, private=after_private)
        if rel.ended:
            items.append(_item("relation", "end" if edges else "skip", text,
                               "end current links" if edges else "no current link to end", details=details))
            continue
        if a == store.user and canon in one_person_roles() and not is_role_name(entity(b).get("name", "")):
            roles = store._q("MATCH (:User {id:$user})-[r:RELATED]->(e:Entity) "
                             "WHERE e.id <> $b AND r.until IS NULL AND r.type IN $types "
                             "AND NOT toLower(coalesce(r.said,'')) CONTAINS 'step' " + store._refresh_visible("e") +
                             " RETURN e.id AS id, e.name AS name",
                             b=b, types=sorted(ontology.lineage(canon) | {canon}))
            roles = [r for r in roles if is_role_name(r["name"]) and "step" not in rel.type.lower()]
            if roles:
                details["merge_entities"] = roles
                pending_merges.extend(roles)
                merge_affected.update([display(b), *(r["id"] for r in roles)])
        reason = "active link already exists; would refresh link metadata" if edges else "proposed link"
        action = "existing" if edges else "new"
        if edges and before_private != after_private:
            action = "update"
            reason = (f"would change visibility from {'private' if before_private else 'shared'} "
                      f"to {'private' if after_private else 'shared'}")
        elif edges and incoming_private and not after_private:
            reason = "active link stays shared; incoming private intent cannot make a shared link private"
        if how == "new":
            reason += "; unknown type would be proposed to ontology"
        items.append(_item("relation", action, text, reason, details=details))
        # The same cardinality and exclusive-type rules used by apply end other links.
        replacement_types = sorted(ontology.lineage(canon) | {canon}) if ontology.rule(canon, "cardinality") == "one" else []
        exclusive = ontology.rule(canon, "exclusive") or []
        for old in store._q("MATCH (:Entity {id:$a, user:$user})-[r:RELATED]->(e:Entity) WHERE r.until IS NULL "
                            "AND ((e.id = $b AND r.type IN $exclusive) OR (r.type IN $types AND (e.id <> $b OR r.type <> $type))) "
                            "RETURN e.id AS id, e.name AS name, r.type AS type", a=a, b=b, type=canon,
                            types=replacement_types, exclusive=exclusive):
            items.append(_item("relation", "end", f"{rel.source} -{old['type']}-> {old['name']}",
                               "cardinality/exclusivity replacement", id=old["id"],
                               details={"source_match": display(a), "target_match": old["id"]}))

    # Apply can delete/transfer these nodes before resolving or writing later relations.
    # Keep the current-graph observation as evidence, but do not promise that link.
    for item in items:
        if (item["category"] == "relation" and item["action"] in {"existing", "new", "update", "end"}
                and {item["details"].get("source_match"), item["details"].get("target_match")} & merge_affected):
            item["details"]["current_graph_match"] = {"action": item["action"], "reason": item["reason"]}
            item["action"] = "unresolved"
            item["reason"] = "depends on proposed entity merge; transferred/deleted links are not simulated"

    def memory(mid):
        rows = store._q("MATCH (m:Memory {user:$user, id:$id}) RETURN CASE WHEN " + visible("m") +
                        " THEN properties(m) ELSE null END AS m", id=mid)
        if not rows:
            return None
        return rows[0]["m"] or {"text": mid, "_hidden": True}

    pending_retirements = set() if protected else set(ex.invalidate)
    for mid in ex.invalidate:
        cur = memory(mid)
        hidden = cur and cur.get("_hidden")
        items.append(_item("retirement", "skip" if protected else "unresolved" if hidden else
                           "end" if cur and cur.get("invalidated_at") is None else "skip",
                           cur["text"] if cur else mid, "deferred retirement during private refresh" if protected else
                           "memory id is not visible to this persona" if hidden else
                           "requested invalidation" if cur else "unknown memory id", id=mid))
    for mid in ex.completed_plans:
        cur = memory(mid)
        hidden = cur and cur.get("_hidden")
        found = cur and not hidden and cur.get("kind") == "plan" and cur.get("status") == "open"
        restated = found and (any(mid in m.supersedes for m in ex.memories) or any(
            m.kind == "plan" and _to_date(m.due_on) is not None and _to_date(m.due_on) == cur.get("due_on")
            and m.text.strip().lower() == cur["text"].strip().lower() for m in ex.memories))
        items.append(_item("completion", "skip" if protected else "unresolved" if hidden else "end" if found and not restated else "skip",
                           cur["text"] if cur else mid, "deferred plan completion during private refresh" if protected else
                           "memory id is not visible to this persona" if hidden else
                           "same extraction re-states plan; keep open" if restated else
                           "would mark plan done" if found else "no open plan with this id", id=mid))

    stored, tombstones = [], store._tombstones(episode_id)
    vectors = embed([m.text for m in ex.memories]) if ex.memories else []
    for mem, vec in zip(ex.memories, vectors):
        details = _fields(mem)
        resolved_about = [(n, lookup(n)) for n in mem.about]
        about_ids = {eid for _, eid in resolved_about if eid and eid != store.user}
        unresolved = [n for n, eid in resolved_about if not eid]
        base_decision = store._memory_decision(mem, vec, about_ids, episode_id, now, stored, tombstones)
        decision = store.refresh_memory_guard(mem, base_decision, protected)
        action, reason = decision["action"], decision["reason"]
        details["incoming_private"] = store._private(mem)
        if base_decision["action"] == "reinforce":
            details.update(private=base_decision["private"], private_before=base_decision["private"],
                           private_after=base_decision["private"])
        if action == "new" and protected:
            details.update(incoming_private=store._private(mem), private=True)
            reason += f"; private to {store.agent} by refresh privacy floor"
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
            hidden = cur and cur.get("_hidden")
            effective = (not protected and action in ("new", "reinforce") and mid != decision.get("id") and cur
                         and not hidden and cur.get("invalidated_at") is None)
            items.append(_item("retirement", "skip" if protected else "unresolved" if hidden else "end" if effective else
                               "unresolved" if action == "unresolved" else "skip",
                               cur["text"] if cur else mid,
                               "deferred replacement during private refresh" if protected else
                               "memory id is not visible to this persona" if hidden else
                               "replaced by proposed memory" if effective else "no effective replacement established", id=mid))
            if effective:
                pending_retirements.add(mid)
    for name in ex.forget_entities:
        eid = lookup(name)
        items.append(_item("forgetting", "skip" if protected else "end" if eid and eid != store.user else "skip", name,
                           "deferred forgetting during private refresh; confirm through normal chat" if protected else
                           "would delete entity, its links and matching memories, redact turns and summaries; not applied"
                           if eid and eid != store.user else "no forgettable entity resolved",
                           id=eid if eid and not eid.startswith("proposal:") else None))
    # Neo4j temporal properties in before/after details become ordinary JSON strings.
    return json.loads(json.dumps(items, ensure_ascii=False, default=str))


def format_preview(items: list[dict]) -> list[str]:
    lines = []
    for item in items:
        match = f" id={item['id']}" if item.get("id") else ""
        score = f" score={item['score']:.3f}" if item.get("score") is not None else ""
        lines.append(f"  [{item['category']}:{item['action']}]{match}{score} {item['text']}")
        lines.append(f"    reason: {item['reason']}")
        if item.get("details"):
            lines.append("    " + json.dumps(item["details"], ensure_ascii=False, default=str, sort_keys=True))
    if any(i["category"] in {"retirement", "forgetting"} and i["action"] == "end" for i in items):
        lines.append("  [effects:unresolved] Downstream relation/provenance effects of retirement or forgetting are not simulated.")
    return lines
