# Privacy during historical refresh

Historical re-extraction is not a new permission to share. Both the dry-run preview
and `episode_pass(..., refresh=True)` use the same stored-evidence privacy floor.
Normal live chat and the ordinary end-of-episode pass retain their existing sharing
rules.

## Choosing the floor

A refresh is protected when any of these applies:

- Its episode has a stored private or sensitive non-boundary memory, including
  expired or retired evidence.
- The incoming extraction explicitly contains private or sensitive material.
- No original non-boundary memory provenance remains, so source privacy is unknown.
- The proposed entities, text or cited memories overlap related stored private
  evidence. Unrelated private facts elsewhere in the user's graph do not, by
  themselves, privatize a public refresh.

Mixed and unknown contexts are treated conservatively as private. This can keep
new material private even when part of a mixed conversation was public. Existing
shared material is left shared; it is not silently privatized.

This preserves recorded privacy. It cannot prove that an earlier extraction
correctly captured every confidentiality intention in the original conversation.

## Protected refresh behavior

- New memories, relations and entities are private to the episode's source
  persona. Newly created unlinked entities also carry an owner/privacy flag so
  their names and summaries do not become globally visible as orphan nodes.
- Existing relations are retained unchanged. Refresh does not turn a private
  relation shared, reactivate ended links, or end another link to satisfy an
  exclusivity/cardinality rule.
- Existing entity metadata and role-entity merges are deferred. Entity summaries
  and aliases are shared storage, so rewriting them with private material could
  expose it through an otherwise public entity.
- An existing private memory belonging to the source persona can be reinforced
  and gain private ABOUT references. Shared or legacy globally visible memories
  are not enriched with private-context metadata or new ABOUT links.
- Retirements, plan completions, replacements, forgetting, new global boundaries,
  and unknown ontology proposals are deferred. A boundary is global by design;
  unknown ontology examples are global too. A historical model output is not
  enough authority to copy private context into either one.

The preview labels these deferrals and shows effective privacy rather than merely
the model's incoming boolean. Ordinary chat or a separately reviewed operation is
the place to confirm a deferred change.

## Shared-source refresh behavior

With stored shared source evidence and no related private evidence, normal graph
backfilling continues. Existing relation visibility is still preserved during
refresh. All historical entity matching and role resolution are restricted to
entities already visible to the source persona; hidden entities are not reused as
a way to reveal their summaries through new memories or links. An independent
visible entity may be created instead, preferring a split to a privacy-leaking
merge.

These lookup restrictions are scoped to refresh and restored even if it fails.
Normal chat's existing matching and explicit-sharing behavior is unchanged.

## Verification and operations

`tests/test_refresh_privacy.py` exercises real Neo4j with synthetic users and a
deterministic extractor. It covers private/shared controls, derived links,
cross-persona matching, unknown and cross-episode provenance, orphan visibility,
shared metadata/ABOUT protection, deferrals, preview/apply agreement, and scope
restoration on failure.

The preview remains read-only. Applying a backfill is a separate operation and
requires separate approval for real users. A later apply reruns extraction and
can differ from the preview.
