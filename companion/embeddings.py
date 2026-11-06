"""Lesson 19 — the companion's embedding model: where it runs, and the thresholds measured for it.

Every similarity cut-off in the memory store (is this a duplicate? the same person? part of a
habit?) was measured on ONE model's scores. Another model spreads its scores differently, so
switching models without re-measuring silently breaks dedupe, entity matching and recall.
A Profile keeps a model and its thresholds together:

    local   BAAI/bge-small-en-v1.5    384 dims, on the CPU via fastembed, no key   (default)
    remote  Qwen/Qwen3-Embedding-8B   4096 dims, any OpenAI-compatible /embeddings endpoint

Each model's vectors live in their own property and vector indexes ("embedding" + memory_vec
for the default, "embedding_qwen3_embedding_8b" + memory_vec_qwen3_embedding_8b for Qwen), so
one database can hold both: the tests run on the local model while your chats use the remote
one, and `python -m companion.reembed` backfills a model's vectors after a switch.

Queries vs passages: Qwen3-Embedding is trained to put an instruction before a QUERY ("what
does this message relate to?") and nothing before a stored text. Without it, its separation
of related from unrelated memories fell from 0.99 to 0.90 AUC (lessons/19). So `embed(...,
query=True)` is for a message searched against memories; comparisons between two stored
things (memory vs memory, entity vs entity) use plain embeddings on both sides.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class Profile:
    model: str
    dims: int
    # Thresholds. "score" = the vector index's 0-1 score, (1 + cosine) / 2; "cosine" = raw.
    duplicate: float          # score: a new memory this close to a current one reinforces it
    forgotten: float          # cosine: this close to a /forget tombstone is not stored again
    entity_match: float       # score: an entity this close (same kind) may be the same one...
    name_match: float         # cosine: ...if their NAMES are this close too ("Mom" ~ "my mother")
    pattern: float            # cosine: every pair in a habit cluster at least this close
    sensitive_floor: float    # score: a sensitive memory needs at least this to be recalled
    sensitive_margin: float   # score: ...and to be within this of the best match
    entity_hop: float         # score: an entity this close to the message is looked up
    query_instruction: str = ""    # put before a query (not before stored text)
    cluster_instruction: str = ""  # put before memories grouped into habits (Lesson 13)
    # score: below this a vector match is "loosely related" (Lesson 18). None: the model can't
    # tell (bge-small's scores overlap), so a cross-encoder decides instead.
    weak_below: float | None = None

    @property
    def slug(self) -> str:
        """'' for the default model (the property and index names from Lessons 8-18)."""
        if self.model == DEFAULT_MODEL:
            return ""
        return re.sub(r"[^a-z0-9]+", "_", self.model.split("/")[-1].lower()).strip("_")

    @property
    def prop(self) -> str:
        return f"embedding_{self.slug}" if self.slug else "embedding"

    def index(self, base: str) -> str:
        return f"{base}_{self.slug}" if self.slug else base


DEFAULT_MODEL = config.LOCAL_EMBED_MODEL

PROFILES = {p.model: p for p in [
    # Lessons 9-13 measured these (see the comments next to each constant in store.py).
    Profile("BAAI/bge-small-en-v1.5", dims=384, duplicate=0.975, forgotten=0.90, entity_match=0.93,
            name_match=0.80, pattern=0.82, sensitive_floor=0.72, sensitive_margin=0.01, entity_hop=0.85),
    # Lesson 19: measured with `python -m companion.calibrate` (results in the lesson).
    # Its scores spread wider than bge-small's: unrelated pairs sit lower (sensitive/hop "no"
    # 0.57-0.62 vs bge's 0.68-0.74), so the floors drop, and a message that only MEANS an
    # entity ("how is my kid doing?" -> Logan) now clears the hop (0.65-0.73 vs no <= 0.59).
    Profile("Qwen/Qwen3-Embedding-8B", dims=4096, duplicate=0.966, forgotten=0.89, entity_match=0.94,
            name_match=0.84, pattern=0.665, sensitive_floor=0.655, sensitive_margin=0.015, entity_hop=0.62,
            query_instruction="Instruct: Given a message from a user, retrieve memories about the "
                              "user's life that it relates to\nQuery: ",
            # without it, "heading to the gym" vs "went rock climbing" (0.77) outscored two
            # gym sentences (0.73): complete-linkage clustering couldn't find the habit
            cluster_instruction="Instruct: Identify the activity or habit described\nQuery: ",
            # one cut on the best match got 36 of 37 test messages right (raw cosine 0.312)
            weak_below=0.656),
]}


def profile_for(model: str) -> Profile:
    if model not in PROFILES:
        raise SystemExit(
            f"No thresholds measured for EMBED_MODEL={model!r}. Add a Profile for it in "
            "companion/embeddings.py, measured with `uv run python -m companion.calibrate` (Lesson 19).")
    return PROFILES[model]


PROFILE = profile_for(config.EMBED_MODEL)

_local = None
_client = None
_lock = threading.Lock()
_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()  # (instruction, text) -> unit vector
CACHE_SIZE = 2048  # names and summaries are embedded again and again during one extraction


def remote() -> bool:
    return bool(config.EMBED_BASE_URL)


def _compute(texts: list[str]) -> np.ndarray:
    global _local, _client
    with _lock:  # the background extractor and the chat can both arrive here first
        if not remote() and _local is None:
            from fastembed import TextEmbedding
            _local = TextEmbedding(PROFILE.model)
        if remote() and _client is None:
            from openai import OpenAI
            _client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY,
                             timeout=60, max_retries=3)
    if not remote():
        return np.array(list(_local.embed(texts)), dtype=np.float32)
    out = []
    for i in range(0, len(texts), 64):  # endpoints cap the batch size
        resp = _client.embeddings.create(model=PROFILE.model, input=texts[i:i + 64], encoding_format="float")
        out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    vecs = np.array(out, dtype=np.float32)
    if vecs.shape[1] != PROFILE.dims:
        raise RuntimeError(f"{PROFILE.model} returned {vecs.shape[1]} dims, expected {PROFILE.dims}")
    return vecs


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Unit-length vectors (so a dot product is the cosine). `query=True`: the text is a message
    searched against stored memories/entities, and gets the model's query instruction."""
    return _embed(texts, PROFILE.query_instruction if query else "")


def embed_for_clustering(texts: list[str]) -> list[list[float]]:
    """Vectors for grouping memories into habits. Not stored: consolidation embeds again
    when the model has a clustering instruction, else it can use the stored vectors."""
    return _embed(texts, PROFILE.cluster_instruction)


def _embed(texts: list[str], instruction: str) -> list[list[float]]:
    keys = [(instruction, t) for t in texts]
    found: dict[tuple[str, str], np.ndarray] = {}
    with _lock:
        for k in keys:
            if k in _cache:
                found[k] = _cache[k]
                _cache.move_to_end(k)
    missing = [k for k in dict.fromkeys(keys) if k not in found]
    if missing:
        vecs = _compute([i + t for i, t in missing])
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)  # an all-zero vector stays zero
        found.update(zip(missing, vecs))
        with _lock:
            _cache.update(zip(missing, vecs))
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
    return [found[k].tolist() for k in keys]
