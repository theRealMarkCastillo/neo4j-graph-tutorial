"""Shared settings for the lesson scripts. Change them here.

Every setting lives in this file: endpoints, model names, temperature, names. Only SECRETS come
from .env (git-ignored, see .env.example): the Neo4j password and the API keys.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _secret(name: str) -> str:
    return os.getenv(name, "").strip()


# --- Neo4j ---------------------------------------------------------------------------------
NEO4J_URI = "neo4j://localhost:7687"
# The user and password also create the database in docker compose, so they stay in .env.
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

# --- Local models (no API key, run on the CPU) ---------------------------------------------
# Embeddings for Lessons 5-7, and for the companion when EMBED_BASE_URL is None.
LOCAL_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LOCAL_EMBED_DIMS = 384
# Lesson 18: a cross-encoder that judges whether a recalled memory is about the message at all
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

# --- Chat LLM: one OpenAI-compatible endpoint for every lesson that calls an LLM (5, 7-20) ---
# Works with Gemini, OpenAI, DeepInfra, OpenRouter, Ollama (http://localhost:11434/v1),
# LM Studio (http://localhost:1234/v1), vLLM... For a local server set LLM_BASE_URL to it;
# no key needed.
LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_MODEL = "gemini-3.8-flash"
LLM_EXTRACTION_MODEL = "gemini-3.8-flash"  # background memory extraction (Lessons 9-20)
LLM_TEMPERATURE = 0.7                       # chat replies; extraction always runs at 0
LLM_API_KEY = _secret("LLM_API_KEY") or "not-needed"  # local servers ignore it

# Lesson 7's tool-calling agent. On Gemini it uses gemini-2.5-flash: Gemini 3 models reject the
# agent's second step over the OpenAI-compatible endpoint, because each tool call carries a
# thought_signature that must be sent back and ChatOpenAI drops it. Elsewhere: LLM_MODEL.
GEMINI_AGENT_MODEL = "gemini-2.5-flash"
LLM_AGENT_MODEL = GEMINI_AGENT_MODEL if "generativelanguage.googleapis.com" in LLM_BASE_URL else LLM_MODEL

# --- Companion embeddings (Lesson 19) ------------------------------------------------------
# A remote OpenAI-compatible /embeddings endpoint, or None for the local model above (offline,
# no key; Lessons 8-18's recorded numbers come from it). Each model needs measured thresholds:
# see companion/embeddings.py. After switching, run: uv run python -m companion.reembed
EMBED_BASE_URL = "https://api.deepinfra.com/v1"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B" if EMBED_BASE_URL else LOCAL_EMBED_MODEL
EMBED_API_KEY = _secret("EMBED_API_KEY") or LLM_API_KEY

# --- Companion -----------------------------------------------------------------------------
COMPANION_USER = "Alex"  # the person the companion talks to (one :User node per name)
COMPANION_NAME = "Sage"

# --- Optional: MLflow traces and eval history (observability.py) ---------------------------
# None: tracing off. "http://localhost:5001" with `uv sync --group observability` and a running
# `mlflow server` (README, "Tests and evals"): every chat turn and eval run is traced there.
MLFLOW_TRACKING_URI = None
MLFLOW_EXPERIMENT = "companion"


def _local(url: str | None) -> bool:
    return bool(url) and any(h in url for h in ("localhost", "127.0.0.1", "[::1]"))


def llm_configured() -> bool:
    """A model and either a key or a local server: what the LLM lessons and evals need."""
    return bool(LLM_MODEL) and (LLM_API_KEY != "not-needed" or _local(LLM_BASE_URL))


def llm_client():
    """The plain OpenAI SDK client for the endpoint above (Lesson 5).

    `chat_model()` below wraps this same SDK for LangChain/LangGraph (Lessons 7-12)."""
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120, max_retries=2)


def chat_model(model: str | None = None, temperature: float | None = None):
    """A LangChain chat model (built on the OpenAI SDK) pointed at the endpoint above."""
    from langchain_openai import ChatOpenAI

    if not llm_configured():
        raise SystemExit("Set LLM_API_KEY in .env (or point LLM_BASE_URL in config.py at a local server).")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=model or LLM_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        timeout=120,
        max_retries=2,
    )


# Optional MLflow tracing (observability.py): on only when MLFLOW_TRACKING_URI is set above.
import observability  # noqa: E402

observability.setup(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)
