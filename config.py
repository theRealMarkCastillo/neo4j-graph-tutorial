"""Shared settings for the lesson scripts, loaded from .env (see .env.example).

This is the ONE place that reads settings: every other module imports them from here, never
from os.environ. .env holds the secrets and every external dependency (endpoints, model names):
nothing here assumes a provider or a model. The defaults below only fill in what .env leaves out.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


# --- Neo4j ---------------------------------------------------------------------------------
NEO4J_URI = _env("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_AUTH = (_env("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

# --- Local models (no API key, run on the CPU) ---------------------------------------------
# Embeddings for Lessons 5-7, and for the companion when EMBED_BASE_URL is empty. Another model
# needs measured thresholds in companion/embeddings.py (Lesson 19) and its own Lesson 5 index.
LOCAL_EMBED_MODEL = _env("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# Lesson 18: a cross-encoder that judges whether a recalled memory is about the message at all
RERANK_MODEL = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# --- Chat LLM: one OpenAI-compatible endpoint for every lesson that calls an LLM (5, 7-20) ---
# Gemini, OpenAI, DeepInfra, OpenRouter, Ollama, LM Studio, vLLM...
LLM_BASE_URL = _env("LLM_BASE_URL") or None
LLM_MODEL = _env("LLM_MODEL")
LLM_API_KEY = _env("LLM_API_KEY", "not-needed")  # local servers ignore it
LLM_EXTRACTION_MODEL = _env("LLM_EXTRACTION_MODEL", LLM_MODEL)
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.7"))
LLM_AGENT_MODEL = _env("LLM_AGENT_MODEL", LLM_MODEL)  # Lesson 7's agent: must support tool calling

# --- Companion embeddings (Lesson 19) ------------------------------------------------------
# Leave EMBED_BASE_URL empty for the local model above; set it to an OpenAI-compatible
# /embeddings endpoint to use a remote model. Each model needs measured thresholds: see
# companion/embeddings.py.
EMBED_BASE_URL = _env("EMBED_BASE_URL") or None
EMBED_MODEL = (_env("EMBED_MODEL") if EMBED_BASE_URL else "") or LOCAL_EMBED_MODEL
EMBED_API_KEY = _env("EMBED_API_KEY", LLM_API_KEY)  # same provider: same key

# --- Companion -----------------------------------------------------------------------------
COMPANION_USER = _env("COMPANION_USER", "Alex")
COMPANION_NAME = _env("COMPANION_NAME", "Sage")
COMPANION_DEBUG_EXTRACT = bool(_env("COMPANION_DEBUG_EXTRACT"))  # print each raw LLM extraction

# --- Optional: MLflow traces and eval history (observability.py) ---------------------------
# Empty: tracing off. http://localhost:5001 with `uv sync --group observability` and a running
# `mlflow server` (README, "Tests and evals").
MLFLOW_TRACKING_URI = _env("MLFLOW_TRACKING_URI") or None
MLFLOW_EXPERIMENT = _env("MLFLOW_EXPERIMENT_NAME", "companion")

# --- Tests (tests/conftest.py) -------------------------------------------------------------
EVAL_TRIALS = int(_env("EVAL_TRIALS", "1"))  # run each LLM eval case this many times
TEST_REMOTE_EMBEDDINGS = bool(_env("TEST_REMOTE_EMBEDDINGS"))  # tests on EMBED_MODEL, not local


def llm_configured() -> bool:
    return bool(LLM_MODEL)


def llm_client():
    """The plain OpenAI SDK client for the endpoint in .env (Lesson 5).

    `chat_model()` below wraps this same SDK for LangChain/LangGraph (Lessons 7-12)."""
    from openai import OpenAI

    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=120, max_retries=2)


def chat_model(model: str | None = None, temperature: float | None = None):
    """A LangChain chat model (built on the OpenAI SDK) pointed at the endpoint in .env."""
    from langchain_openai import ChatOpenAI

    if not (model or LLM_MODEL):
        raise SystemExit("Set LLM_MODEL (and LLM_BASE_URL / LLM_API_KEY) in .env first.")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=model or LLM_MODEL,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        timeout=120,
        max_retries=2,
    )


# Optional MLflow tracing (observability.py): on only when MLFLOW_TRACKING_URI is set.
import observability  # noqa: E402

observability.setup(MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT)
