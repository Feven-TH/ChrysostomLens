"""Runtime configuration for the ChrysostomLens pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


DEFAULT_GEMINI_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
)


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    huggingface_token: str | None = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_fallback_models: tuple[str, ...] = _csv_env("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_MODELS)
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    pdf_path: Path = Path(
        os.getenv("CHRYSOSTOM_PDF", "data/Homilies on the Gospel of Matthew.pdf")
    )
    cache_path: Path = Path(os.getenv("CHRYSOSTOM_CACHE", "data/enriched_homilies.json"))
    parsed_cache_path: Path = Path(os.getenv("CHRYSOSTOM_PARSED_CACHE", "data/parsed_paragraphs.json"))
    index_path: Path = Path(os.getenv("CHRYSOSTOM_INDEX", "faiss_homilies_index"))
    request_cooldown_seconds: float = float(os.getenv("GEMINI_COOLDOWN_SECONDS", "4"))
    max_retries: int = int(os.getenv("MAX_LLM_RETRIES", "6"))
    max_prompt_tokens: int = int(os.getenv("MAX_BATCH_PROMPT_TOKENS", "28000"))
