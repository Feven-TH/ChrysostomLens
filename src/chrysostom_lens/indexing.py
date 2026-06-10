"""Step 5: Dense vector indexing with LangChain and FAISS."""

from __future__ import annotations

from pathlib import Path
import random
import time
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

from chrysostom_lens.config import Settings
from chrysostom_lens.models import EnrichedParagraph
from chrysostom_lens.synthesis import load_enriched_payloads


class RetryingEmbeddings(Embeddings):
    """Small retry wrapper for transient Hugging Face serverless throttling."""

    def __init__(self, inner: Embeddings, max_retries: int, base_delay: float) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._with_retry(self.inner.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        return self._with_retry(self.inner.embed_query, text)

    def _with_retry(self, fn, *args: Any):
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args)
            except Exception as exc:
                last_error = exc
                text = f"{type(exc).__name__}: {exc}".lower()
                retryable = (
                    "429" in text
                    or "rate limit" in text
                    or "too many requests" in text
                    or "temporarily unavailable" in text
                    or "timeout" in text
                )
                if not retryable or attempt == self.max_retries - 1:
                    raise
                delay = min(90.0, self.base_delay * (2**attempt)) + random.uniform(0.0, 1.5)
                time.sleep(delay)
        raise RuntimeError("Embedding request failed after retries.") from last_error


def create_documents(payloads: list[EnrichedParagraph]) -> list[Document]:
    return [
        Document(
            page_content=item.stacked_payload,
            metadata={
                "homily": item.homily,
                "paragraph_index": item.paragraph_index,
                "batch_start_paragraph": item.batch_start_paragraph,
            },
        )
        for item in payloads
    ]


def initialize_embeddings(settings: Settings, provider: str = "serverless"):
    """Initialize BAAI/bge-large-en-v1.5 embeddings.

    ``serverless`` uses Hugging Face Inference API free tier. ``local`` uses the
    class named in the requirements and downloads the model to the machine.
    """

    if provider == "serverless":
        if not settings.huggingface_token:
            raise RuntimeError(
                "HUGGINGFACEHUB_API_TOKEN or HF_TOKEN is required for Hugging Face serverless embeddings."
            )
        endpoint_embeddings = HuggingFaceEndpointEmbeddings(
            model=settings.embedding_model,
            task="feature-extraction",
            huggingfacehub_api_token=settings.huggingface_token,
        )
        return RetryingEmbeddings(
            endpoint_embeddings,
            max_retries=settings.max_retries,
            base_delay=settings.request_cooldown_seconds,
        )
    if provider == "local":
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    raise ValueError("Embedding provider must be 'serverless' or 'local'.")


def build_faiss_index(
    settings: Settings,
    embedding_provider: str = "serverless",
    cache_path: Path | None = None,
) -> FAISS:
    payloads = load_enriched_payloads(cache_path or settings.cache_path)
    documents = create_documents(payloads)
    if not documents:
        raise RuntimeError("No enriched payloads found for indexing.")
    embeddings = initialize_embeddings(settings, provider=embedding_provider)

    if embedding_provider == "serverless":
        print(f"\n[Cloud Ingestion] Total documents to index: {len(documents)}")
        print("Processing in batches of 32 to prevent Hugging Face 504 Timeouts...\n", flush=True)
        
        chunk_size = 32
        
        # Initialize the FAISS index container with the very first batch of 32
        first_batch = documents[:chunk_size]
        print(f"Initializing index container with docs 0 to {len(first_batch)}...", flush=True)
        index = FAISS.from_documents(first_batch, embeddings)
        
        for i in range(chunk_size, len(documents), chunk_size):
            batch = documents[i : i + chunk_size]
            print(f"→ Pushing cloud batch: items {i} to {i + len(batch)}...", flush=True)
            
            index.add_documents(batch)
            
            time.sleep(0.5)
            
    else:
        print("\n[Local Ingestion] Processing bulk array directly on CPU...", flush=True)
        index = FAISS.from_documents(documents, embeddings)

    print(f"\n[Success] Vector matrices compiled. Saving index to: {settings.index_path}")
    index.save_local(str(settings.index_path))
    return index


def load_faiss_index(settings: Settings, embedding_provider: str = "serverless") -> FAISS:
    if not settings.index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {settings.index_path}")
    embeddings = initialize_embeddings(settings, provider=embedding_provider)
    return FAISS.load_local(
        str(settings.index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
