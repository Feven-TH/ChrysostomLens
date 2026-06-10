"""Unified ingestion pipeline for ChrysostomLens."""

from __future__ import annotations

from pathlib import Path

from chrysostom_lens.config import Settings
from chrysostom_lens.indexing import build_faiss_index
from chrysostom_lens.parsing import batch_paragraphs, load_parsed_cache, parse_pdf_to_paragraphs, save_parsed_cache
from chrysostom_lens.synthesis import build_enriched_payloads


def run_ingestion(
    pdf_path: str,
    settings: Settings,
    embedding_provider: str = "serverless",
    force_parse: bool = False,
    force_synthesis: bool = False,
) -> None:
    """Run all six architecture steps through FAISS index creation."""

    parsed_path = Path(settings.parsed_cache_path)
    if parsed_path.exists() and not force_parse:
        paragraphs = load_parsed_cache(parsed_path)
    else:
        paragraphs = parse_pdf_to_paragraphs(pdf_path)
        save_parsed_cache(paragraphs, parsed_path)

    if not paragraphs:
        raise RuntimeError("PDF parsing produced no paragraphs.")

    batches = batch_paragraphs(paragraphs, batch_size=8)
    build_enriched_payloads(batches, settings, force=force_synthesis)
    build_faiss_index(settings, embedding_provider=embedding_provider)
