"""Command-line entry point for ingestion and chat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chrysostom_lens.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chrysostom-lens",
        description="Hierarchical global-to-local context-stack RAG for Homilies on Matthew.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Parse PDF, synthesize context, and build FAISS index.")
    ingest.add_argument(
        "pdf_path",
        nargs="?",
        help="Path to the Homilies on Matthew PDF. Defaults to CHRYSOSTOM_PDF or data/Homilies on the Gospel of Matthew.pdf.",
    )
    ingest.add_argument(
        "--embedding-provider",
        choices=("serverless", "local"),
        default="serverless",
        help="Use Hugging Face serverless embeddings or local HuggingFaceEmbeddings.",
    )
    ingest.add_argument("--force-parse", action="store_true", help="Re-parse the PDF even if parsed cache exists.")
    ingest.add_argument("--force-synthesis", action="store_true", help="Re-run Gemini synthesis even if cache exists.")

    chat = subparsers.add_parser("chat", help="Run the terminal RAG chatbot against the local FAISS index.")
    chat.add_argument(
        "--embedding-provider",
        choices=("serverless", "local"),
        default="serverless",
        help="Provider used to embed the query; must match the indexed embeddings.",
    )

    subparsers.add_parser("status", help="Show local PDF, cache, and FAISS index status.")
    subparsers.add_parser("models", help="Show Gemini fallback models and models visible to your API key.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = Settings()

    if args.command == "ingest":
        from chrysostom_lens.pipeline import run_ingestion

        pdf_path = Path(args.pdf_path) if args.pdf_path else settings.pdf_path
        run_ingestion(
            pdf_path=str(pdf_path),
            settings=settings,
            embedding_provider=args.embedding_provider,
            force_parse=args.force_parse,
            force_synthesis=args.force_synthesis,
        )
        print(f"Done. Enriched cache: {settings.cache_path}; FAISS index: {settings.index_path}")
        return

    if args.command == "chat":
        from chrysostom_lens.chat import run_chat_loop

        run_chat_loop(settings=settings, embedding_provider=args.embedding_provider)
        return

    if args.command == "status":
        print_status(settings)
        return

    if args.command == "models":
        print_models(settings)
        return


def _json_count(path: Path) -> int | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else None


def _first_json_item(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def print_status(settings: Settings) -> None:
    parsed_count = _json_count(settings.parsed_cache_path)
    enriched_count = _json_count(settings.cache_path)
    index_files = sorted(p.name for p in settings.index_path.glob("*")) if settings.index_path.exists() else []

    print("ChrysostomLens status")
    print(f"PDF: {settings.pdf_path} ({'found' if settings.pdf_path.exists() else 'missing'})")
    print(f"Parsed cache: {settings.parsed_cache_path} ({parsed_count if parsed_count is not None else 'missing'} records)")
    print(f"Enriched cache: {settings.cache_path} ({enriched_count if enriched_count is not None else 'missing'} records)")
    print(f"FAISS index: {settings.index_path} ({', '.join(index_files) if index_files else 'missing'})")

    first_parsed = _first_json_item(settings.parsed_cache_path)
    if first_parsed:
        text = first_parsed.get("paragraph_text", "")
        print("\nFirst parsed paragraph preview:")
        print(f"Homily: {first_parsed.get('homily')}")
        print(text[:500] + ("..." if len(text) > 500 else ""))

    first_enriched = _first_json_item(settings.cache_path)
    if first_enriched:
        payload = first_enriched.get("stacked_payload", "")
        print("\nFirst enriched payload preview:")
        print(payload[:700] + ("..." if len(payload) > 700 else ""))


def print_models(settings: Settings) -> None:
    configured = []
    for model in (settings.gemini_model, *settings.gemini_fallback_models):
        if model not in configured:
            configured.append(model)

    print("Configured Gemini fallback order:")
    for index, model in enumerate(configured, start=1):
        print(f"{index}. {model}")

    if not settings.google_api_key:
        print("\nSet GOOGLE_API_KEY or GEMINI_API_KEY to list models visible to your account.")
        return

    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        visible = []
        for model in client.models.list():
            name = getattr(model, "name", "")
            actions = getattr(model, "supported_actions", []) or []
            if "generateContent" in actions or name.startswith("models/gemini"):
                visible.append(name.replace("models/", ""))

        print("\nGemini models visible to your API key:")
        for name in sorted(set(visible)):
            print(f"- {name}")
    except Exception as exc:
        print(f"\nCould not list Gemini models with this key: {exc}")


if __name__ == "__main__":
    main()
