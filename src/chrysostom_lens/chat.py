"""Step 6: Terminal RAG chatbot loop with Groq streaming."""

from __future__ import annotations

import random
import time

from groq import Groq

from chrysostom_lens.config import Settings
from chrysostom_lens.indexing import load_faiss_index


SYSTEM_PROMPT = """You are a careful theological research assistant.
Answer strictly from the supplied context drawn from St. John Chrysostom's Homilies on Matthew.
Use the global context, local context, and raw text together. If the context is insufficient, say so plainly.
Do not invent citations, patristic claims, historical facts, or cross-references that are not supported by the retrieved text.
"""


def _format_retrieved_context(docs) -> str:
    chunks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        homily = doc.metadata.get("homily", "Unknown homily")
        paragraph_index = doc.metadata.get("paragraph_index", "unknown")
        chunks.append(
            f"<context source=\"{idx}\" homily=\"{homily}\" paragraph_index=\"{paragraph_index}\">\n"
            f"{doc.page_content}\n"
            f"</context>"
        )
    return "\n\n".join(chunks)


def stream_groq_answer(client: Groq, settings: Settings, query: str, context: str) -> None:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Retrieved context:\n{context}\n\nQuestion: {query}",
        },
    ]

    for attempt in range(settings.max_retries):
        try:
            stream = client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    print(delta, end="", flush=True)
            print()
            return
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}".lower()
            retryable = "429" in text or "rate limit" in text or "too many requests" in text or "timeout" in text
            if not retryable or attempt == settings.max_retries - 1:
                raise
            delay = min(90.0, settings.request_cooldown_seconds * (2**attempt)) + random.uniform(0.0, 1.5)
            print(f"\nRate limited; retrying in {delay:.1f}s...", flush=True)
            time.sleep(delay)


def run_chat_loop(settings: Settings, embedding_provider: str = "serverless") -> None:
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for the interactive chatbot.")

    vector_store = load_faiss_index(settings, embedding_provider=embedding_provider)
    groq_client = Groq(api_key=settings.groq_api_key)

    print("ChrysostomLens RAG chat. Type 'exit', 'quit', or press Ctrl-C to leave.")
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", ":q"}:
            print("Goodbye.")
            break

        docs = vector_store.similarity_search(query, k=3)
        if not docs:
            print("No relevant context found in the local FAISS index.")
            continue

        context = _format_retrieved_context(docs)
        print("\nAnswer> ", end="", flush=True)
        stream_groq_answer(groq_client, settings, query, context)

if __name__ == "__main__":
    settings = Settings()
    run_chat_loop(settings, embedding_provider="serverless")