"""Step 6: Terminal RAG chatbot loop with Groq streaming and Gemini fallback."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, AsyncGenerator, Generator
from groq import AsyncGroq, Groq
from chrysostom_lens.config import Settings
from chrysostom_lens.indexing import load_faiss_index


SYSTEM_PROMPT = """You are ChrysostomLens, a careful theological research assistant specialising \
in St. John Chrysostom's Homilies on the Gospel of Matthew.

Your task is to answer the user's question using ONLY the retrieved context passages provided \
below. Each passage is tagged with its homily number and paragraph index.

Guidelines:
- Draw on the global context, local context, and raw translation together.
- Quote or paraphrase directly from the retrieved text when relevant.
- Always cite the source of your information inline using the format [H<homily_number> §<paragraph_index>] \
matching the homily and paragraph_index from the source tags (e.g., [H1 §12]). Make sure to use this \
exact format so the user can click the citation to inspect the original translation.
- If the retrieved passages do not contain enough information, say so plainly and invite the \
user to narrow or rephrase the question.
- Never invent citations, patristic claims, historical facts, or cross-references that are \
not supported by the retrieved text.
- Write with humility and clarity, reflecting the wisdom and sobriety of St. John Chrysostom.
- Refer to the saint as "St. John Chrysostom" or "the Golden-Mouthed" on first mention, then \
"he" or "Chrysostom" thereafter.
- Use plain, accessible language suitable for a general reader with no prior theological training.
"""


def format_retrieved_context(docs: list[Any]) -> str:
    """Format retrieved document chunks as XML-tagged context blocks."""
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


def retrieve_context(vector_store: Any, query: str, k: int = 3) -> tuple[list[Any], str]:
    """Retrieve top-k relevant documents and format them as context."""
    docs = vector_store.similarity_search(query, k=k)
    return docs, format_retrieved_context(docs)


def _get_gemini_models(settings: Settings) -> list[str]:
    """Get the unique list of Gemini models configured."""
    models: list[str] = []
    for model in (settings.gemini_model, *settings.gemini_fallback_models):
        if model and model not in models:
            models.append(model)
    return models


def prepare_gemini_inputs(messages: list[dict[str, str]]) -> tuple[str | None, list[Any]]:
    """Format messages array to system instructions + contents for Gemini client."""
    from google.genai import types
    
    system_instruction: str | None = None
    contents: list[Any] = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            system_instruction = content
        elif role == "user":
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)]
                )
            )
        elif role in ("assistant", "model"):
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=content)]
                )
            )
    return system_instruction, contents


def stream_chat_response(
    client: Groq | None,
    settings: Settings,
    messages: list[dict],
) -> Generator[str, None, None]:
    """Synchronously stream an answer from Groq or fall back to Gemini."""
    tried_groq = False
    
    if client is not None and settings.groq_api_key:
        tried_groq = True
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
                        yield delta
                return  
            except Exception as exc:
                print(f"Groq sync attempt {attempt + 1} failed: {exc}", flush=True)
                text = f"{type(exc).__name__}: {exc}".lower()
                retryable = (
                    "429" in text
                    or "rate limit" in text
                    or "too many requests" in text
                    or "timeout" in text
                )
                if not retryable or attempt == settings.max_retries - 1:
                    break
                delay = (
                    min(90.0, settings.request_cooldown_seconds * (2**attempt))
                    + random.uniform(0.0, 1.5)
                )
                print(f"\nRate limited on Groq; retrying in {delay:.1f}s...", flush=True)
                time.sleep(delay)

    # Fallback to Gemini if Groq fails
    if settings.google_api_key:
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=settings.google_api_key)
        system_instruction, contents = prepare_gemini_inputs(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )

        models = _get_gemini_models(settings)
        last_error = None
        for model in models:
            try:
                print(f"Falling back to Gemini model: {model}", flush=True)
                response_stream = gemini_client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return 
            except Exception as exc:
                print(f"Gemini model {model} stream failed: {exc}", flush=True)
                last_error = exc
        if last_error:
            raise last_error
    else:
        if tried_groq:
            raise RuntimeError("Groq failed and no GOOGLE_API_KEY/GEMINI_API_KEY is configured for fallback.")
        else:
            raise RuntimeError("Neither Groq nor Gemini API keys are configured.")


async def stream_chat_response_async(
    client: AsyncGroq | None,
    settings: Settings,
    messages: list[dict],
) -> AsyncGenerator[str, None]:
    """Asynchronously stream an answer from Groq or fall back to Gemini."""
    tried_groq = False
    
    if client is not None and settings.groq_api_key:
        tried_groq = True
        for attempt in range(settings.max_retries):
            try:
                stream = await client.chat.completions.create(
                    model=settings.groq_model,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  
            except Exception as exc:
                print(f"Groq async attempt {attempt + 1} failed: {exc}", flush=True)
                text = str(exc).lower()
                retryable = (
                    "429" in text
                    or "rate limit" in text
                    or "too many requests" in text
                    or "timeout" in text
                )
                if not retryable or attempt == settings.max_retries - 1:
                    break
                delay = min(5.0, settings.request_cooldown_seconds * (2**attempt))
                await asyncio.sleep(delay)

    if settings.google_api_key:
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=settings.google_api_key)
        system_instruction, contents = prepare_gemini_inputs(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        )

        models = _get_gemini_models(settings)
        last_error = None
        for model in models:
            try:
                print(f"Falling back to Gemini model: {model}", flush=True)
                response_stream = await gemini_client.aio.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=config,
                )
                async for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text
                return  
            except Exception as exc:
                print(f"Gemini model {model} stream failed: {exc}", flush=True)
                last_error = exc
        if last_error:
            raise last_error
    else:
        if tried_groq:
            raise RuntimeError("Groq failed and no GOOGLE_API_KEY/GEMINI_API_KEY is configured for fallback.")
        else:
            raise RuntimeError("Neither Groq nor Gemini API keys are configured.")


def stream_groq_answer(
    client: Groq | None,
    settings: Settings,
    messages: list[dict],
) -> None:
    """Consume the stream generator and print chunks in real-time."""
    for delta in stream_chat_response(client, settings, messages):
        print(delta, end="", flush=True)
    print()


def run_chat_loop(settings: Settings, embedding_provider: str = "serverless") -> None:
    if not settings.groq_api_key and not settings.google_api_key:
        raise RuntimeError("Either GROQ_API_KEY or GOOGLE_API_KEY/GEMINI_API_KEY is required for the interactive chatbot.")

    vector_store = load_faiss_index(settings, embedding_provider=embedding_provider)
    groq_client = None
    if settings.groq_api_key:
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

        docs, context = retrieve_context(vector_store, query, k=3)
        if not docs:
            print("No relevant context found in the local FAISS index.")
            continue

        # Build the full messages array for the CLI (no history in CLI mode)
        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            "Retrieved context from St. John Chrysostom's Homilies on Matthew:\n"
            f"{context}"
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        print("\nAnswer> ", end="", flush=True)
        stream_groq_answer(groq_client, settings, messages)


if __name__ == "__main__":
    settings = Settings()
    run_chat_loop(settings, embedding_provider="serverless")