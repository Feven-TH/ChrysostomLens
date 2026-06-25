"""FastAPI backend server for the ChrysostomLens RAG chatbot."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator
from time import perf_counter as system_perf_counter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from pydantic import BaseModel

from chrysostom_lens.chat import (
    SYSTEM_PROMPT,
    retrieve_context,
    stream_chat_response_async,
)
from chrysostom_lens.config import Settings
from chrysostom_lens.indexing import load_faiss_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chrysostom-server")

settings = Settings()

app = FastAPI(
    title="ChrysostomLens API",
    description="Backend API for querying St. John Chrysostom's Homilies on Matthew.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chrysostom-lens.vercel.app", 
        "http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of FAISS index to allow startup even if index isn't ready
vector_store = None
loading_error = None


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: str       
    content: str


class QueryRequest(BaseModel):
    """Incoming POST body for /api/query."""
    query: str
    history: list[ChatMessage] = []


def get_vector_store():
    global vector_store, loading_error
    if vector_store is not None:
        return vector_store
    if loading_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"FAISS index failed to load: {loading_error}. Please ingest the Homilies first.",
        )
    try:
        provider = "serverless"
        logger.info(f"Loading FAISS index with provider={provider}...")
        vector_store = load_faiss_index(settings, embedding_provider=provider)
        logger.info("FAISS index loaded successfully.")
        return vector_store
    except Exception as e:
        loading_error = str(e)
        logger.error(f"Error loading FAISS index: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"FAISS index not found or failed to load: {e}. Please ingest the Homilies first.",
        )


@app.api_route("/api/status" , methods=["GET", "HEAD"])
def get_status() -> dict[str, Any]:
    """Check system and database health/status."""
    global vector_store
    index_exists = settings.index_path.exists()
    parsed_exists = settings.parsed_cache_path.exists()
    enriched_exists = settings.cache_path.exists()

    status_info = {
        "pdf_found": settings.pdf_path.exists(),
        "parsed_cache_found": parsed_exists,
        "enriched_cache_found": enriched_exists,
        "index_found": index_exists,
        "index_path": str(settings.index_path),
        "groq_model": settings.groq_model,
        "embedding_model": settings.embedding_model,
        "groq_key_configured": bool(settings.groq_api_key),
        "google_key_configured": bool(settings.google_api_key),
    }

    if index_exists:
        try:
            if vector_store is None:
                get_vector_store()
            status_info["status"] = "ready"
            status_info["message"] = "Library is loaded and ready for study."
        except Exception as e:
            status_info["status"] = "error"
            status_info["message"] = f"Library exists but failed to load: {e}"
    else:
        status_info["status"] = "missing_index"
        status_info["message"] = "Library is missing. Please run the ingestion script."

    return status_info


async def stream_rag_response(
    query: str,
    history: list[ChatMessage],
) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events with sources then streamed answer with robust Gemini fallback processing."""
    try:
        store = get_vector_store()

        loop = asyncio.get_running_loop()
        logger.info(f"Querying FAISS for: '{query}'")
        
        start_retrieval = system_perf_counter()
        docs, context = await loop.run_in_executor(
            None, lambda: retrieve_context(store, query, k=3)
        )
        retrieval_duration = system_perf_counter() - start_retrieval

        if not docs:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant passages found in the library.'})}\n\n"
            yield 'data: {"type": "done"}\n\n'
            return

        sources = []
        for idx, doc in enumerate(docs, start=1):
            sources.append({
                "id": idx,
                "homily": doc.metadata.get("homily", "Unknown homily"),
                "paragraph_index": doc.metadata.get("paragraph_index", "unknown"),
                "batch_start_paragraph": doc.metadata.get("batch_start_paragraph", "unknown"),
                "content": doc.page_content,
            })

        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        await asyncio.sleep(0.05) 

        system_content = (
            f"{SYSTEM_PROMPT}\n\n"
            "--- START CONTEXT ---\n"
            f"{context}\n"
            "--- END CONTEXT ---"
        )

        stream_successful = False

        if settings.groq_api_key:
            messages: list[dict] = [{"role": "system", "content": system_content}]
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": query})

            # 20-second timeout allows quick pivot to Gemini without leaving the user hanging
            groq_client = AsyncGroq(api_key=settings.groq_api_key, timeout=20.0)
            logger.info(f"Initiating primary Groq stream with model: {settings.groq_model}")
            
            start_llm = system_perf_counter()
            try:
                async for delta in stream_chat_response_async(groq_client, settings, messages):
                    yield f"data: {json.dumps({'type': 'content', 'delta': delta})}\n\n"
                
                llm_duration = system_perf_counter() - start_llm
                logger.info(f"PERF [Groq]: Retrieval: {retrieval_duration:.3f}s | LLM Generation: {llm_duration:.3f}s")
                stream_successful = True
            except Exception as groq_exc:
                logger.error(f"Primary Groq stream failed or timed out: {groq_exc}. Initiating Gemini recovery fallback...")
               
        if not stream_successful:
            if settings.google_api_key:
                from google import genai
                from google.genai import types

                gemini_client = genai.Client(api_key=settings.google_api_key)
                
                fallback_models = settings.gemini_fallback_models
                if not fallback_models:
                    fallback_models = ("gemini-2.5-flash-lite", "gemini-2.5-flash")

                gemini_contents = []
                for msg in history:
                    role = "user" if msg.role == "user" else "model"
                    gemini_contents.append(
                        types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
                    )
                gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

                start_llm = system_perf_counter()
                
                for model_name in fallback_models:
                    logger.info(f"Attempting Gemini fallback stream profile: {model_name}")
                    try:
                        config_spec = types.GenerateContentConfig(
                            system_instruction=system_content,
                            temperature=0.3
                        )
                        
                        response_stream = await gemini_client.aio.models.generate_content_stream(
                            model=model_name,
                            contents=gemini_contents,
                            config=config_spec
                        )

                        async for chunk in response_stream:
                            if chunk.text:
                                yield f"data: {json.dumps({'type': 'content', 'delta': chunk.text})}\n\n"
                        
                        llm_duration = system_perf_counter() - start_llm
                        logger.info(f"PERF [Gemini - {model_name}]: Retrieval: {retrieval_duration:.3f}s | LLM Generation: {llm_duration:.3f}s")
                        stream_successful = True
                        break 
                    except Exception as gemini_exc:
                        logger.error(f"Gemini model variant '{model_name}' stream failed: {gemini_exc}")
                        continue
            else:
                logger.error("Groq failed and settings.google_api_key is not configured for fallback.")

      
        if stream_successful:
            yield 'data: {"type": "done"}\n\n'
        else:
            friendly_message = (
                "We're experiencing a temporary inconvenience and couldn't generate a summary, "
                "but your matching excerpts have loaded successfully below. Please try again in a few moments."
            )
            yield f"data: {json.dumps({'type': 'error', 'message': friendly_message})}\n\n"
            yield 'data: {"type": "done"}\n\n'

    except Exception as e:
        logger.exception("Global failure inside stream_rag_response pipeline workflow")
        yield f"data: {json.dumps({'type': 'error', 'message': f'Server Error: {str(e)}'})}\n\n"
        yield 'data: {"type": "done"}\n\n'

@app.post("/api/query")
async def query_rag(request: QueryRequest) -> StreamingResponse:
    """Accept a query + conversation history, stream back SSE chunks."""
    return StreamingResponse(
        stream_rag_response(request.query, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chrysostom_lens.server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,   # single worker — prevents index duplication in memory
    )
