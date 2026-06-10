"""Step 3 and Step 4: Gemini synthesis and stacked payload construction."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import tiktoken
from google import genai
from google.genai import types

from chrysostom_lens.config import Settings
from chrysostom_lens.models import BatchSynthesis, EnrichedParagraph, ParagraphBatch, ParagraphNote
import httpx

def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "rate limit" in text or "resource_exhausted" in text


def _is_retryable_gemini_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "resource_exhausted",
        "unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "deadline",
        "internal",
        "overloaded",
    )
    return any(marker in text for marker in retry_markers)


def _is_model_unavailable_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    unavailable_markers = (
        "404",
        "not found",
        "not supported",
        "permission",
        "does not exist",
        "not available",
    )
    return any(marker in text for marker in unavailable_markers)


def _gemini_models(settings: Settings) -> list[str]:
    models: list[str] = []
    for model in (settings.gemini_model, *settings.gemini_fallback_models):
        if model and model not in models:
            models.append(model)
    return models


def _sleep_for_retry(attempt: int, base_seconds: float) -> None:
    delay = min(90.0, base_seconds * (2**attempt)) + random.uniform(0.0, 1.5)
    print(f"Gemini request retrying in {delay:.1f}s...", flush=True)
    time.sleep(delay)


def _count_tokens(text: str) -> int:
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def format_batch_prompt(batch: ParagraphBatch) -> str:
    tagged_paragraphs = "\n\n".join(
        f'<p index="{idx}">{paragraph.paragraph_text}</p>'
        for idx, paragraph in enumerate(batch.paragraphs)
    )
    return f"""You are building a retrieval context stack for St. John Chrysostom's Homilies on Matthew.

For the supplied consecutive paragraphs from {batch.homily}, produce:
1. A two-sentence macro_summary naming the shared theological theme and scriptural or rhetorical scope.
2. One paragraph note per supplied paragraph. Each micro_context must be one or two sentences explaining the core situational, argumentative, or metaphorical insight that would help later retrieval.

Return only JSON conforming to the response schema. Preserve the exact integer indexes from the XML-like tags.

{tagged_paragraphs}
"""


def synthesize_batch(
    client: genai.Client,
    batch: ParagraphBatch,
    settings: Settings,
    current_model_index: int,  
) -> tuple[BatchSynthesis, int]:  
    prompt = format_batch_prompt(batch)
    token_count = _count_tokens(prompt)
    if token_count > settings.max_prompt_tokens:
        raise ValueError(
            f"Batch starting at paragraph {batch.start_paragraph} has {token_count} prompt tokens; "
            f"limit is {settings.max_prompt_tokens}."
        )

    last_error: Exception | None = None
    models = _gemini_models(settings)
    num_models = len(models)
    max_model_retries = min(3, settings.max_retries)

    for model_try in range(num_models):
        active_index = (current_model_index + model_try) % num_models
        model = models[active_index]

        for attempt in range(max_model_retries):
            try:
                print(
                    f"Synthesizing batch at paragraph {batch.start_paragraph} with {model} "
                    f"(Model pool position {active_index + 1}/{num_models}, attempt {attempt + 1}/{max_model_retries})",
                    flush=True,
                )
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=BatchSynthesis,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, BatchSynthesis):
                    return _normalize_synthesis(parsed, len(batch.paragraphs)), active_index

                response_text = getattr(response, "text", None)
                if not response_text:
                    raise ValueError("Gemini returned no parseable text.")
                return _normalize_synthesis(BatchSynthesis.model_validate_json(response_text), len(batch.paragraphs)), active_index
                
            except Exception as exc:
                last_error = exc
                if _is_model_unavailable_error(exc):
                    print(f"Gemini model {model} is unavailable; shifting to next fallback.", flush=True)
                    break 
                if attempt < max_model_retries - 1 and _is_retryable_gemini_error(exc):
                    _sleep_for_retry(attempt, settings.request_cooldown_seconds)
                    continue
                if attempt < max_model_retries - 1 and isinstance(exc, (TimeoutError, ConnectionError, httpx.HTTPError)):
                    _sleep_for_retry(attempt, settings.request_cooldown_seconds)
                    continue
                
                if _is_retryable_gemini_error(exc):
                    print(f"Gemini model {model} failed after {max_model_retries} retries; shifting to next fallback.", flush=True)
                    break
                raise

    model_list = ", ".join(models)
    raise RuntimeError(f"Gemini synthesis failed for all configured models: {model_list}") from last_error

def _normalize_synthesis(synthesis: BatchSynthesis, paragraph_count: int) -> BatchSynthesis:
    notes_by_index: dict[int, ParagraphNote] = {}
    for note in synthesis.paragraph_notes:
        if 0 <= note.paragraph_index < paragraph_count and note.paragraph_index not in notes_by_index:
            notes_by_index[note.paragraph_index] = note

    missing = [idx for idx in range(paragraph_count) if idx not in notes_by_index]
    if missing:
        raise ValueError(f"Gemini response omitted paragraph note indexes: {missing}")

    return BatchSynthesis(
        macro_summary=synthesis.macro_summary.strip(),
        paragraph_notes=[notes_by_index[idx] for idx in range(paragraph_count)],
    )


def build_enriched_payloads(
    batches: list[ParagraphBatch],
    settings: Settings,
    force: bool = False,
) -> list[EnrichedParagraph]:
    """Run single-pass synthesis and cache stacked paragraph payloads."""

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for synthesis.")

    enriched: list[EnrichedParagraph] = []
    processed_paragraphs: set[int] = set()

    if settings.cache_path.exists() and not force:
        try:
            existing_data = json.loads(settings.cache_path.read_text(encoding="utf-8"))
            enriched = [EnrichedParagraph.model_validate(item) for item in existing_data]
            processed_paragraphs = {item.paragraph_index for item in enriched}
            print(f"Loaded {len(enriched)} previously enriched paragraphs from cache.", flush=True)
        except Exception:
            print("Cache file corrupted or empty; starting fresh.", flush=True)
            enriched = []

    client = genai.Client(api_key=settings.google_api_key)
    model_tracker_idx = 0

    for batch_number, batch in enumerate(batches, start=1):
        batch_indices = [batch.start_paragraph + i for i in range(len(batch.paragraphs))]
        if all(idx in processed_paragraphs for idx in batch_indices) and not force:
            continue

        synthesis, model_tracker_idx = synthesize_batch(client, batch, settings, model_tracker_idx)
        notes = {note.paragraph_index: note.micro_context.strip() for note in synthesis.paragraph_notes}

        for local_index, paragraph in enumerate(batch.paragraphs):
            macro_summary = synthesis.macro_summary.strip()
            micro_context = notes[local_index]
            stacked_payload = (
                f"[GLOBAL CONTEXT]: {macro_summary}\n"
                f"[LOCAL CONTEXT]: {micro_context}\n"
                f"[RAW TEXT]: {paragraph.paragraph_text}"
            )
            enriched.append(
                EnrichedParagraph(
                    homily=paragraph.homily,
                    paragraph_index=batch.start_paragraph + local_index,
                    batch_start_paragraph=batch.start_paragraph,
                    macro_summary=macro_summary,
                    micro_context=micro_context,
                    raw_text=paragraph.paragraph_text,
                    stacked_payload=stacked_payload,
                )
            )

        _write_enriched_cache(enriched, settings.cache_path)
        print(f"Successfully cached up to paragraph {batch.start_paragraph + len(batch.paragraphs)}.", flush=True)

        if batch_number < len(batches):
            time.sleep(settings.request_cooldown_seconds)
    
    return enriched

def _write_enriched_cache(enriched: list[EnrichedParagraph], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: list[dict[str, Any]] = [item.model_dump() for item in enriched]
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_enriched_payloads(cache_path: Path) -> list[EnrichedParagraph]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Enriched cache not found: {cache_path}")
    return [
        EnrichedParagraph.model_validate(item)
        for item in json.loads(cache_path.read_text(encoding="utf-8"))
    ]
