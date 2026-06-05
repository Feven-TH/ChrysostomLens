"""Pydantic schemas and domain records used by the RAG pipeline."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ParagraphNote(BaseModel):
    paragraph_index: int = Field(ge=0, description="The zero-based index supplied in the prompt.")
    micro_context: str = Field(
        min_length=20,
        max_length=900,
        description="One or two sentences capturing the paragraph's situational or metaphorical insight.",
    )


class BatchSynthesis(BaseModel):
    macro_summary: str = Field(
        min_length=40,
        max_length=1600,
        description="Two sentences summarizing the batch's overarching theme and scriptural scope.",
    )
    paragraph_notes: list[ParagraphNote] = Field(
        min_length=1,
        max_length=8,
        description="Per-paragraph notes keyed to prompt paragraph indexes.",
    )


class ParsedParagraph(BaseModel):
    homily: str
    paragraph_text: str = Field(min_length=1)
    scripture_reference: Optional[str] = None
    verse_id: Optional[str] = None


class ParagraphBatch(BaseModel):
    homily: str
    start_paragraph: int
    paragraphs: list[ParsedParagraph] = Field(min_length=1, max_length=8)


class EnrichedParagraph(BaseModel):
    homily: str
    paragraph_index: int
    batch_start_paragraph: int
    macro_summary: str
    micro_context: str
    raw_text: str
    stacked_payload: str
