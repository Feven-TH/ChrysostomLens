# ChrysostomLens

Production-ready, free-tier-friendly RAG pipeline for St. John Chrysostom's
*Homilies on Matthew*. The system builds a hierarchical global-to-local context
stack for every paragraph, indexes it with FAISS, and serves a terminal chatbot
through Groq.

## Architecture

1. Structural PDF parsing with `pdfplumber`
2. Homily-aware operational batching in exact 8-paragraph windows
3. Single-pass Gemini `gemini-2.5-flash` structured synthesis with Pydantic v2
4. Stacked paragraph payload cache:
   `[GLOBAL CONTEXT]`, `[LOCAL CONTEXT]`, `[RAW TEXT]`
5. Dense FAISS indexing with `BAAI/bge-large-en-v1.5`
6. Interactive Groq `llama-3.3-70b-versatile` RAG chat

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
cp .env.example .env
```

Export the values from `.env` in your shell before running:

```bash
export GOOGLE_API_KEY=...
export GROQ_API_KEY=...
export HUGGINGFACEHUB_API_TOKEN=...
export GEMINI_FALLBACK_MODELS="gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite"
```

## Ingest

```bash
chrysostom-lens status
chrysostom-lens ingest
chrysostom-lens status
```

This writes:

- `data/parsed_paragraphs.json`
- `data/enriched_homilies.json`
- `faiss_homilies_index/`

The default embedding provider is Hugging Face serverless. To index locally
instead, install the optional local embedding dependencies and pass:

```bash
pip install -r requirements-local-embeddings.txt
chrysostom-lens ingest --embedding-provider local
```

The default PDF path is:

```text
data/Homilies on the Gospel of Matthew.pdf
```

Override it only when needed:

```bash
chrysostom-lens ingest "data/another-file.pdf"
```

## Step-by-step run

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

Set your API keys:

```bash
export GOOGLE_API_KEY="your-google-key"
export HUGGINGFACEHUB_API_TOKEN="your-huggingface-token"
export GROQ_API_KEY="your-groq-key"
export GEMINI_FALLBACK_MODELS="gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite"
```

Confirm the project sees the PDF:

```bash
chrysostom-lens status
```

Expected before ingestion: the PDF should say `found`; parsed cache, enriched
cache, and FAISS index may say `missing`.

Confirm Gemini fallback order:

```bash
chrysostom-lens models
```

Run ingestion:

```bash
chrysostom-lens ingest
```

Confirm outputs:

```bash
chrysostom-lens status
```

Expected after ingestion:

- `Parsed cache` shows a record count
- `Enriched cache` shows a record count
- `FAISS index` lists `index.faiss` and `index.pkl`
- The previews show one parsed paragraph and one stacked payload

## Chat

```bash
chrysostom-lens chat
```

Use the same `--embedding-provider` value that was used during ingestion.
