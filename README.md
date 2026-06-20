# ChrysostomLens
[![Deployment Status](https://img.shields.io/badge/Live_Demo-Hosted_on_Render-brightgreen.svg)](https://chrysostom-lens.vercel.app/)

Production-grade RAG for St. John Chrysostom's *Homilies on the Gospel of Matthew*, built with Python 3.11, Pydantic v2, FAISS, Hugging Face `BAAI/bge-large-en-v1.5` embeddings, Gemini synthesis, Groq/Gemini chat generation, FastAPI, and a React/Vite study interface.

---

## 1. What Is This About? (Purpose & Capabilities)

ChrysostomLens turns a long patristic PDF into a searchable theological research assistant. It parses the homilies into natural paragraphs, enriches each paragraph with a retrieval-oriented context stack, indexes those payloads in FAISS, and answers questions only from retrieved passages.

The current local corpus contains:

- Source PDF: `data/Homilies on the Gospel of Matthew.pdf` (`5.4 MB`)
- Parsed paragraph cache: `data/parsed_paragraphs.json` (`3,900` records)
- Enriched retrieval cache: `data/enriched_homilies.json` (`3,900` records)
- FAISS index: `faiss_homilies_index/index.faiss` and `faiss_homilies_index/index.pkl`

Technical pipeline:

- **Parsing:** `pdfplumber` extracts layout-preserving PDF lines. The parser tracks homily headings, page numbers, Matthew references, verse markers, and paragraph continuations.
- **Chunking:** paragraphs are batched in exact 8-paragraph operational windows. Batches never cross homily boundaries.
- **Synthesis:** Gemini receives each 8-paragraph batch and returns JSON matching Pydantic schemas. Every batch gets one `macro_summary`; every paragraph gets one `micro_context`.
- **Index payload:** each paragraph is stored as one stacked text unit:

```text
[GLOBAL CONTEXT]: batch-level macro summary
[LOCAL CONTEXT]: paragraph-level retrieval note
[RAW TEXT]: original paragraph text
```

- **Embeddings:** `BAAI/bge-large-en-v1.5` creates dense vectors through Hugging Face serverless embeddings by default, or local CPU embeddings when requested.
- **Vector database:** LangChain FAISS stores the enriched paragraphs with metadata for `homily`, `paragraph_index`, and `batch_start_paragraph`.
- **LLM orchestration:** the chatbot retrieves top-3 FAISS matches, formats them as bounded context, and streams an answer through Groq `llama-3.3-70b-versatile` with Gemini fallback.
- **Interfaces:** use the terminal chatbot, FastAPI backend, or React/Vite frontend.

Data architecture:

```text
Raw PDF
  data/Homilies on the Gospel of Matthew.pdf
        |
        v
Structural parsing
  pdfplumber -> homily tracking -> paragraph reconstruction
        |
        v
Parsed cache
  data/parsed_paragraphs.json
  ParsedParagraph(homily, paragraph_text, scripture_reference, verse_id)
        |
        v
Homily-bounded batching
  exact 8-paragraph windows
        |
        v
Gemini synthesis
  macro_summary + paragraph_notes validated by Pydantic
        |
        v
Stacked payload cache
  data/enriched_homilies.json
  [GLOBAL CONTEXT] + [LOCAL CONTEXT] + [RAW TEXT]
        |
        v
Embeddings
  BAAI/bge-large-en-v1.5 via Hugging Face serverless or local CPU
        |
        v
Vector database
  faiss_homilies_index/index.faiss
  faiss_homilies_index/index.pkl
        |
        v
Prompt loop
  query -> top-3 retrieval -> source blocks -> Groq stream -> Gemini fallback
        |
        v
Response
  terminal chat, FastAPI SSE stream, or React frontend
```

## 2. Why Should I Care? (Value Proposition)

St. John Chrysostom's homilies are difficult retrieval material. The paragraphs are long, rhetorically dense, and full of theological movement: scriptural interpretation, moral exhortation, metaphor, historical argument, and pastoral application often appear in the same passage. A plain paragraph-only RAG index can miss the point because the paragraph's literal wording may not contain the modern search terms a user asks with.

ChrysostomLens improves retrieval by indexing each paragraph with two extra layers of model-generated context:

- **Global context:** the shared theological and scriptural scope of an 8-paragraph window.
- **Local context:** a short explanation of the specific paragraph's argumentative or metaphorical role.
- **Raw text:** the original paragraph remains present so answers stay anchored to the source.

This is more reliable than a generic "split by character count and embed" pipeline because the index preserves the paragraph as the atomic source while adding semantic handles for retrieval. The 8-paragraph window is large enough to capture Chrysostom's argument flow and small enough to keep Gemini prompts under the configured `MAX_BATCH_PROMPT_TOKENS=28000` ceiling.

The project is also free-tier-conscious:

- Hugging Face serverless embeddings are the default to avoid local model setup.
- Serverless indexing pushes documents in batches of `32` to reduce timeout risk.
- Local embeddings are available when you want repeatable offline indexing.
- Cached parsing and synthesis prevent paying for repeated Gemini calls unless `--force-parse` or `--force-synthesis` is used.

## 3. Can I Trust It? (Production Quality & Engineering Standards)

Repository structure:

```text
ChrysostomLens/
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
|-- requirements-local-embeddings.txt
|-- .env_example
|-- data/
|   |-- Homilies on the Gospel of Matthew.pdf
|   |-- parsed_paragraphs.json
|   `-- enriched_homilies.json
|-- faiss_homilies_index/
|   |-- index.faiss
|   `-- index.pkl
|-- src/
|   `-- chrysostom_lens/
|       |-- cli.py
|       |-- config.py
|       |-- models.py
|       |-- parsing.py
|       |-- synthesis.py
|       |-- indexing.py
|       |-- pipeline.py
|       |-- chat.py
|       `-- server.py
`-- frontend/
    |-- package.json
    |-- vite.config.js
    |-- .env_example
    `-- src/
        |-- App.jsx
        |-- main.jsx
        `-- index.css
```

Reliability elements:

- **Schema validation:** `models.py` defines Pydantic v2 models for parsed paragraphs, paragraph batches, Gemini synthesis, paragraph notes, and enriched payloads.
- **Structured Gemini output:** synthesis requests set `response_mime_type="application/json"` and `response_schema=BatchSynthesis`.
- **Completeness checks:** Gemini output is normalized and rejected if any paragraph note index is missing.
- **Deterministic chunking:** `batch_paragraphs(..., batch_size=8)` rejects any batch size other than `8`.
- **Prompt budget guard:** synthesis fails fast when a batch exceeds `MAX_BATCH_PROMPT_TOKENS`.
- **Retry behavior:** Gemini and Hugging Face calls retry transient `429`, timeout, unavailable, and server-side errors with exponential backoff plus jitter.
- **Fallback model chain:** chat generation uses Groq first when `GROQ_API_KEY` is present, then tries configured Gemini fallback models.
- **Cache-first ingestion:** parsed and enriched caches are reused by default; destructive regeneration requires explicit force flags.
- **Source-bounded answers:** the system prompt instructs the assistant to answer only from retrieved context and to admit when the retrieved passages are insufficient.
- **API health endpoint:** FastAPI exposes `/api/status` for frontend and operator checks.

Configured defaults live in `src/chrysostom_lens/config.py`:

```text
GEMINI_MODEL=gemini-2.5-flash-lite
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
GEMINI_COOLDOWN_SECONDS=8
MAX_LLM_RETRIES=2
MAX_BATCH_PROMPT_TOKENS=28000
CHRYSOSTOM_INDEX=faiss_homilies_index
```

## 4. Can I Use It? (Reproducibility & Execution)

### Prerequisites

- Python `3.11` or newer. The package declares `requires-python = ">=3.11"`.
- Node.js `18` or newer for the React frontend.
- API keys you provide yourself:
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Gemini synthesis and fallback chat.
  - `GROQ_API_KEY` for primary streamed chat responses.
  - `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, or `HUGGING_FACE_HUB_TOKEN` for serverless embeddings.

### Security Protocol

Do not commit real credentials. Put secrets in a local `.env` file at the repository root. This project loads `.env` automatically through `python-dotenv`, and `.gitignore` excludes `.env`.

Create your local backend environment file:

```bash
cp .env_example .env
```

Then edit `.env` and replace every placeholder value:

```bash
GOOGLE_API_KEY=insert_your_google_or_gemini_key_here
GROQ_API_KEY=insert_your_groq_key_here
HUGGINGFACEHUB_API_TOKEN=insert_your_huggingface_token_here
GEMINI_FALLBACK_MODELS=gemini-2.5-flash,gemini-2.5-flash-lite,gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.0-flash,gemini-2.0-flash-lite
```

For the frontend, copy its example file only if you need a non-default backend URL:

```bash
cp frontend/.env_example frontend/.env
```

If the backend runs locally on port `8000`, the React app works without changing `VITE_API_URL`.

### Sequential Setup Protocol

Run these commands from a clean clone:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e . --no-deps
cp .env_example .env
```

If your machine only exposes Python as `python3`, use this equivalent first command:

```bash
python3 -m venv .venv
```

If you want local CPU embeddings instead of Hugging Face serverless embeddings:

```bash
pip install -r requirements-local-embeddings.txt
```

### Operational Sequences

Check local system status:

```bash
chrysostom-lens status
```

Expected output when the committed caches and FAISS index are present:

```text
ChrysostomLens status
PDF: data/Homilies on the Gospel of Matthew.pdf (found)
Parsed cache: data/parsed_paragraphs.json (3900 records)
Enriched cache: data/enriched_homilies.json (3900 records)
FAISS index: faiss_homilies_index (index.faiss, index.pkl)
```

Show configured Gemini fallback order:

```bash
chrysostom-lens models
```

Expected local output before live model listing:

```text
Configured Gemini fallback order:
1. gemini-2.5-flash-lite
2. gemini-2.5-flash
3. gemini-3.5-flash
4. gemini-3.1-flash-lite
5. gemini-2.0-flash
6. gemini-2.0-flash-lite
```

If your network and Google key are available, the same command also attempts to list Gemini models visible to your account.

Run ingestion with serverless embeddings:

```bash
chrysostom-lens ingest
```

Expected output shape:

```text
Loaded 3900 previously enriched paragraphs from cache.

[Cloud Ingestion] Total documents to index: 3900
Processing in batches of 32 to prevent Hugging Face 504 Timeouts...

Initializing index container with docs 0 to 32...
-> Pushing cloud batch: items 32 to 64...
...
[Success] Vector matrices compiled. Saving index to: faiss_homilies_index
Done. Enriched cache: data/enriched_homilies.json; FAISS index: faiss_homilies_index
```

Run ingestion with local CPU embeddings:

```bash
chrysostom-lens ingest --embedding-provider local
```
> ℹ️ **Note on Ingestion:** The pre-compiled vector database and enriched caches are already included in this repository. You can skip ingestion and jump directly to `chrysostom-lens chat`. 
> If you wish to test the full ingestion pipeline from scratch, download the source public-domain PDF from [this link](https://drive.google.com/file/d/1QAG6YC8pW3JjRfu7ivq3_lhM6UVlwK_w/view?usp=sharing) and place it in the `data/` directory before running `chrysostom-lens ingest`.

Expected output shape:

```text
Loaded 3900 previously enriched paragraphs from cache.

[Local Ingestion] Processing bulk array directly on CPU...

[Success] Vector matrices compiled. Saving index to: faiss_homilies_index
Done. Enriched cache: data/enriched_homilies.json; FAISS index: faiss_homilies_index
```

Re-parse and re-run Gemini synthesis only when you intentionally want to regenerate caches:

```bash
chrysostom-lens ingest --force-parse --force-synthesis
```

Launch the terminal chatbot:

```bash
chrysostom-lens chat
```

Expected startup:

```text
ChrysostomLens RAG chat. Type 'exit', 'quit', or press Ctrl-C to leave.

Question>
```

Run the FastAPI backend:

```bash
uvicorn chrysostom_lens.server:app --host 0.0.0.0 --port 8000
```

Check backend health:

```bash
curl http://localhost:8000/api/status
```

Expected JSON shape:

```json
{
  "pdf_found": true,
  "parsed_cache_found": true,
  "enriched_cache_found": true,
  "index_found": true,
  "index_path": "faiss_homilies_index",
  "groq_model": "llama-3.3-70b-versatile",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "groq_key_configured": true,
  "google_key_configured": true,
  "status": "ready",
  "message": "Library is loaded and ready for study."
}
```

Run the React frontend:

```bash
cd frontend
npm install
npm run dev
```

Expected Vite output:

```text
VITE v5.x.x  ready
Local:   http://localhost:5173/
```

## 5. License & Legal Compliance

This repository is released under the MIT License. See `LICENSE` for the full license text.

Data and model rights:

- **Primary text:** the source material is St. John Chrysostom's *Homilies on the Gospel of Matthew*. The included English text is treated as public-domain patristic material, commonly associated with 19th-century public-domain translations. If you replace the PDF with a modern edition, confirm that edition's license before redistribution.
- **Generated caches:** `parsed_paragraphs.json` and `enriched_homilies.json` are derived from the included source text and model-generated retrieval notes.
- **Embeddings:** vectors are produced with `BAAI/bge-large-en-v1.5`; follow the model provider's license and usage terms.
- **LLMs:** Gemini and Groq-hosted models are external services. You must supply your own API keys and comply with Google, Groq, Hugging Face, and any model-specific terms.
- **Secrets:** API keys belong in `.env` only. Never commit `.env`, terminal logs containing keys, or screenshots exposing credentials.
