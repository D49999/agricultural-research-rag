# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Common commands

```bash
# install dependencies
pip install -r requirements.txt

# configure local environment
cp .env.example .env
# then set DASHSCOPE_API_KEY and any model/runtime overrides in .env

# run tests
pytest tests/ -v

# run a single test file / test case
pytest tests/test_parser.py -v
pytest tests/test_parser.py::TestStructureAwareChunker::test_table_is_atomic -v

# diagnose VL embedding environment and text-image similarity alignment
python tests/diagnose_vl_embedding.py

# start FastAPI backend
python run_server.py --reload
python run_server.py --host 0.0.0.0 --port 8000 --workers 4

# start Streamlit frontend
streamlit run frontend/app.py

# OCR preprocessing and ingestion pipeline
python scripts/mineru_ocr.py --input-dir ./data/documents
python scripts/ingest_ocr_output.py
python scripts/ingest_ocr_output.py --reset
python scripts/ingest_ocr_output.py --chunk-strategy markdown
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output --reset

# evaluation
python scripts/evaluate.py --dataset ./data/eval_dataset.jsonl --corpus ./data/eval_corpus --k 1 3 5 10 --output ./results/eval_results.json
python scripts/evaluate.py --online --dataset ./data/eval_dataset.jsonl
python scripts/evaluate.py --chunk-ablation all
```

## Runtime configuration

Configuration is centralized in `config/settings.py` via `pydantic-settings`; values can be overridden by environment variables or `.env`. Key switches:

- `LLM_MODE=api|local`: API mode uses DashScope OpenAI-compatible API; local mode uses `LOCAL_LLM_BASE_URL`.
- `DASHSCOPE_API_KEY`: required for DashScope-backed LLM/VL calls.
- `EMBEDDING_MODE=local|api`: text embeddings use local `Qwen/Qwen3-Embedding-0.6B` by default or DashScope embedding API.
- `LOCAL_VL_EMBEDDING_MODEL`: image/cross-modal embeddings use local `Qwen/Qwen3-VL-Embedding-2B`.
- `CHUNK_STRATEGY=fixed|sentence|paragraph|markdown|recursive|semantic`: default is `semantic`.
- `ENABLE_RERANK` and `RERANKER_MODE=local|api`: control optional Qwen reranking.
- Chroma persists under `CHROMA_PERSIST_DIR` (default `./data/chroma_db`) with separate text and image collections.

Be aware that many tests mock heavy components, but running the server, ingestion, online evaluation, reranking, local embeddings, or VL diagnostics may load large models or call external APIs.

## Architecture overview

This is a Python multimodal Agentic RAG system for agricultural research, with separate text and image retrieval paths coordinated by a LangGraph ReAct agent.

### End-to-end flow

1. Complex PDFs/scans are expected to be preprocessed by MinerU into structured Markdown plus extracted images.
2. Markdown/TXT enters `DocumentRouter`, which delegates to `TextParser`; standalone image files go to `VisionParser` for Qwen-VL transcription.
3. Parsed blocks flow into `StructureAwareChunker`, which preserves headings as context and keeps tables/figures/formulas atomic.
4. Text chunks are embedded and stored in `ChromaVectorStore`; Markdown-referenced images are separately embedded by `QwenVLLocalEmbeddings` and stored in `ImageChromaStore`.
5. `HybridRetriever.retrieve()` handles text retrieval with dense Chroma + sparse BM25 + RRF fusion + optional reranker.
6. `HybridRetriever.retrieve_images()` handles image retrieval only, using the VL image collection for text-to-image search.
7. `MultimodalRAGAgent` exposes both retrieval paths and auxiliary skills as LangChain tools, then streams tool events and final answers through FastAPI.
8. The Streamlit frontend calls the REST/SSE API for chat, image search, document browsing, ingestion, and collection management.

### Important boundaries

- Text and image retrieval are intentionally separate collections and separate tools. Do not merge image results into the text vector collection or RRF pipeline unless the architecture is intentionally being redesigned.
- `src/api/main.py` initializes heavyweight singleton components during FastAPI lifespan: text Chroma store, image Chroma store, dense/sparse retrievers, reranker, hybrid retriever, and agent. API routes should reuse these dependencies rather than reloading models per request.
- BM25 state lives in `SparseRetriever` and is rebuilt from stored Chroma documents at startup and after ingestion.
- `src/agent/tools.py` is the tool boundary exposed to the ReAct agent. Changes here affect both synchronous `/v1/query` and SSE `/v1/query/stream` behavior.
- `frontend/app.py` assumes the backend is at `http://localhost:8000` and consumes SSE events from `/v1/query/stream` with `tool_start`, `tool_done`, `answer`, and `error` event types.

### Data and side effects

- Chroma data and uploaded/processed corpus artifacts live under `data/`; avoid deleting or resetting collections unless explicitly requested.
- `/v1/collection`, `scripts/ingest_ocr_output.py --reset`, and `scripts/ingest_images_from_md.py --reset` are destructive to local indexed data.
- Image Chroma IDs default to resolved absolute image paths, making repeated image ingestion idempotent.
- `scripts/cleanup_unused_images.py` can delete files; run with `--dry-run` first unless the user explicitly asks for deletion.

## Testing notes

- Unit tests currently cover parser/chunker, retrieval, and agent behavior under lightweight mocks.
- Use targeted `pytest path::Class::test_name -v` while iterating, then run `pytest tests/ -v` before claiming completion.
- For UI/API changes, run the backend and Streamlit frontend and exercise the affected flow manually; test suites do not validate the Streamlit experience or SSE rendering.
