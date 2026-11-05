# AGENTS.md

Python 3.13 multimodal Agentic RAG for agricultural research: FastAPI + Streamlit + Chroma + LangGraph ReAct. `README.md` is user-facing Chinese and partly stale; treat `config/settings.py`, `scripts/`, and tests as sources of truth. `CLAUDE.md` / `CLAUDE1.md` are older copies with wrong commands.

## Commands

```bash
# install (requirements.txt is incomplete — see Setup)
pip install -r requirements.txt

# tests: currently 238 pass; heavy components are mocked
pytest tests/ -q --basetemp=.pytest_cache/tmp
pytest tests/test_parser.py::TestStructureAwareChunker::test_table_is_atomic -v

# backend / frontend
python run_server.py --reload
streamlit run frontend/app.py
bash launch.sh   # hardcodes D:/Anaconda3/envs/py313 and an absolute PYTHONPATH

# diagnose VL embedding environment and text/image similarity alignment
python tests/diagnose_vl_embedding.py

# OCR preprocessing: defaults data/pdfs -> data/ocr_output
python scripts/mineru_ocr.py
python scripts/mineru_ocr.py --input ./data/pdfs --output ./data/ocr_output

# ingestion (--reset is destructive)
python scripts/ingest_ocr_output.py --reset
python scripts/ingest_ocr_output.py --chunk-strategy markdown
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output --reset
python scripts/ingest_documents.py --input-dir ./data/documents
python scripts/pdf_to_txt.py   # PyMuPDF text-only fallback; no images

# Chinese agri corpus: crawl MOA column pages into Markdown, then ingest (no --reset)
python scripts/fetch_zh_literature.py --pages 2 --max-docs 40
python scripts/ingest_documents.py --input-dir ./data/documents/zh --chunk-strategy semantic

# backfill agri metadata for a pre-existing index (metadata only; dry-run by default)
python scripts/backfill_agri_metadata.py
python scripts/backfill_agri_metadata.py --apply
python scripts/backfill_agri_metadata.py --apply --overwrite --prune   # drop stale fields (delete+re-add)

# after switching embedding backend: recompute every text vector (dry-run by default)
python scripts/reembed_collection.py
python scripts/reembed_collection.py --apply --batch-size 16

# evaluation (Chinese agri corpus; the old eval_dataset.jsonl is an unrelated HR/template set)
python scripts/evaluate.py --online --dataset ./data/eval_dataset_zh.jsonl --k 1 3 5 10
python scripts/evaluate.py --online --dataset ./data/eval_dataset_zh_paraphrase.jsonl --k 1 3 5 10
python scripts/evaluate.py --online --ablation dense --dataset ./data/eval_dataset_zh_paraphrase.jsonl
python scripts/evaluate.py --dataset ./data/eval_dataset.jsonl --corpus ./data/eval_corpus --k 1 3 5 10
python scripts/evaluate.py --chunk-ablation all --chunk-size 1024
python scripts/evaluate.py --online --ablation sparse
```

- Plain `pytest tests/` fails on this Windows machine with `PermissionError` on `%TEMP%\pytest-of-*`. Always pass `--basetemp` (the path above is already gitignored).
- `scripts/evaluate.py` has no `--output` flag; older docs are wrong. Offline eval defaults to `--chunk-size 256`, so pass `--chunk-size 1024` to match the README benchmark. `--online` needs a populated Chroma index (it builds the embedder/retriever directly, no HTTP server); `--ablation` accepts only `dense|sparse` — plain `--online` is the hybrid path; `--filters` additionally applies `understand_query` slot filters.
- `data/eval_dataset_zh_negative.jsonl` is for refusal checks via `/v1/query`, not for `evaluate.py` (metrics would always count it as a miss).
- `scripts/mineru_ocr.py` only accepts `--input` / `--output` (README's `--input-dir` is wrong). Backend, parse mode, language, formula toggle, and page range are hardcoded in `main()`.
- No lint, formatter, typecheck, or CI config exists; pytest is the only automated gate.

## Setup and runtime config

- All configuration is in `config/settings.py` via `pydantic-settings`. Use `get_settings()` (an `lru_cache` singleton); restart after changing env vars.
- `CHUNK_STRATEGY` (default `semantic`) accepts `fixed|sentence|paragraph|markdown|recursive|semantic`.
- `ENABLE_AGRI_METADATA` (default `true`) controls rule-based crop/disease/region/year extraction at chunk time. It is pure local regex + `config/agri_lexicon.json`; no LLM or network. Document-level `crop` prefers the Markdown H1/filename over body frequency, ignores crop prefixes inside polyphagous pest names (`棉铃虫`, `甜菜夜蛾`), and ASCII aliases require whole-word matches (`tea` must not hit `instead`).
- `.env.example` does not exist. A local gitignored `.env` contains a live `DASHSCOPE_API_KEY`; never print or commit it.
- `requirements.txt` omits several runtime imports: `langchain-openai` and `PyYAML` are needed just to import the app; `torch` is needed because `src/api/main.py` imports `image_store` → `qwen_vl_embedding` at module load; `transformers` / `huggingface-hub` are needed when local embedding, VL, or reranking actually loads; `mineru` is OCR-only. Install them separately for a fresh environment.
- Local text/VL/reranker models use `local_files_only=True`: they must already exist in the HuggingFace cache or a configured local path; there is no auto-download. The VL embedder additionally requires the official `scripts/qwen3_vl_embedding.py` inside the model repo snapshot.
- Image vectors always come from local `Qwen/Qwen3-VL-Embedding-2B`, regardless of `EMBEDDING_MODE`. If that model is unavailable, the API logs a warning and starts with image search disabled (503), but a missing `torch` prevents startup entirely.
- Text vectors are tied to the embedding backend. Chroma does not record which model produced them, so changing `EMBEDDING_MODE` / embedding model against an existing `data/chroma_db` silently mixes incompatible vectors — either `scripts/reembed_collection.py --apply` (keeps documents, recomputes vectors) or reset and re-ingest.
- This machine runs CPU-only `torch`, so `QwenLocalEmbeddings` loads the model in float32 (fp16 on CPU is ~10x slower). `LOCAL_EMBEDDING_MODEL` currently points at `F:/models/Qwen3-Embedding-0.6B`, downloaded via `modelscope download` (huggingface.co is unreachable, `hf-mirror.com` works). Expect roughly 5-7 s per chunk on CPU when re-embedding.
- Reranking is off by default (`ENABLE_RERANK=false`); `QwenReranker()` returns `None` and loads nothing until enabled.

## Architecture and boundaries

Flow: OCR Markdown → `DocumentRouter` → `TextParser` → `StructureAwareChunker` (+ `src/agri` metadata) → `ChromaVectorStore`; referenced images → `QwenVLLocalEmbeddings` → `ImageChromaStore`; text query → `understand_query` slots → dense + BM25 with metadata filters → RRF → optional rerank; image query → VL collection; `MultimodalRAGAgent` exposes both paths as tools and streams events.

- **PDFs are not parsed by the retrieval stack.** `DocumentRouter` only handles `.md` / `.markdown` / `.txt` and images. Despite README, `ingest_documents.py`'s PDF extension list, and the UI's PDF upload widget, `router.route()` raises on PDFs. The working path is MinerU OCR (or `pdf_to_txt.py`) first.
- Text and image retrieval intentionally use separate collections and separate agent tools. Do not merge image results into the text Chroma collection or RRF pipeline.
- `src/api/main.py` builds all heavyweight singletons once in the FastAPI lifespan (stores, retrievers, agent). Routes must reuse them via dependencies, not reload models per request.
- BM25 is in-memory only. The API rebuilds it at startup and after ingestion from at most 10,000 Chroma documents; `paper_ingest` uses the same cap. CLI ingest scripts build a local BM25 index that dies with the process.
- `src/agri/` is the domain layer: `lexicon` (aliases, entity hits, agri tokenizer), `metadata_extractor` (doc/chunk metadata), `filters` (normalize + Chroma `where` + BM25 matching), `query_understanding` (slots/intent/year expansion). It must stay free of LLM/network calls and must not import retrieval or agent modules.
- Metadata filters use one canonical shape (`{"crop": "水稻", "year_min": 2020, ...}`); `build_chroma_where` translates it for Chroma and `metadata_matches` for BM25. Both paths must receive the same normalized dict to stay consistent.
- `HybridRetriever.retrieve_with_info()` returns `RetrievalOutcome` and relaxes over-strict filters to an unfiltered search with `relaxed=True`. Tools surface this so the agent can tell the user results are approximate. `retrieve()` stays backward compatible.
- `src/agent/tools.py` is the retrieval tool boundary (`knowledge_base_search`, `knowledge_base_search_structured`, `knowledge_base_search_with_filter`, `multi_round_search`, `image_search`, optional `query_rewrite`). Skills live in `src/agent/skills/` (`calculator`, `web_search`, `table_analyzer`, `image_describer`, `summarizer`, `paper_search`, `paper_ingest`, `paper_polishing`).
- `paper_ingest` is registered only when both `store` and `sparse_retriever` are passed (the API lifespan does this). `paper_search` and `paper_ingest` hit arXiv / Semantic Scholar live; `paper_ingest` writes Markdown under `data/literature` and is idempotent per `paper_id`.
- `paper_polishing` loads an external skill from `settings.nature_polishing_skill_dir`, defaulting to the absolute path `E:/agent-learning/nature-skills/skills/nature-polishing`; override with `NATURE_POLISHING_SKILL_DIR`. If missing, the tool remains registered but returns `status=error`.
- `ingest_ocr_output.py` expects `{ocr_dir}/{doc}/{backend}/{doc}.md` and backfills page numbers from `*_content_list.json`; `ingest_images_from_md.py` resolves Markdown image links relative to the Markdown file.
- `frontend/app.py` hardcodes `API_BASE = "http://localhost:8000"` and consumes `/v1/query/stream` SSE events `tool_start`, `tool_done`, `answer`, `error`, terminated by `[DONE]`. Changing `API_PORT` does not propagate to the UI.

## Data and side effects

- Chroma, OCR output, and uploaded corpus artifacts live under `data/` (gitignored). Do not delete or reset unless explicitly requested.
- Destructive operations: `DELETE /v1/collection`, `ingest_ocr_output.py --reset`, `ingest_images_from_md.py --reset`, `ingest_documents.py --reset`; resetting text and image collections are separate operations.
- Image Chroma IDs are resolved absolute image paths, so re-ingesting the same files is idempotent — but moving the image tree creates duplicates.
- Run `scripts/cleanup_unused_images.py --dry-run` before any real deletion.
- `scripts/backfill_agri_metadata.py` updates metadata only (no re-embedding, no deletion) and is idempotent. It defaults to dry-run; `--apply` writes to Chroma, so stop the backend first and restart it afterwards so the in-memory BM25 index picks up the new metadata.

## Testing notes

- Unit tests mock LLMs, retrieval, and HTTP; they do not validate real model quality, the Streamlit UI, or SSE rendering.
- `tests/test_agri_*.py` and `tests/test_query_understanding.py` cover the domain layer; they are pure rule tests and need no models. If you change `config/agri_lexicon.json` or the filter shape, update them together.
- `tests/test_paper_polishing.py` includes a real-skill smoke test that skips when the external skill directory is absent.
- For API/UI changes, run the backend and Streamlit frontend and exercise the affected flow manually in addition to pytest.
