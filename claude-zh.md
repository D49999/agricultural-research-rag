# claude-zh.md

本文件是 `CLAUDE.md` 的中文参考版，用于帮助后续 Claude Code 实例快速理解并操作本仓库。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 配置本地环境
cp .env.example .env
# 然后在 .env 中设置 DASHSCOPE_API_KEY 以及需要覆盖的模型/运行时配置

# 运行全部测试
pytest tests/ -v

# 运行单个测试文件 / 单个测试用例
pytest tests/test_parser.py -v
pytest tests/test_parser.py::TestStructureAwareChunker::test_table_is_atomic -v

# 诊断 VL embedding 环境与文本-图像相似度对齐
python tests/diagnose_vl_embedding.py

# 启动 FastAPI 后端
python run_server.py --reload
python run_server.py --host 0.0.0.0 --port 8000 --workers 4

# 启动 Streamlit 前端
streamlit run frontend/app.py

# OCR 预处理与入库流程
python scripts/mineru_ocr.py --input-dir ./data/documents
python scripts/ingest_ocr_output.py
python scripts/ingest_ocr_output.py --reset
python scripts/ingest_ocr_output.py --chunk-strategy markdown
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output --reset

# 评估
python scripts/evaluate.py --dataset ./data/eval_dataset.jsonl --corpus ./data/eval_corpus --k 1 3 5 10 --output ./results/eval_results.json
python scripts/evaluate.py --online --dataset ./data/eval_dataset.jsonl
python scripts/evaluate.py --chunk-ablation all
```

## 运行时配置

项目配置集中在 `config/settings.py`，基于 `pydantic-settings` 管理；所有配置都可以通过环境变量或 `.env` 覆盖。关键配置包括：

- `LLM_MODE=api|local`：API 模式使用 DashScope OpenAI-compatible 接口；本地模式使用 `LOCAL_LLM_BASE_URL`。
- `DASHSCOPE_API_KEY`：DashScope 后端的 LLM / VL 调用需要此密钥。
- `EMBEDDING_MODE=local|api`：文本 embedding 默认使用本地 `Qwen/Qwen3-Embedding-0.6B`，也可切换到 DashScope embedding API。
- `LOCAL_VL_EMBEDDING_MODEL`：图像/跨模态 embedding 使用本地 `Qwen/Qwen3-VL-Embedding-2B`。
- `CHUNK_STRATEGY=fixed|sentence|paragraph|markdown|recursive|semantic`：默认使用 `semantic`。
- `ENABLE_RERANK` 与 `RERANKER_MODE=local|api`：控制是否启用 Qwen reranker 精排以及使用本地/API 模式。
- `CHROMA_PERSIST_DIR`：Chroma 持久化目录，默认是 `./data/chroma_db`；文本和图片使用不同 collection。

注意：很多单元测试使用 mock，避免加载重型模型。但运行服务、入库、在线评估、reranker、本地 embedding 或 VL 诊断脚本时，可能会加载大模型或调用外部 API。

## 架构概览

这是一个 Python 多模态 Agentic RAG 系统。核心特点是：文本检索和图像检索分成两条独立路径，再由 LangGraph ReAct Agent 统一编排。

### 端到端流程

1. PDF / 扫描件等复杂文档先通过 MinerU 预处理，输出结构化 Markdown 和提取出的图片。
2. Markdown / TXT 文件进入 `DocumentRouter`，再路由到 `TextParser`；独立图片文件进入 `VisionParser`，由 Qwen-VL 转写为文本。
3. 解析得到的 `ParsedBlock` 交给 `StructureAwareChunker` 切分。该切分器会把标题作为上下文保留，并将表格、图片、公式等原子块整体保留。
4. 文本 chunk 经 embedding 后写入 `ChromaVectorStore`；Markdown 中引用的本地图片则经 `QwenVLLocalEmbeddings` 编码后写入 `ImageChromaStore`。
5. `HybridRetriever.retrieve()` 负责文本检索：Chroma 稠密检索 + BM25 稀疏检索 + RRF 融合 + 可选 reranker 精排。
6. `HybridRetriever.retrieve_images()` 只负责图像检索：使用 VL 图像 collection 进行文本搜图。
7. `MultimodalRAGAgent` 将文本检索、图像检索和扩展技能封装成 LangChain tools，并通过 FastAPI 返回普通响应或 SSE 流式响应。
8. Streamlit 前端通过 REST / SSE API 实现智能问答、图片检索、知识库浏览、文档入库和系统管理。

### 关键边界

- 文本检索和图像检索有意使用两个独立 collection 和两个独立工具。除非明确要重构架构，否则不要把图片结果混入文本向量库或文本 RRF 流程。
- `src/api/main.py` 在 FastAPI lifespan 中初始化重型单例：文本 Chroma store、图片 Chroma store、dense/sparse retriever、reranker、hybrid retriever 和 agent。API 路由应复用这些依赖，不要在每次请求时重新加载模型。
- BM25 状态保存在 `SparseRetriever` 中，服务启动时会从 Chroma 已有文档重建索引，入库后也会重建。
- `src/agent/tools.py` 是暴露给 ReAct Agent 的工具边界。修改这里会同时影响同步接口 `/v1/query` 和流式接口 `/v1/query/stream`。
- `frontend/app.py` 默认后端地址是 `http://localhost:8000`，并消费 `/v1/query/stream` 的 SSE 事件：`tool_start`、`tool_done`、`answer`、`error`。

## 数据与副作用

- Chroma 数据和处理后的语料通常位于 `data/` 下；除非用户明确要求，不要删除或重置 collection。
- `/v1/collection`、`scripts/ingest_ocr_output.py --reset` 和 `scripts/ingest_images_from_md.py --reset` 都会破坏本地索引数据。
- 图片 Chroma ID 默认使用图片解析后的绝对路径，因此重复执行图片入库是幂等的。
- `scripts/cleanup_unused_images.py` 可能删除文件；除非用户明确要求实际删除，否则应先使用 `--dry-run` 预览。

## 测试说明

- 当前单元测试主要覆盖 parser/chunker、retrieval 和 agent 行为，并大量使用轻量级 mock。
- 开发时优先运行定向测试，例如 `pytest path::Class::test_name -v`；完成前再运行 `pytest tests/ -v`。
- 如果修改 UI 或 API 行为，应启动后端和 Streamlit 前端，并手动验证受影响流程；单元测试无法覆盖 Streamlit 交互和 SSE 渲染效果。
