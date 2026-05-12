# AgRAG — 农业科研智能问答助手

> 把散落在 PDF、论文、报告里的农业科研资料，变成一个能对话的知识伙伴。

农业科研人员每天面对大量文献——实验报告里的表格数据、论文中的作物生长曲线图、扫描版的品种对比图谱。传统关键词搜索难以跨模态理解这些内容。**AgRAG** 用 Agentic RAG 技术将这些异构资料统一索引，让你用自然语言直接提问，系统自动判断该查文本还是找图片，像一个熟悉你全部资料库的科研助理一样回答问题。

---

## 它能做什么

**一句话概括**：你问，它答——并且告诉你答案来自哪篇文献的哪一页。

### 实际使用场景

| 你可以这样问 | 系统会做什么 |
|---|---|
| "小麦锈病的防治措施有哪些？" | 在科研文本库中检索相关段落，汇总多个来源的回答 |
| "找一张水稻分蘖期的照片" | 用跨模态向量匹配，在图像库中找到最相关的图片并展示 |
| "对比一下氮肥和磷肥对产量的影响" | 改写为多个子查询，多轮检索后综合分析 |
| "这个表格里的数据帮我算一下均值" | 调用表格分析工具，自动计算统计指标 |
| "论文里那张折线图是什么意思" | 用视觉模型理解图像内容，给出解读 |

### 五项核心能力

1. **科研文献问答** — 支持多轮对话，Agent 自主选择检索策略
2. **以文搜图** — 输入描述文字，跨模态匹配论文中的图表和照片
3. **文档管理** — 上传、浏览、检索、删除，全生命周期管理
4. **流式思考** — 实时展示 Agent 的每一步工具调用过程
5. **灵活部署** — 全部本地推理或 DashScope API，一键切换

---

## 快速上手

### 环境准备

```bash
# 克隆项目
git clone <repo-url> && cd agricultural-research-rag

# 安装依赖
pip install -r requirements.txt

# 配置 API Key（本地推理模式可跳过）
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
```

### 三步跑通

```bash
# 1. 预处理：PDF/扫描件 → 结构化 Markdown
python scripts/mineru_ocr.py --input-dir ./data/documents

# 2. 入库：文本 + 图片分别向量化
python scripts/ingest_ocr_output.py
python scripts/ingest_images_from_md.py --input-dir ./data/ocr_output

# 3. 启动：后端 + 前端
python run_server.py --reload          # 后端 :8000
streamlit run frontend/app.py          # 前端 :8501
```

打开浏览器访问 `http://localhost:8501`，在「科研问答」页面开始对话。

### 一键启动（开发模式）

```bash
bash launch.sh
# 自动清理旧进程 → 启动后端 → 等待就绪 → 启动前端
```

---

## 技术架构

### 数据流：从原始文档到智能问答

```
  ┌──────────────────────────────────────────────────────┐
  │                  农业科研资料                         │
  │         PDF · 扫描件 · 实验报告 · 图表               │
  └─────────────────────┬────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │   MinerU OCR 预处理    │
            │   布局识别 + 文字识别   │
            └───────────┬───────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
        结构化 Markdown      提取的图片文件
              │                   │
    ┌─────────▼─────────┐  ┌──────▼──────────────┐
    │   TextParser      │  │  图片向量化          │
    │   结构化解析       │  │  Qwen3-VL-Embedding  │
    │   ↓               │  │  文本↔图像同空间      │
    │   StructureAware  │  └──────┬──────────────┘
    │   Chunker         │         │
    └─────────┬─────────┘         │
              ▼                   ▼
      ┌───────────────┐    ┌───────────────┐
      │  agri_rag     │    │ agri_rag_     │
      │  文本集合      │    │ images        │
      │  Chroma       │    │ 图像集合       │
      └───────┬───────┘    └───────┬───────┘
              │                    │
     ┌────────┴────────┐          │
     ▼                 ▼          │
  稠密检索          稀疏检索       │
  (向量相似度)      (BM25)        │
     └────────┬────────┘          │
              ▼                   │
         RRF 融合                 │
              │                   │
         (可选) 精排              │
              │                   │
              └────────┬──────────┘
                       ▼
            ┌─────────────────────┐
            │   LangGraph Agent   │
            │   自主决策调用哪个   │
            │   工具回答问题       │
            └─────────┬───────────┘
                      ▼
              Streamlit 前端
              科研问答 · 图片检索 · 文档管理
```

### 关键设计选择

**为什么文本和图像分开两个集合？**

实验发现，文本向量和图像向量的分布差异很大。如果强行用 RRF 融合，图像结果会被文本稀释。让 Agent 根据问题语义自行决定调用 `knowledge_base_search`（文本）还是 `image_search`（图像），效果更稳定。

**为什么用结构感知切分？**

农业科研文档里有大量表格、公式和图表。传统按字符数切分会把表格从中间截断，破坏信息完整性。`StructureAwareChunker` 识别 Markdown 元素类型，表格/图表作为原子块整体输出，标题作为上下文拼入后续 chunk。

**为什么支持本地推理？**

农业科研数据可能涉及未公开的实验数据和品种信息。本地推理模式让整个 RAG 流水线完全离线运行，数据不出本地环境。

---

## 模型配置

| 用途 | 默认模型 | 备选 |
|------|---------|------|
| 问答推理（LLM） | `qwen3.6-plus`（API） | `qwen3-8b`（本地） |
| 视觉理解 | `qwen3-vl-plus`（API） | — |
| 文本向量化 | `Qwen3-Embedding-0.6B`（本地） | `text-embedding-v4`（API） |
| 图像向量化 | `Qwen3-VL-Embedding-2B`（本地固定） | — |
| 结果精排 | `Qwen3-Reranker-0.6B`（本地） | `qwen3-rerank`（API） |

关键环境变量：

```env
LLM_MODE=api                    # api（DashScope）或 local
EMBEDDING_MODE=local            # local 或 api
RERANKER_MODE=local             # local 或 api
ENABLE_RERANK=true              # false 可关闭精排降低延迟
CHUNK_STRATEGY=semantic         # 切分策略
```

---

## Agent 工具

Agent 可以调用以下工具，根据问题自主选择检索策略：

### 检索类

| 工具 | 功能 |
|------|------|
| `knowledge_base_search` | 文本混合检索（稠密+稀疏+RRF+精排） |
| `knowledge_base_search_with_filter` | 限定在特定文件内检索 |
| `image_search` | 跨模态图像检索（文本→图片） |
| `query_rewrite` | 复杂问题拆解为多个子查询 |
| `multi_round_search` | 多个子查询依次检索并合并去重 |

### 辅助类

| 工具 | 功能 |
|------|------|
| `calculator` | 安全数学表达式求值 |
| `web_search` | 实时网络搜索（知识库无结果时补充） |
| `table_analyzer` | Markdown/CSV 表格统计分析 |
| `image_describer` | 视觉模型描述图像内容 |
| `summarizer` | 长文本摘要压缩 |

---

## 前端功能

Streamlit 前端提供五个页面：

| 页面 | 说明 |
|------|------|
| **科研问答** | 对话式问答，实时展示 Agent 思考过程，支持多轮对话和来源引用 |
| **图片检索** | 输入描述文字检索相关图片，画廊式展示，显示相似度分数 |
| **科研文档浏览** | 分页浏览所有已入库文档，支持展开查看和批量删除 |
| **文档入库** | 拖拽上传 PDF/TXT/MD/PNG/JPG，自动解析入库 |
| **系统管理** | 健康状态监控、统计数据、重置知识库 |

---

## API 接口速查

后端基于 FastAPI，启动后访问 `http://localhost:8000/docs` 查看交互式文档。

```bash
# 健康检查
curl http://localhost:8000/health

# 提问
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "水稻适宜的播种温度是多少？"}'

# 流式问答（实时推送工具调用过程）
curl -N -X POST http://localhost:8000/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "小麦有哪些主要病害？"}'

# 上传文档
curl -X POST http://localhost:8000/v1/ingest/upload \
  -F "files=@experiment_report.pdf"

# 跨模态图片检索
curl -X POST http://localhost:8000/v1/images/search \
  -H "Content-Type: application/json" \
  -d '{"query": "玉米果穗剖面图", "top_k": 5}'

# 文档 CRUD
curl http://localhost:8000/v1/documents?offset=0&limit=20
curl -X DELETE http://localhost:8000/v1/documents/{doc_id}
```

### SSE 流式事件格式

```
data: {"type": "tool_start", "tool": "knowledge_base_search", ...}
data: {"type": "tool_done",  "tool": "knowledge_base_search", "elapsed_ms": 312, ...}
data: {"type": "answer", "answer": "...", "sources": [...]}
data: [DONE]
```

---

## 评估与调优

### 检索评估

```bash
# 离线评估（BM25，无需 API）
python scripts/evaluate.py \
  --dataset ./data/eval_dataset.jsonl \
  --corpus ./data/eval_corpus \
  --k 1 3 5 10

# 在线评估（完整流水线）
python scripts/evaluate.py --online --dataset ./data/eval_dataset.jsonl
```

### 切分策略对比

```bash
# 单策略评估
python scripts/evaluate.py --chunk-ablation markdown

# 全策略对比（输出汇总表）
python scripts/evaluate.py --chunk-ablation all
```

内置六种策略：`fixed` | `sentence` | `paragraph` | `markdown` | `recursive` | `semantic`（默认）

### 基准结果

173 条问答对，`chunk_size=1024`，`semantic` 切分：

| 方法 | Recall@1 | Recall@5 | MRR@5 | NDCG@5 |
|------|----------|----------|-------|--------|
| 混合检索（Dense+Sparse+RRF） | 76.30% | 94.80% | 0.8357 | 0.8638 |
| 混合检索 + Reranker 精排 | **86.71%** | **95.95%** | **0.9094** | **0.9223** |

---

## 目录结构

```
agricultural-research-rag/
├── config/settings.py              # 全局配置（pydantic-settings）
├── src/
│   ├── document_parser/            # 文档解析层
│   │   ├── text_parser.py          #   Markdown 结构化解析
│   │   ├── vision_parser.py        #   Qwen-VL 图像转文本
│   │   ├── chunker.py              #   结构感知切分（6种策略）
│   │   └── router.py               #   文件类型路由
│   ├── embeddings/                 # 向量化层
│   │   ├── qwen_embedding.py       #   文本 Embedding（本地/API）
│   │   └── qwen_vl_embedding.py    #   跨模态 Embedding（本地）
│   ├── retrieval/                  # 检索层
│   │   ├── dense_retriever.py      #   Chroma 稠密检索
│   │   ├── sparse_retriever.py     #   BM25 稀疏检索
│   │   ├── hybrid_retriever.py     #   RRF 融合 + 精排编排
│   │   └── reranker.py             #   Qwen3-Reranker
│   ├── vectorstore/                # 存储层
│   │   ├── chroma_store.py         #   文本向量库
│   │   └── image_store.py          #   图像向量库
│   ├── agent/                      # Agent 层
│   │   ├── rag_agent.py            #   LangGraph ReAct Agent
│   │   ├── tools.py                #   检索工具定义
│   │   └── skills/                 #   扩展技能（计算/搜索/分析）
│   └── api/                        # API 层
│       ├── main.py                 #   FastAPI 应用
│       └── schemas.py              #   请求/响应模型
├── frontend/
│   ├── app.py                      # Streamlit 前端
│   └── style.css                   # 自定义样式
├── scripts/                        # 数据处理脚本
│   ├── mineru_ocr.py               #   MinerU OCR 预处理
│   ├── ingest_ocr_output.py        #   文本入库
│   ├── ingest_images_from_md.py    #   图片入库
│   └── evaluate.py                 #   检索评估
├── tests/                          # 单元测试
├── run_server.py                   # 后端启动入口
├── launch.sh                       # 一键启动脚本
└── requirements.txt
```

---

## 测试

```bash
pytest tests/ -v

# VL embedding 环境诊断
python tests/diagnose_vl_embedding.py
```

---

## 技术栈

- **LLM**: Qwen3 系列（DashScope API / 本地推理）
- **向量数据库**: ChromaDB
- **Agent 框架**: LangGraph (ReAct)
- **Embedding**: Qwen3-Embedding-0.6B + Qwen3-VL-Embedding-2B
- **OCR**: MinerU
- **后端**: FastAPI + SSE
- **前端**: Streamlit
- **稀疏检索**: BM25 + RRF 融合
