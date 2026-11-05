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
python scripts/mineru_ocr.py --input ./data/documents

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

> 架构取舍、实测数据与优化路线的完整说明见 [`docs/技术亮点与项目总结.md`](docs/技术亮点与项目总结.md)。

### 关键设计选择

**为什么文本和图像分开两个集合？**

实验发现，文本向量和图像向量的分布差异很大。如果强行用 RRF 融合，图像结果会被文本稀释。让 Agent 根据问题语义自行决定调用 `knowledge_base_search`（文本）还是 `image_search`（图像），效果更稳定。

**为什么用结构感知切分？**

农业科研文档里有大量表格、公式和图表。传统按字符数切分会把表格从中间截断，破坏信息完整性。`StructureAwareChunker` 识别 Markdown 元素类型，表格/图表作为原子块整体输出，标题作为上下文拼入后续 chunk。

**为什么支持本地推理？**

农业科研数据可能涉及未公开的实验数据和品种信息。本地推理模式让整个 RAG 流水线完全离线运行，数据不出本地环境。

### 农业元数据与结构化检索（P0）

入库时为每个 chunk 抽取农业元数据，检索时先用问题中的明确条件收窄范围，再走稠密 + 稀疏融合：

| 字段 | 示例 | 用途 |
|------|------|------|
| `crop` | 水稻 | 作物过滤 |
| `disease` / `pest` | 稻瘟病 / 二化螟 | 病害、虫害过滤 |
| `pesticide` | 三环唑 | 农药过滤 |
| `region` | 长江中下游 | 适用区域过滤 |
| `growth_stage` | 分蘖期 | 生育期过滤 |
| `variety` | 郑单958 | 品种过滤 |
| `year` | 2021 | 年份下限过滤 |
| `doc_type` / `authority_level` | 标准 / 1 | 资料类型与权威等级过滤 |

- 抽取完全基于规则，离线可跑，不调用 LLM，也不新增外部 API 依赖。
- 别名自动归一（稻热病 → 稻瘟病、土豆 → 马铃薯），保证 Chroma 过滤能精确命中。
- 过滤条件同时作用于 Chroma `where` 和内存 BM25；条件过严导致零召回时自动放宽，并在工具返回中标记 `relaxed=true`。
- 问题里没有明确出现的条件不会加入过滤，避免过度约束。
- 开关：`ENABLE_AGRI_METADATA=true`（默认）。旧索引没有农业元数据也能正常检索，重新入库后过滤收益更明显。

```bash
# 查询理解示例：自动得到 crop=水稻、disease=稻瘟病、region=长江中下游、year_min=2024
python -c "from src.agri import understand_query; print(understand_query('近三年长江中下游水稻稻热病怎么防治').as_dict())"
```

#### 给旧索引回填农业元数据

在农业元数据上线之前入库的 chunk 没有这些字段，带条件的检索只会「命中 0 条 → 放宽为全库」。
回填脚本按 `source` 把 chunk 归并成文档重新抽取，**只更新 metadata，不重算向量、不删除文档**：

```bash
# 默认 dry-run，只打印统计与样例
python scripts/backfill_agri_metadata.py

# 真正写回（建议先停止后端，避免 Chroma 写锁争用）
python scripts/backfill_agri_metadata.py --apply

# 覆盖已有农业字段 / 只处理前 N 条
python scripts/backfill_agri_metadata.py --apply --overwrite --limit 500

# 抽取规则变更后：覆盖并删除旧规则误标的字段（如 crop=棉花）
python scripts/backfill_agri_metadata.py --apply --overwrite --prune
```

脚本可反复执行（幂等）。回填后需**重启后端**，让内存 BM25 索引带上新元数据。

作物判定规则：
- 优先取标题（Markdown 一级标题或文件名）里的作物；正文频繁出现的轮作/间套作作物不会抢走主题作物；
- 「棉铃虫」「甜菜夜蛾」等多食性害虫名里的作物前缀不会被误判成 crop；
- 英文别名按整词匹配，避免 `tea` 命中 `instead` 这类子串误报。

#### 补充中文农业语料

`scripts/fetch_zh_literature.py` 直接抓取农业农村部「农事指导」栏目与历年「春耕技术指导」专栏里公开的技术方案 / 指导意见，存成带来源链接的 Markdown：

```bash
# 抓取到 data/documents/zh（默认每个列表页翻 2 页，最多 40 篇）
python scripts/fetch_zh_literature.py --pages 2 --max-docs 40

# 入库（注意：不要加 --reset，否则会清空整个集合）
python scripts/ingest_documents.py --input-dir ./data/documents/zh --chunk-strategy semantic
```

- 文件头保留「资料来源 / 原文链接 / 发布年份」，`_sources.json` 记录已抓 URL，重复运行自动跳过；
- 只抓公开发布的政府/事业单位技术资料，正文里的章节标题会转成二级标题，切分按章节边界进行。

#### 切换 embedding 模型后重刷向量

Chroma 不记录向量由哪个模型生成。把 `EMBEDDING_MODE` 从 `api` 换成 `local`（或反向）后，库里的旧向量和新查询向量不在同一空间，必须先整体重算：

```bash
python scripts/reembed_collection.py            # dry-run：只打印配置与集合规模
python scripts/reembed_collection.py --apply    # 只重算向量，不改内容、不删文档
```

重算期间建议停掉后端，完成后重启，让检索与 BM25 索引都加载新向量。

---

## 模型配置

| 用途 | 默认模型 | 备选 |
|------|---------|------|
| 问答推理（LLM） | `qwen3.8-2.4t-a95b`（API） | `qwen3-8b`（本地） |
| 视觉理解 | `qwen3-vl-plus`（API） | — |
| 文本向量化 | `Qwen3-Embedding-0.6B`（本地） | `text-embedding-v4`（API） |
| 图像向量化 | `Qwen3-VL-Embedding-2B`（本地固定） | — |
| 结果精排 | `Qwen3-Reranker-0.6B`（本地） | `qwen3-rerank`（API） |

关键环境变量：

```env
LLM_MODE=api                    # api（DashScope）或 local
EMBEDDING_MODE=local            # local 或 api
LOCAL_EMBEDDING_MODEL=F:/models/Qwen3-Embedding-0.6B   # 本地向量模型路径（mode=local 时必填）
RERANKER_MODE=local             # local 或 api
ENABLE_RERANK=true              # false 可关闭精排降低延迟
CHUNK_STRATEGY=semantic         # 切分策略
ENABLE_AGRI_METADATA=true       # 入库时抽取农业元数据（作物/病害/年份等）
```

---

## Agent 工具

Agent 可以调用以下工具，根据问题自主选择检索策略：

### 检索类

| 工具 | 功能 |
|------|------|
| `knowledge_base_search` | 文本混合检索（稠密+稀疏+RRF+精排），自动识别农业槽位并过滤 |
| `knowledge_base_search_structured` | 显式指定作物/病害/地区/年份/资料类型的结构化检索 |
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
# 中文农业评测集（在线，完整流水线；需要已建好的 Chroma 索引）
python scripts/evaluate.py --online \
  --dataset ./data/eval_dataset_zh.jsonl \
  --k 1 3 5 10

# 口语化问法（更接近真实用户）
python scripts/evaluate.py --online \
  --dataset ./data/eval_dataset_zh_paraphrase.jsonl \
  --k 1 3 5 10

# 消融：仅 Dense / 仅 Sparse
python scripts/evaluate.py --online --ablation dense \
  --dataset ./data/eval_dataset_zh_paraphrase.jsonl
python scripts/evaluate.py --online --ablation sparse \
  --dataset ./data/eval_dataset_zh_paraphrase.jsonl

# 打开农业元数据过滤（按问题自动抽取 crop/disease/growth_stage）
python scripts/evaluate.py --online --filters \
  --dataset ./data/eval_dataset_zh_paraphrase.jsonl
```

`data/eval_dataset.jsonl` 是早期的人事/技术模板，与当前农业语料不匹配，不要再作为基线。

### 中文农业评测集

三份文件，均以 `data/documents/zh/` 的 36 篇农业农村部方案为事实来源：

| 文件 | 条数 | 用途 |
|------|------|------|
| `data/eval_dataset_zh.jsonl` | 70 | 直述式问题，覆盖 35/36 篇文档，用于回归和分类型统计 |
| `data/eval_dataset_zh_paraphrase.jsonl` | 30 | 口语化问法，不带文档标题词，用于衡量真实用户检索效果 |
| `data/eval_dataset_zh_negative.jsonl` | 10 | 知识库外问题，用于检验“拒答而非编造”，不参与 Recall 统计 |

字段沿用 `evaluate.py` 的 `question / answer / source / page / category`：

- `source` 是真实 `.md` 文件名，评测按文件名（去扩展名）匹配；
- `category` 取值 `factual / summary / reasoning / comparison / paraphrase / negative`；
- 负例题额外带 `expected_behavior: "refuse"`，`source` 为空，不能直接跑 `evaluate.py`；
  应通过 `/v1/query` 逐条提问，检查回答是否明确说明“知识库未收录”。
- 负例题实测（2026-11）：模型会先说明“知识库未找到”，但随后仍可能用互联网资料补充
  （例如苹果腐烂病会给出刮治时期和药剂）。是否允许这种“标注来源的补充回答”属于产品策略，
  当前按“先声明未收录、不得把外部内容说成知识库依据”判定。
- `2026_玉米大豆水肥一体化单产提升技术方案.md` 因原文只剩表格单元格、语句不完整，未出题。

标注校验方式：每条答案都要求能从原文逐字定位（quote 子串匹配），答案中的数字还需在原文按顺序出现；70 条直述题全部通过，因此答案可追溯到具体方案。

### 基准结果

环境：562 个 chunk（其中中文 280 个），`semantic` 切分，`chunk_size=1024`，
本地 `Qwen/Qwen3-Embedding-0.6B`，`ENABLE_RERANK=false`，未加农业过滤。

**直述式（70 条）**

| 方法 | Recall@1 | Recall@3 | Recall@10 | MRR@10 | NDCG@10 | Precision@10 |
|------|----------|----------|-----------|--------|---------|--------------|
| 混合检索（Dense+Sparse+RRF） | 90.00% | 100.00% | 100.00% | 0.9500 | 0.9631 | 40.00% |
| 仅 Dense | **91.43%** | 100.00% | 100.00% | **0.9524** | **0.9646** | 40.00% |
| 仅 Sparse（BM25） | 82.86% | 100.00% | 100.00% | 0.9095 | 0.9330 | 30.29% |

**口语化（30 条）**

| 方法 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | NDCG@10 |
|------|----------|----------|----------|-----------|--------|---------|
| 混合检索（Dense+Sparse+RRF） | **63.33%** | **83.33%** | **93.33%** | **96.67%** | 0.7417 | 0.7970 |
| 仅 Dense | 56.67% | 76.67% | 90.00% | 96.67% | 0.7039 | 0.7682 |
| 仅 Sparse（BM25） | **63.33%** | 80.00% | 93.33% | 93.33% | **0.7444** | **0.7915** |

**农业元数据过滤（混合检索，开/关对比，`--filters` 复现）**

| 数据集 | Recall@1 关 → 开 | Recall@5 关 → 开 | Recall@10 关 → 开 | MRR@3 关 → 开 |
|--------|------------------|------------------|-------------------|---------------|
| 直述式 | 90.0% → 88.6% | 100% → 97.1% | 100% → 97.1% | 0.9500 → 0.9262 |
| 口语化 | 63.3% → 63.3% | 93.3% → 86.7% | 96.7% → 90.0% | 0.7111 → 0.7278 |

### 结果解读与优化方向

1. **直述式已饱和，口语化才是瓶颈。** 直述题在 K=3 全部命中，但口语化题 Recall@1 只有 63%，
   Recall@10 才到 97%：检索“找得到”，但“排不准”。口语题上 BM25（63.3%）反而强于 Dense（56.7%），
   说明真实问法更依赖“小麦/蓟马”这类字面词；Hybrid 的 Recall@5 最高（93.3%），融合主要改善中段排序。
   优化重点应放在 top-1 排序与词面归一，而不是继续扩召回。
2. **超长文档挤占 top-k。** `2026_小麦秋冬种技术明白纸.md` 占 99/562 个 chunk（17.6%），
   在 11 条非 top-1 口语题中有 9 条的 top-3 出现该文档。建议在 RRF 后加“每来源最多 N 块”配额或 MMR。
3. **近似重复版本互相抢答。** 直述题 7 条非 top-1 全部是版本或区域差异：
   2023/2024 番茄潜叶蛾、豇豆、棉花，2024/2025 果树冻害，以及“全国版 vs 西南版”大豆玉米，
   排名均为第 2。需要年份/区域仲裁（显式过滤或同版本优先），仅靠文本相似度无法区分。
4. **过滤条件宁少勿多。** 当前 `understand_query` 会抽出 `crop / disease / growth_stage`，
   而 chunk 的 `crop` 是单值主作物、`growth_stage` 也是单值。硬过滤 `growth_stage`（如“灌浆期”）
   或 `disease` 会让正确文档直接掉出 top-10（直述集 3 条、口语集 3 条）；口语集 MRR@3 反而略升
   （0.711 → 0.728），说明过滤提升了“排得准”，代价是“找得全”。
   建议：`crop` 改匹配多值 `crops` 字段，`growth_stage` 默认不参与硬过滤，空结果时逐槽位放松。
5. **别名与时令词仍有缺口。** `西红柿 → 番茄`、`豆角 → 豇豆` 已能正确归一；
   “送嫁肥”“闷棚”等农事俗语没有进入词典，口语题里仍会命中小麦/蔬菜通用文档。
6. **Reranker 尚未启用。** 上表 `ENABLE_RERANK=false`。top-1 的 37% 失败正是精排的用武之地，
   下一步可在口语集上对比开启前后的 MRR@1。

### 切分策略对比

```bash
# 单策略评估
python scripts/evaluate.py --chunk-ablation markdown

# 全策略对比（输出汇总表）
python scripts/evaluate.py --chunk-ablation all
```

内置六种策略：`fixed` | `sentence` | `paragraph` | `markdown` | `recursive` | `semantic`（默认）

---

## 目录结构

```
agricultural-research-rag/
├── config/
│   ├── settings.py                 # 全局配置（pydantic-settings）
│   └── agri_lexicon.json           # 农业词典（作物/病害/虫害/农药/地区/生育期）
├── src/
│   ├── agri/                       # 农业语义层（纯规则，无外部依赖）
│   │   ├── lexicon.py              #   别名归一 / 实体识别 / 分词增强
│   │   ├── metadata_extractor.py   #   文档与 chunk 元数据抽取
│   │   ├── filters.py              #   Chroma where + BM25 元数据过滤
│   │   └── query_understanding.py  #   查询槽位、意图与查询扩展
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
│   ├── backfill_agri_metadata.py   #   旧索引农业元数据回填（非破坏性）
│   └── evaluate.py                 #   检索评估
├── tests/                          # 单元测试
├── run_server.py                   # 后端启动入口
├── launch.sh                       # 一键启动脚本
└── requirements.txt
```

---

## 测试

```bash
# Windows 下必须指定 --basetemp，否则会因临时目录权限失败
pytest tests/ -q --basetemp=.pytest_cache/tmp

# VL embedding 环境诊断
python tests/diagnose_vl_embedding.py
```

当前共 238 个测试，重组件（LLM、Chroma、HTTP）均已 mock；其中 `tests/test_agri_*.py`
与 `tests/test_query_understanding.py` 是纯规则测试，不需要任何模型即可运行。

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
