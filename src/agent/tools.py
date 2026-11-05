"""
RAG Agent 的 LangChain 工具定义
──────────────────────────────────────────────────────────────────────────────
工具将检索子系统暴露给 LangChain Agent 循环。
Agent 在生成答案前可调用这些工具获取上下文。

工具清单
--------
- knowledge_base_search            : 农业科研文本知识库检索（稠密 + 稀疏 + 精排）
                                     自动识别作物/病害/地区/年份并收窄检索范围
- knowledge_base_search_structured : 由 Agent 显式指定农业条件的结构化检索
- knowledge_base_search_with_filter: 限定来源文件的文本检索
- image_search                     : 农业图像库跨模态检索（文本 query → 相关图片）
- query_rewrite                    : 将复杂问题改写为多个子查询（需要 LLM）
- multi_round_search               : 对多个子查询依次检索并合并去重结果（文本）
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from loguru import logger

from config.settings import get_settings
from src.agri.filters import describe_filters, normalize_filters
from src.agri.query_understanding import QueryUnderstanding, understand_query

if TYPE_CHECKING:
    from src.retrieval.hybrid_retriever import HybridRetriever

_settings = get_settings()


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _retrieve_documents(
    retriever: "HybridRetriever",
    query: str,
    filters: dict[str, Any] | None,
) -> tuple[list, bool]:
    """
    执行带农业元数据过滤的检索，返回 (documents, relaxed)。

    优先使用 HybridRetriever.retrieve_with_info() 获取「过滤是否被放宽」；
    对测试替身或其他检索器，退化为普通 retrieve() 调用。
    """
    filters = normalize_filters(filters)
    if not filters:
        return retriever.retrieve(query), False

    supports_info = getattr(type(retriever), "retrieve_with_info", None) is not None
    if supports_info:
        outcome = retriever.retrieve_with_info(query, filters=filters)
        return outcome.documents, outcome.relaxed

    try:
        docs = retriever.retrieve(query, filters=filters)
    except TypeError:
        docs = retriever.retrieve(query)

    if docs:
        return docs, False

    # 过滤后为空：放宽一次，避免「有资料但条件太窄」导致空回答
    try:
        return retriever.retrieve(query), True
    except TypeError:
        return docs, True


def _doc_to_result(
    doc: Any,
    rank: int,
    matched_query: str | None = None,
) -> dict[str, Any]:
    """把 Document 转成工具返回的统一结果结构（含农业元数据）。"""
    metadata = doc.metadata or {}
    result: dict[str, Any] = {
        "rank": rank,
        "content": doc.page_content,
        "source": metadata.get("source", "unknown"),
        "page": metadata.get("page_num", "?"),
        "block_type": metadata.get("block_type", "text"),
        "header_context": metadata.get("header_context", ""),
    }
    for field in (
        "crop", "disease", "pest", "pesticide",
        "region", "growth_stage", "variety", "year", "doc_type",
    ):
        if metadata.get(field) not in (None, ""):
            result[field] = metadata[field]
    if matched_query:
        result["matched_query"] = matched_query
    return result


def _search_payload(
    results: list[dict[str, Any]],
    understanding: QueryUnderstanding | None = None,
    relaxed: bool = False,
    empty_message: str = "未找到相关内容。",
) -> str:
    """组装检索工具的 JSON 返回，附上查询理解与过滤状态。"""
    payload: dict[str, Any] = {"results": results}

    if not results:
        payload["message"] = empty_message

    if understanding is not None:
        payload["query_understanding"] = understanding.as_dict()
        if understanding.filters:
            payload["applied_filters"] = understanding.filters
            payload["filter_description"] = describe_filters(understanding.filters)

    if relaxed:
        payload["relaxed"] = True
        payload["relaxation_note"] = (
            "严格条件下未命中，已放宽为全库检索；回答时应提示用户"
            "「未找到完全匹配的资料，以下为相近内容」。"
        )

    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 工厂函数——在启动时绑定检索器，返回可用工具列表
# ---------------------------------------------------------------------------

def build_retrieval_tools(retriever: "HybridRetriever", llm: Any = None) -> list:
    """
    返回绑定到指定 HybridRetriever 的 LangChain 工具列表。

    参数
    ----
    retriever : 已初始化的 HybridRetriever。
    llm       : 可选，LangChain LLM 实例。提供后会额外注册
                query_rewrite 工具，用于复杂问题的查询改写。

    使用工厂模式（而非模块级全局变量）的原因：
    保持工具的可测试性，避免导入时的副作用。
    """

    @tool
    def knowledge_base_search(query: str) -> str:
        """
        在农业科研文本知识库中检索与查询相关的段落（不含图片）。

        当需要从已上传的农业科研文档中获取事实性文本信息时使用此工具。
        会自动识别问题中的作物、病害、虫害、农药、地区、年份和资料类型，
        并据此收窄检索范围；若条件过严未命中，会自动放宽并在结果中标记
        relaxed=true。

        参数
        ----
        query : 自然语言问题或搜索短语。

        返回
        ----
        包含来源元数据、农业元数据（作物/病害/年份/资料类型）与
        query_understanding（槽位识别结果）的 JSON。
        """
        logger.info(f"[Tool:knowledge_base_search] query='{query[:80]}'")

        understanding = understand_query(query)
        if understanding.filters:
            logger.info(
                f"[Tool:knowledge_base_search] 过滤条件："
                f"{describe_filters(understanding.filters)}"
            )

        docs, relaxed = _retrieve_documents(retriever, query, understanding.filters)
        results = [_doc_to_result(doc, i) for i, doc in enumerate(docs, start=1)]
        return _search_payload(results, understanding, relaxed)

    @tool
    def knowledge_base_search_structured(
        query: str,
        crop: str = "",
        disease: str = "",
        pest: str = "",
        pesticide: str = "",
        region: str = "",
        growth_stage: str = "",
        doc_type: str = "",
        variety: str = "",
        year_from: str = "",
    ) -> str:
        """
        按显式农业条件检索文本知识库。

        当用户明确给出「作物 + 病害/虫害 + 地区 + 年份 + 资料类型」等条件时，
        用此工具比 knowledge_base_search 更精确。所有参数均可留空，
        留空时自动回退到问题本身的槽位识别结果。

        参数
        ----
        query        : 自然语言问题。
        crop         : 作物，如「水稻」「小麦」。
        disease      : 病害，如「稻瘟病」。
        pest         : 虫害，如「二化螟」。
        pesticide    : 农药，如「吡虫啉」。
        region       : 地区，如「长江中下游」「黑龙江」。
        growth_stage : 生育期，如「分蘖期」「抽穗期」。
        doc_type     : 资料类型：标准/审定公告/农药标签/技术规程/论文/试验报告。
        variety      : 品种，如「郑单958」。
        year_from    : 起始年份（字符串或数字），如「2020」。
        """
        logger.info(
            f"[Tool:knowledge_base_search_structured] query='{query[:60]}' "
            f"crop={crop} disease={disease} region={region} doc_type={doc_type}"
        )

        raw_filters: dict[str, Any] = {
            "crop": crop,
            "disease": disease,
            "pest": pest,
            "pesticide": pesticide,
            "region": region,
            "growth_stage": growth_stage,
            "doc_type": doc_type,
            "variety": variety,
        }
        if str(year_from).strip():
            raw_filters["year_min"] = year_from

        filters = normalize_filters(raw_filters)
        understanding: QueryUnderstanding | None = None
        if not filters:
            understanding = understand_query(query)
            filters = understanding.filters
        else:
            understanding = understand_query(query)
            # 显式条件优先；仅在问题明确给出时间/资料类型时补充
            for key in ("year_min", "doc_type"):
                if key in understanding.filters and key not in filters:
                    filters[key] = understanding.filters[key]
            understanding.filters = filters

        docs, relaxed = _retrieve_documents(retriever, query, filters)
        results = [_doc_to_result(doc, i) for i, doc in enumerate(docs, start=1)]
        return _search_payload(results, understanding, relaxed)

    @tool
    def knowledge_base_search_with_filter(query: str, source_file: str) -> str:
        """
        在农业科研知识库中检索，结果限定在指定的源文档范围内。

        当用户询问特定文件或文档时使用此工具。

        参数
        ----
        query       : 自然语言问题。
        source_file : 限定检索的文件名（例如 'annual_report.pdf'）。
        """
        logger.info(
            f"[Tool:filtered_search] query='{query[:60]}' filter='{source_file}'"
        )
        docs = retriever.dense.retrieve(
            query, filter={"source": {"$eq": source_file}}
        )

        if not docs:
            return json.dumps(
                {
                    "results": [],
                    "message": f"在 '{source_file}' 中未找到相关内容。",
                },
                ensure_ascii=False,
            )

        results = [
            _doc_to_result(doc, i)
            for i, (doc, _) in enumerate(docs[:5], start=1)
        ]
        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    @tool
    def multi_round_search(queries: str) -> str:
        """
        对多个查询依次检索知识库，合并并去重结果。

        通常与 query_rewrite 配合使用：
        先调用 query_rewrite 获得子查询列表，
        再将 rewritten_queries 序列化为 JSON 字符串传入此工具。
        每个子查询都会自动做农业槽位识别与元数据过滤。

        参数
        ----
        queries : JSON 编码的查询字符串列表，
                  例如 '["子查询1", "子查询2", "子查询3"]'。
        """
        logger.info(f"[Tool:multi_round_search] queries='{queries[:120]}'")

        try:
            query_list = json.loads(queries)
            if not isinstance(query_list, list):
                query_list = [str(query_list)]
        except json.JSONDecodeError:
            # 容错：当作单个普通查询处理
            query_list = [queries]

        seen_keys: set[str] = set()
        merged: list[dict] = []
        rank = 1
        relaxed_queries: list[str] = []

        for q in query_list:
            if not q or not q.strip():
                continue

            understanding = understand_query(q)
            docs, relaxed = _retrieve_documents(retriever, q, understanding.filters)
            if relaxed:
                relaxed_queries.append(q)

            for doc in docs:
                # 用内容前 120 字符作去重 key，避免重复片段
                dedup_key = doc.page_content[:120]
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                merged.append(_doc_to_result(doc, rank, matched_query=q))
                rank += 1

        if not merged:
            return json.dumps(
                {"results": [], "message": "多轮检索未找到相关内容。"},
                ensure_ascii=False,
            )

        payload: dict[str, Any] = {
            "results": merged,
            "total_queries": len(query_list),
        }
        if relaxed_queries:
            payload["relaxed_queries"] = relaxed_queries
            payload["relaxation_note"] = (
                "部分子查询在严格条件下未命中，已放宽为全库检索。"
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)

    _image_top_k_default = _settings.image_top_k

    @tool
    def image_search(query: str, top_k: int = _image_top_k_default) -> str:
        """
        在农业图像知识库中用自然语言检索最相关的图片（跨模态检索）。

        当用户问题涉及图表、示意图、截图、曲线、架构图等视觉内容，
        或者需要展示某个概念对应的图片时使用此工具。
        底层使用 Qwen3-VL-Embedding 将文本与图片映射到同一向量空间。

        参数
        ----
        query  : 描述要找的图片的自然语言（例如「模型架构图」「训练损失曲线」）。
        top_k  : 返回图片数量（默认来自 settings.image_top_k，建议 3~10）。
        """
        logger.info(f"[Tool:image_search] query='{query[:80]}' top_k={top_k}")

        pairs = retriever.retrieve_images(query, k=top_k)
        if not pairs:
            return json.dumps(
                {"results": [], "message": "图片库为空或未找到相关图片。"},
                ensure_ascii=False,
            )

        results = []
        for rank, (doc, score) in enumerate(pairs, start=1):
            meta = doc.metadata or {}
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 4),
                    "source_path": meta.get("source_path", ""),
                    "file_name": meta.get("file_name", ""),
                    "alt_text": meta.get("alt_text", ""),
                    "caption": meta.get("caption", ""),
                    "source": meta.get("source", ""),
                    "page": str(meta.get("page_num", "-")),
                    "block_type": "figure",
                }
            )

        return json.dumps({"results": results}, ensure_ascii=False, indent=2)

    tools = [
        knowledge_base_search,
        knowledge_base_search_structured,
        knowledge_base_search_with_filter,
        multi_round_search,
        image_search,
    ]

    # query_rewrite 依赖 LLM，仅在提供时注册
    if llm is not None:

        @tool
        def query_rewrite(query: str) -> str:
            """
            将复杂问题改写为多个不同角度的检索子查询，提升农业科研知识库检索命中率。

            适用场景：
            - 问题涉及多个实体或概念
            - 问题表述模糊或过于口语化
            - 单次检索结果不理想时

            使用流程：
            1. 调用此工具生成 rewritten_queries 列表。
            2. 将列表序列化为 JSON 字符串，传给 multi_round_search。
            3. 基于合并后的检索结果生成答案。

            参数
            ----
            query : 原始用户问题。
            """
            logger.info(f"[Tool:query_rewrite] 原始问题='{query[:80]}'")

            prompt = (
                "请将以下农业科研问题改写为3个不同角度的检索查询，"
                "以提高农业知识库检索命中率。\n"
                f"原始问题：{query}\n\n"
                "要求：\n"
                "1. 每个查询从不同角度或使用不同关键词表达相同需求\n"
                "2. 保留作物、病害、虫害、农药、品种、地区等专业术语，"
                "必要时补充规范名或同义名（如 稻热病→稻瘟病）\n"
                "3. 保持语义一致，不引入无关内容\n"
                "4. 查询应简洁，适合关键词检索\n\n"
                '严格以 JSON 格式返回，例如：{"queries": ["查询1", "查询2", "查询3"]}\n'
                "只输出 JSON，不要任何其他内容。"
            )

            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.strip()

                # 兼容 LLM 输出 markdown 代码块的情况
                if "```" in content:
                    parts = content.split("```")
                    # 取第一个代码块内容
                    content = parts[1].lstrip("json").strip()

                data = json.loads(content)
                rewritten: list[str] = data.get("queries", [])
                if not rewritten:
                    raise ValueError("queries 列表为空")
            except Exception as exc:
                logger.warning(f"[Tool:query_rewrite] 改写失败，回退至原始问题：{exc}")
                rewritten = [query]

            logger.info(f"[Tool:query_rewrite] 生成 {len(rewritten)} 个子查询")
            return json.dumps(
                {"original_query": query, "rewritten_queries": rewritten},
                ensure_ascii=False,
            )

        tools.append(query_rewrite)

    return tools
