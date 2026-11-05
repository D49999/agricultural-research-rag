"""
农业领域语义层
═══════════════════════════════════════════════════════════════════════════════
将「农业」从一组通用文本，变成可识别、可过滤、可改写的领域对象。

模块划分
--------
- lexicon            : 领域词典、别名归一、实体识别、分词增强
- metadata_extractor : 文档 / chunk 级农业元数据抽取
- filters            : 元数据过滤条件的规范化、Chroma where 构造与本地匹配
- query_understanding: 查询槽位抽取（作物/病害/地区/年份/意图）与查询扩展

设计原则
--------
1. 纯规则、无外部 API：入库和检索都可离线运行，成本为零。
2. 规范名唯一：所有别名最终落到一个 canonical 名称，保证 Chroma 过滤可精确匹配。
3. 保守输出：抽不到就留空，不用 LLM 猜，避免污染元数据。
"""
from .filters import (
    build_chroma_where,
    describe_filters,
    metadata_matches,
    normalize_filters,
)
from .lexicon import (
    CATEGORY_FIELD,
    AgriLexicon,
    EntityHit,
    expand_query,
    find_entities,
    find_hits,
    get_lexicon,
    normalize_term,
    resolve_value,
    tokenize,
)
from .metadata_extractor import (
    AUTHORITY_LEVEL,
    DOC_TYPES,
    classify_document,
    extract_chunk_metadata,
    extract_document_metadata,
    extract_variety_candidates,
    extract_year,
    format_metadata,
)
from .query_understanding import QueryUnderstanding, understand_query

__all__ = [
    "AUTHORITY_LEVEL",
    "CATEGORY_FIELD",
    "DOC_TYPES",
    "AgriLexicon",
    "EntityHit",
    "QueryUnderstanding",
    "build_chroma_where",
    "describe_filters",
    "expand_query",
    "extract_chunk_metadata",
    "extract_document_metadata",
    "extract_variety_candidates",
    "extract_year",
    "classify_document",
    "find_entities",
    "find_hits",
    "format_metadata",
    "get_lexicon",
    "metadata_matches",
    "normalize_filters",
    "normalize_term",
    "resolve_value",
    "tokenize",
    "understand_query",
]
