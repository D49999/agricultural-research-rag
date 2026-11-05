"""
元数据过滤条件
─────────────────────────────────────────────────────────────────────────────
同一份过滤条件需要同时服务两条检索路径：

  • Chroma 稠密检索：翻译为 Chroma `where` 语法；
  • BM25 稀疏检索：在 Python 侧对 metadata 做匹配（BM25 索引是内存的，
    无法复用 Chroma 过滤）。

因此这里定义一套与存储无关的规范化过滤条件：

    {"crop": ["水稻"], "disease": ["稻瘟病"], "year_min": 2020}

值可以是单个字符串/整数，也可以是列表（列表内为 OR）。
"""
from __future__ import annotations

from typing import Any

from .lexicon import normalize_term, resolve_value

ENTITY_FIELDS: tuple[str, ...] = (
    "crop",
    "disease",
    "pest",
    "pesticide",
    "region",
    "growth_stage",
)
EXACT_FIELDS: tuple[str, ...] = ENTITY_FIELDS + ("variety", "doc_type", "source")

# 过滤字段名 → 词典类别名（lexicon 使用复数类别键）
_FIELD_TO_CATEGORY: dict[str, str] = {
    "crop": "crops",
    "disease": "diseases",
    "pest": "pests",
    "pesticide": "pesticides",
    "region": "regions",
    "growth_stage": "growth_stages",
}

_DOC_TYPE_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("标准", "标准"),
    ("国标", "标准"),
    ("行标", "标准"),
    ("审定", "审定公告"),
    ("公告", "审定公告"),
    ("标签", "农药标签"),
    ("登记证", "农药标签"),
    ("规程", "技术规程"),
    ("论文", "论文"),
    ("文献", "论文"),
    ("试验", "试验报告"),
    ("科普", "科普文章"),
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item not in (None, "")]
    if isinstance(value, str):
        # 允许调用方传 "水稻,小麦"
        parts = [part.strip() for part in value.replace("，", ",").split(",")]
        return [part for part in parts if part]
    return [value]


def normalize_doc_type(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for keyword, canonical in _DOC_TYPE_SYNONYMS:
        if keyword in text:
            return canonical
    return text


def normalize_filters(raw: dict[str, Any] | None) -> dict[str, Any]:
    """规范化过滤条件：去空、别名归一、年份转 int。"""
    if not raw:
        return {}

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in raw.items():
        key = {
            "year_from": "year_min",
            "year_to": "year_max",
            "year": "year_min",
        }.get(raw_key, raw_key)

        values = _as_list(raw_value)
        if not values:
            continue

        if key in ENTITY_FIELDS:
            category = _FIELD_TO_CATEGORY.get(key)
            resolved = []
            for value in values:
                canonical = resolve_value(str(value), category)
                if canonical and canonical not in resolved:
                    resolved.append(canonical)
            if resolved:
                normalized[key] = resolved[0] if len(resolved) == 1 else resolved
        elif key == "variety":
            varieties = [str(v).strip() for v in values if str(v).strip()]
            if varieties:
                normalized[key] = varieties[0] if len(varieties) == 1 else varieties
        elif key == "doc_type":
            doc_types = []
            for value in values:
                canonical = normalize_doc_type(str(value))
                if canonical and canonical not in doc_types:
                    doc_types.append(canonical)
            if doc_types:
                normalized[key] = doc_types[0] if len(doc_types) == 1 else doc_types
        elif key == "source":
            sources = [str(v).strip() for v in values if str(v).strip()]
            if sources:
                normalized[key] = sources[0] if len(sources) == 1 else sources
        elif key in ("year_min", "year_max", "authority_max"):
            try:
                normalized[key] = int(values[0])
            except (TypeError, ValueError):
                continue
    return normalized


def build_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """把规范化过滤条件翻译成 Chroma `where` 语法；空条件返回 None。"""
    filters = normalize_filters(filters)
    if not filters:
        return None

    clauses: list[dict[str, Any]] = []

    for field in EXACT_FIELDS:
        if field not in filters:
            continue
        values = _as_list(filters[field])
        if not values:
            continue
        if len(values) == 1:
            clauses.append({field: {"$eq": values[0]}})
        else:
            clauses.append({field: {"$in": values}})

    year_clause: dict[str, Any] = {}
    if "year_min" in filters:
        year_clause["$gte"] = filters["year_min"]
    if "year_max" in filters:
        year_clause["$lte"] = filters["year_max"]
    if year_clause:
        clauses.append({"year": year_clause})

    if "authority_max" in filters:
        clauses.append({"authority_level": {"$lte": filters["authority_max"]}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _value_equal(actual: Any, expected: Any) -> bool:
    if actual is None:
        return False
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    actual_text = normalize_term(str(actual))
    expected_text = normalize_term(str(expected))
    if not actual_text or not expected_text:
        return False
    if actual_text == expected_text:
        return True
    # 兼容旧数据把多值存成 "水稻, 小麦" 的情况
    return expected_text in {
        normalize_term(part) for part in str(actual).replace("，", ",").split(",")
    }


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metadata_matches(
    metadata: dict[str, Any] | None,
    filters: dict[str, Any] | None,
    pre_normalized: bool = False,
) -> bool:
    """BM25 侧过滤：判断一条文档的 metadata 是否满足全部过滤条件。"""
    filters = filters if pre_normalized else normalize_filters(filters)
    if not filters:
        return True
    metadata = metadata or {}

    for field in EXACT_FIELDS:
        if field not in filters:
            continue
        expected_values = _as_list(filters[field])
        actual = metadata.get(field)
        # 兼容只写了复数聚合字段的旧 chunk（如 diseases="稻瘟病, 稻曲病"）
        if actual is None and field in ENTITY_FIELDS:
            actual = metadata.get(f"{field}s")
        if not any(_value_equal(actual, expected) for expected in expected_values):
            return False

    if "year_min" in filters or "year_max" in filters:
        actual_year = _to_int(metadata.get("year"))
        if actual_year is None:
            return False
        if "year_min" in filters and actual_year < filters["year_min"]:
            return False
        if "year_max" in filters and actual_year > filters["year_max"]:
            return False

    if "authority_max" in filters:
        actual_level = _to_int(metadata.get("authority_level"))
        if actual_level is None or actual_level > filters["authority_max"]:
            return False

    return True


def describe_filters(filters: dict[str, Any] | None) -> str:
    """把过滤条件转成一句中文，用于工具返回与前端展示。"""
    filters = normalize_filters(filters)
    if not filters:
        return "无"

    labels = {
        "crop": "作物",
        "disease": "病害",
        "pest": "虫害",
        "pesticide": "农药",
        "region": "地区",
        "growth_stage": "生育期",
        "variety": "品种",
        "doc_type": "资料类型",
        "source": "来源文件",
    }
    parts: list[str] = []
    for field, label in labels.items():
        if field in filters:
            values = _as_list(filters[field])
            parts.append(f"{label}={'/'.join(str(v) for v in values)}")

    if "year_min" in filters:
        parts.append(f"年份≥{filters['year_min']}")
    if "year_max" in filters:
        parts.append(f"年份≤{filters['year_max']}")
    if "authority_max" in filters:
        parts.append(f"权威等级≤{filters['authority_max']}")
    return "；".join(parts) if parts else "无"
