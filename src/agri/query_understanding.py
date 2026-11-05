"""
查询理解（Query Understanding）
─────────────────────────────────────────────────────────────────────────────
在检索之前，把一句自然语言问题拆成结构化槽位：

    "近三年长江中下游水稻稻瘟病怎么防治？"
      → crop=水稻, disease=稻瘟病, region=长江中下游,
        year_from=2024, intent=control

用途
----
1. 元数据过滤：作物/病害/地区/年份/资料类型 → 直接缩小检索范围；
2. 查询扩展：补充规范名（稻热病 → 稻瘟病）与「作物+病害」组合；
3. 提示词注入：让 LLM 知道用户问题的适用条件，回答时主动说明边界。

设计取舍
--------
只使用规则，不调用 LLM：
  • 零延迟、零成本，适合每个请求都跑；
  • 只抽取「明确出现」的槽位，抽不到就不加过滤，避免过度约束；
  • 过滤过严时由检索层自动放宽（relaxed），保证不会查不到。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .filters import normalize_filters
from .lexicon import expand_query, find_entities
from .metadata_extractor import extract_variety_candidates

# 意图 → 中文说明
INTENT_LABELS: dict[str, str] = {
    "control": "防治/防控",
    "dosage": "用药剂量",
    "variety": "品种选择",
    "comparison": "对比",
    "yield": "产量",
    "mechanism": "机理/原理",
    "standard": "标准/限量",
    "general": "通用问答",
}

_DOC_TYPE_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("农药标签", "登记证", "标签"), "农药标签"),
    (("审定公告", "审定", "品种公告", "引种"), "审定公告"),
    (("技术规程", "规程", "栽培技术", "防治技术"), "技术规程"),
    (("标准", "国标", "行标", "NY/T", "GB/T", "限量", "残留"), "标准"),
    (("试验报告", "试验"), "试验报告"),
    (("论文", "文献", "研究进展"), "论文"),
)

_YEAR_FROM_RE = re.compile(
    r"(?:自|从)?\s*((?:19|20)\d{2})\s*(?:年)?\s*(?:以后|之后|以来|起|至今)"
)
_YEAR_RANGE_RE = re.compile(
    r"((?:19|20)\d{2})\s*(?:年)?\s*[-~—–至到]\s*((?:19|20)\d{2})\s*(?:年)?"
)
_RECENT_YEARS_RE = re.compile(r"(?:最近|近|过去)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:年|年来|年内)")

_CN_NUMBERS: dict[str, int] = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十五": 15, "二十": 20,
}

_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dosage", ("剂量", "用量", "亩用", "稀释", "配药", "打多少", "兑水", "喷雾倍数", "安全间隔期")),
    ("standard", ("标准", "限量", "残留", "国标", "行标", "NY/T", "GB/T", "登记证")),
    ("control", ("防治", "防控", "怎么治", "如何治", "打什么药", "用什么药", "打药", "喷药", "防治措施", "综合防治")),
    ("variety", ("品种", "推荐", "适合种", "引种", "选种", "主栽", "抗性品种", "什么品种")),
    ("comparison", ("对比", "比较", "区别", "差异", "哪个好", "优劣", "优势")),
    ("yield", ("产量", "增产", "减产", "产量构成", "千粒重", "穗粒数")),
    ("mechanism", ("机理", "原理", "为什么", "机制", "通路", "基因")),
)


@dataclass
class QueryUnderstanding:
    """一次查询的结构化理解结果。"""

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    intent: str = "general"
    year_from: int | None = None
    doc_type: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    expanded_queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def crop(self) -> str | None:
        return self._first("crops")

    @property
    def disease(self) -> str | None:
        return self._first("diseases")

    @property
    def pest(self) -> str | None:
        return self._first("pests")

    @property
    def region(self) -> str | None:
        return self._first("regions")

    def _first(self, category: str) -> str | None:
        values = self.entities.get(category, [])
        return values[0] if values else None

    @property
    def intent_label(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent)

    def as_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "intent": self.intent,
            "intent_label": self.intent_label,
            "year_from": self.year_from,
            "doc_type": self.doc_type,
            "filters": self.filters,
            "expanded_queries": self.expanded_queries,
            "notes": self.notes,
        }

    def prompt_context(self) -> str:
        """生成一行中文上下文，供工具返回或提示词注入。"""
        parts: list[str] = []
        labels = {
            "crops": "作物",
            "diseases": "病害",
            "pests": "虫害",
            "pesticides": "农药",
            "regions": "地区",
            "growth_stages": "生育期",
        }
        for category, label in labels.items():
            values = self.entities.get(category, [])
            if values:
                parts.append(f"{label}={'/'.join(values[:3])}")
        if self.doc_type:
            parts.append(f"资料类型={self.doc_type}")
        if self.year_from:
            parts.append(f"年份≥{self.year_from}")
        parts.append(f"意图={self.intent_label}")
        return "；".join(parts)


def _parse_cn_number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    return _CN_NUMBERS.get(text)


def _extract_year_from(query: str, now_year: int) -> int | None:
    recent = _RECENT_YEARS_RE.search(query)
    if recent:
        amount = _parse_cn_number(recent.group(1))
        if amount:
            return max(now_year - amount + 1, 1900)

    year_from = _YEAR_FROM_RE.search(query)
    if year_from:
        return int(year_from.group(1))

    year_range = _YEAR_RANGE_RE.search(query)
    if year_range:
        return int(year_range.group(1))

    return None


def _detect_doc_type(query: str) -> str | None:
    upper = query.upper()
    for keywords, doc_type in _DOC_TYPE_HINTS:
        if any(keyword.upper() in upper for keyword in keywords):
            return doc_type
    return None


def _detect_intent(query: str) -> str:
    for intent, keywords in _INTENT_RULES:
        if any(keyword in query for keyword in keywords):
            return intent
    return "general"


def understand_query(query: str, now: datetime | None = None) -> QueryUnderstanding:
    """把用户问题解析为结构化槽位，并生成过滤条件与扩展查询。"""
    query = (query or "").strip()
    now = now or datetime.now()

    entities = find_entities(query)
    intent = _detect_intent(query)
    year_from = _extract_year_from(query, now.year)
    doc_type = _detect_doc_type(query)

    raw_filters: dict[str, Any] = {}
    if entities.get("crops"):
        raw_filters["crop"] = entities["crops"][:2]
    if entities.get("diseases"):
        raw_filters["disease"] = entities["diseases"][:3]
    if entities.get("pests"):
        raw_filters["pest"] = entities["pests"][:3]
    if entities.get("pesticides"):
        raw_filters["pesticide"] = entities["pesticides"][:2]
    if entities.get("regions"):
        raw_filters["region"] = entities["regions"][:2]
    if entities.get("growth_stages"):
        raw_filters["growth_stage"] = entities["growth_stages"][:2]

    varieties = extract_variety_candidates(query, strict=False)
    if varieties:
        raw_filters["variety"] = varieties[0]

    if doc_type:
        raw_filters["doc_type"] = doc_type
    if year_from:
        raw_filters["year_min"] = year_from

    filters = normalize_filters(raw_filters)

    expansions = expand_query(query)
    if intent == "control" and (entities.get("diseases") or entities.get("pests")):
        target = (entities.get("diseases") or entities.get("pests"))[0]
        crop = (entities.get("crops") or [""])[0]
        base = target if crop and crop in target else f"{crop}{target}"
        combo = f"{base} 防治"
        if combo not in expansions and combo not in query:
            expansions.append(combo)

    notes: list[str] = []
    if filters:
        notes.append("已按查询中的明确条件收窄检索范围")
    if year_from:
        notes.append(f"时间范围：{year_from} 年及以后")

    return QueryUnderstanding(
        query=query,
        entities=entities,
        intent=intent,
        year_from=year_from,
        doc_type=doc_type,
        filters=filters,
        expanded_queries=expansions[:4],
        notes=notes,
    )
