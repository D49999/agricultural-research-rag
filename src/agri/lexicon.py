"""
农业领域词典与实体识别
─────────────────────────────────────────────────────────────────────────────
职责：
  • 加载 config/agri_lexicon.json（作物/病害/虫害/农药/地区/生育期）。
  • 别名 → 规范名归一（稻热病 → 稻瘟病）。
  • 长词优先的实体识别（小麦条锈病不会被拆成 小麦 + 锈病）。
  • 增强 jieba 分词，保证「郑单958」「二化螟」等专名不被切碎。
  • 查询扩展：补充规范名，提升长尾术语召回。

所有函数均为纯本地规则，不依赖任何模型或网络。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import jieba
from loguru import logger

_LEXICON_PATH = Path(__file__).resolve().parents[2] / "config" / "agri_lexicon.json"

# 需要参与实体识别的类别；键名与 JSON 顶层一致
CATEGORIES: tuple[str, ...] = (
    "crops",
    "diseases",
    "pests",
    "pesticides",
    "regions",
    "growth_stages",
)

# 类别 → 文档 metadata 字段名
CATEGORY_FIELD: dict[str, str] = {
    "crops": "crop",
    "diseases": "disease",
    "pests": "pest",
    "pesticides": "pesticide",
    "regions": "region",
    "growth_stages": "growth_stage",
}

# 归一化时视为分隔符的字符：去掉后仍能匹配「rice blast」这类英文别名
_SEPARATOR_RE = re.compile(
    r"[\s\-_·•,，。;；:：!！?？\"'“”‘’()（）\[\]【】{}<>《》/\\|+*=~`]+"
)

_TOKEN_SEPARATOR_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)

# 品种名保护：句首/空格/标点之后的「郑单958」「济麦22」整体保留。
# 不追求覆盖所有上下文，只避免最影响 BM25 的切碎问题。
_PROTECTED_VARIETY_RE = re.compile(
    r"(?<![\u4e00-\u9fffA-Za-z])([\u4e00-\u9fffA-Za-zⅡⅢⅣ]{2,4}(?:\d{1,4}号|\d{2,4}))"
    r"(?!\s*(?:年|月|日|天|次|个|倍|%|℃|公斤|千克|克|米|厘米|毫米|亩|公顷|hm|kg))"
)


def normalize_term(text: str) -> str:
    """把术语/文本归一化为可比较形式：NFKC、小写、去空白与标点。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _SEPARATOR_RE.sub("", text)


# 纯 ASCII 别名（英文名/拉丁名）必须整词匹配，
# 否则 "tea" 会命中 "insTEAad" 之类的巧合。
_ASCII_ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9 ._\-]*$")


@dataclass(frozen=True)
class EntityHit:
    """一次实体命中。start 为归一化文本中的字符位置，用于排序与去重。"""

    category: str
    canonical: str
    matched: str
    start: int


class AgriLexicon:
    """农业词典的只读视图。"""

    def __init__(self, data: dict) -> None:
        self.version = int(data.get("version", 0))
        self.by_category: dict[str, dict[str, list[str]]] = {}
        # norm(alias) -> (category, canonical)
        self._alias_index: dict[str, tuple[str, str]] = {}
        # 长词优先，保证最长匹配
        aliases: list[str] = []
        # 原始形式的 ASCII 别名，用于在原文上做整词校验
        ascii_aliases: set[str] = set()

        for category in CATEGORIES:
            raw = data.get(category, {}) or {}
            table: dict[str, list[str]] = {}
            for canonical, alias_list in raw.items():
                canonical = str(canonical).strip()
                if not canonical:
                    continue
                aliases_for_term = [canonical, *(alias_list or [])]
                table[canonical] = [str(a).strip() for a in aliases_for_term if str(a).strip()]
                for alias in table[canonical]:
                    key = normalize_term(alias)
                    if not key:
                        continue
                    if _ASCII_ALIAS_RE.match(str(alias).lower()):
                        ascii_aliases.add(str(alias).lower())
                    existing = self._alias_index.get(key)
                    if existing and existing != (category, canonical):
                        # 同一个别名指向两个实体时不覆盖，保留先出现的（词典顺序即优先级）
                        logger.debug(
                            f"[AgriLexicon] 别名冲突：'{alias}' "
                            f"{existing} vs {(category, canonical)}，保留前者"
                        )
                        continue
                    self._alias_index[key] = (category, canonical)
                    aliases.append(key)
            self.by_category[category] = table

        # 去重并按长度降序，长词优先
        unique_aliases = sorted(set(aliases), key=len, reverse=True)
        self._regex = (
            re.compile("|".join(re.escape(a) for a in unique_aliases))
            if unique_aliases
            else None
        )

        # ASCII 别名的整词校验正则：在保留空格/标点的原文上运行，
        # 命中集合用于过滤「tea 命中 instead」这类子串误报。
        self._ascii_boundary_regex = (
            re.compile(
                r"(?<![a-z0-9])(?:"
                + "|".join(
                    re.escape(a) for a in sorted(ascii_aliases, key=len, reverse=True)
                )
                + r")(?![a-z0-9])"
            )
            if ascii_aliases
            else None
        )

        # 作物别名单独编译：用于在「玉米大斑病」这类长实体内部补出作物，
        # 避免最长匹配吃掉了作物名，导致 crop 元数据缺失。
        crop_aliases = sorted(
            {
                alias
                for alias, (category, _canonical) in self._alias_index.items()
                if category == "crops"
            },
            key=len,
            reverse=True,
        )
        self._crop_regex = (
            re.compile("|".join(re.escape(a) for a in crop_aliases))
            if crop_aliases
            else None
        )
        self._register_jieba_words()

    # ── 分词 ──────────────────────────────────────────────────────────────

    def _register_jieba_words(self) -> None:
        """把词典中的专名注册进 jieba，避免被切碎。"""
        added = 0
        for alias in self._alias_index:
            if len(alias) >= 2:
                jieba.add_word(alias, freq=100_000, tag="nz")
                added += 1
        logger.debug(f"[AgriLexicon] 已向 jieba 注册 {added} 个领域词")

    # ── 实体识别 ──────────────────────────────────────────────────────────

    def find_hits(
        self,
        text: str,
        categories: Iterable[str] | None = None,
    ) -> list[EntityHit]:
        """返回文本中所有实体命中（按出现位置排序，保留重复出现）。"""
        if not text or self._regex is None:
            return []

        allowed = set(categories) if categories else set(CATEGORIES)
        norm = normalize_term(text)
        ascii_ok = self._ascii_boundary_hits(text)
        hits: list[EntityHit] = []
        for match in self._regex.finditer(norm):
            info = self._alias_index.get(match.group(0))
            if not info or info[0] not in allowed:
                continue
            if not self._is_valid_ascii(match.group(0), ascii_ok):
                continue
            hits.append(
                EntityHit(
                    category=info[0],
                    canonical=info[1],
                    matched=match.group(0),
                    start=match.start(),
                )
            )

        # 补出被长实体覆盖的作物名（如「玉米大斑病」→ crop=玉米）
        if self._crop_regex is not None and "crops" in allowed:
            seen_crops = {hit.canonical for hit in hits if hit.category == "crops"}
            for match in self._crop_regex.finditer(norm):
                info = self._alias_index.get(match.group(0))
                if not info or info[0] != "crops":
                    continue
                if not self._is_valid_ascii(match.group(0), ascii_ok):
                    continue
                canonical = info[1]
                if canonical in seen_crops:
                    continue
                seen_crops.add(canonical)
                hits.append(
                    EntityHit(
                        category="crops",
                        canonical=canonical,
                        matched=match.group(0),
                        start=match.start(),
                    )
                )

        hits.sort(key=lambda hit: hit.start)
        return hits

    def _ascii_boundary_hits(self, text: str) -> set[str]:
        """原文中「整词出现」的 ASCII 别名集合（已归一化）。"""
        if self._ascii_boundary_regex is None:
            return set()
        lowered = unicodedata.normalize("NFKC", text).lower()
        return {
            normalize_term(match.group(0))
            for match in self._ascii_boundary_regex.finditer(lowered)
        }

    @staticmethod
    def _is_valid_ascii(matched: str, ascii_ok: set[str]) -> bool:
        """ASCII 别名必须整词出现；中文别名不受影响。"""
        if not _ASCII_ALIAS_RE.match(matched):
            return True
        return matched in ascii_ok

    def find_entities(
        self,
        text: str,
        categories: Iterable[str] | None = None,
    ) -> dict[str, list[str]]:
        """返回 {category: [canonical, ...]}，每个类别内按首次出现排序去重。"""
        result: dict[str, list[str]] = {}
        for hit in self.find_hits(text, categories):
            bucket = result.setdefault(hit.category, [])
            if hit.canonical not in bucket:
                bucket.append(hit.canonical)
        return result

    def resolve(self, value: str, category: str | None = None) -> str:
        """把别名/规范名解析为规范名；无法识别时原样返回。"""
        key = normalize_term(value)
        if not key:
            return value
        info = self._alias_index.get(key)
        if info and (category is None or info[0] == category):
            return info[1]
        # 不是完整别名时，尝试在字符串中找实体（如「水稻稻瘟病」）
        hits = self.find_hits(value)
        for hit in hits:
            if category is None or hit.category == category:
                return hit.canonical
        return value

    def aliases_of(self, canonical: str, category: str | None = None) -> list[str]:
        """返回规范名的全部别名（含规范名本身）。"""
        cats = [category] if category else list(CATEGORIES)
        for cat in cats:
            table = self.by_category.get(cat, {})
            if canonical in table:
                return list(table[canonical])
        return []

    # ── 查询扩展 ──────────────────────────────────────────────────────────

    def expand_query(self, query: str, limit: int = 4) -> list[str]:
        """
        在原查询基础上补充规范名与「作物+病害/虫害」组合。

        只做保守扩展：不引入查询中不存在的作物或病害组合，避免语义漂移。
        """
        if not query:
            return []
        norm_query = normalize_term(query)
        entities = self.find_entities(query)
        expansions: list[str] = []

        for values in entities.values():
            for canonical in values:
                if normalize_term(canonical) not in norm_query:
                    expansions.append(canonical)

        crops = entities.get("crops", [])
        targets = entities.get("diseases", []) + entities.get("pests", [])
        if crops and targets:
            for crop in crops[:1]:
                for target in targets[:2]:
                    # 病害名本身已含作物名时（如 小麦赤霉病），不要再拼一次
                    if normalize_term(crop) in normalize_term(target):
                        continue
                    combo = f"{crop}{target}"
                    if normalize_term(combo) not in norm_query:
                        expansions.append(combo)

        # 去重保序
        seen: set[str] = set()
        result: list[str] = []
        for item in expansions:
            key = normalize_term(item)
            if not key or key in seen or key in norm_query:
                continue
            seen.add(key)
            result.append(item)
            if len(result) >= limit:
                break
        return result


def tokenize(text: str) -> list[str]:
    """
    农业增强分词：
      1. 标点转空格后交给 jieba；
      2. 把词典命中但被切碎的规范名补回 token 列表。

    第二步保证「稻热病」与「稻瘟病」在 BM25 中都能命中同一规范词。
    """
    if not text:
        return []
    cleaned = _TOKEN_SEPARATOR_RE.sub(" ", str(text).lower())
    tokens = [tok.strip() for tok in jieba.cut(cleaned) if tok.strip()]

    for hit in find_hits(text):
        canonical = hit.canonical.lower()
        if canonical not in tokens:
            tokens.append(canonical)

    # 品种名兜底：被 jieba 切成「郑单」「958」时补回整体
    for match in _PROTECTED_VARIETY_RE.finditer(text):
        term = match.group(1).lower()
        if term not in tokens:
            tokens.append(term)
    return tokens


@lru_cache(maxsize=1)
def get_lexicon() -> AgriLexicon:
    """加载并缓存农业词典单例。"""
    if not _LEXICON_PATH.exists():
        logger.warning(f"[AgriLexicon] 词典文件不存在，退化为空词典：{_LEXICON_PATH}")
        return AgriLexicon({})

    data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    lexicon = AgriLexicon(data)
    logger.info(
        f"[AgriLexicon] 已加载 {len(lexicon._alias_index)} 个别名，"
        f"覆盖 {len(lexicon.by_category)} 个类别"
    )
    return lexicon


def find_hits(text: str, categories: Iterable[str] | None = None) -> list[EntityHit]:
    return get_lexicon().find_hits(text, categories)


def find_entities(text: str, categories: Iterable[str] | None = None) -> dict[str, list[str]]:
    return get_lexicon().find_entities(text, categories)


def resolve_value(value: str, category: str | None = None) -> str:
    return get_lexicon().resolve(value, category)


def expand_query(query: str, limit: int = 4) -> list[str]:
    return get_lexicon().expand_query(query, limit)


def aliases_of(canonical: str, category: str | None = None) -> list[str]:
    return get_lexicon().aliases_of(canonical, category)
