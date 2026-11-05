"""
农业文档 / chunk 元数据抽取
─────────────────────────────────────────────────────────────────────────────
目标：让每个 chunk 都能回答「这是什么作物、什么问题、哪一年、什么类型的资料」。

抽取字段（均为 Chroma 可存储的扁平标量）
----------------------------------------
crop / crops              主要作物、全部作物
variety                   品种（郑单958、济麦22……）
disease / diseases        病害
pest / pests              虫害
pesticide / pesticides    农药
region / regions          适宜/试验区域
growth_stage(s)           生育期
year                      发布/发表/实施年份
doc_type                  标准 | 审定公告 | 农药标签 | 技术方案 | 技术规程 | 论文 | 试验报告 | 科普文章 | 其他
authority_level           1=标准/审定/标签，2=方案/规程/试验报告，3=论文，4=科普/其他
metadata_version          元数据版本，便于后续重刷

说明
----
• 文档级元数据从「文件名 + 正文前部」抽取，文件名权重更高。
• chunk 级优先使用 chunk 自身命中的实体；自身没有时，继承文档级单值实体。
• 所有阈值刻意保守：抽不到就留空，宁可少标，不可错标。
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from .lexicon import CATEGORY_FIELD, EntityHit, find_hits, get_lexicon

METADATA_VERSION = 1

DOC_TYPES: tuple[str, ...] = (
    "标准",
    "审定公告",
    "农药标签",
    "技术方案",
    "技术规程",
    "试验报告",
    "论文",
    "科普文章",
    "其他",
)

AUTHORITY_LEVEL: dict[str, int] = {
    "标准": 1,
    "审定公告": 1,
    "农药标签": 1,
    "技术方案": 2,
    "技术规程": 2,
    "试验报告": 2,
    "论文": 3,
    "科普文章": 4,
    "其他": 4,
}

# 文档类型信号：(正则, 权重)。文件名中的命中会再乘 2。
_DOC_TYPE_SIGNALS: dict[str, list[tuple[str, int]]] = {
    "标准": [
        (r"GB\s*/?\s*T?\s*\d+", 5),
        (r"NY\s*/?\s*T\s*\d+", 5),
        (r"DB\s*\d{2}", 4),
        (r"ISO\s*\d+", 4),
        (r"国家标准|行业标准|地方标准|强制性标准", 4),
        (r"最大残留限量|限量标准|安全间隔期", 2),
    ],
    "审定公告": [
        (r"审定编号|品种审定|审定公告|国审|省审|引种备案|品种登记", 5),
        (r"育种者|适宜区域|特征特性|产量表现", 2),
    ],
    "农药标签": [
        (r"农药登记证|登记证号|PD\s*\d{8}", 5),
        (r"有效成分含量|剂型|农药名称|防治对象|施药方法", 3),
    ],
    "技术方案": [
        (r"防控技术方案|防控方案|技术方案|治理方案", 5),
        (r"指导意见|技术意见|技术要点|技术指导意见|明白纸", 4),
    ],
    "技术规程": [
        (r"技术规程|栽培技术|防治技术|操作规程|生产技术|施肥技术|灌溉技术", 4),
        (r"适用范围|规范性引用文件", 2),
    ],
    "试验报告": [
        (r"试验报告|田间试验|小区试验|随机区组|裂区设计|方差分析", 4),
        (r"试验设计|处理组|对照区|重复\s*\d+\s*次", 2),
    ],
    "论文": [
        (r"摘要|关键词|参考文献|中图分类号|收稿日期|基金项目|作者简介", 3),
        (r"Abstract|Keywords|doi|DOI", 3),
    ],
    "科普文章": [
        (r"科普|栽培要点|种植户|技术问答", 3),
    ],
}

_YEAR_CONTEXT_RE = re.compile(
    r"(?:出版|发表|收稿|修回|接受|发布|实施|印发|颁布|审定|登记|公示)"
    r"[^\n]{0,16}?((?:19|20)\d{2})"
)
_YEAR_CN_RE = re.compile(r"((?:19|20)\d{2})\s*年\s*\d{1,2}\s*月")
_YEAR_PAREN_RE = re.compile(r"[（(]((?:19|20)\d{2})[)）]")
_YEAR_ANY_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")

# 品种候选：郑单958 / 济麦22 / 龙粳31 / 中稻1号 / 先玉335
# 文档级要求有「品种/审定/选育……」等上下文，保证精度；
# 查询级允许裸名，再通过前缀裁剪去掉「品种为」等噪音。
_VARIETY_CONTEXT_RE = re.compile(
    r"(?:品种|品系|组合|杂交种|常规种|审定|选育|推广|种植|示范|主栽|对照|参试|名为|叫做)"
    r"[：:\s“”\"']*(?:为|是)?"
    r"([\u4e00-\u9fffA-Za-zⅡⅢⅣ]{2,6}(?:\d{1,4}号|\d{2,4}))"
    r"(?!\s*(?:年|月|日|天|次|个|倍|%|℃|公斤|千克|克|米|厘米|毫米|亩|公顷|hm|kg))"
)
_VARIETY_BARE_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-zⅡⅢⅣ]{2,6}(?:\d{1,4}号|\d{2,4}))"
    r"(?!\s*(?:年|月|日|天|次|个|倍|%|℃|公斤|千克|克|米|厘米|毫米|亩|公顷|hm|kg))"
)

_VARIETY_STOP_WORDS = (
    "第", "年", "月", "日", "表", "图", "页", "章", "节",
    "吨", "公斤", "千克", "克", "米", "厘米", "毫米", "度",
)

# 裸品种名前面常被粘上的动词/量词，逐层裁掉
_VARIETY_PREFIX_WORDS = (
    "品种为", "品种是", "品种", "品系", "组合", "种植", "选用", "推广",
    "示范", "主栽", "对照", "参试", "推荐", "适合", "采用",
    "高产", "优质", "抗病", "抗倒", "耐密", "杂交", "常规",
)

_VARIETY_PREFIX_CHARS = "为是的和与及或在对从到由以用于种系等把将向"

# 品种名前后出现这些词，说明是正式提及，可作为文档级强信号
_VARIETY_STRONG_CONTEXT = (
    "品种", "品系", "审定", "选育", "推广", "示范",
    "主栽", "对照", "参试", "种植", "引种", "育种",
)

_FILENAME_SEPARATOR_RE = re.compile(r"[_\-—–.]+")


def _source_stem(source_name: str) -> str:
    name = Path(str(source_name or "")).name
    stem = Path(name).stem if Path(name).suffix else name
    return _FILENAME_SEPARATOR_RE.sub(" ", stem).strip()


# 名字里带作物、实际却是多食性害虫：不能据此判定作物。
# 例如「棉铃虫」可危害番茄/玉米/花生，「甜菜夜蛾」也不是甜菜专属。
_POLYPHAGOUS_PESTS = frozenset({"棉铃虫", "甜菜夜蛾", "斜纹夜蛾", "草地贪夜蛾"})

# 正文（非标题）里出现次数太少的作物多半只是顺带提及，不足以代表文档主题。
_MIN_BODY_CROP_COUNT = 3

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")


def _entity_hits(text: str) -> list[EntityHit]:
    """实体命中，并剔除「多食性害虫名中的作物前缀」造成的误报。"""
    hits = find_hits(text)
    blocked = [
        (hit.start, hit.start + len(hit.matched))
        for hit in hits
        if hit.category == "pests" and hit.canonical in _POLYPHAGOUS_PESTS
    ]
    if not blocked:
        return hits
    kept: list[EntityHit] = []
    for hit in hits:
        end = hit.start + len(hit.matched)
        if hit.category == "crops" and any(
            start <= hit.start and end <= stop for start, stop in blocked
        ):
            continue
        kept.append(hit)
    return kept


def _document_title(text: str, filename: str) -> str:
    """取 Markdown 一级标题；没有标题时退回文件名。"""
    for line in (text or "").splitlines()[:40]:
        match = _HEADING_RE.match(line)
        if match:
            return match.group(1)
    return filename


def _title_crops(text: str, filename: str) -> list[str]:
    """标题/文件名里的作物（按出现顺序去重），最强的主题信号。"""
    ordered: list[str] = []
    for hit in _entity_hits(_document_title(text, filename)):
        if hit.category == "crops" and hit.canonical not in ordered:
            ordered.append(hit.canonical)
    return ordered


def _body_crops(text: str) -> list[str]:
    """正文里出现足够多次的作物，按频次降序。"""
    counts = Counter(
        hit.canonical for hit in _entity_hits((text or "")[:30000])
        if hit.category == "crops"
    )
    return [name for name, count in counts.most_common() if count >= _MIN_BODY_CROP_COUNT]


def _looks_like_year(year: int, now_year: int) -> bool:
    return 1950 <= year <= now_year + 1


def extract_year(text: str, now_year: int | None = None) -> int | None:
    """从正文前部抽取年份；优先带语义上下文的年份，其次取最大合理年份。"""
    if not text:
        return None
    now_year = now_year or datetime.now().year
    head = text[:6000]

    for pattern in (_YEAR_CONTEXT_RE, _YEAR_CN_RE, _YEAR_PAREN_RE):
        for match in pattern.finditer(head):
            year = int(match.group(1))
            if _looks_like_year(year, now_year):
                return year

    years = [
        candidate
        for candidate in _YEAR_ANY_RE.findall(head)
        if _looks_like_year(int(candidate), now_year)
    ]
    return max(int(y) for y in years) if years else None


def classify_document(text: str, source_name: str = "") -> tuple[str, dict[str, int]]:
    """按关键词打分判定文档类型，返回 (doc_type, 各类得分)。"""
    stem = _source_stem(source_name)
    head = (text or "")[:12000]
    haystack = f"{stem} {stem} {head}"

    scores: dict[str, int] = {}
    for doc_type, signals in _DOC_TYPE_SIGNALS.items():
        score = 0
        for pattern, weight in signals:
            score += len(re.findall(pattern, haystack, flags=re.IGNORECASE)) * weight
        scores[doc_type] = score

    best_type = max(scores, key=lambda k: scores[k]) if scores else "其他"
    if scores.get(best_type, 0) < 3:
        return "其他", scores
    return best_type, scores


def _trim_variety_prefix(name: str) -> str:
    """裁掉裸匹配时粘在品种名前面的动词/量词。"""
    changed = True
    while changed and len(name) > 2:
        changed = False
        for word in _VARIETY_PREFIX_WORDS:
            if name.startswith(word) and len(name) > len(word) + 1:
                name = name[len(word):]
                changed = True
                break
        if changed:
            continue
        if name[0] in _VARIETY_PREFIX_CHARS and len(name) > 3:
            name = name[1:]
            changed = True

    # 贪心匹配可能把更长的句子片段粘在前面：保留最后 2 个汉字作为品种名前缀
    match = re.match(r"^([^\d]+)(\d{2,4}号?)$", name)
    if match and len(match.group(1)) > 3:
        name = match.group(1)[-2:] + match.group(2)
    return name


def extract_variety_candidates(text: str, strict: bool = True) -> list[str]:
    """
    抽取品种候选，按上下文强度与出现频次排序。

    strict=True（文档级）：只接受「品种/审定/选育……」上下文中的候选，
    以及出现在强上下文或带「号」的候选，优先保证精度。
    strict=False（查询级）：允许裸名，并裁掉前缀噪音。
    """
    if not text:
        return []
    counts: Counter[str] = Counter()
    strong: set[str] = set()

    for match in _VARIETY_CONTEXT_RE.finditer(text):
        candidate = _trim_variety_prefix(match.group(1).strip())
        if candidate and not any(word in candidate for word in _VARIETY_STOP_WORDS):
            counts[candidate] += 1
            strong.add(candidate)

    for match in _VARIETY_BARE_RE.finditer(text):
        candidate = _trim_variety_prefix(match.group(1).strip())
        if not candidate or any(word in candidate for word in _VARIETY_STOP_WORDS):
            continue
        if len(candidate) < 3:
            continue
        window = text[max(0, match.start() - 8): match.end() + 30]
        strong_context = any(word in window for word in _VARIETY_STRONG_CONTEXT)
        if strong_context:
            strong.add(candidate)
        if strong_context or candidate.endswith("号"):
            counts[candidate] += 1
        if not strict:
            counts[candidate] += 1

    if not counts:
        return []

    def _score(item: tuple[str, int]) -> tuple[int, int, int]:
        name, count = item
        return (1 if name in strong else 0, count, len(name))

    return [name for name, _ in sorted(counts.items(), key=_score, reverse=True)]


def _pick_entities(text: str, filename: str) -> dict[str, list[str]]:
    """
    统计实体出现次数，返回每个类别按频次降序的规范名列表。

    文件名命中权重 ×3：文件名通常最能代表文档主题。
    """
    counter: Counter[tuple[str, str]] = Counter()
    first_seen: dict[tuple[str, str], int] = {}

    for weight, chunk in ((3, filename), (1, text[:30000])):
        if not chunk:
            continue
        for hit in _entity_hits(chunk):
            key = (hit.category, hit.canonical)
            counter[key] += weight
            first_seen.setdefault(key, hit.start)

    grouped: dict[str, list[str]] = {}
    for (category, canonical), _count in sorted(
        counter.items(),
        key=lambda item: (-item[1], first_seen.get(item[0], 0)),
    ):
        grouped.setdefault(category, []).append(canonical)
    return grouped


def extract_document_metadata(
    text: str,
    source_name: str = "",
    now_year: int | None = None,
) -> dict:
    """抽取文档级农业元数据。返回的 dict 只包含非空字段。"""
    filename = _source_stem(source_name)
    meta: dict = {}

    doc_type, _scores = classify_document(text, source_name)
    meta["doc_type"] = doc_type
    meta["authority_level"] = AUTHORITY_LEVEL.get(doc_type, 4)

    entities = _pick_entities(text, filename)
    for category, values in entities.items():
        field = CATEGORY_FIELD.get(category)
        if not field or not values:
            continue
        meta[field] = values[0]
        meta[f"{field}s"] = ", ".join(values[:5])

    # 作物以标题为主：正文常顺带提到轮作/间套作的其他作物，频次容易反超主题作物。
    crops = _title_crops(text, filename) or _body_crops(text)
    if crops:
        meta["crop"] = crops[0]
        meta["crops"] = ", ".join(crops[:5])
    else:
        meta.pop("crop", None)
        meta.pop("crops", None)

    varieties = extract_variety_candidates(f"{filename} {filename}\n{(text or '')[:20000]}")
    if varieties:
        meta["variety"] = varieties[0]

    # 文件名里的年份通常就是发布/实施年份，优先于正文（正文可能引用其他年份）
    year = extract_year(filename, now_year=now_year) or extract_year(
        text, now_year=now_year
    )
    if year:
        meta["year"] = year

    meta["metadata_version"] = METADATA_VERSION
    return meta


def extract_chunk_metadata(text: str, doc_metadata: dict | None = None) -> dict:
    """
    抽取 chunk 级元数据。

    规则：
      • chunk 自身命中的实体优先（更精确）；
      • chunk 没提到、而文档级只有一个明确值时，继承文档级值（提升过滤召回）；
      • 文档级单值字段（doc_type/year/authority_level/variety）总是继承。
    """
    doc_metadata = doc_metadata or {}
    meta: dict = {}

    entities = _pick_entities(text or "", "")
    for category, values in entities.items():
        field = CATEGORY_FIELD.get(category)
        if not field or not values:
            continue
        meta[field] = values[0]
        meta[f"{field}s"] = ", ".join(values[:5])

    for field in (
        "crop", "crops", "disease", "diseases", "pest", "pests",
        "pesticide", "pesticides", "region", "regions",
        "growth_stage", "growth_stages", "variety",
    ):
        if field not in meta and doc_metadata.get(field):
            value = doc_metadata[field]
            # 仅继承单值类别；多值字段在 chunk 上不继承，避免过度标注
            if "," not in str(value):
                meta[field] = value

    for field in ("doc_type", "authority_level", "year", "metadata_version"):
        if doc_metadata.get(field) not in (None, ""):
            meta[field] = doc_metadata[field]

    return meta


def format_metadata(meta: dict) -> str:
    """把元数据格式化为一行日志文本。"""
    if not meta:
        return "（无）"
    keys = (
        "crop", "disease", "pest", "pesticide", "region",
        "growth_stage", "variety", "year", "doc_type", "authority_level",
    )
    parts = [f"{key}={meta[key]}" for key in keys if meta.get(key) not in (None, "")]
    return " | ".join(parts)
