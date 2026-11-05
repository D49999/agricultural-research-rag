"""
农业文档 / chunk 元数据抽取的单元测试。

覆盖：年份、资料类型、权威等级、作物/病害等实体、品种候选、chunk 继承规则。
"""
from __future__ import annotations

from src.agri import (
    AUTHORITY_LEVEL,
    DOC_TYPES,
    classify_document,
    extract_chunk_metadata,
    extract_document_metadata,
    extract_variety_candidates,
    extract_year,
    format_metadata,
)
from src.document_parser.chunker import chunk_blocks
from src.document_parser.text_parser import TextParser


# ── 年份 ──────────────────────────────────────────────────────────────────

class TestExtractYear:
    def test_year_with_context(self):
        assert extract_year("出版日期：2018年3月，本标准适用于水稻。") == 2018

    def test_year_in_parentheses(self):
        assert extract_year("水稻栽培技术（2019）") == 2019

    def test_latest_reasonable_year(self):
        assert extract_year("2017年试验，2019年修订，2021年发布") == 2021

    def test_invalid_year_ignored(self):
        assert extract_year("编号 1234，数量 5678") is None

    def test_empty(self):
        assert extract_year("") is None


# ── 资料类型 ──────────────────────────────────────────────────────────────

class TestClassifyDocument:
    def test_standard(self):
        doc_type, scores = classify_document(
            "NY/T 393-2020 绿色食品 农药使用准则\n本标准规定了水稻稻瘟病防治要求。"
        )
        assert doc_type == "标准"
        assert scores["标准"] > 0

    def test_pesticide_label(self):
        doc_type, _ = classify_document(
            "农药登记证号 PD20090001\n有效成分含量：25%\n剂型：悬浮剂\n防治对象：稻飞虱"
        )
        assert doc_type == "农药标签"

    def test_paper(self):
        doc_type, _ = classify_document(
            "摘要：本文研究水稻稻瘟病抗性。\n关键词：水稻；稻瘟病\n参考文献：..."
        )
        assert doc_type == "论文"

    def test_unknown_is_other(self):
        doc_type, _ = classify_document("今天天气不错，适合出门散步。")
        assert doc_type == "其他"

    def test_government_technical_plan(self):
        """部级「防控技术方案 / 技术意见」应识别为技术方案，而不是科普或其他。"""
        doc_type, _ = classify_document(
            "2024年水稻重大病虫害防控技术方案\n据全国农技中心预测，稻瘟病总体中等发生。",
        )
        assert doc_type == "技术方案"
        assert AUTHORITY_LEVEL[doc_type] == 2

    def test_guidance_opinion(self):
        doc_type, _ = classify_document(
            "小麦秋冬种技术指导意见\n一、品种选择\n二、播种质量"
        )
        assert doc_type == "技术方案"

    def test_filename_signal_has_higher_weight(self):
        doc_type, _ = classify_document("正文内容", "GB_T_8321_农药合理使用准则.pdf")
        assert doc_type == "标准"


# ── 品种 ──────────────────────────────────────────────────────────────────

class TestVarietyCandidates:
    def test_context_candidate(self):
        names = extract_variety_candidates("推荐品种为郑单958，郑单958表现稳定", strict=True)
        assert names[0] == "郑单958"

    def test_no_prefix_noise(self):
        names = extract_variety_candidates("推荐品种为郑单958", strict=False)
        assert "为郑单958" not in names
        assert "郑单958" in names

    def test_numbered_variety_not_truncated(self):
        assert extract_variety_candidates("深两优5814在长江中下游种植", strict=False)[0] == "深两优5814"

    def test_number_suffix_variety(self):
        assert extract_variety_candidates("中浙优1号在浙江推广", strict=False)[0] == "中浙优1号"

    def test_strict_requires_context(self):
        # 没有品种/审定/推广等上下文时，文档级不轻易标注
        assert extract_variety_candidates("2021年产量比2020年提高", strict=True) == []

    def test_strict_accepts_trial_context(self):
        assert extract_variety_candidates("较对照郑单958增产5.2%", strict=True)[0] == "郑单958"

    def test_empty(self):
        assert extract_variety_candidates("") == []


# ── 文档级元数据 ──────────────────────────────────────────────────────────

class TestDocumentMetadata:
    def _doc(self) -> str:
        return (
            "NY/T 393-2020 绿色食品 农药使用准则\n"
            "本标准规定了水稻稻瘟病防治中三环唑的使用限量，2020年发布实施。\n"
            "适宜区域：长江中下游。"
        )

    def test_core_fields(self):
        meta = extract_document_metadata(self._doc(), "NY_T_393_2020_水稻稻瘟病.pdf")
        assert meta["doc_type"] == "标准"
        assert meta["authority_level"] == AUTHORITY_LEVEL["标准"]
        assert meta["crop"] == "水稻"
        assert meta["disease"] == "稻瘟病"
        assert meta["pesticide"] == "三环唑"
        assert meta["region"] == "长江中下游"
        assert meta["year"] == 2020

    def test_metadata_version_present(self):
        meta = extract_document_metadata(self._doc(), "x.pdf")
        assert meta["metadata_version"] >= 1

    def test_filename_year_takes_precedence(self):
        # 正文引用其他年份时，以文件名中的发布年份为准
        meta = extract_document_metadata("根据2025年最新研究修订。", "水稻技术规程_2021.pdf")
        assert meta["year"] == 2021

    def test_doc_types_constant(self):
        assert "标准" in DOC_TYPES
        assert "论文" in DOC_TYPES

    def test_unknown_document_still_typed(self):
        meta = extract_document_metadata("一段没有任何农业术语的普通文本。", "note.txt")
        assert meta["doc_type"] == "其他"
        assert meta["authority_level"] == AUTHORITY_LEVEL["其他"]


# ── 作物判定 ──────────────────────────────────────────────────────────────

class TestDocumentCrop:
    def test_title_crop_beats_body_frequency(self):
        """正文里轮作作物出现更多次，也不能盖过标题里的主题作物。"""
        text = (
            "# 2023年大豆油菜花生春季科学施肥指导意见\n\n"
            + "玉米秸秆还田后种植大豆，玉米茬口注意氮肥用量。\n" * 12
        )
        meta = extract_document_metadata(text, "2023_大豆油菜花生施肥指导意见.md")
        assert meta["crop"] == "大豆"

    def test_umbrella_vegetable_doc_has_no_false_crop(self):
        """蔬菜类方案没有具体作物时，不能因为「棉铃虫」就标成棉花。"""
        text = (
            "# 保护地蔬菜重要害虫生物防治技术方案\n\n"
            "重点防治粉虱、蓟马和棉铃虫，甜菜夜蛾仅在局部发生。\n"
        )
        meta = extract_document_metadata(text, "保护地蔬菜害虫防治方案.md")
        assert "crop" not in meta
        assert meta["pest"] == "蓟马"

    def test_polyphagous_pest_does_not_set_chunk_crop(self):
        meta = extract_chunk_metadata("棉铃虫和甜菜夜蛾可用生物农药防治。", None)
        assert "crop" not in meta

    def test_crop_prefix_disease_still_sets_crop(self):
        """「玉米大斑病」这类以作物命名的病害仍应带出作物。"""
        meta = extract_chunk_metadata("玉米大斑病可用吡唑醚菌酯防治。", None)
        assert meta["crop"] == "玉米"

    def test_sunflower_and_cowpea_recognized(self):
        sunflower = extract_document_metadata(
            "# 向日葵主要病虫害防控技术方案\n\n重点防控菌核病。", "向日葵防控方案.md"
        )
        cowpea = extract_document_metadata(
            "# 豇豆病虫害绿色防控技术方案\n\n重点防治蓟马。", "豇豆防控方案.md"
        )
        assert sunflower["crop"] == "向日葵"
        assert cowpea["crop"] == "豇豆"


# ── chunk 级元数据 ────────────────────────────────────────────────────────

class TestChunkMetadata:
    def test_chunk_entities_extracted(self):
        meta = extract_chunk_metadata(
            "在长江中下游稻区，分蘖期防治二化螟可选用氯虫苯甲酰胺。"
        )
        assert meta["crop"] == "水稻"
        assert meta["pest"] == "二化螟"
        assert meta["pesticide"] == "氯虫苯甲酰胺"
        assert meta["growth_stage"] == "分蘖期"

    def test_inherits_document_fields(self):
        doc_meta = {
            "crop": "水稻",
            "doc_type": "标准",
            "authority_level": 1,
            "year": 2020,
        }
        meta = extract_chunk_metadata("本条规定了施药方法。", doc_meta)
        assert meta["crop"] == "水稻"
        assert meta["doc_type"] == "标准"
        assert meta["authority_level"] == 1
        assert meta["year"] == 2020

    def test_chunk_entity_overrides_document(self):
        doc_meta = {"crop": "水稻", "year": 2020}
        meta = extract_chunk_metadata("玉米大斑病可用吡唑醚菌酯防治。", doc_meta)
        assert meta["crop"] == "玉米"
        assert meta["disease"] == "玉米大斑病"

    def test_no_doc_metadata(self):
        meta = extract_chunk_metadata("普通段落。", None)
        assert meta == {}


# ── 展示 ──────────────────────────────────────────────────────────────────

class TestFormatMetadata:
    def test_format(self):
        text = format_metadata({"crop": "水稻", "year": 2021, "doc_type": "标准"})
        assert "crop=水稻" in text
        assert "year=2021" in text
        assert "doc_type=标准" in text

    def test_empty(self):
        assert format_metadata({}) == "（无）"


# ── 入库管线集成 ──────────────────────────────────────────────────────────

class TestChunkerIntegration:
    _DOC = (
        "# 水稻稻瘟病防治技术规程\n\n"
        "## 适用范围\n\n"
        "2021年发布，适用于长江中下游稻区。\n\n"
        "分蘖期可选用三环唑防治稻瘟病。\n"
    )

    def _chunks(self, tmp_path, strategy: str):
        path = tmp_path / "水稻稻瘟病防治技术规程.md"
        path.write_text(self._DOC, encoding="utf-8")
        blocks = TextParser().parse(path)
        return chunk_blocks(blocks, strategy, 512, 32, path.name)

    def test_semantic_strategy_attaches_metadata(self, tmp_path):
        chunks = self._chunks(tmp_path, "semantic")
        assert chunks
        meta = chunks[0].metadata
        assert meta["crop"] == "水稻"
        assert meta["disease"] == "稻瘟病"
        assert meta["year"] == 2021
        assert meta["doc_type"] == "技术规程"
        assert meta["authority_level"] == 2

    def test_markdown_strategy_attaches_metadata(self, tmp_path):
        chunks = self._chunks(tmp_path, "markdown")
        assert chunks
        assert all(chunk.metadata.get("crop") == "水稻" for chunk in chunks)

    def test_all_chunks_have_doc_type(self, tmp_path):
        chunks = self._chunks(tmp_path, "semantic")
        assert all("doc_type" in chunk.metadata for chunk in chunks)
