"""
查询理解（Query Understanding）的单元测试。

覆盖槽位抽取、年份解析、意图识别、资料类型识别与过滤条件生成。
纯规则实现，固定 now 以保证年份断言稳定。
"""
from __future__ import annotations

from datetime import datetime

import pytest

from src.agri import understand_query


NOW = datetime(2026, 1, 1)


class TestSlotExtraction:
    def test_crop_disease_region(self):
        u = understand_query("长江中下游水稻稻瘟病怎么防治", now=NOW)
        assert u.crop == "水稻"
        assert u.disease == "稻瘟病"
        assert u.region == "长江中下游"
        assert u.filters["crop"] == "水稻"
        assert u.filters["disease"] == "稻瘟病"
        assert u.filters["region"] == "长江中下游"

    def test_alias_normalized(self):
        u = understand_query("稻热病怎么防治", now=NOW)
        assert u.disease == "稻瘟病"
        assert u.filters["disease"] == "稻瘟病"

    def test_pest_and_pesticide(self):
        u = understand_query("二化螟用氯虫苯甲酰胺防治", now=NOW)
        assert u.pest == "二化螟"
        assert u.filters["pesticide"] == "氯虫苯甲酰胺"

    def test_growth_stage(self):
        u = understand_query("水稻分蘖期怎么施肥", now=NOW)
        assert u.entities["growth_stages"] == ["分蘖期"]
        assert u.filters["growth_stage"] == "分蘖期"

    def test_variety(self):
        u = understand_query("郑单958在黄淮海地区的抗病性", now=NOW)
        assert u.filters["variety"] == "郑单958"
        assert u.filters["region"] == "黄淮海"

    def test_no_explicit_slot_no_filter(self):
        u = understand_query("今天天气怎么样", now=NOW)
        assert u.filters == {}
        assert u.intent == "general"


class TestYearParsing:
    def test_recent_years(self):
        u = understand_query("近三年水稻稻瘟病防治", now=NOW)
        assert u.year_from == 2024
        assert u.filters["year_min"] == 2024

    def test_since_year(self):
        u = understand_query("2020年以后的小麦条锈病研究", now=NOW)
        assert u.year_from == 2020

    def test_year_range_takes_start(self):
        u = understand_query("2020-2022年玉米大斑病研究", now=NOW)
        assert u.year_from == 2020

    def test_no_year(self):
        u = understand_query("水稻稻瘟病", now=NOW)
        assert u.year_from is None
        assert "year_min" not in u.filters


class TestIntent:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("水稻稻瘟病怎么防治", "control"),
            ("小麦赤霉病什么时候打药", "control"),
            ("吡虫啉每亩用多少克", "dosage"),
            ("三环唑的残留限量标准", "standard"),
            ("推荐适合黄淮海种植的玉米品种", "variety"),
            ("两个处理的差异对比", "comparison"),
            ("稻瘟病抗性机理是什么", "mechanism"),
        ],
    )
    def test_intent(self, query, expected):
        assert understand_query(query, now=NOW).intent == expected

    def test_intent_label(self):
        u = understand_query("水稻稻瘟病怎么防治", now=NOW)
        assert u.intent_label == "防治/防控"


class TestDocType:
    def test_standard(self):
        u = understand_query("NY/T 393 标准里三环唑的限量是多少", now=NOW)
        assert u.doc_type == "标准"
        assert u.filters["doc_type"] == "标准"

    def test_paper(self):
        u = understand_query("水稻稻瘟病研究进展论文", now=NOW)
        assert u.doc_type == "论文"

    def test_pesticide_label(self):
        u = understand_query("这个农药标签上的安全间隔期", now=NOW)
        assert u.doc_type == "农药标签"


class TestExpansionAndNotes:
    def test_alias_expansion(self):
        u = understand_query("稻热病怎么防治", now=NOW)
        assert "稻瘟病" in u.expanded_queries

    def test_control_combo_expansion(self):
        u = understand_query("水稻稻热病怎么防治", now=NOW)
        assert any("水稻稻瘟病" in item for item in u.expanded_queries)

    def test_control_combo_not_duplicated(self):
        u = understand_query("小麦赤霉病怎么防治", now=NOW)
        assert not any(item.count("小麦") > 1 for item in u.expanded_queries)

    def test_notes_mention_filter_and_year(self):
        u = understand_query("近三年水稻稻瘟病防治", now=NOW)
        assert any("收窄" in note for note in u.notes)
        assert any("2024" in note for note in u.notes)

    def test_prompt_context(self):
        u = understand_query("长江中下游水稻稻瘟病怎么防治", now=NOW)
        context = u.prompt_context()
        assert "作物=水稻" in context
        assert "病害=稻瘟病" in context
        assert "地区=长江中下游" in context

    def test_as_dict_serializable(self):
        u = understand_query("水稻稻瘟病", now=NOW)
        data = u.as_dict()
        assert data["entities"]["crops"] == ["水稻"]
        assert data["intent"] == "general"
        assert isinstance(data["filters"], dict)
