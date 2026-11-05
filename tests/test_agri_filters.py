"""
农业元数据过滤条件的单元测试。

同一套过滤条件需要同时驱动 Chroma `where`（稠密检索）和
本地 metadata 匹配（BM25 稀疏检索），这里分别验证。
"""
from __future__ import annotations

from src.agri import (
    build_chroma_where,
    describe_filters,
    metadata_matches,
    normalize_filters,
)


# ── 规范化 ────────────────────────────────────────────────────────────────

class TestNormalizeFilters:
    def test_empty(self):
        assert normalize_filters(None) == {}
        assert normalize_filters({}) == {}

    def test_alias_resolution(self):
        filters = normalize_filters({"crop": "土豆", "disease": "稻热病"})
        assert filters == {"crop": "马铃薯", "disease": "稻瘟病"}

    def test_year_aliases_and_str(self):
        assert normalize_filters({"year_from": "2020"}) == {"year_min": 2020}
        assert normalize_filters({"year_to": "2022"}) == {"year_max": 2022}
        assert normalize_filters({"year": 2021}) == {"year_min": 2021}

    def test_invalid_year_dropped(self):
        assert normalize_filters({"year_min": "去年"}) == {}

    def test_list_values(self):
        filters = normalize_filters({"crop": ["水稻", "小麦"]})
        assert filters["crop"] == ["水稻", "小麦"]

    def test_comma_string(self):
        filters = normalize_filters({"crop": "水稻，小麦"})
        assert filters["crop"] == ["水稻", "小麦"]

    def test_doc_type_synonym(self):
        assert normalize_filters({"doc_type": "国标"}) == {"doc_type": "标准"}
        assert normalize_filters({"doc_type": "登记证"}) == {"doc_type": "农药标签"}

    def test_unknown_keys_ignored(self):
        assert normalize_filters({"unknown": "x"}) == {}

    def test_empty_values_dropped(self):
        assert normalize_filters({"crop": "", "year_min": None}) == {}


# ── Chroma where ──────────────────────────────────────────────────────────

class TestBuildChromaWhere:
    def test_no_filters(self):
        assert build_chroma_where(None) is None
        assert build_chroma_where({}) is None

    def test_single_equality(self):
        where = build_chroma_where({"crop": "水稻"})
        assert where == {"crop": {"$eq": "水稻"}}

    def test_multiple_values_use_in(self):
        where = build_chroma_where({"crop": ["水稻", "小麦"]})
        assert where == {"crop": {"$in": ["水稻", "小麦"]}}

    def test_year_range(self):
        where = build_chroma_where({"year_min": 2020, "year_max": 2022})
        assert where == {"year": {"$gte": 2020, "$lte": 2022}}

    def test_authority(self):
        where = build_chroma_where({"authority_max": 2})
        assert where == {"authority_level": {"$lte": 2}}

    def test_combined_uses_and(self):
        where = build_chroma_where(
            {"crop": "水稻", "disease": "稻瘟病", "year_min": 2020}
        )
        assert "$and" in where
        assert {"crop": {"$eq": "水稻"}} in where["$and"]
        assert {"disease": {"$eq": "稻瘟病"}} in where["$and"]
        assert {"year": {"$gte": 2020}} in where["$and"]


# ── 本地 metadata 匹配（BM25 侧）─────────────────────────────────────────

class TestMetadataMatches:
    def test_no_filters_always_matches(self):
        assert metadata_matches({"crop": "玉米"}, None) is True
        assert metadata_matches(None, {}) is True

    def test_exact_match(self):
        assert metadata_matches({"crop": "水稻"}, {"crop": "水稻"}) is True
        assert metadata_matches({"crop": "玉米"}, {"crop": "水稻"}) is False

    def test_missing_field_fails(self):
        assert metadata_matches({"crop": "水稻"}, {"disease": "稻瘟病"}) is False

    def test_list_value_in_metadata(self):
        assert metadata_matches({"crop": ["玉米", "水稻"]}, {"crop": "水稻"}) is True

    def test_comma_separated_metadata(self):
        assert metadata_matches({"diseases": "稻瘟病, 稻曲病"}, {"disease": "稻曲病"}) is True

    def test_multiple_expected_values_is_or(self):
        assert metadata_matches({"crop": "小麦"}, {"crop": ["水稻", "小麦"]}) is True

    def test_year_bounds(self):
        meta = {"year": 2021}
        assert metadata_matches(meta, {"year_min": 2020}) is True
        assert metadata_matches(meta, {"year_max": 2020}) is False
        assert metadata_matches({"year": "2021"}, {"year_min": "2020"}) is True

    def test_missing_year_fails_when_year_filtered(self):
        assert metadata_matches({"crop": "水稻"}, {"year_min": 2020}) is False

    def test_authority_level(self):
        assert metadata_matches({"authority_level": 1}, {"authority_max": 2}) is True
        assert metadata_matches({"authority_level": 3}, {"authority_max": 2}) is False
        assert metadata_matches({}, {"authority_max": 2}) is False

    def test_pre_normalized_skips_resolution(self):
        # 已规范化时不再做别名解析，直接按规范名比较
        assert metadata_matches(
            {"crop": "马铃薯"}, {"crop": "马铃薯"}, pre_normalized=True
        ) is True


# ── 展示 ──────────────────────────────────────────────────────────────────

class TestDescribeFilters:
    def test_empty(self):
        assert describe_filters({}) == "无"

    def test_fields(self):
        text = describe_filters({"crop": "水稻", "year_min": 2020, "authority_max": 2})
        assert "作物=水稻" in text
        assert "年份≥2020" in text
        assert "权威等级≤2" in text

    def test_multi_values(self):
        assert "作物=水稻/小麦" in describe_filters({"crop": ["水稻", "小麦"]})
