"""
农业语义层：词典、别名归一、实体识别与分词增强的单元测试。

全部为本地规则，不依赖模型、网络或 API Key。
"""
from __future__ import annotations

from src.agri import (
    CATEGORY_FIELD,
    expand_query,
    find_entities,
    find_hits,
    get_lexicon,
    normalize_term,
    resolve_value,
    tokenize,
)


# ── 归一化 ────────────────────────────────────────────────────────────────

class TestNormalizeTerm:
    def test_lowercase_and_remove_punctuation(self):
        assert normalize_term("Rice Blast！") == "riceblast"

    def test_fullwidth_normalized(self):
        # NFKC 会把全角字母/数字转换为半角
        assert normalize_term("ＡＢＣ１２３") == "abc123"

    def test_empty_input(self):
        assert normalize_term("") == ""


# ── 实体识别 ──────────────────────────────────────────────────────────────

class TestEntityRecognition:
    def test_crop_and_disease(self):
        entities = find_entities("水稻稻瘟病怎么防治")
        assert entities["crops"] == ["水稻"]
        assert entities["diseases"] == ["稻瘟病"]

    def test_alias_maps_to_canonical(self):
        assert find_entities("稻热病")["diseases"] == ["稻瘟病"]
        assert find_entities("土豆")["crops"] == ["马铃薯"]
        assert find_entities("甲维盐")["pesticides"] == ["甲氨基阿维菌素苯甲酸盐"]

    def test_longest_match_wins(self):
        # 不应被拆成「小麦」+「锈病」
        entities = find_entities("小麦条锈病")
        assert "小麦条锈病" in entities["diseases"]

    def test_region_and_growth_stage(self):
        entities = find_entities("长江中下游水稻分蘖期")
        assert entities["regions"] == ["长江中下游"]
        assert entities["growth_stages"] == ["分蘖期"]

    def test_multiple_values_keep_order(self):
        entities = find_entities("水稻和小麦")
        assert entities["crops"] == ["水稻", "小麦"]

    def test_no_entity_returns_empty(self):
        assert find_entities("今天天气不错") == {}

    def test_category_filter(self):
        hits = find_hits("水稻稻瘟病", categories=["diseases"])
        assert [hit.canonical for hit in hits] == ["稻瘟病"]

    def test_ascii_alias_requires_word_boundary(self):
        # "instead" 里包含 "tea"，不能据此判成茶树
        assert "crops" not in find_entities("We use a CNN instead of a transformer.")
        assert find_entities("tea plantation")["crops"] == ["茶树"]
        assert find_entities("potato disease detection")["crops"] == ["马铃薯"]


class TestResolve:
    def test_alias(self):
        assert resolve_value("稻热病", "diseases") == "稻瘟病"

    def test_unknown_passthrough(self):
        assert resolve_value("某种未知术语", "crops") == "某种未知术语"

    def test_entity_inside_sentence(self):
        assert resolve_value("水稻稻瘟病的防治", "diseases") == "稻瘟病"


class TestLexiconData:
    def test_lexicon_loaded(self):
        lexicon = get_lexicon()
        assert lexicon.version >= 1
        assert set(CATEGORY_FIELD) <= set(lexicon.by_category)

    def test_each_category_has_entries(self):
        lexicon = get_lexicon()
        for category in ("crops", "diseases", "pests", "pesticides", "regions", "growth_stages"):
            assert lexicon.by_category[category], f"{category} 词典为空"


# ── 分词增强 ──────────────────────────────────────────────────────────────

class TestTokenize:
    def test_protects_variety(self):
        tokens = tokenize("郑单958在黄淮海地区的稻瘟病抗性")
        assert "郑单958" in tokens
        assert "稻瘟病" in tokens
        assert "黄淮海" in tokens

    def test_protects_numbered_variety(self):
        tokens = tokenize("中浙优1号在浙江推广")
        assert "中浙优1号" in tokens

    def test_alias_and_canonical_both_tokenized(self):
        tokens = tokenize("稻热病")
        assert "稻瘟病" in tokens

    def test_empty(self):
        assert tokenize("") == []


# ── 查询扩展 ──────────────────────────────────────────────────────────────

class TestExpandQuery:
    def test_adds_canonical_alias(self):
        expansions = expand_query("稻热病怎么防治")
        assert "稻瘟病" in expansions

    def test_adds_crop_disease_combo(self):
        expansions = expand_query("水稻稻热病怎么防治")
        assert "水稻稻瘟病" in expansions

    def test_no_duplicate_crop_in_disease_name(self):
        # 小麦赤霉病已包含作物名，不应扩展出「小麦小麦赤霉病」
        expansions = expand_query("小麦赤霉病怎么防治")
        assert not any(item.count("小麦") > 1 for item in expansions)

    def test_no_expansion_for_unknown(self):
        assert expand_query("完全无关的问题") == []
