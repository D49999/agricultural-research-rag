"""
农业增强检索工具与过滤链路的单元测试。

覆盖：
  • knowledge_base_search 自动槽位过滤
  • knowledge_base_search_structured 显式条件检索
  • multi_round_search 去重与放宽标记
  • SparseRetriever 的 BM25 侧元数据过滤
  • HybridRetriever 过滤过严时的自动放宽

所有外部调用（Chroma、LLM）均已 mock 或使用轻量替身。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.agent.tools import build_retrieval_tools
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.sparse_retriever import SparseRetriever


# ── 测试替身 ──────────────────────────────────────────────────────────────

class _FakeOutcome:
    def __init__(self, documents: list[Document], relaxed: bool = False) -> None:
        self.documents = documents
        self.relaxed = relaxed
        self.filters: dict = {}
        self.filter_description = ""


class _FakeRetriever:
    """只实现工具层需要的方法，用来断言过滤条件被正确传入。"""

    def __init__(self, docs: list[Document], relaxed: bool = False) -> None:
        self.docs = docs
        self.relaxed = relaxed
        self.calls: list[dict] = []

    def retrieve(self, query: str, filters=None) -> list[Document]:
        self.calls.append({"query": query, "filters": filters, "info": False})
        return self.docs

    def retrieve_with_info(self, query: str, filters=None) -> _FakeOutcome:
        self.calls.append({"query": query, "filters": filters, "info": True})
        return _FakeOutcome(self.docs, relaxed=self.relaxed)


def _tool(tools, name: str):
    return next(t for t in tools if t.name == name)


def _doc(content: str = "水稻稻瘟病防治技术", **metadata) -> Document:
    base = {"source": "test.pdf", "page_num": 1, "block_type": "text"}
    base.update(metadata)
    return Document(page_content=content, metadata=base)


# ── knowledge_base_search ────────────────────────────────────────────────

class TestKnowledgeBaseSearchAutoFilter:
    def test_query_slots_become_filters(self):
        retriever = _FakeRetriever([_doc(crop="水稻", disease="稻瘟病", year=2021)])
        search = _tool(build_retrieval_tools(retriever), "knowledge_base_search")

        data = json.loads(search.invoke({"query": "水稻稻瘟病怎么防治"}))

        assert data["applied_filters"]["crop"] == "水稻"
        assert data["applied_filters"]["disease"] == "稻瘟病"
        assert "作物=水稻" in data["filter_description"]

        call = retriever.calls[-1]
        assert call["info"] is True
        assert call["filters"]["crop"] == "水稻"

    def test_results_include_agri_metadata(self):
        retriever = _FakeRetriever([_doc(crop="水稻", year=2021, doc_type="标准")])
        search = _tool(build_retrieval_tools(retriever), "knowledge_base_search")

        data = json.loads(search.invoke({"query": "水稻稻瘟病防治"}))
        result = data["results"][0]

        assert result["crop"] == "水稻"
        assert result["year"] == 2021
        assert result["doc_type"] == "标准"
        assert result["source"] == "test.pdf"

    def test_relaxed_flag_surfaces_note(self):
        retriever = _FakeRetriever([_doc(crop="玉米")], relaxed=True)
        search = _tool(build_retrieval_tools(retriever), "knowledge_base_search")

        data = json.loads(search.invoke({"query": "水稻稻瘟病防治"}))

        assert data["relaxed"] is True
        assert "放宽" in data["relaxation_note"]

    def test_no_slots_uses_plain_retrieve(self):
        retriever = _FakeRetriever([_doc()])
        search = _tool(build_retrieval_tools(retriever), "knowledge_base_search")

        json.loads(search.invoke({"query": "你好世界"}))

        assert retriever.calls[-1]["info"] is False
        assert retriever.calls[-1]["filters"] is None

    def test_empty_results_message(self):
        retriever = _FakeRetriever([])
        search = _tool(build_retrieval_tools(retriever), "knowledge_base_search")

        data = json.loads(search.invoke({"query": "水稻稻瘟病"}))
        assert data["results"] == []
        assert "message" in data


# ── knowledge_base_search_structured ─────────────────────────────────────

class TestStructuredSearch:
    def test_explicit_conditions(self):
        retriever = _FakeRetriever([_doc(crop="小麦", disease="小麦赤霉病")])
        search = _tool(
            build_retrieval_tools(retriever), "knowledge_base_search_structured"
        )

        data = json.loads(
            search.invoke(
                {
                    "query": "怎么防治",
                    "crop": "小麦",
                    "disease": "赤霉病",
                    "year_from": "2020",
                }
            )
        )

        filters = retriever.calls[-1]["filters"]
        assert filters["crop"] == "小麦"
        assert filters["disease"] == "小麦赤霉病"
        assert filters["year_min"] == 2020
        assert data["applied_filters"]["crop"] == "小麦"

    def test_empty_params_fall_back_to_query_slots(self):
        retriever = _FakeRetriever([_doc(crop="水稻")])
        search = _tool(
            build_retrieval_tools(retriever), "knowledge_base_search_structured"
        )

        data = json.loads(search.invoke({"query": "水稻稻瘟病怎么防治"}))
        assert data["applied_filters"]["crop"] == "水稻"

    def test_doc_type_only(self):
        retriever = _FakeRetriever([_doc(doc_type="标准")])
        search = _tool(
            build_retrieval_tools(retriever), "knowledge_base_search_structured"
        )

        json.loads(search.invoke({"query": "农药残留限量", "doc_type": "标准"}))
        filters = retriever.calls[-1]["filters"]
        assert filters["doc_type"] == "标准"


# ── multi_round_search ───────────────────────────────────────────────────

class TestMultiRoundSearch:
    def test_deduplicates_across_queries(self):
        retriever = _FakeRetriever([_doc("重复内容" * 30)])
        search = _tool(build_retrieval_tools(retriever), "multi_round_search")

        data = json.loads(
            search.invoke({"queries": json.dumps(["水稻稻瘟病", "水稻稻瘟病防治"])})
        )

        assert data["total_queries"] == 2
        assert len(data["results"]) == 1

    def test_relaxed_queries_reported(self):
        retriever = _FakeRetriever([_doc("内容" * 40)], relaxed=True)
        search = _tool(build_retrieval_tools(retriever), "multi_round_search")

        data = json.loads(search.invoke({"queries": json.dumps(["水稻稻瘟病"])}))

        assert data["relaxed_queries"] == ["水稻稻瘟病"]
        assert "放宽" in data["relaxation_note"]

    def test_invalid_json_treated_as_single_query(self):
        retriever = _FakeRetriever([_doc("内容" * 40)])
        search = _tool(build_retrieval_tools(retriever), "multi_round_search")

        data = json.loads(search.invoke({"queries": "水稻稻瘟病防治"}))
        assert data["results"]


# ── BM25 侧过滤 ──────────────────────────────────────────────────────────

class TestSparseRetrieverFilter:
    def _retriever(self) -> SparseRetriever:
        docs = [
            Document(page_content="水稻稻瘟病三环唑防治", metadata={"crop": "水稻"}),
            Document(page_content="小麦条锈病戊唑醇防治", metadata={"crop": "小麦"}),
            Document(page_content="玉米大斑病吡唑醚菌酯防治", metadata={"crop": "玉米"}),
        ]
        retriever = SparseRetriever()
        retriever.build_index(docs)
        return retriever

    def test_filter_keeps_matching_documents(self):
        results = self._retriever().retrieve(
            "三环唑", k=10, filters={"crop": "水稻"}
        )
        assert [doc.metadata["crop"] for doc, _ in results] == ["水稻"]

    def test_filter_with_no_match_returns_empty(self):
        results = self._retriever().retrieve(
            "三环唑", k=10, filters={"crop": "马铃薯"}
        )
        assert results == []

    def test_no_filter_returns_both(self):
        results = self._retriever().retrieve("三环唑 戊唑醇", k=10)
        assert {doc.metadata["crop"] for doc, _ in results} == {"水稻", "小麦"}


# ── Hybrid 自动放宽 ──────────────────────────────────────────────────────

class TestHybridRetrieverRelaxation:
    def _build(self, doc: Document) -> HybridRetriever:
        dense = MagicMock()

        def dense_retrieve(query, k=None, filter=None):
            return [] if filter else [(doc, 0.9)]

        dense.retrieve.side_effect = dense_retrieve

        sparse = MagicMock()
        sparse.is_ready = True

        def sparse_retrieve(query, k=None, filters=None):
            return [] if filters else [(doc, 0.8)]

        sparse.retrieve.side_effect = sparse_retrieve

        return HybridRetriever(dense, sparse, reranker=None)

    def test_over_strict_filter_relaxes(self):
        doc = _doc("水稻稻瘟病防治技术")
        retriever = self._build(doc)

        outcome = retriever.retrieve_with_info(
            "水稻稻瘟病", filters={"crop": "水稻"}
        )

        assert outcome.relaxed is True
        assert outcome.filters["crop"] == "水稻"
        assert [d.page_content for d in outcome.documents] == ["水稻稻瘟病防治技术"]

    def test_without_filters_not_relaxed(self):
        retriever = self._build(_doc("水稻稻瘟病防治技术"))

        outcome = retriever.retrieve_with_info("水稻稻瘟病")

        assert outcome.relaxed is False
        assert outcome.filters == {}
        assert outcome.documents

    def test_retrieve_delegates_to_retrieve_with_info(self):
        retriever = self._build(_doc("水稻稻瘟病防治技术"))
        docs = retriever.retrieve("水稻稻瘟病")
        assert [d.page_content for d in docs] == ["水稻稻瘟病防治技术"]
