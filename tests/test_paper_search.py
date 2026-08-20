"""
文献搜索 Skill（paper_search / paper_ingest）的单元测试。
所有 HTTP 请求均 mock，不打外网；向量库与 BM25 用 MagicMock 替代。
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage

import src.agent.skills.paper_search as psm
from src.agent.skills.paper_search import (
    _parse_arxiv_entries,
    _s2_to_paper,
    build_paper_tools,
    search_papers,
)
from src.document_parser.base_parser import BlockType
from src.document_parser.html_converter import arxiv_html_to_markdown
from src.document_parser.text_parser import TextParser


# ── 测试夹具与样本数据 ──────────────────────────────────────────────────────

ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.12345v2</id>
    <published>2025-01-22T18:59:00Z</published>
    <title>Deep Learning for Crop Disease Detection</title>
    <summary>We propose a novel method for detecting crop diseases.</summary>
    <author><name>Alice Zhang</name></author>
    <author><name>Bob Li</name></author>
    <link href="http://arxiv.org/abs/2501.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2501.12345v2" rel="related" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2005.54321v1</id>
    <published>2020-05-01T00:00:00Z</published>
    <title>Old Paper on Agriculture</title>
    <summary>Old summary.</summary>
    <author><name>Carol Wu</name></author>
  </entry>
</feed>"""

CANNED_PAPER = {
    "paper_id": "arxiv:2501.12345",
    "title": "Deep Learning for Crop Disease Detection",
    "authors": ["Alice Zhang", "Bob Li"],
    "year": 2025,
    "abstract": "We propose a novel method for detecting crop diseases.",
    "url": "https://arxiv.org/abs/2501.12345",
    "pdf_url": "https://arxiv.org/pdf/2501.12345",
    "venue": "arXiv",
    "citations": None,
    "source": "arxiv",
}


@pytest.fixture
def mock_stores():
    store = MagicMock()
    sparse = MagicMock()
    store.get_all_documents.return_value = []
    return store, sparse


@pytest.fixture
def tools(mock_stores):
    store, sparse = mock_stores
    return build_paper_tools(store=store, sparse_retriever=sparse)


@pytest.fixture
def tmp_literature_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(psm.settings, "literature_dir", str(tmp_path))
    return tmp_path


# ── arXiv Atom 解析 ────────────────────────────────────────────────────────

class TestArxivParsing:
    def test_parse_entries(self):
        papers = _parse_arxiv_entries(ATOM_XML)
        assert len(papers) == 2
        first = papers[0]
        assert first["paper_id"] == "arxiv:2501.12345"      # 版本号被剥离
        assert first["title"] == "Deep Learning for Crop Disease Detection"
        assert first["year"] == 2025
        assert first["authors"] == ["Alice Zhang", "Bob Li"]
        assert first["pdf_url"].endswith("/pdf/2501.12345v2")
        assert first["url"] == "https://arxiv.org/abs/2501.12345"

    def test_parse_invalid_xml(self):
        assert _parse_arxiv_entries("<not-xml>") == []


# ── Semantic Scholar 映射 ──────────────────────────────────────────────────

class TestS2Parsing:
    def test_with_arxiv_external_id(self):
        item = {
            "paperId": "abc123",
            "title": "Some Paper",
            "abstract": "Abs.",
            "year": 2024,
            "authors": [{"name": "A"}],
            "citationCount": 10,
            "externalIds": {"ArXiv": "2401.99999"},
            "url": "https://www.semanticscholar.org/paper/abc123",
            "venue": "CVPR",
        }
        paper = _s2_to_paper(item)
        assert paper["paper_id"] == "arxiv:2401.99999"
        assert paper["source"] == "arxiv"
        assert paper["citations"] == 10

    def test_without_arxiv_id(self):
        paper = _s2_to_paper({
            "paperId": "deadbeef",
            "title": "Another Paper",
            "year": 2025,
            "externalIds": {},
        })
        assert paper["paper_id"] == "s2:deadbeef"
        assert paper["source"] == "semanticscholar"

    def test_no_title_returns_none(self):
        assert _s2_to_paper({"paperId": "x", "title": ""}) is None


# ── 双源融合去重 ───────────────────────────────────────────────────────────

class TestSearchMerge:
    def test_merge_dedup_and_year_filter(self, monkeypatch):
        arxiv_paper = dict(CANNED_PAPER)
        old_paper = {**CANNED_PAPER, "paper_id": "arxiv:2005.54321",
                     "year": 2020, "title": "Old Paper"}
        # S2 返回同一篇论文（带引用数）+ 一篇 s2 独有论文
        s2_dup = {**CANNED_PAPER, "citations": 42, "venue": "CVPR"}
        s2_only = {
            "paper_id": "s2:unique1", "title": "Unique S2 Paper",
            "authors": ["D"], "year": 2024, "abstract": "abs",
            "url": "u", "pdf_url": "", "venue": "", "citations": 5,
            "source": "semanticscholar",
        }
        monkeypatch.setattr(psm, "_search_arxiv", lambda *a, **k: [arxiv_paper, old_paper])
        monkeypatch.setattr(
            psm, "_search_semanticscholar", lambda *a, **k: [s2_dup, s2_only]
        )

        papers = search_papers("crop disease", max_results=5,
                               year_from=2024, year_to=2026)
        # 2020 年旧论文被过滤；arxiv/S2 重复条目合并为一条并补充引用数
        ids = [p["paper_id"] for p in papers]
        assert ids == ["arxiv:2501.12345", "s2:unique1"]
        assert papers[0]["citations"] == 42
        assert papers[0]["venue"] == "CVPR"

    def test_sort_by_citations(self, monkeypatch):
        papers = [
            {"paper_id": "arxiv:1", "title": "A", "year": 2024, "citations": 1},
            {"paper_id": "arxiv:2", "title": "B", "year": 2025, "citations": 99},
        ]
        monkeypatch.setattr(psm, "_search_arxiv", lambda *a, **k: papers)
        monkeypatch.setattr(psm, "_search_semanticscholar", lambda *a, **k: [])
        result = search_papers("t", max_results=5, year_from=2024,
                               year_to=2026, sort="cited")
        assert result[0]["paper_id"] == "arxiv:2"


# ── paper_search 工具 ──────────────────────────────────────────────────────

class TestPaperSearchTool:
    def test_invoke_returns_structured_results(self, tools, monkeypatch):
        monkeypatch.setattr(
            psm, "_search_arxiv", lambda *a, **k: [dict(CANNED_PAPER)]
        )
        monkeypatch.setattr(psm, "_search_semanticscholar", lambda *a, **k: [])

        search_tool = next(t for t in tools if t.name == "paper_search")
        raw = search_tool.invoke({"topic": "crop disease detection"})
        data = json.loads(raw)

        assert data["total"] == 1
        result = data["results"][0]
        assert result["paper_id"] == "arxiv:2501.12345"
        assert result["block_type"] == "literature"
        assert result["fulltext_available"] is True
        assert "abstract" in result

    def test_invoke_no_results(self, tools, monkeypatch):
        monkeypatch.setattr(psm, "_search_arxiv", lambda *a, **k: [])
        monkeypatch.setattr(psm, "_search_semanticscholar", lambda *a, **k: [])

        search_tool = next(t for t in tools if t.name == "paper_search")
        data = json.loads(search_tool.invoke({"topic": "nonexistent"}))
        assert data["results"] == []
        assert "未找到" in data["message"]


# ── paper_ingest 工具 ──────────────────────────────────────────────────────

class TestPaperIngestTool:
    def test_abstract_ingest(self, tools, mock_stores, tmp_literature_dir, monkeypatch):
        store, sparse = mock_stores
        monkeypatch.setattr(psm, "_fetch_paper_meta", lambda pid: dict(CANNED_PAPER))

        ingest_tool = next(t for t in tools if t.name == "paper_ingest")
        data = json.loads(
            ingest_tool.invoke({"paper_ids": '["arxiv:2501.12345"]'})
        )

        assert data["status"] == "success"
        entry = data["ingested"][0]
        assert entry["paper_id"] == "arxiv:2501.12345"
        assert entry["depth"] == "abstract"
        assert entry["chunks"] >= 1

        # 文件已落盘到 literature 目录
        md_file = tmp_literature_dir / "arxiv-2501.12345.md"
        assert md_file.exists()
        assert "Deep Learning for Crop Disease Detection" in md_file.read_text(encoding="utf-8")

        # 幂等入库：先按 paper_id 删除旧数据，再以确定性 ID 写入
        store.delete_documents_by_filter.assert_called_once_with(
            {"paper_id": {"$eq": "arxiv:2501.12345"}}
        )
        store.add_documents.assert_called_once()
        chunks = store.add_documents.call_args.args[0]
        ids = store.add_documents.call_args.kwargs["ids"]
        assert len(chunks) == len(ids)
        assert all(i.startswith("paper::arxiv-2501.12345::") for i in ids)

        # chunk 元数据包含文献信息
        meta = chunks[0].metadata
        assert meta["paper_id"] == "arxiv:2501.12345"
        assert meta["entry_type"] == "literature"
        assert meta["source"] == "arxiv-2501.12345.md"
        assert meta["ingest_depth"] == "abstract"

        # BM25 索引已重建
        sparse.build_index.assert_called_once()

    def test_fulltext_fallback_to_abstract(
        self, tools, mock_stores, tmp_literature_dir, monkeypatch
    ):
        store, sparse = mock_stores
        monkeypatch.setattr(psm, "_fetch_paper_meta", lambda pid: dict(CANNED_PAPER))
        monkeypatch.setattr(
            psm, "_fetch_arxiv_fulltext_markdown", lambda aid: None
        )

        ingest_tool = next(t for t in tools if t.name == "paper_ingest")
        data = json.loads(
            ingest_tool.invoke({
                "paper_ids": '["arxiv:2501.12345"]', "depth": "fulltext",
            })
        )

        entry = data["ingested"][0]
        assert entry["depth"] == "abstract"
        assert "降级" in entry["note"]

    def test_fulltext_non_arxiv_degrades(
        self, tools, tmp_literature_dir, monkeypatch
    ):
        s2_paper = {**CANNED_PAPER, "paper_id": "s2:deadbeef"}
        monkeypatch.setattr(psm, "_fetch_paper_meta", lambda pid: dict(s2_paper))

        ingest_tool = next(t for t in tools if t.name == "paper_ingest")
        data = json.loads(
            ingest_tool.invoke({"paper_ids": '["s2:deadbeef"]', "depth": "fulltext"})
        )
        assert data["ingested"][0]["depth"] == "abstract"
        assert "非 arXiv" in data["ingested"][0]["note"]

    def test_invalid_paper_ids(self, tools):
        ingest_tool = next(t for t in tools if t.name == "paper_ingest")
        data = json.loads(ingest_tool.invoke({"paper_ids": "not-json"}))
        assert data["status"] == "error"

    def test_failed_paper_recorded(self, tools, tmp_literature_dir, monkeypatch):
        monkeypatch.setattr(psm, "_fetch_paper_meta", lambda pid: None)
        ingest_tool = next(t for t in tools if t.name == "paper_ingest")
        data = json.loads(
            ingest_tool.invoke({"paper_ids": '["arxiv:0000.0000"]'})
        )
        assert data["status"] == "error"
        assert data["failed"][0]["paper_id"] == "arxiv:0000.0000"


# ── HTML → Markdown 转换器 ─────────────────────────────────────────────────

LATEXML_SAMPLE = """<html><head><title>doc</title></head><body>
<nav class="ltx_TOC">table of contents</nav>
<article class="ltx_document">
<h1 class="ltx_title">Crop Disease Detection with Deep Learning</h1>
<section class="ltx_section">
<h2 class="ltx_title_section"><span class="ltx_title_section">1 Introduction</span></h2>
<p class="ltx_p">We study the <math class="ltx_Math" alttext="\\alpha"><mi>α</mi></math> coefficient in crops.</p>
<div class="ltx_equation ltx_math_display">
<table class="ltx_equation"><tr><td><math alttext="y = f(x)" display="block"><mi>y</mi></math></td></tr></table>
</div>
<table class="ltx_tabular"><tr><td class="ltx_td">Acc</td><td class="ltx_td">0.95</td></tr></table>
<figure class="ltx_figure">
<img src="assets/fig1.png" alt="Architecture"/>
<figcaption class="ltx_caption">Figure 1: Overall architecture.</figcaption>
</figure>
</section>
</article>
</body></html>"""


class TestHtmlConverter:
    def test_convert_produces_parser_compatible_markdown(self, tmp_path):
        md = arxiv_html_to_markdown(
            LATEXML_SAMPLE, base_url="https://arxiv.org/html/2501.12345"
        )

        # nav 内容被丢弃
        assert "table of contents" not in md
        # 标题 / 行内公式 / 独立公式
        assert "# Crop Disease Detection with Deep Learning" in md
        assert "## 1 Introduction" in md
        assert "$\\alpha$" in md
        assert "$$y = f(x)$$" in md
        # 图片地址被补全为绝对 URL
        assert (
            "![Architecture](https://arxiv.org/html/2501.12345/assets/fig1.png)"
            in md
        )

        # 经 TextParser 解析后块类型正确
        md_file = tmp_path / "paper.md"
        md_file.write_text(md, encoding="utf-8")
        blocks = TextParser().parse(md_file)
        types = [b.block_type for b in blocks]

        assert BlockType.HEADER in types
        assert BlockType.FORMULA in types
        assert BlockType.TABLE in types
        assert BlockType.FIGURE in types

        figure = next(b for b in blocks if b.block_type == BlockType.FIGURE)
        assert figure.metadata["caption"].startswith("Figure 1:")
        assert figure.metadata["image_url"].startswith("https://arxiv.org/html/")

        table = next(b for b in blocks if b.block_type == BlockType.TABLE)
        assert table.metadata.get("subtype") == "html_table"
        assert "Acc" in table.content


# ── 来源提取（literature 类型） ────────────────────────────────────────────

class TestLiteratureSourceExtraction:
    def test_literature_sources_extracted(self):
        from src.agent.rag_agent import MultimodalRAGAgent

        msg = ToolMessage(
            tool_call_id="t1",
            content=json.dumps({
                "results": [{
                    "paper_id": "arxiv:2501.12345",
                    "title": "Deep Learning for Crop Disease Detection",
                    "year": 2025,
                    "block_type": "literature",
                }]
            }),
        )
        sources = MultimodalRAGAgent._extract_sources([msg])
        assert len(sources) == 1
        assert sources[0]["source"] == "arxiv:2501.12345"
        assert sources[0]["block_type"] == "literature"
        assert sources[0]["page"] == "2025"
