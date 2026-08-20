"""
文献搜索 Skill（paper_search / paper_ingest）
──────────────────────────────────────────────────────────────────────────────
联网检索近几年与某主题相关的学术论文（arXiv + Semantic Scholar 双源），
并支持将检索到的论文以轻量方式入库（无需 MinerU 离线 OCR）：

  • paper_search : 按主题/年份检索论文，返回标题、作者、摘要等结构化结果
  • paper_ingest : 将指定论文入库到文本知识库（Chroma + BM25）

入库深度分级
────────────
  • abstract  — 摘要级（默认）：标题+作者+年份+摘要 构造小文档，秒级入库
  • fulltext  — 全文级（仅 arXiv）：抓取 arxiv.org/html 官方 HTML 转 Markdown，
                复用现有 TextParser + StructureAwareChunker 管线，
                失败时自动降级为摘要级；非 arXiv 论文自动降级为摘要级

入库幂等性：chunk ID 为 `paper::{paper_id}::{i}` 的确定性 ID，
入库前先按 paper_id 删除旧 chunks，重复入库或切换深度不会产生重复数据。
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
from langchain_core.tools import tool
from loguru import logger

from config.settings import get_settings
from src.document_parser.chunker import chunk_blocks
from src.document_parser.html_converter import arxiv_html_to_markdown
from src.document_parser.text_parser import TextParser

if TYPE_CHECKING:
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()

# ── API 端点 ────────────────────────────────────────────────────────────────
_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
_AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_S2_API = "https://api.semanticscholar.org/graph/v1"
_HEADERS = {"User-Agent": "agri-rag-literature-search/1.0"}

_ATOM = "{http://www.w3.org/2005/Atom}"

_ABSTRACT_MAX_CHARS = 900          # 工具返回给 LLM 的摘要截断长度
_MIN_FULLTEXT_CHARS = 2_000        # 全文抓取结果低于此长度视为失败


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    """标题归一化（小写、仅保留字母数字），用于跨源去重。"""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _http_get(url: str, params: dict | None = None) -> requests.Response | None:
    """带超时的 GET 请求，失败返回 None（不抛异常）。"""
    try:
        resp = requests.get(
            url, params=params, headers=_HEADERS,
            timeout=settings.paper_fetch_timeout,
        )
        return resp
    except requests.RequestException as exc:
        logger.warning(f"[PaperSearch] 请求失败 {url}：{exc}")
        return None


def _strip_version(arxiv_id: str) -> str:
    """去除 arXiv ID 末尾的版本号（如 2401.12345v2 → 2401.12345）。"""
    return re.sub(r"v\d+$", "", arxiv_id)


# ─────────────────────────────────────────────────────────────────────────────
# arXiv API 客户端
# ─────────────────────────────────────────────────────────────────────────────

def _parse_arxiv_entries(xml_text: str) -> list[dict]:
    """解析 arXiv Atom feed，返回论文字典列表。"""
    papers: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning(f"[PaperSearch] arXiv XML 解析失败：{exc}")
        return papers

    for entry in root.findall(f"{_ATOM}entry"):
        arxiv_id = ""
        title = ""
        abstract = ""
        year = None
        authors: list[str] = []
        pdf_url = ""

        id_el = entry.find(f"{_ATOM}id")
        if id_el is not None and id_el.text:
            # http://arxiv.org/abs/2401.12345v2 → 2401.12345
            arxiv_id = _strip_version(id_el.text.strip().rstrip("/").split("/")[-1])

        title_el = entry.find(f"{_ATOM}title")
        if title_el is not None and title_el.text:
            title = re.sub(r"\s+", " ", title_el.text).strip()

        summary_el = entry.find(f"{_ATOM}summary")
        if summary_el is not None and summary_el.text:
            abstract = re.sub(r"\s+", " ", summary_el.text).strip()

        pub_el = entry.find(f"{_ATOM}published")
        if pub_el is not None and pub_el.text:
            year = int(pub_el.text[:4])

        for author in entry.findall(f"{_ATOM}author"):
            name_el = author.find(f"{_ATOM}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        for link in entry.findall(f"{_ATOM}link"):
            if link.get("type") == "application/pdf" and link.get("href"):
                pdf_url = link.get("href", "")

        if not (arxiv_id and title):
            continue
        papers.append({
            "paper_id": f"arxiv:{arxiv_id}",
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url,
            "venue": "arXiv",
            "citations": None,
            "source": "arxiv",
        })
    return papers


def _search_arxiv(topic: str, max_results: int) -> list[dict]:
    """在 arXiv 检索（按提交时间倒序，过量拉取后客户端过滤年份）。"""
    resp = _http_get(_ARXIV_API, params={
        "search_query": f'all:"{topic}"',
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": min(max_results * 4, 60),
    })
    if resp is None or resp.status_code != 200:
        logger.warning(f"[PaperSearch] arXiv 搜索失败：HTTP {resp.status_code if resp else 'N/A'}")
        return []
    return _parse_arxiv_entries(resp.text)


def _fetch_arxiv_meta(arxiv_id: str) -> dict | None:
    """按 arXiv ID 获取单篇论文元数据（瞬时网络错误重试一次）。"""
    for attempt in range(2):
        resp = _http_get(_ARXIV_API, params={"id_list": arxiv_id, "max_results": 1})
        if resp is not None and resp.status_code == 200:
            papers = _parse_arxiv_entries(resp.text)
            if papers:
                return papers[0]
        if attempt == 0:
            time.sleep(1.0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Semantic Scholar API 客户端
# ─────────────────────────────────────────────────────────────────────────────

_S2_FIELDS = "title,abstract,year,authors,citationCount,externalIds,url,venue"


def _s2_to_paper(item: dict) -> dict | None:
    """将 Semantic Scholar API 的论文对象映射为统一论文字典。"""
    title = (item.get("title") or "").strip()
    if not title:
        return None
    ext_ids = item.get("externalIds") or {}
    arxiv_id = (ext_ids.get("ArXiv") or "").strip()
    paper_id = f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{item.get('paperId', '')}"
    if paper_id in ("arxiv:", "s2:"):
        return None
    abstract = (item.get("abstract") or "").strip()
    authors = [
        a.get("name", "") for a in (item.get("authors") or []) if a.get("name")
    ]
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": item.get("year"),
        "abstract": abstract,
        "url": item.get("url", "") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
        "pdf_url": (item.get("openAccessPdf") or {}).get("url", ""),
        "venue": item.get("venue", "") or "",
        "citations": item.get("citationCount"),
        "source": "arxiv" if arxiv_id else "semanticscholar",
    }


def _s2_request(path: str, params: dict) -> dict | None:
    """Semantic Scholar 请求（429 限速时退避重试一次）。"""
    url = f"{_S2_API}{path}"
    for attempt in range(2):
        resp = _http_get(url, params=params)
        if resp is None:
            return None
        if resp.status_code == 429 and attempt == 0:
            time.sleep(2.0)
            continue
        if resp.status_code != 200:
            logger.warning(f"[PaperSearch] S2 请求失败 {path}：HTTP {resp.status_code}")
            return None
        try:
            return resp.json()
        except ValueError:
            return None
    return None


def _search_semanticscholar(
    topic: str, max_results: int, year_from: int, year_to: int,
) -> list[dict]:
    """在 Semantic Scholar 检索（服务端过滤年份）。"""
    data = _s2_request("/paper/search", params={
        "query": topic,
        "limit": min(max_results * 2, 40),
        "fields": _S2_FIELDS,
        "year": f"{year_from}-{year_to}",
    })
    if not data:
        return []
    papers = []
    for item in data.get("data", []):
        paper = _s2_to_paper(item)
        if paper:
            papers.append(paper)
    return papers


def _fetch_s2_meta(s2_id: str) -> dict | None:
    """按 Semantic Scholar paperId 获取单篇论文元数据。"""
    data = _s2_request(f"/paper/{s2_id}", params={"fields": _S2_FIELDS})
    if not data:
        return None
    return _s2_to_paper(data)


# ─────────────────────────────────────────────────────────────────────────────
# 检索融合
# ─────────────────────────────────────────────────────────────────────────────

def search_papers(
    topic: str,
    max_results: int,
    year_from: int,
    year_to: int,
    sort: str = "recent",
) -> list[dict]:
    """
    双源检索论文：arXiv + Semantic Scholar，按 paper_id 与标题去重合并。

    sort == "recent"：年份降序优先；sort == "cited"：引用数降序优先。
    """
    arxiv_papers = _search_arxiv(topic, max_results)
    s2_papers = _search_semanticscholar(topic, max_results, year_from, year_to)

    merged: dict[str, dict] = {}
    title_map: dict[str, str] = {}   # 归一化标题 → paper_id

    for paper in arxiv_papers + s2_papers:
        pid = paper["paper_id"]
        ntitle = _normalize_title(paper["title"])
        # 同 ID 或同标题（跨源重复）时，保留已有条目但补充缺失字段
        existing = merged.get(pid) or (
            merged.get(title_map.get(ntitle, "")) if ntitle in title_map else None
        )
        if existing:
            for key in ("citations", "abstract", "pdf_url"):
                if not existing.get(key) and paper.get(key):
                    existing[key] = paper[key]
            # venue：S2 的正式发表信息比 arXiv 预印本标记更有价值
            if paper.get("venue") and paper["venue"] not in ("", "arXiv"):
                existing["venue"] = paper["venue"]
            continue
        merged[pid] = paper
        if ntitle:
            title_map[ntitle] = pid

    papers = [p for p in merged.values() if p.get("year") and year_from <= p["year"] <= year_to]
    if sort == "cited":
        papers.sort(key=lambda p: (p.get("citations") or 0, p.get("year") or 0), reverse=True)
    else:
        papers.sort(key=lambda p: (p.get("year") or 0, p.get("citations") or 0), reverse=True)
    return papers[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# 论文抓取与入库
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_arxiv_fulltext_markdown(arxiv_id: str) -> str | None:
    """
    抓取 arXiv 论文官方 HTML（arxiv.org/html，失败回退 ar5iv）并转 Markdown。
    """
    for template in (_ARXIV_HTML_URL, _AR5IV_HTML_URL):
        url = template.format(arxiv_id=arxiv_id)
        resp = _http_get(url)
        if resp is None or resp.status_code != 200:
            continue
        html = resp.text
        if "<article" not in html:
            continue
        try:
            markdown = arxiv_html_to_markdown(html, base_url=url)
        except Exception as exc:
            logger.warning(f"[PaperIngest] HTML 转换失败 {url}：{exc}")
            continue
        if len(markdown) >= _MIN_FULLTEXT_CHARS:
            return markdown[: settings.paper_fulltext_max_chars]
    return None


def _build_abstract_markdown(paper: dict) -> str:
    """由元数据构造摘要级 Markdown 文档。"""
    authors = ", ".join(paper.get("authors", [])) or "未知"
    abstract = paper.get("abstract", "").strip() or "（暂无摘要）"
    return (
        f"# {paper['title']}\n\n"
        f"- 作者：{authors}\n"
        f"- 年份：{paper.get('year', '未知')}\n"
        f"- 来源：{paper['paper_id']}"
        + (f"（{paper['venue']}）" if paper.get("venue") and paper["venue"] != "arXiv" else "")
        + "\n"
        f"- 链接：{paper.get('url', '')}\n\n"
        f"## 摘要\n\n{abstract}\n"
    )


def _build_fulltext_markdown(paper: dict, body_markdown: str) -> str:
    """在全文正文前拼上元数据头部（去掉正文中重复的标题行）。"""
    body = body_markdown
    first_block = body.split("\n\n", 1)[0].strip()
    if first_block.startswith("#") and (
        _normalize_title(first_block.lstrip("# ")) == _normalize_title(paper["title"])
    ):
        body = body.split("\n\n", 1)[1] if "\n\n" in body else ""

    authors = ", ".join(paper.get("authors", [])) or "未知"
    header = (
        f"# {paper['title']}\n\n"
        f"- 作者：{authors}\n"
        f"- 年份：{paper.get('year', '未知')}\n"
        f"- 来源：{paper['paper_id']}\n"
        f"- 链接：{paper.get('url', '')}\n"
    )
    return f"{header}\n{body.strip()}\n"


def _paper_filename(paper_id: str) -> str:
    """论文 markdown 文件名（确定性，作为 chunk 的 source 元数据）。"""
    return f"{paper_id.replace(':', '-')}.md"


def _fetch_paper_meta(paper_id: str) -> dict | None:
    """按 paper_id（arxiv:xxx / s2:xxx）获取论文元数据。"""
    prefix, _, oid = paper_id.partition(":")
    if prefix == "arxiv" and oid:
        return _fetch_arxiv_meta(oid)
    if prefix == "s2" and oid:
        return _fetch_s2_meta(oid)
    return None


def _build_paper_chunks(
    paper: dict,
    depth: str,
    markdown: str,
    filename: str,
    parser: TextParser,
) -> list:
    """将论文 Markdown 落盘并切分为带文献元数据的 Document 列表。"""
    lit_dir = Path(settings.literature_dir)
    lit_dir.mkdir(parents=True, exist_ok=True)
    file_path = lit_dir / filename
    file_path.write_text(markdown, encoding="utf-8")

    blocks = parser.parse(file_path)
    chunks = chunk_blocks(
        blocks=blocks,
        strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        source_name=filename,
    )
    extra_meta = {
        "paper_id": paper["paper_id"],
        "paper_title": paper["title"],
        "paper_year": paper.get("year"),
        "paper_authors": ", ".join(paper.get("authors", [])),
        "paper_url": paper.get("url", ""),
        "entry_type": "literature",
        "ingest_depth": depth,
    }
    for chunk in chunks:
        chunk.metadata.update(extra_meta)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 工具工厂
# ─────────────────────────────────────────────────────────────────────────────

def build_paper_tools(
    store: "ChromaVectorStore | None" = None,
    sparse_retriever: "SparseRetriever | None" = None,
) -> list:
    """
    构建文献搜索工具列表。

    参数
    ----
    store            : 文本向量库（paper_ingest 入库目标）。
    sparse_retriever : BM25 稀疏检索器（入库后重建索引）。
                       两者任一为 None 时仅注册 paper_search。
    """

    can_ingest = store is not None and sparse_retriever is not None
    _default_max_results = settings.paper_search_max_results
    _default_year_window = settings.paper_search_year_window

    @tool
    def paper_search(
        topic: str,
        max_results: int = _default_max_results,
        year_from: int | None = None,
        sort: str = "recent",
    ) -> str:
        """
        联网检索近几年与指定主题相关的学术论文（arXiv + Semantic Scholar 双源）。

        适用场景：
        - 用户想调研/搜索某个主题近几年的最新论文
        - 用户想了解某领域的研究进展、需要论文推荐

        注意：
        - topic 请使用英文关键词（中文主题需先翻译为英文），可用空格组合多个关键词
        - 返回结果中每篇论文都有唯一 paper_id，用户后续要求入库时必须传该 ID
        - 检索完成后请基于标题和摘要为每篇论文写 2~3 句简短介绍

        参数
        ----
        topic       : 英文搜索关键词，例如 "crop disease detection"。
        max_results : 返回论文数量（默认 8，建议 3~15）。
        year_from   : 起始年份（含）；不传则默认近 3 年。
        sort        : "recent" 按时间排序（默认），"cited" 按引用数排序。
        """
        from datetime import date

        now = date.today()
        year_from = year_from or (now.year - _default_year_window + 1)
        year_to = now.year
        logger.info(
            f"[Tool:paper_search] topic='{topic}' years={year_from}-{year_to} "
            f"max={max_results} sort={sort}"
        )

        try:
            papers = search_papers(
                topic, max_results=max_results,
                year_from=year_from, year_to=year_to, sort=sort,
            )
        except Exception as exc:
            logger.exception("[Tool:paper_search] 搜索异常")
            return json.dumps(
                {"results": [], "message": f"论文检索失败：{exc}"},
                ensure_ascii=False,
            )

        if not papers:
            return json.dumps(
                {
                    "results": [],
                    "message": (
                        f"未找到 {year_from}-{year_to} 年间与 “{topic}” 相关的论文，"
                        "建议更换英文关键词或放宽年份范围。"
                    ),
                },
                ensure_ascii=False,
            )

        results = []
        for rank, p in enumerate(papers, start=1):
            abstract = p.get("abstract", "")
            if len(abstract) > _ABSTRACT_MAX_CHARS:
                abstract = abstract[:_ABSTRACT_MAX_CHARS] + "…"
            results.append({
                "rank": rank,
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": ", ".join(p.get("authors", []))[:200],
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "citations": p.get("citations"),
                "url": p.get("url", ""),
                "pdf_url": p.get("pdf_url", ""),
                "fulltext_available": p["paper_id"].startswith("arxiv:"),
                "abstract": abstract,
                "source": p.get("source", ""),
                "block_type": "literature",
            })

        return json.dumps(
            {"results": results, "total": len(results)},
            ensure_ascii=False, indent=2,
        )

    tools = [paper_search]

    if not can_ingest:
        logger.debug("[Skills] paper_ingest 跳过（未提供 store / sparse_retriever）")
        return tools

    @tool
    def paper_ingest(paper_ids: str, depth: str = "abstract") -> str:
        """
        将之前 paper_search 检索到的论文入库到农业科研知识库。

        入库深度分级（轻量快速，无需 MinerU OCR）：
        - "abstract"（默认）：标题+作者+年份+摘要入库，秒级完成
        - "fulltext"：抓取 arXiv 官方 HTML 全文转 Markdown 入库，检索质量更好
          （仅支持 arXiv 论文；抓取失败或非 arXiv 论文自动降级为摘要级）

        注意：
        - paper_ids 必须是 paper_search 结果中返回的 paper_id 列表（JSON 数组），
          例如 '["arxiv:2401.12345", "s2:4f3e2d1c..."]'
        - 入库幂等：同一论文重复入库或切换深度不会产生重复数据

        参数
        ----
        paper_ids : JSON 编码的 paper_id 字符串列表。
        depth     : 入库深度，"abstract" 或 "fulltext"。
        """
        logger.info(f"[Tool:paper_ingest] paper_ids='{paper_ids[:200]}' depth={depth}")

        try:
            ids = json.loads(paper_ids)
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list) or not ids:
                raise ValueError("paper_ids 须为非空 JSON 数组")
        except (json.JSONDecodeError, ValueError) as exc:
            return json.dumps(
                {"status": "error", "message": f"paper_ids 参数无效：{exc}"},
                ensure_ascii=False,
            )

        if depth not in ("abstract", "fulltext"):
            depth = "abstract"

        router = TextParser()
        ingested: list[dict] = []
        failed: list[dict] = []
        all_chunks: list = []
        all_ids: list[str] = []

        for pid in ids:
            pid = str(pid).strip()
            if not pid:
                continue
            try:
                paper = _fetch_paper_meta(pid)
                if paper is None:
                    raise ValueError("论文元数据获取失败（ID 无效或网络错误）")

                actual_depth = depth
                note = ""
                if depth == "fulltext":
                    if pid.startswith("arxiv:"):
                        body = _fetch_arxiv_fulltext_markdown(pid.split(":", 1)[1])
                        if body:
                            markdown = _build_fulltext_markdown(paper, body)
                        else:
                            markdown = _build_abstract_markdown(paper)
                            actual_depth = "abstract"
                            note = "全文 HTML 抓取失败，已降级为摘要级入库"
                    else:
                        markdown = _build_abstract_markdown(paper)
                        actual_depth = "abstract"
                        note = "非 arXiv 论文，已降级为摘要级入库"
                else:
                    markdown = _build_abstract_markdown(paper)

                filename = _paper_filename(pid)
                chunks = _build_paper_chunks(paper, actual_depth, markdown, filename, router)

                # 幂等入库：先删旧 chunks，再以确定性 ID 写入
                store.delete_documents_by_filter({"paper_id": {"$eq": pid}})
                all_chunks.extend(chunks)
                all_ids.extend(f"paper::{pid.replace(':', '-')}::{i}" for i in range(len(chunks)))

                ingested.append({
                    "paper_id": pid,
                    "title": paper["title"],
                    "depth": actual_depth,
                    "chunks": len(chunks),
                    "file": filename,
                    "note": note,
                })
                logger.info(
                    f"[Tool:paper_ingest] ✓ {pid}（{actual_depth}）→ {len(chunks)} chunks"
                )
            except Exception as exc:
                logger.warning(f"[Tool:paper_ingest] ✗ {pid}：{exc}")
                failed.append({"paper_id": pid, "error": str(exc)})

        if all_chunks:
            store.add_documents(all_chunks, ids=all_ids)
            # 重建 BM25 索引（与 /v1/ingest 行为一致）
            try:
                corpus = store.get_all_documents(limit=10_000)
                sparse_retriever.build_index(corpus)
            except Exception as exc:
                logger.warning(f"[Tool:paper_ingest] BM25 重建失败：{exc}")

        return json.dumps(
            {
                "status": "success" if ingested else "error",
                "ingested": ingested,
                "failed": failed,
                "total_chunks": len(all_chunks),
                "message": (
                    f"已入库 {len(ingested)} 篇论文（{len(all_chunks)} 个 chunks），"
                    "现已可通过 knowledge_base_search 检索。"
                    if ingested else "入库失败，请检查 paper_id 是否来自 paper_search 结果。"
                ),
            },
            ensure_ascii=False, indent=2,
        )

    tools.append(paper_ingest)
    return tools
