"""
OCR 解析结果入库 CLI 脚本
──────────────────────────────────────────────────────────────────────────────
将 MinerU OCR 解析生成的 Markdown 文件批量写入 Chroma 向量数据库。

OCR 输出目录结构（由 mineru_ocr.py 生成）：
    {ocr_dir}/
      {文档名}/
        {backend}/
          {文档名}.md      ← 入库目标
          images/          ← 提取出的图片（暂不入库）

用法
────
# 入库默认目录 data/ocr_output 下所有文档
python scripts/ingest_ocr_output.py

# 指定其他 OCR 输出目录
python scripts/ingest_ocr_output.py --ocr-dir ./my_ocr_results

# 先清空集合再入库
python scripts/ingest_ocr_output.py --reset

# 指定切分策略（fixed / sentence / paragraph / markdown / recursive / semantic）
python scripts/ingest_ocr_output.py --chunk-strategy markdown
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from tqdm import tqdm

from config.settings import get_settings
from src.document_parser.base_parser import BlockType
from src.document_parser.chunker import SUPPORTED_STRATEGIES, chunk_blocks
from src.document_parser.text_parser import TextParser
from src.retrieval.sparse_retriever import SparseRetriever
from src.vectorstore.chroma_store import ChromaVectorStore

settings = get_settings()

DEFAULT_OCR_DIR = Path(__file__).resolve().parents[1] / "data" / "ocr_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 MinerU OCR 解析结果（Markdown）入库到农业科研 RAG 知识库。"
    )
    parser.add_argument(
        "--ocr-dir",
        type=Path,
        default=DEFAULT_OCR_DIR,
        help=f"OCR 输出根目录（默认：{DEFAULT_OCR_DIR}）。",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="入库前先清空现有集合。",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help=f"chunk 大小（近似 token 数，默认：{settings.chunk_size}）。",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help=f"chunk 重叠量（近似 token 数，默认：{settings.chunk_overlap}）。",
    )
    parser.add_argument(
        "--chunk-strategy",
        type=str,
        default=settings.chunk_strategy,
        choices=sorted(SUPPORTED_STRATEGIES),
        help=f"切分策略（默认：{settings.chunk_strategy}）。",
    )
    return parser.parse_args()


def collect_md_files(ocr_dir: Path) -> list[tuple[Path, str]]:
    """
    扫描 OCR 输出目录，返回 (md文件路径, 文档名) 列表。

    目录结构：{ocr_dir}/{文档名}/{backend}/{文档名}.md
    """
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR 输出目录不存在：{ocr_dir}")

    results: list[tuple[Path, str]] = []

    for doc_dir in sorted(ocr_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_name = doc_dir.name
        # 在所有 backend 子目录中查找 .md 文件
        md_files = sorted(doc_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"[扫描] {doc_name}：未找到 Markdown 文件，跳过")
            continue
        # 若同一文档有多个 .md（多个 backend），取第一个
        md_path = md_files[0]
        if len(md_files) > 1:
            logger.warning(
                f"[扫描] {doc_name}：发现 {len(md_files)} 个 Markdown 文件，"
                f"使用：{md_path.relative_to(ocr_dir)}"
            )
        results.append((md_path, doc_name))

    return results


# ── 页码回填 ──────────────────────────────────────────────────────────────
# MinerU 的 content_list.json 中每个块携带 page_idx（0 基），
# 而 Markdown 本身不含页码。这里通过内容前缀匹配（双指针顺序推进）
# 将页码回填到 TextParser 解析出的 block，供下游 chunk 元数据使用。

_NORM_KEEP_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")
_PREFIX_LEN = 60
_SEARCH_WINDOW = 150


def _normalize_text(text: str) -> str:
    """去空白/标点后仅保留字母数字与中文，用于模糊前缀匹配。"""
    return _NORM_KEEP_RE.sub("", text or "")


def _find_content_list(md_path: Path) -> Path | None:
    """在 md 所在目录及文档根目录中查找 *_content_list.json。"""
    for directory in (md_path.parent, *md_path.parents):
        candidates = sorted(directory.glob("*_content_list.json"))
        if candidates:
            return candidates[0]
    return None


def assign_page_numbers(md_path: Path, blocks: list) -> int:
    """
    根据 content_list.json 回填 blocks 的 page_num（1 基页码）。

    返回成功回填页码的 block 数量；无 content_list 时不做任何修改。
    """
    cl_path = _find_content_list(md_path)
    if cl_path is None:
        logger.warning(f"[页码] 未找到 content_list，页码保持默认：{md_path.name}")
        return 0

    try:
        content_list = json.loads(cl_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[页码] content_list 解析失败（{exc}）：{cl_path.name}")
        return 0

    # 按文件顺序保留带页码的文本/表格条目（归一化后非空）
    entries: list[tuple[str, int]] = []
    # 图片 basename → 页码
    image_pages: dict[str, int] = {}
    for item in content_list:
        page_idx = int(item.get("page_idx", 0))
        itype = item.get("type", "")
        # MinerU 会把插图分为 image 与 chart（图表/曲线图）两种类型
        if itype in ("image", "chart"):
            img_path = item.get("img_path", "")
            if img_path:
                image_pages[Path(img_path).name] = page_idx + 1
        elif itype in ("text", "table"):
            norm = _normalize_text(item.get("text", ""))
            if norm:
                entries.append((norm, page_idx + 1))

    assigned = 0
    cursor = 0
    last_page = 1

    for block in blocks:
        # 图片块：按 image_url 的文件名匹配
        if block.block_type == BlockType.FIGURE:
            url = block.metadata.get("image_url", "")
            if url:
                page = image_pages.get(Path(url).name)
                if page:
                    block.page_num = page
                    last_page = page
                    assigned += 1
                    continue

        # 文本/表格/标题块：用归一化前缀在条目序列中向前搜索
        if block.block_type == BlockType.HEADER:
            probe = _normalize_text(block.content.lstrip("#").strip())
        else:
            probe = _normalize_text(block.content)

        if probe:
            probe = probe[:_PREFIX_LEN]
            hit = None
            for j in range(cursor, min(cursor + _SEARCH_WINDOW, len(entries))):
                if entries[j][0].startswith(probe):
                    hit = j
                    break
            if hit is None:
                # 容错：短块（标题等）可能被合并在长条目内部
                for j in range(cursor, min(cursor + _SEARCH_WINDOW, len(entries))):
                    if probe and probe in entries[j][0][: len(probe) * 4]:
                        hit = j
                        break
            if hit is not None:
                block.page_num = entries[hit][1]
                last_page = entries[hit][1]
                cursor = hit + 1
                assigned += 1
                continue

        # 未匹配到：继承上一个已知页码
        block.page_num = last_page

    logger.info(
        f"[页码] {md_path.name}：{assigned}/{len(blocks)} 个 block 完成页码回填"
    )
    return assigned


def main() -> None:
    _start_time = time.time()
    args = parse_args()

    # ── 初始化 ────────────────────────────────────────────────────────────
    store = ChromaVectorStore()
    if args.reset:
        logger.warning("按要求重置集合……")
        store.delete_collection()

    text_parser = TextParser()
    sparse = SparseRetriever()

    # ── 收集文件 ──────────────────────────────────────────────────────────
    try:
        md_files = collect_md_files(args.ocr_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not md_files:
        logger.error(f"在 {args.ocr_dir} 中未找到任何 Markdown 文件，退出。")
        sys.exit(1)

    logger.info(f"发现 {len(md_files)} 个 OCR 文档，切分策略：{args.chunk_strategy}")

    # ── 处理循环 ──────────────────────────────────────────────────────────
    total_chunks = 0
    failed: list[str] = []

    for md_path, doc_name in tqdm(md_files, desc="入库中", unit="文档"):
        try:
            blocks = text_parser.parse(md_path)
            assign_page_numbers(md_path, blocks)
            chunks = chunk_blocks(
                blocks=blocks,
                strategy=args.chunk_strategy,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                source_name=doc_name,
            )

            if chunks:
                store.add_documents(chunks)
                total_chunks += len(chunks)
                logger.info(
                    f"✓ {doc_name}：{len(blocks)} blocks → {len(chunks)} chunks"
                    f"（来源：{md_path.relative_to(args.ocr_dir)}）"
                )
            else:
                logger.warning(f"⚠ {doc_name}：未生成任何 chunk")

        except Exception as exc:
            logger.error(f"✗ {doc_name}：{exc}")
            failed.append(doc_name)

    # ── 构建 BM25 索引 ────────────────────────────────────────────────────
    logger.info("正在构建 BM25 稀疏索引……")
    corpus = store.get_all_documents(limit=100_000)
    if corpus:
        sparse.build_index(corpus)

    # ── 汇总输出 ──────────────────────────────────────────────────────────
    elapsed = time.time() - _start_time
    print("\n" + "=" * 60)
    print(f"  OCR 输出目录 : {args.ocr_dir}")
    print(f"  已处理文档数 : {len(md_files) - len(failed)} / {len(md_files)}")
    print(f"  生成 chunk 数 : {total_chunks}")
    print(f"  库中总文档数 : {store.count()}")
    if failed:
        print(f"  失败文档数   : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"  总耗时       : {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
