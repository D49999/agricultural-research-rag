"""
Chroma 文本向量重刷 CLI
──────────────────────────────────────────────────────────────────────────────
切换 embedding 后端（例如 `EMBEDDING_MODE=api` → `local`）后，集合里已有的向量
来自旧模型，与新模型不在同一向量空间，必须整体重算，否则检索结果会被污染。

本脚本只重算 embedding：**不删除文档、不改内容、不改元数据**。

用法
────
# 先看当前配置与集合规模（默认 dry-run）
python scripts/reembed_collection.py

# 真正执行重算（需要几分钟，取决于模型与文档量）
python scripts/reembed_collection.py --apply

# 调整批大小（默认 16；CPU 上可调小，显存充足可调大）
python scripts/reembed_collection.py --apply --batch-size 8

注意
────
• 运行前建议停止后端，避免写入过程中有并发查询读到新旧混合的向量。
• 重算完成后重启后端，让 BM25 与向量索引都重新加载。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from config.settings import get_settings
from src.embeddings.qwen_embedding import QwenEmbeddings
from src.vectorstore.chroma_store import ChromaVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重算 Chroma 文本集合的全部向量。")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写入向量；不加此参数时只打印配置与规模。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="每批重算的文档数（默认 16）。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    logger.info(
        f"embedding_mode={settings.embedding_mode} | "
        f"model={settings.local_embedding_model if settings.embedding_mode == 'local' else settings.embedding_model} | "
        f"dim={settings.embedding_dim}"
    )

    store = ChromaVectorStore(embedding_fn=QwenEmbeddings())
    total = store.count()
    logger.info(f"集合 '{settings.chroma_collection_name}' 现有 {total} 条文档")

    if not args.apply:
        logger.info("dry-run：未写入任何向量。确认无误后加 --apply 执行。")
        return 0
    if total == 0:
        logger.warning("集合为空，无需重算。")
        return 0

    start = time.time()
    processed = store.reembed_all_documents(batch_size=args.batch_size)
    logger.info(
        f"完成：重算 {processed} 条，耗时 {time.time() - start:.1f}s，"
        f"集合文档数仍为 {store.count()}"
    )
    logger.info("下一步：重启后端，让检索与 BM25 索引加载新向量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
