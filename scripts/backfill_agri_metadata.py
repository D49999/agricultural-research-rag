"""
已入库文档的农业元数据回填 CLI 脚本
──────────────────────────────────────────────────────────────────────────────
P0 农业结构化检索依赖 chunk 上的 crop / disease / year / doc_type 等元数据。
但在这套元数据上线之前入库的旧数据没有这些字段，带过滤条件的检索只能
「命中 0 条 → 自动放宽为全库检索」，结构化过滤实际不起作用。

本脚本按 `source` 把 chunk 分组还原成「伪文档」，用与入库时完全相同的
`extract_document_metadata` / `extract_chunk_metadata` 重新抽取，并只更新
Chroma 的 metadata：

  • 不重算向量（向量与文本都没变，语义检索结果不受影响）；
  • 不删除、不重建集合（非破坏性，可反复执行，结果幂等）；
  • 默认只做 dry-run，只打印统计与样例，加 --apply 才真正写回。

用法
────
# 只看会补出什么（默认 dry-run）
python scripts/backfill_agri_metadata.py

# 真正写回（建议先停掉后端，避免 Chroma 写锁争用）
python scripts/backfill_agri_metadata.py --apply

# 覆盖已有的农业字段（默认只补缺失字段）
python scripts/backfill_agri_metadata.py --apply --overwrite

# 抽取规则变更后：覆盖并删除旧规则误标的字段（如 crop=棉花）
python scripts/backfill_agri_metadata.py --apply --overwrite --prune

# 指定其它集合 / 目录，或只处理前 N 条 chunk
python scripts/backfill_agri_metadata.py --collection agri_rag --limit 200
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from loguru import logger

from config.settings import get_settings
from src.agri.metadata_extractor import (
    extract_chunk_metadata,
    extract_document_metadata,
)

settings = get_settings()

# 回填的农业字段（与 metadata_extractor 的输出保持一致）
AGRI_FIELDS: tuple[str, ...] = (
    "crop",
    "crops",
    "variety",
    "disease",
    "diseases",
    "pest",
    "pests",
    "pesticide",
    "pesticides",
    "region",
    "regions",
    "growth_stage",
    "growth_stages",
    "year",
    "doc_type",
    "authority_level",
    "metadata_version",
)

_FETCH_BATCH = 500
_UPDATE_BATCH = 100
# 伪文档正文上限：与入库时文档级抽取的取样量级接近即可
_DOC_TEXT_LIMIT = 40000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为已入库的 Chroma 文档回填农业元数据（默认 dry-run）。"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="真正写回 Chroma metadata；不加此参数只做 dry-run。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的农业字段（默认只补缺失字段）。",
    )
    parser.add_argument(
        "--collection",
        default=settings.chroma_collection_name,
        help=f"Chroma 集合名（默认 {settings.chroma_collection_name}）。",
    )
    parser.add_argument(
        "--persist-dir",
        default=settings.chroma_persist_dir,
        help="Chroma 持久化目录（默认取配置）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理多少条 chunk（0 表示全量）。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="dry-run 时打印的样例条数（默认 5）。",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help=(
            "删除新规则不再产出的农业字段（如旧规则误标的 crop）；"
            "常与 --overwrite 一起使用。默认关闭，避免意外丢字段。"
        ),
    )
    return parser.parse_args()


def fetch_all(
    collection, limit: int
) -> tuple[list[str], list[str], list[dict], list[list[float]]]:
    """分页读取全部 chunk（文档文本 + 元数据 + 向量）。"""
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []
    offset = 0
    while True:
        size = _FETCH_BATCH
        if limit:
            size = min(size, limit - len(ids))
            if size <= 0:
                break
        result = collection.get(
            limit=size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        batch_ids = result.get("ids") or []
        if not batch_ids:
            break
        ids.extend(batch_ids)
        documents.extend(result.get("documents") or [])
        metadatas.extend(result.get("metadatas") or [{} for _ in batch_ids])
        batch_embeddings = result.get("embeddings")
        embeddings.extend(
            [list(vec) for vec in batch_embeddings]
            if batch_embeddings is not None
            else [[] for _ in batch_ids]
        )
        offset += len(batch_ids)
    return ids, documents, metadatas, embeddings


def build_doc_metadata(
    documents: list[str], indices: list[int], source: str
) -> dict:
    """把一个 source 下的所有 chunk 拼成伪文档，抽取文档级元数据。"""
    text = "\n\n".join(documents[i] or "" for i in indices)[:_DOC_TEXT_LIMIT]
    return extract_document_metadata(text, source)


def merge_metadata(
    existing: dict, extracted: dict, overwrite: bool, prune: bool = False
) -> tuple[dict, list[str], list[str]]:
    """把抽取结果合并进原 metadata，返回 (新 metadata, 新增/覆盖字段, 删除字段)。"""
    merged = dict(existing or {})
    added: list[str] = []
    removed: list[str] = []
    for field, value in extracted.items():
        if value in (None, "", []):
            continue
        old = merged.get(field)
        if old not in (None, "") and not overwrite:
            continue
        if old == value:
            continue
        merged[field] = value
        added.append(field)
    if prune:
        # 新规则不再产出的字段说明旧标注有误（例如把「棉铃虫」标成 crop=棉花）
        for field in AGRI_FIELDS:
            if field == "metadata_version":
                continue
            if field in merged and field not in extracted:
                merged.pop(field)
                removed.append(field)
    return merged, added, removed


def main() -> int:
    args = parse_args()
    persist_dir = Path(args.persist_dir)
    if not persist_dir.exists():
        logger.error(f"Chroma 目录不存在：{persist_dir}")
        return 1

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(args.collection)
    except Exception as exc:  # 集合不存在
        logger.error(f"读取集合 '{args.collection}' 失败：{exc}")
        return 1

    total = collection.count()
    logger.info(f"[回填] 集合 '{args.collection}' 共 {total} 条 chunk，目录 {persist_dir}")
    if total == 0:
        logger.warning("集合为空，无事可做。")
        return 0

    t0 = time.time()
    ids, documents, metadatas, embeddings = fetch_all(collection, args.limit)
    logger.info(f"[回填] 已读取 {len(ids)} 条 chunk（{time.time() - t0:.1f}s）")

    # 按 source 分组，尽量还原「文档 → chunk」的原始结构
    groups: dict[str, list[int]] = {}
    for idx, meta in enumerate(metadatas):
        source = str((meta or {}).get("source") or "")
        groups.setdefault(source, []).append(idx)
    logger.info(f"[回填] 归并出 {len(groups)} 个来源文档")

    updates: list[tuple[str, dict]] = []
    # 需要删除字段的 chunk：Chroma 的 update 是「合并」语义，删不掉键，只能删后重加
    rewrites: list[tuple[str, str, dict, list[float]]] = []
    field_hits: Counter[str] = Counter()
    field_added: Counter[str] = Counter()
    field_removed: Counter[str] = Counter()
    unchanged = 0
    samples: list[tuple[str, dict]] = []

    for source, indices in groups.items():
        doc_metadata = build_doc_metadata(documents, indices, source)
        for idx in indices:
            extracted = extract_chunk_metadata(documents[idx] or "", doc_metadata)
            merged, added, removed = merge_metadata(
                metadatas[idx], extracted, args.overwrite, args.prune
            )
            for field in extracted:
                if extracted.get(field) not in (None, ""):
                    field_hits[field] += 1
            if not added and not removed:
                unchanged += 1
                continue
            for field in added:
                field_added[field] += 1
            for field in removed:
                field_removed[field] += 1
            if removed:
                rewrites.append(
                    (
                        ids[idx],
                        documents[idx] or "",
                        merged,
                        embeddings[idx] if idx < len(embeddings) else [],
                    )
                )
            else:
                updates.append((ids[idx], merged))
            if len(samples) < max(args.samples, 0):
                samples.append((ids[idx], merged))

    logger.info(
        f"[回填] {len(updates) + len(rewrites)} 条 chunk 需要更新"
        f"（其中 {len(rewrites)} 条需删除字段后重写），{unchanged} 条已是最新"
    )
    if field_added:
        logger.info(
            "[回填] 将写入字段："
            + "，".join(f"{k}={v}" for k, v in field_added.most_common())
        )
    if field_removed:
        logger.info(
            "[回填] 将删除字段："
            + "，".join(f"{k}={v}" for k, v in field_removed.most_common())
        )
    if field_hits and not args.overwrite:
        logger.info(
            "[回填] 识别命中（含已有字段）："
            + "，".join(f"{k}={v}" for k, v in field_hits.most_common())
        )

    for chunk_id, meta in samples:
        preview = {k: meta[k] for k in AGRI_FIELDS if meta.get(k) not in (None, "")}
        logger.info(f"[样例] {chunk_id[:24]}… source={meta.get('source')} → {preview}")

    if not args.apply:
        logger.warning(
            "[回填] dry-run 结束，未写入任何数据；确认无误后加 --apply 执行。"
        )
        logger.warning("[回填] 正式写回前建议先停止后端服务，避免 Chroma 写锁争用。")
        return 0

    written = 0
    for start in range(0, len(updates), _UPDATE_BATCH):
        batch = updates[start : start + _UPDATE_BATCH]
        collection.update(
            ids=[chunk_id for chunk_id, _ in batch],
            metadatas=[meta for _, meta in batch],
        )
        written += len(batch)
        logger.debug(f"[回填] 已写入 {written}/{len(updates)} 条")

    rewritten = 0
    for chunk_id, content, meta, vector in rewrites:
        if not vector:
            logger.warning(f"[回填] {chunk_id[:12]}… 缺少向量，跳过重写（删除字段未生效）")
            continue
        collection.delete(ids=[chunk_id])
        collection.add(
            ids=[chunk_id],
            documents=[content],
            metadatas=[meta],
            embeddings=[vector],
        )
        rewritten += 1
        if rewritten % _UPDATE_BATCH == 0:
            logger.debug(f"[回填] 已重写 {rewritten}/{len(rewrites)} 条")

    logger.info(
        f"[回填] 完成：更新 {written} 条、重写 {rewritten} 条 chunk"
        f"（耗时 {time.time() - t0:.1f}s）"
    )
    logger.info(
        "[回填] 提示：BM25 索引是内存索引，需重启后端（或重新执行入库）后才会带上新元数据。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
