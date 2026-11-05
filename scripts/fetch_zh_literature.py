"""
中文农业文献抓取 CLI 脚本
──────────────────────────────────────────────────────────────────────────────
用途：为知识库补充**中文农业权威资料**（农业农村部 / 全国农技推广服务中心等
公开发布的技术方案、防控意见、技术指导意见），下载后保存为 Markdown，
再由 `scripts/ingest_documents.py` 入库。

工作方式
────────
1. 直接抓取农业农村部「农事指导」栏目和历年「春耕技术指导」专栏的**列表页**
   （带翻页），比搜索引擎稳定：不依赖第三方检索结果，也不触发反爬；
2. 按标题关键词筛选出种植业生产/病虫害防控类文档，逐篇抓取正文；
3. 清洗导航/版权等噪音，写入 `data/documents/zh/{年份}_{标题}.md`，
   文件头保留来源、发布日期、原文链接，便于溯源；
4. `data/documents/zh/_sources.json` 记录已抓取的 URL，重复运行会跳过。

用法
────
# 默认入口（农事指导 + 2023/2024 春耕技术指导专栏，每页翻 2 页）
python scripts/fetch_zh_literature.py

# 只看会抓哪些页面，不落盘
python scripts/fetch_zh_literature.py --dry-run

# 自定义列表页（可重复传），翻 5 页，最多抓 60 篇
python scripts/fetch_zh_literature.py --index-page https://www.moa.gov.cn/gk/nszd_1/ --pages 5 --max-docs 60

# 只保留正文不少于 3000 字的文档
python scripts/fetch_zh_literature.py --min-chars 3000

说明
────
• 仅抓取公开发布的政府/事业单位技术资料，并保留原文链接；请遵守目标站点
  的 robots 与使用条款，控制抓取频率（默认每次请求间隔 1.5 秒）。
• 正文抽取基于启发式规则（正文容器 + 段落/表格块），建议抓完先抽查几篇。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "documents" / "zh"
MANIFEST_NAME = "_sources.json"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9"}

# 默认列表页：农业农村部「农事指导」栏目 + 2023/2024 春耕技术指导专栏
DEFAULT_INDEX_PAGES: tuple[str, ...] = (
    "https://www.moa.gov.cn/gk/nszd_1/",
    "https://www.moa.gov.cn/ztzl/2024cg/jszd_29651/",
    "https://www.moa.gov.cn/ztzl/2023cg/jszd_29356/",
)

# 列表页里的正文链接形如 ./202403/t20240308_6450966.htm
_ARTICLE_HREF_RE = re.compile(r"t20\d{6}_\d+\.s?html?$", re.IGNORECASE)

# 章节标题（一、防控目标 / 二、防控策略 ……），用来补 Markdown 二级标题
_SECTION_RE = re.compile(r"^[一二三四五六七八九十]+、\S")

# 标题命中任一关键词才抓：主要粮油作物 + 病虫害防控 + 生产技术指导
_KEEP_TITLE_RE = re.compile(
    r"(水稻|稻瘟病|稻飞虱|二化螟|小麦|条锈病|赤霉病|玉米|草地贪夜蛾|"
    r"马铃薯|晚疫病|油菜|菌核病|大豆|病虫害|病虫|防控|防治|"
    r"技术意见|技术方案|技术要点|技术指导|生产技术|田间管理|单产提升)"
)

# 明显的新闻/政策/非技术内容，命中即跳过
_SKIP_TITLE_RE = re.compile(
    r"(价格|生猪|猪肉|牛肉|羊肉|饲料|奶牛|蛋鸡|肉鸡|水产|渔船|"
    r"博览会|论坛|会议|座谈|培训班的通知|人事|任免|招标|采购|"
    r"首页|登录|注册|版权|联系我们|网站地图|更多|查看全部|搜索结果)"
)

# 正文容器候选（命中越多越好，取文字最长的那个）
_MAIN_SELECTORS = (
    "div.TRS_Editor",
    "div.arc_body",
    "div.article",
    "div.article-content",
    "div.article_content",
    "div.articleContent",
    "div.content_body",
    "div.pages_content",
    "div.zwxl-article",
    "div#zoom",
    "div.zoom",
    "div.content",
    "div#content",
    "article",
)

# 导航/版权/推荐位等噪音容器，按 class/id 关键字整体剔除
_NOISE_ATTR_RE = re.compile(
    r"(nav|menu|footer|copyright|crumb|breadcrumb|sidebar|side-bar|"
    r"share|related|recommend|previous|next|search|login|banner|"
    r"header|top_bar|toolbar)",
    re.IGNORECASE,
)
_NOISE_TAGS = ("script", "style", "nav", "footer", "header", "form", "iframe")

# 抽取后如果仍以这些词开头，说明抓到的是站点导航而不是正文
_NAV_MARKERS = (
    "机关子站",
    "直属单位网站",
    "国务院各部门网站",
    "地方农业管理部门网站",
    "关于我们",
    "网站声明",
)

_SOURCE_HINTS = (
    ("moa.gov.cn", "农业农村部"),
    ("natesc.org.cn", "全国农业技术推广服务中心"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抓取中文农业权威资料并保存为 Markdown（供入库使用）。"
    )
    parser.add_argument(
        "--index-page",
        action="append",
        dest="index_pages",
        help="列表页 URL，可重复传入；默认使用内置的农事指导/春耕专栏。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUT_DIR}）。",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=2,
        help="每个列表页翻页数（默认 2，即列表页本体 + index_1.htm）。",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=40,
        help="本次最多保存的文档数（默认 40）。",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1500,
        help="正文字数下限，低于该值的页面视为无效（默认 1500）。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="请求间隔秒数（默认 1.5）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印候选标题/链接，不下载不落盘。",
    )
    return parser.parse_args()


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _decode_html(raw: bytes) -> str:
    """按内容而不是响应头解码：UTF-8 优先，失败再退回 GB18030。"""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace")


def _text(node) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _title_ok(title: str) -> bool:
    if len(title) < 6:
        return False
    if not re.search(r"[\u4e00-\u9fff]", title):
        return False
    if _SKIP_TITLE_RE.search(title):
        return False
    return bool(_KEEP_TITLE_RE.search(title))


def list_article_links(
    session: requests.Session, index_url: str, pages: int
) -> list[tuple[str, str]]:
    """抓取列表页（含翻页），返回 [(标题, 正文链接)]。"""
    base = index_url if index_url.endswith("/") else f"{index_url}/"
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for page in range(max(1, pages)):
        url = base if page == 0 else urljoin(base, f"index_{page}.htm")
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            logger.debug(f"[列表页失败] {url}：{exc}")
            continue

        soup = BeautifulSoup(_decode_html(resp.content), "lxml")
        raw_count = 0
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if not _ARTICLE_HREF_RE.search(href):
                continue
            raw_count += 1
            title = _text(anchor)
            if not _title_ok(title):
                continue
            absolute = urljoin(url, href)
            if absolute in seen:
                continue
            seen.add(absolute)
            results.append((title, absolute))
        logger.debug(f"[列表页] {url} 正文链接 {raw_count} 条，保留 {len(results)} 条候选")

    return results


def _strip_noise(soup: BeautifulSoup) -> None:
    for name in _NOISE_TAGS:
        for tag in soup.find_all(name):
            tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag is None or tag.attrs is None:  # 祖先已被移除
            continue
        attrs = " ".join(tag.get("class") or []) + " " + (tag.get("id") or "")
        if attrs.strip() and _NOISE_ATTR_RE.search(attrs):
            try:
                tag.decompose()
            except Exception:
                continue


def extract_article(html: str) -> tuple[str, str]:
    """抽取正文与标题，返回 (标题, 正文 Markdown 近似文本)。"""
    soup = BeautifulSoup(html, "lxml")
    title = _text(soup.title) if soup.title else ""
    _strip_noise(soup)

    candidates = []
    for selector in _MAIN_SELECTORS:
        candidates.extend(soup.select(selector))
    if not candidates and soup.body is not None:
        # 兜底：取文字最多的 div（很多政府站点没有统一 class）
        candidates = [
            div for div in soup.find_all("div") if len(div.get_text(strip=True)) > 600
        ]
    if not candidates:
        return title, ""
    best = max(candidates, key=lambda node: len(node.get_text(strip=True)))

    blocks: list[str] = []
    for element in best.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = _text(element)
        if not text:
            continue
        if len(text) < 6 and element.name not in ("h1", "h2", "h3", "h4"):
            continue
        if element.name in ("h1", "h2", "h3", "h4"):
            level = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}[element.name]
            blocks.append(f"{level} {text}")
        elif element.name == "li":
            blocks.append(f"- {text}")
        elif _SECTION_RE.match(text):
            # 政府文档的章节标题多是「一、防控目标」这类加粗段落，补成二级标题，
            # 让入库切分能按章节边界切开，而不是整篇挤成一个 chunk
            blocks.append(f"## {text}")
        else:
            blocks.append(text)

    # 去掉与页面标题重复的首个标题行（文件头已经写了标题）
    while blocks and blocks[0].lstrip("# ").strip() == title.strip():
        blocks.pop(0)

    # 去掉连续重复行
    deduped: list[str] = []
    for block in blocks:
        if deduped and deduped[-1] == block:
            continue
        deduped.append(block)
    return title, "\n\n".join(deduped)


def looks_like_nav_boilerplate(body: str) -> bool:
    head = body[:200]
    return any(marker in head for marker in _NAV_MARKERS)


def guess_source(url: str) -> str:
    for hint, name in _SOURCE_HINTS:
        if hint in url:
            return name
    return "网络公开资料"


def guess_year(title: str, text: str) -> int | None:
    for candidate in re.findall(r"(20\d{2})\s*年", f"{title} {text[:1500]}"):
        year = int(candidate)
        if 2000 <= year <= 2035:
            return year
    return None


def year_from_url(url: str) -> int | None:
    """政府站点正文链接形如 .../t20240308_6450966.htm，可直接取发布年份。"""
    match = re.search(r"/t(20\d{2})\d{4}_\d+\.", url)
    return int(match.group(1)) if match else None


def safe_filename(year: int | None, title: str) -> str:
    stem = re.sub(r"[\\/:*?\"<>|\s]+", "_", title).strip("_")
    stem = stem[:60] or "zh_doc"
    return f"{year}_{stem}.md" if year else f"{stem}.md"


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"清单文件损坏，将重建：{path}")
    return {"sources": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def existing_body_hashes(out_dir: Path) -> set[str]:
    hashes: set[str] = set()
    if not out_dir.exists():
        return hashes
    for path in out_dir.glob("*.md"):
        body = path.read_text(encoding="utf-8", errors="ignore").split("---", 1)[-1]
        hashes.add(hashlib.md5(body.strip().encode("utf-8")).hexdigest())
    return hashes


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    index_pages = args.index_pages or list(DEFAULT_INDEX_PAGES)
    manifest_path = out_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path) if out_dir.exists() else {"sources": {}}
    known = set(manifest.get("sources", {}))
    body_hashes = existing_body_hashes(out_dir)

    session = _session()
    saved = 0
    skipped_known = 0
    failed: list[str] = []

    for index_url in index_pages:
        logger.info(f"[列表页] {index_url}")
        candidates = list_article_links(session, index_url, args.pages)
        if not candidates:
            logger.warning(f"[列表页] 未找到符合条件的正文链接：{index_url}")
        time.sleep(args.delay)

        for title, target in candidates:
            if saved >= args.max_docs:
                break
            if target in known:
                skipped_known += 1
                continue
            if args.dry_run:
                print(f"  - {title}\n    {target}")
                continue

            try:
                resp = session.get(target, timeout=25)
                resp.raise_for_status()
            except Exception as exc:
                logger.warning(f"[抓取失败] {target}：{exc}")
                failed.append(target)
                continue
            finally:
                time.sleep(args.delay)

            page_title, body = extract_article(_decode_html(resp.content))
            if len(body) < args.min_chars:
                logger.debug(f"[跳过] 正文过短（{len(body)} 字）：{target}")
                continue
            if looks_like_nav_boilerplate(body):
                logger.debug(f"[跳过] 抽到站点导航而非正文：{target}")
                continue

            body_hash = hashlib.md5(body.strip().encode("utf-8")).hexdigest()
            if body_hash in body_hashes:
                logger.debug(f"[跳过] 内容重复：{target}")
                continue

            real_title = page_title or title
            year = guess_year(real_title, body) or year_from_url(target)
            source = guess_source(target)
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(year, real_title)
            path = out_dir / filename
            counter = 2
            while path.exists():
                path = out_dir / f"{Path(filename).stem}_{counter}.md"
                counter += 1

            header = [
                f"# {real_title}",
                "",
                f"- 资料来源：{source}",
                f"- 原文链接：{target}",
            ]
            if year:
                header.append(f"- 发布年份：{year}")
            header.extend(["", "---", ""])

            path.write_text("\n".join(header) + body + "\n", encoding="utf-8")
            manifest.setdefault("sources", {})[target] = {
                "title": real_title,
                "file": path.name,
                "source": source,
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_manifest(manifest_path, manifest)
            known.add(target)
            body_hashes.add(body_hash)
            saved += 1
            logger.info(f"  ✓ {path.name}（{len(body)} 字）")

        if saved >= args.max_docs:
            logger.info(f"[停止] 已达到 --max-docs {args.max_docs}")
            break

    logger.info(
        f"[完成] 新保存 {saved} 篇，跳过已抓取 {skipped_known} 篇，失败 {len(failed)} 篇"
    )
    if saved and not args.dry_run:
        logger.info(
            f"下一步：python scripts/ingest_documents.py --input-dir {out_dir} "
            f"--chunk-strategy markdown"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
