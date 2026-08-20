"""
arXiv / LaTeXML HTML → Markdown 转换器
──────────────────────────────────────────────────────────────────────────────
将 arxiv.org/html（或 ar5iv）渲染的 LaTeXML 论文 HTML 转为 Markdown，
输出格式与 TextParser 的 Markdown 解析规则对齐：

  • <h1>~<h6>            → `#` 标题行（HEADER 块）
  • <math alttext="..."> → 行内 `$...$` / 独立块 `$$...$$`（FORMULA 块）
  • <table>（数据表）    → 原样保留 HTML 表格（TABLE 块，subtype=html_table）
  • <img> + <figcaption> → `![alt](src)` 紧跟图注行（FIGURE 块，自动合并图注）
  • <ul>/<ol>/<li>       → Markdown 列表
  • <pre>                → 围栏代码块（TEXT 块，subtype=code_block）
  • 导航/页眉/页脚/脚本   → 丢弃

仅依赖标准库（html.parser），针对 LaTeXML 生成的规整 HTML 设计，
不追求通用 HTML 转换的完备性。
"""
from __future__ import annotations

import re
from html.parser import HTMLParser

# 无条件丢弃的标签（含其全部子树）
_SKIP_TAGS = {
    "script", "style", "noscript", "svg", "head", "nav", "header",
    "footer", "button", "form", "select", "canvas", "iframe",
}

# HTML void 标签（无闭合标签）
_VOID_TAGS = {
    "br", "img", "hr", "input", "meta", "link", "col", "area",
    "base", "embed", "source", "track", "wbr",
}

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# 提取 <article> 子树（LaTeXML 论文正文容器）
_ARTICLE_RE = re.compile(r"<article\b.*?</article>", re.DOTALL | re.IGNORECASE)

# 触发段落边界的块级标签
_BLOCK_BOUNDARY_TAGS = {
    "p", "div", "section", "article", "figure", "blockquote",
    "table", "ul", "ol", "dl", "hr", "aside", "main",
} | set(_HEADING_TAGS)


class _LatexmlToMarkdown(HTMLParser):
    """LaTeXML HTML → Markdown 的有状态转换器。"""

    def __init__(self, base_url: str = "") -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        # 输出的块级元素列表（最终以空行连接）
        self._blocks: list[str] = []
        # 当前段落/标题正在累积的内联文本
        self._inline: list[str] = []
        # 丢弃子树状态
        self._skip_stack: list[str] = []
        # 数学公式：深度 > 0 表示在 <math> 内（子内容全部抑制）
        self._math_depth = 0
        self._math_alttext = ""
        self._math_display = False
        # 表格原始 HTML 重建模式
        self._table_raw: list[str] | None = None
        self._table_depth = 0
        # 标题级别（0 表示不在标题内）
        self._heading_level = 0
        # 图注捕获
        self._in_caption = False
        self._caption: list[str] = []
        # 列表状态
        self._list_stack: list[dict] = []   # {"ordered": bool, "index": int}
        self._li_marker = ""
        # 代码块捕获
        self._pre_raw: list[str] | None = None

    # ──────────────────────────────────────────────────────────────────────
    # 对外入口
    # ──────────────────────────────────────────────────────────────────────

    def result(self) -> str:
        """返回转换后的 Markdown 文本。"""
        self._flush_paragraph()
        text = "\n\n".join(b for b in self._blocks if b.strip())
        # 清理不间断空格与多余空行
        text = text.replace("\xa0", " ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ──────────────────────────────────────────────────────────────────────
    # HTMLParser 回调
    # ──────────────────────────────────────────────────────────────────────

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_stack:
            if tag not in _VOID_TAGS:
                self._skip_stack.append(tag)
            return

        attr_map = {k: (v or "") for k, v in attrs}
        classes = attr_map.get("class", "")

        # ── 数学公式：优先于表格处理（公式可能嵌在公式表格内） ──────────
        if tag == "math":
            self._math_alttext = attr_map.get("alttext", "").strip()
            self._math_display = attr_map.get("display") == "block"
            if self._table_raw is not None:
                # 表格单元格内的公式：直接以 LaTeX 文本写入原始 HTML
                if self._math_alttext:
                    self._table_raw.append(f"${self._math_alttext}$")
            self._math_depth = 1
            return

        if self._math_depth > 0:
            # math 子树全部抑制
            if tag not in _VOID_TAGS:
                self._math_depth += 1
            return

        # ── 原始表格重建模式 ────────────────────────────────────────────
        if self._table_raw is not None:
            self._table_raw.append(self.get_starttag_text() or f"<{tag}>")
            if tag == "table":
                self._table_depth += 1
            return

        # ── 丢弃标签 ────────────────────────────────────────────────────
        if tag in _SKIP_TAGS:
            self._flush_paragraph()
            self._skip_stack.append(tag)
            return

        # ── 标题 ────────────────────────────────────────────────────────
        if tag in _HEADING_TAGS:
            self._flush_paragraph()
            self._heading_level = _HEADING_TAGS[tag]
            return

        # ── 数据表格：进入原始重建模式 ─────────────────────────────────
        # LaTeXML 的行间公式也用 <table> 包裹（class 含 ltx_equation），
        # 这类"表格"不是数据表，跳过以让内部 math 正常处理。
        if tag == "table":
            if "ltx_equation" in classes or "ltx_eqn_table" in classes:
                self._flush_paragraph()
                return
            self._flush_paragraph()
            self._table_raw = [self.get_starttag_text() or "<table>"]
            self._table_depth = 1
            return

        # ── 图片 ────────────────────────────────────────────────────────
        if tag == "img":
            self._flush_paragraph()
            alt = re.sub(r"[\[\]\n\r]", " ", attr_map.get("alt", "")).strip()
            src = self._absolutize(attr_map.get("src", ""))
            if src:
                self._blocks.append(f"![{alt}]({src})")
            return

        # ── 图注 ────────────────────────────────────────────────────────
        if tag == "figcaption":
            self._flush_paragraph()
            self._in_caption = True
            self._caption = []
            return

        # ── 代码块 ──────────────────────────────────────────────────────
        if tag == "pre":
            self._flush_paragraph()
            self._pre_raw = []
            return

        # ── 列表 ────────────────────────────────────────────────────────
        if tag in ("ul", "ol"):
            self._flush_paragraph()
            self._list_stack.append(
                {"ordered": tag == "ol", "index": 0, "marker": self._li_marker}
            )
            self._li_marker = ""
            return
        if tag == "li":
            self._flush_paragraph()
            if self._list_stack:
                top = self._list_stack[-1]
                if top["ordered"]:
                    top["index"] += 1
                    self._li_marker = f"{top['index']}. "
                else:
                    self._li_marker = "- "
            return

        # ── 块级边界：刷新当前段落 ──────────────────────────────────────
        if tag in _BLOCK_BOUNDARY_TAGS:
            self._flush_paragraph()
            return

        if tag == "br":
            self._inline.append("\n")

    def handle_endtag(self, tag: str) -> None:
        # ── 丢弃模式退出 ─────────────────────────────────────────────────
        if self._skip_stack:
            if self._skip_stack and tag == self._skip_stack[-1]:
                self._skip_stack.pop()
            return

        # ── 数学公式结束 ─────────────────────────────────────────────────
        if tag == "math":
            if self._math_depth > 0:
                self._math_depth -= 1
                if self._math_depth == 0 and self._table_raw is None:
                    self._emit_math(self._math_alttext, self._math_display)
                self._math_alttext = ""
                self._math_display = False
            return
        if self._math_depth > 0:
            self._math_depth -= 1
            return

        # ── 表格重建模式退出 ─────────────────────────────────────────────
        if self._table_raw is not None:
            self._table_raw.append(f"</{tag}>")
            if tag == "table":
                self._table_depth -= 1
                if self._table_depth <= 0:
                    self._blocks.append("\n".join(self._table_raw).strip())
                    self._table_raw = None
                    self._table_depth = 0
            return

        # ── 标题结束 ─────────────────────────────────────────────────────
        if tag in _HEADING_TAGS:
            text = self._collapse("".join(self._inline))
            self._inline = []
            if text:
                self._blocks.append("#" * self._heading_level + " " + text)
            self._heading_level = 0
            return

        if tag == "figcaption":
            self._in_caption = False
            caption = self._collapse("".join(self._caption))
            self._caption = []
            if not caption:
                return
            # 图片块紧跟图注行 → TextParser 会合并为带 caption 的 FIGURE 块
            if self._blocks and self._blocks[-1].startswith("!["):
                self._blocks[-1] = f"{self._blocks[-1]}\n{caption}"
            else:
                self._blocks.append(caption)
            return

        if tag == "pre":
            if self._pre_raw is not None:
                code = "".join(self._pre_raw).strip("\n")
                if code.strip():
                    self._blocks.append(f"```\n{code}\n```")
                self._pre_raw = None
            return

        if tag == "li":
            self._flush_paragraph()
            self._li_marker = ""
            return

        if tag in ("ul", "ol"):
            if self._list_stack:
                self._li_marker = self._list_stack.pop()["marker"]
            return

        if tag == "p":
            self._flush_paragraph()
            return

        if tag in _BLOCK_BOUNDARY_TAGS:
            self._flush_paragraph()

    def handle_data(self, data: str) -> None:
        if self._skip_stack or self._math_depth > 0:
            return
        if self._table_raw is not None:
            self._table_raw.append(data)
            return
        if self._pre_raw is not None:
            self._pre_raw.append(data)
            return
        if self._in_caption:
            self._caption.append(data)
            return
        self._inline.append(data)

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _flush_paragraph(self) -> None:
        """将累积的内联文本刷新为块级段落。"""
        text = self._collapse("".join(self._inline))
        self._inline = []
        if not text:
            return
        if self._li_marker:
            text = self._li_marker + text
        self._blocks.append(text)

    def _emit_math(self, alttext: str, display: bool) -> None:
        """输出数学公式：行内 $...$ 或独立 $$...$$ 块。"""
        alt = alttext.strip()
        if not alt:
            return
        if display:
            if "\n" in alt:
                self._blocks.append(f"$$\n{alt}\n$$")
            else:
                self._blocks.append(f"$${alt}$$")
        elif self._in_caption:
            self._caption.append(f"${alt}$")
        else:
            self._inline.append(f"${alt}$")

    def _absolutize(self, src: str) -> str:
        """将相对图片地址补全为绝对 URL（远程 URL 不做本地化）。"""
        if not src:
            return ""
        if src.startswith(("http://", "https://", "data:")):
            return src
        if src.startswith("//"):
            return "https:" + src
        base = self._base_url.rstrip("/") + "/"
        # 简易 urljoin（避免引入 urllib.parse 对 ./ ../ 的边界差异）
        from urllib.parse import urljoin
        return urljoin(base, src)

    @staticmethod
    def _collapse(text: str) -> str:
        """压缩内联文本中的空白为单个空格。"""
        return re.sub(r"\s+", " ", text).strip()


def arxiv_html_to_markdown(html: str, base_url: str = "") -> str:
    """
    将 arXiv / ar5iv 的 LaTeXML 论文 HTML 转换为 Markdown。

    参数
    ----
    html     : 论文页面原始 HTML。
    base_url : 页面 URL，用于将图片相对地址补全为绝对地址。
    """
    # 只取 <article> 子树，丢弃页面横幅/导航等 article 之外的内容
    match = _ARTICLE_RE.search(html)
    if match:
        html = match.group(0)
    parser = _LatexmlToMarkdown(base_url=base_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        # 解析中断时尽量返回已转换部分
        pass
    return parser.result()
