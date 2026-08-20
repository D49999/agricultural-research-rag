"""
Nature 风格学术润色 Skill
──────────────────────────────────────────────────────────────────────────────
将外部 nature-polishing skill（静态/动态双层结构）封装为 LangChain Tool。

实现 skill 的路由协议（SKILL.md）：
1. 构建时加载 manifest.yaml 与 always_load 核心片段；
2. 每次调用按入参解析各轴（paper_type / section / language / journal），
   仅加载命中的片段（未提供或非法时回退到 manifest 默认值）；
3. 按优先级组装润色规范 prompt，调用 LLM 完成润色；
4. 以 JSON 返回润色结果与所使用的轴信息。

skill 目录通过 settings.nature_polishing_skill_dir 配置，
可用环境变量 NATURE_POLISHING_SKILL_DIR 覆盖。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from loguru import logger

from config.settings import get_settings

# 中日韩字符范围，用于 language 轴自动检测
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 中文占比超过该阈值时判定为 zh-to-en
_CJK_RATIO_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# 片段加载
# ---------------------------------------------------------------------------

def _load_manifest(skill_dir: Path) -> dict:
    """读取并解析 skill 的 manifest.yaml，失败时抛出异常。"""
    manifest_path = skill_dir / "manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    if "axes" not in manifest:
        raise ValueError("manifest.yaml 缺少 axes 定义")
    return manifest


def _read_fragment(skill_dir: Path, rel_path: str) -> str:
    """读取单个片段文件；路径相对 skill 根目录（支持 ../_shared 跨目录引用）。"""
    path = (skill_dir / rel_path).resolve()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning(f"[Skill:paper_polishing] 片段读取失败 {rel_path}：{exc}")
        return ""


def _detect_language(text: str) -> str:
    """根据 CJK 字符占比检测源稿语言：中文草稿 → zh-to-en，否则 en。"""
    if not text:
        return "en"
    cjk_count = len(_CJK_RE.findall(text))
    ratio = cjk_count / len(text)
    return "zh-to-en" if ratio > _CJK_RATIO_THRESHOLD else "en"


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def build_paper_polishing_tool(llm: Any, skill_dir: str | Path | None = None):
    """
    返回 Nature 风格学术润色 LangChain Tool。

    参数
    ----
    llm      : LangChain ChatModel 实例，用于执行润色生成。
    skill_dir: 可选，nature-polishing skill 根目录（含 manifest.yaml）。
               未提供时使用 settings.nature_polishing_skill_dir。
    """

    settings = get_settings()
    base_dir = Path(skill_dir) if skill_dir else Path(settings.nature_polishing_skill_dir)

    # 构建期一次性加载 manifest 与 always_load 核心片段（skill 路由协议第 1 步）
    try:
        manifest = _load_manifest(base_dir)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error(f"[Skill:paper_polishing] manifest 加载失败，工具不可用：{exc}")
        manifest = None

    core_fragments: list[tuple[str, str]] = []  # (标题, 内容)
    if manifest is not None:
        for rel in manifest.get("always_load", []):
            content = _read_fragment(base_dir, rel)
            if content:
                core_fragments.append((f"core: {rel}", content))
        logger.info(
            f"[Skill:paper_polishing] skill 就绪，核心片段 {len(core_fragments)} 个"
        )

    def _resolve_axis(axis: str, raw: str, text: str) -> list[str]:
        """
        解析单个轴的取值列表（skill 路由协议第 2 步）。

        - 未提供 → manifest 默认值（language 轴额外支持从文稿自动检测）
        - 非法值 → 回退默认值并告警
        - multi 轴（section）支持逗号分隔多值
        """
        spec = (manifest or {}).get("axes", {}).get(axis, {})
        values: dict = spec.get("values", {})
        default = spec.get("default", "")

        # language 轴未指定时从文稿自动检测
        if axis == "language" and not raw.strip():
            raw = _detect_language(text)

        candidates = [v.strip() for v in raw.split(",") if v.strip()]
        if not spec.get("multi", False):
            candidates = candidates[:1]

        resolved: list[str] = []
        for v in candidates:
            if v in values and v not in resolved:
                resolved.append(v)
            elif v not in values:
                logger.warning(f"[Skill:paper_polishing] 轴 {axis} 非法值 '{v}'，已忽略")
                # 仅有默认值的轴才回退；无默认值的轴（如 section）跳过非法值
                if default in values and default not in resolved:
                    resolved.append(default)

        # 单值轴未提供取值时应用默认值（section 轴留空表示跳过，属正常）
        if not resolved and default in values and not raw.strip():
            resolved = [default]
        return resolved

    @tool
    def paper_polishing(
        text: str,
        paper_type: str = "",
        section: str = "",
        language: str = "",
        journal: str = "",
    ) -> str:
        """
        对学术文稿进行 Nature 风格润色、重构或中译英，返回润色后文本与修改说明。

        适用场景：
        - 用户要求润色论文的摘要/引言/结果/讨论/结论/标题/方法等章节
        - 将中文学术草稿改写为发表级英文（zh-to-en）
        - 一般性的学术写作、SCI 写作、语言润色请求

        参数
        ----
        text       : 待润色的文稿内容。
        paper_type : 论文类型 research|methods|hypothesis|algorithmic|review，
                     留空默认 research。
        section    : 目标章节，可逗号分隔多个：
                     abstract|intro|results|discussion|conclusion|title|methods，
                     留空表示无章节上下文的自由文本。
        language   : en 或 zh-to-en，留空时根据文稿语言自动检测。
        journal    : nature|nat-comms|generic，留空默认 generic。
        """
        logger.info(
            f"[Skill:paper_polishing] text_len={len(text)} "
            f"paper_type='{paper_type}' section='{section}' "
            f"language='{language}' journal='{journal}'"
        )

        if not text.strip():
            return json.dumps(
                {"status": "error", "message": "输入文稿为空，无法润色。"},
                ensure_ascii=False,
            )

        if manifest is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": (
                        f"nature-polishing skill 未就绪：在 {base_dir} 找不到有效的 "
                        "manifest.yaml，请检查 NATURE_POLISHING_SKILL_DIR 配置。"
                    ),
                },
                ensure_ascii=False,
            )

        # ── 路由协议第 2 步：解析各轴 ────────────────────────────────────────
        axes_resolved = {
            "paper_type": _resolve_axis("paper_type", paper_type, text),
            "section": _resolve_axis("section", section, text),
            "language": _resolve_axis("language", language, text),
            "journal": _resolve_axis("journal", journal, text),
        }

        # ── 路由协议第 3 步：按需加载轴片段 ──────────────────────────────────
        def load_axis(axis: str) -> str:
            spec = manifest["axes"][axis]
            values = spec["values"]
            parts: list[str] = []
            for v in axes_resolved[axis]:
                content = _read_fragment(base_dir, values[v])
                if content:
                    parts.append(f"--- {axis}={v} ---\n{content}")
            return "\n\n".join(parts)

        core_text = "\n\n".join(
            f"--- {title} ---\n{content}" for title, content in core_fragments
        )
        paper_type_text = load_axis("paper_type")
        section_text = load_axis("section")
        journal_text = load_axis("journal")
        language_text = load_axis("language")

        # ── 路由协议第 4 步：按 skill 优先级组装润色规范 ──────────────────────
        axes_desc: dict[str, Any] = {}
        for axis_name, vals in axes_resolved.items():
            if not vals:
                continue
            # multi 轴（section）保持列表，单值轴折叠为标量
            if manifest["axes"][axis_name].get("multi", False):
                axes_desc[axis_name] = vals
            else:
                axes_desc[axis_name] = vals[0]
        prompt = (
            "你是 Nature 风格学术润色助手。请严格按照下方【润色规范】处理【待润色文稿】。\n\n"
            "【润色规范】（来自 nature-polishing skill，按优先级从高到低应用："
            "论文类型 → 章节职责 → 期刊风格 → 语言规则，核心立场贯穿始终）\n\n"
            f"=== 核心立场与通用规则 ===\n{core_text}\n\n"
            f"=== 论文类型规范 ===\n{paper_type_text}\n\n"
            f"=== 章节规范 ===\n{section_text or '（无章节上下文，跳过章节轴）'}\n\n"
            f"=== 期刊风格 ===\n{journal_text}\n\n"
            f"=== 语言规则 ===\n{language_text}\n\n"
            "【执行要求】\n"
            "1. 先诊断文稿的主要问题（结构层面 vs 句子层面），修复顺序："
            "论文类型逻辑 → 章节职责 → 段落逻辑 → 论断/证据/边界 → 句子润色。\n"
            "2. 不编造数据、参考文献、机制或新颖性声明；若段落的结构问题无法在不发明"
            "内容的前提下修复，请在修改说明中明确指出，而不是用润色掩盖。\n"
            "3. 建立术语表并保持术语、缩写、单位、符号全篇一致，不为行文变化引入同义词。\n"
            "4. 输出格式：\n"
            "   (1) 先给出润色后的完整正文（英文，纯文本，不要放在代码块中）；\n"
            "   (2) 然后以 'Revision notes:' 开头，用 3-5 条简短中文要点说明主要修改；\n"
            "   (3) 若重写改变了章节逻辑，请明确说明。\n\n"
            f"【待润色文稿】\n{text}"
        )

        # ── 调用 LLM 生成润色结果 ──────────────────────────────────────────
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            polished = response.content.strip()
        except Exception as exc:
            logger.warning(f"[Skill:paper_polishing] 润色失败：{exc}")
            return json.dumps(
                {"status": "error", "message": f"润色生成失败：{exc}"},
                ensure_ascii=False,
            )

        logger.info(f"[Skill:paper_polishing] 润色完成，输出 {len(polished)} 字")
        return json.dumps(
            {
                "status": "ok",
                "axes": axes_desc,
                "polished": polished,
            },
            ensure_ascii=False,
            indent=2,
        )

    return paper_polishing
