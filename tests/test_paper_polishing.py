"""
paper_polishing skill（nature-polishing 集成）的单元测试。
所有 LLM 调用均已 mock；skill 目录用 tempfile 下的最小化假目录模拟，
另有针对真实 skill 目录的冒烟测试（目录不存在时自动跳过）。
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.skills import build_skill_tools
from src.agent.skills.nature_polishing import (
    _detect_language,
    build_paper_polishing_tool,
)

# ── 测试用假 skill 目录 ───────────────────────────────────────────────────

MANIFEST_YAML = """
always_load:
  - ../_shared/core/reader-workflow.md
  - static/core/stance.md
axes:
  paper_type:
    values:
      research: static/fragments/paper_type/research.md
      methods: static/fragments/paper_type/methods.md
    default: research
    multi: false
  section:
    values:
      abstract: static/fragments/section/abstract.md
      intro: static/fragments/section/intro.md
    multi: true
  language:
    values:
      en: static/fragments/language/en.md
      zh-to-en: static/fragments/language/zh-to-en.md
    default: en
    multi: false
  journal:
    values:
      nature: static/fragments/journal/nature.md
      generic: static/fragments/journal/generic.md
    default: generic
    multi: false
"""

FRAGMENTS = {
    # 与 manifest 中引用路径完全一致（../_shared 相对 skill 根解析到 base）
    "../_shared/core/reader-workflow.md": "MARKER_SHARED_READER",
    "static/core/stance.md": "MARKER_CORE_STANCE",
    "static/fragments/paper_type/research.md": "MARKER_PAPER_RESEARCH",
    "static/fragments/paper_type/methods.md": "MARKER_PAPER_METHODS",
    "static/fragments/section/abstract.md": "MARKER_SECTION_ABSTRACT",
    "static/fragments/section/intro.md": "MARKER_SECTION_INTRO",
    "static/fragments/language/en.md": "MARKER_LANG_EN",
    "static/fragments/language/zh-to-en.md": "MARKER_LANG_ZH2EN",
    "static/fragments/journal/generic.md": "MARKER_JOURNAL_GENERIC",
    "static/fragments/journal/nature.md": "MARKER_JOURNAL_NATURE",
}


@pytest.fixture
def fake_skill_dir() -> Path:
    """用 tempfile 构建最小化的 nature-polishing 目录结构（绕开 pytest tmp_path
    在部分 Windows 环境下的 basetemp 权限问题），测试结束后自动清理。"""
    base = Path(tempfile.mkdtemp(prefix="nature_polishing_test_"))
    skill_root = base / "skill"
    try:
        for rel_path, content in FRAGMENTS.items():
            # 与 loader 相同的解析方式：../_shared 落到 base，其余落到 skill 根
            frag = (skill_root / rel_path).resolve()
            frag.parent.mkdir(parents=True, exist_ok=True)
            frag.write_text(content, encoding="utf-8")
        (skill_root / "manifest.yaml").parent.mkdir(parents=True, exist_ok=True)
        (skill_root / "manifest.yaml").write_text(MANIFEST_YAML, encoding="utf-8")
        yield skill_root
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _make_llm(response_content: str = "Polished prose.\n\nRevision notes:\n- 修改1") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=response_content)
    return llm


def _last_prompt(llm: MagicMock) -> str:
    """提取最近一次 llm.invoke 收到的 prompt 文本。"""
    call = llm.invoke.call_args
    return call[0][0][0].content


EN_TEXT = "We study the effect of drought on maize yield in field experiments."
ZH_TEXT = "本研究通过田间试验分析了干旱胁迫对玉米产量及其构成因子的影响，结果表明干旱显著降低了穗粒数。"


# ── 语言自动检测 ───────────────────────────────────────────────────────────

class TestDetectLanguage:
    def test_chinese_text_detected_as_zh_to_en(self):
        assert _detect_language(ZH_TEXT) == "zh-to-en"

    def test_english_text_detected_as_en(self):
        assert _detect_language(EN_TEXT) == "en"

    def test_mostly_english_with_few_cjk_chars_is_en(self):
        text = "The 结果 shows a significant increase in yield under treatment."
        assert _detect_language(text) == "en"

    def test_empty_text_is_en(self):
        assert _detect_language("") == "en"


# ── 工具注册 ───────────────────────────────────────────────────────────────

class TestToolRegistration:
    def test_registered_via_build_skill_tools(self):
        tools = build_skill_tools(llm=_make_llm())
        names = [t.name for t in tools]
        assert "paper_polishing" in names

    def test_not_registered_without_llm(self):
        tools = build_skill_tools(llm=None)
        names = [t.name for t in tools]
        assert "paper_polishing" not in names

    def test_dedicated_polishing_llm_is_used(self, fake_skill_dir):
        """传入 polishing_llm 时，paper_polishing 应使用专用 LLM 而非通用 llm。"""
        general_llm = _make_llm("from general")
        polishing_llm = _make_llm("from polishing")
        tools = build_skill_tools(llm=general_llm, polishing_llm=polishing_llm)

        tool = next(t for t in tools if t.name == "paper_polishing")
        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["status"] == "ok"
        assert data["polished"] == "from polishing"
        polishing_llm.invoke.assert_called_once()
        general_llm.invoke.assert_not_called()

    def test_polishing_llm_falls_back_to_general_llm(self, fake_skill_dir):
        """未传 polishing_llm 时，回退到通用 llm（向后兼容）。"""
        general_llm = _make_llm("from general")
        tools = build_skill_tools(llm=general_llm)

        tool = next(t for t in tools if t.name == "paper_polishing")
        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["status"] == "ok"
        assert data["polished"] == "from general"
        general_llm.invoke.assert_called_once()


# ── 工具核心行为 ───────────────────────────────────────────────────────────

class TestPaperPolishingTool:
    def test_returns_polished_json_and_loads_fragments(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        result = tool.invoke({
            "text": EN_TEXT,
            "paper_type": "methods",
            "section": "abstract",
            "language": "en",
            "journal": "nature",
        })
        data = json.loads(result)

        assert data["status"] == "ok"
        assert data["polished"] == "Polished prose.\n\nRevision notes:\n- 修改1"
        assert data["axes"] == {
            "paper_type": "methods",
            "section": ["abstract"],
            "language": "en",
            "journal": "nature",
        }

        # prompt 中应包含核心片段与全部指定轴的片段
        prompt = _last_prompt(llm)
        for marker in (
            "MARKER_SHARED_READER",
            "MARKER_CORE_STANCE",
            "MARKER_PAPER_METHODS",
            "MARKER_SECTION_ABSTRACT",
            "MARKER_LANG_EN",
            "MARKER_JOURNAL_NATURE",
        ):
            assert marker in prompt, f"prompt 缺少片段 {marker}"
        # 未指定的轴值不应被加载
        assert "MARKER_PAPER_RESEARCH" not in prompt
        assert "MARKER_JOURNAL_GENERIC" not in prompt
        # 原文应注入 prompt
        assert EN_TEXT in prompt

    def test_defaults_applied_when_axes_omitted(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["axes"]["paper_type"] == "research"
        assert data["axes"]["journal"] == "generic"
        assert data["axes"]["language"] == "en"
        assert "section" not in data["axes"]  # 留空跳过

        prompt = _last_prompt(llm)
        assert "跳过章节轴" in prompt
        assert "MARKER_PAPER_RESEARCH" in prompt
        assert "MARKER_JOURNAL_GENERIC" in prompt
        assert "MARKER_LANG_EN" in prompt

    def test_invalid_axis_value_falls_back_to_default(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": EN_TEXT, "paper_type": "bogus"}))
        assert data["axes"]["paper_type"] == "research"

    def test_invalid_section_value_skipped(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": EN_TEXT, "section": "bogus"}))
        assert "section" not in data["axes"]

    def test_multi_section_comma_separated(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(
            tool.invoke({"text": EN_TEXT, "section": "abstract, intro"})
        )
        assert data["axes"]["section"] == ["abstract", "intro"]
        prompt = _last_prompt(llm)
        assert "MARKER_SECTION_ABSTRACT" in prompt
        assert "MARKER_SECTION_INTRO" in prompt

    def test_chinese_draft_auto_detects_zh_to_en(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": ZH_TEXT}))
        assert data["axes"]["language"] == "zh-to-en"
        assert "MARKER_LANG_ZH2EN" in _last_prompt(llm)

    def test_explicit_language_overrides_detection(self, fake_skill_dir):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": ZH_TEXT, "language": "en"}))
        assert data["axes"]["language"] == "en"

    def test_empty_text_returns_error(self, fake_skill_dir):
        tool = build_paper_polishing_tool(_make_llm(), skill_dir=fake_skill_dir)
        data = json.loads(tool.invoke({"text": "   "}))
        assert data["status"] == "error"
        assert "为空" in data["message"]

    def test_missing_skill_dir_returns_error(self, fake_skill_dir):
        missing = fake_skill_dir.parent / "no_such_skill"
        tool = build_paper_polishing_tool(_make_llm(), skill_dir=missing)
        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["status"] == "error"
        assert "manifest.yaml" in data["message"]

    def test_llm_exception_returns_graceful_error(self, fake_skill_dir):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["status"] == "error"
        assert "LLM unavailable" in data["message"]

    def test_missing_fragment_file_does_not_crash(self, fake_skill_dir):
        # 删除一个轴片段，工具应跳过并继续工作
        (fake_skill_dir / "static/fragments/journal/generic.md").unlink()
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=fake_skill_dir)

        data = json.loads(tool.invoke({"text": EN_TEXT}))
        assert data["status"] == "ok"


# ── 前端工具结果摘要 ──────────────────────────────────────────────────────

class TestPolishingSummary:
    def test_summary_for_ok_result(self):
        from src.agent.rag_agent import _summarize_tool_result

        raw = json.dumps({
            "status": "ok",
            "axes": {"paper_type": "research", "language": "zh-to-en",
                     "journal": "generic"},
            "polished": "Polished prose here.\n\nRevision notes:\n- 修改1",
        }, ensure_ascii=False)
        summary = _summarize_tool_result("paper_polishing", raw)
        assert "完成 Nature 风格润色" in summary
        assert "research/zh-to-en/generic" in summary
        assert "Polished prose here." in summary

    def test_summary_for_error_result(self):
        from src.agent.rag_agent import _summarize_tool_result

        raw = json.dumps({"status": "error", "message": "润色生成失败"},
                         ensure_ascii=False)
        summary = _summarize_tool_result("paper_polishing", raw)
        assert summary == "润色生成失败"


# ── 真实 skill 目录冒烟测试（目录不存在时跳过） ─────────────────────────────

_REAL_SKILL_DIR = Path("E:/agent-learning/nature-skills/skills/nature-polishing")


@pytest.mark.skipif(
    not _REAL_SKILL_DIR.exists(), reason="真实 nature-polishing skill 目录不存在"
)
class TestRealSkillDir:
    def test_manifest_loads_and_core_fragments_present(self):
        llm = _make_llm()
        tool = build_paper_polishing_tool(llm, skill_dir=_REAL_SKILL_DIR)

        data = json.loads(tool.invoke({
            "text": EN_TEXT,
            "section": "abstract",
            "language": "en",
        }))
        assert data["status"] == "ok"
        assert data["axes"]["paper_type"] == "research"
        assert data["axes"]["journal"] == "generic"

        prompt = _last_prompt(llm)
        # 真实核心片段内容应进入 prompt
        assert "Default stance" in prompt            # static/core/stance.md
        assert "Section: Abstract" in prompt          # section/abstract.md
        assert "Language: English source" in prompt  # language/en.md

    def test_all_manifest_fragment_files_exist(self):
        """校验 manifest 中声明的所有片段文件真实存在。"""
        import yaml

        manifest = yaml.safe_load(
            (_REAL_SKILL_DIR / "manifest.yaml").read_text(encoding="utf-8")
        )
        all_paths = list(manifest.get("always_load", []))
        for spec in manifest["axes"].values():
            all_paths.extend(spec.get("values", {}).values())

        missing = [
            p for p in all_paths
            if not (_REAL_SKILL_DIR / p).resolve().exists()
        ]
        assert not missing, f"manifest 声明但缺失的片段：{missing}"
