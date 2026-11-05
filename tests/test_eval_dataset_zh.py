"""中文农业评测集的格式与来源校验。

数据位于 data/（运行时不入库），因此目录缺失时直接 skip；
只要文件存在，就保证 JSONL 可解析、字段齐全、source 指向真实文档。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DOC_DIR = DATA_DIR / "documents" / "zh"

POSITIVE_FILES = ("eval_dataset_zh.jsonl", "eval_dataset_zh_paraphrase.jsonl")
NEGATIVE_FILE = "eval_dataset_zh_negative.jsonl"
CATEGORIES = {"factual", "summary", "reasoning", "comparison", "paraphrase"}


def _load(name: str) -> list[dict]:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"评测集不存在：{path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture(scope="module")
def positives() -> list[dict]:
    rows: list[dict] = []
    for name in POSITIVE_FILES:
        rows.extend(_load(name))
    return rows


@pytest.fixture(scope="module")
def negatives() -> list[dict]:
    return _load(NEGATIVE_FILE)


class TestPositiveDataset:
    def test_size(self, positives):
        assert len(positives) >= 90

    def test_required_fields(self, positives):
        for row in positives:
            assert row["question"].strip()
            assert row["answer"].strip()
            assert row["source"].strip()
            assert row["category"] in CATEGORIES
            assert row["page"] == 0

    def test_questions_are_unique(self, positives):
        questions = [row["question"] for row in positives]
        assert len(questions) == len(set(questions))

    def test_sources_exist(self, positives):
        if not DOC_DIR.exists():
            pytest.skip("中文语料目录不存在")
        names = {p.name for p in DOC_DIR.glob("*.md")}
        missing = {row["source"] for row in positives} - names
        assert not missing, f"source 不在语料目录：{sorted(missing)}"


class TestNegativeDataset:
    def test_size(self, negatives):
        assert len(negatives) >= 8

    def test_refusal_contract(self, negatives):
        for row in negatives:
            assert row["category"] == "negative"
            assert row["expected_behavior"] == "refuse"
            assert row["source"] == ""
            assert "未收录" in row["answer"] or "没有" in row["answer"]
