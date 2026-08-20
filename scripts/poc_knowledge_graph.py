"""
方向二技术预研：农业 AI 研究图谱 PoC
──────────────────────────────────────────────────────────────────────────────
从已入库论文的 OCR Markdown 中抽取实体与关系，构建轻量研究知识图谱，
验证「图谱路线」在现有三篇农业深度学习文献上的可行性。

图谱 Schema（PoC 版）
────────────────────
实体类型:  model / dataset / disease / crop / task
关系类型:
  • evaluated_on : model → dataset   （属性：metric、value、notes）
  • based_on     : model → model     （改进/派生关系）
  • targets      : model → disease/crop（面向的病害或作物）
  • outperforms  : model → model     （属性：metric、value）

流水线
──────
1. TextParser 解析 OCR Markdown → ParsedBlock 列表（表格为原子块）。
2. 高价值块筛选：TABLE 块 + 含百分比指标且提及模型名的 TEXT 块。
3. LLM 结构化抽取（严格 JSON 输出），按块批量调用，控制成本。
4. 实体归一化合并（别名映射），确定性 ID（sha1），写入 SQLite + JSON。

用法
────
# 查看抽取计划（不调用 API）
python scripts/poc_knowledge_graph.py extract --dry-run

# 执行抽取（默认处理 data/ocr_output 下全部论文）
python scripts/poc_knowledge_graph.py extract

# 只处理某篇论文 / 限制每篇最多抽取块数
python scripts/poc_knowledge_graph.py extract --paper Plant_Disease_Detection_from_Images --max-blocks 6

# 图谱统计
python scripts/poc_knowledge_graph.py stats

# GraphRAG-lite 问答（关键词召回三元组 → LLM 回答）
python scripts/poc_knowledge_graph.py ask "HvT 和 MobileNetV3 在 PlantVillage 上的准确率谁更高？"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from src.document_parser.base_parser import BlockType
from src.document_parser.text_parser import TextParser

settings = get_settings()

# ── 常量 ────────────────────────────────────────────────────────────────────

DEFAULT_OCR_DIR = Path("./data/ocr_output")
GRAPH_DIR = Path("./data/knowledge_graph")
DB_PATH = GRAPH_DIR / "poc_graph.db"
JSON_PATH = GRAPH_DIR / "poc_graph.json"

ENTITY_TYPES = ["model", "dataset", "disease", "crop", "task"]
RELATION_TYPES = ["evaluated_on", "based_on", "targets", "outperforms"]

_PCT_RE = re.compile(r"\d{2}(?:\.\d+)?\s?%")

_EXTRACT_PROMPT = """\
你是农业 AI 文献知识抽取专家。请从给定的论文片段中抽取结构化知识图谱信息。

【当前片段所属章节】{header}
【来源论文】{paper}

【片段内容】
{content}

【抽取要求】
1. 实体类型仅限：model（深度学习模型/网络）、dataset（数据集）、disease（植物病害或虫害）、crop（作物）、task（任务类型如分类/检测/分割）。
2. 关系类型仅限：
   - evaluated_on：模型在数据集上评测（必须尽量附 metric 与 value，如 accuracy/99.3%）
   - based_on：某模型基于/改进自另一模型
   - targets：模型面向某病害或作物
   - outperforms：某模型性能超过另一模型（附 metric 与 value）
3. head/tail 必须使用你抽取的实体 name 原文；只抽取片段中明确陈述的事实，禁止推测。
4. evidence 用片段中的原句（不超过 60 词）。
5. 忽略与上述类型无关的信息（如背景介绍中的泛泛陈述）。

【输出格式】严格输出如下 JSON，不要任何其他内容：
{{"entities": [{{"name": "...", "type": "model|dataset|disease|crop|task", "aliases": []}}],
 "relations": [{{"head": "...", "type": "evaluated_on|based_on|targets|outperforms", "tail": "...", "metric": "", "value": "", "evidence": ""}}]}}
"""

_ASK_PROMPT = """\
你是农业科研助手。请仅根据下面给出的知识图谱三元组回答问题。
如果三元组中没有足够信息，明确说明"图谱中暂无相关记录"，不要编造。

【图谱三元组】
{triples}

【问题】{question}

请用简洁的中文回答，涉及数值对比时给出具体数字，并注明来源论文。
"""


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """实体名归一化：去空白、转小写，用于合并判重。"""
    return re.sub(r"\s+", " ", name.strip()).lower()


def _entity_id(etype: str, name: str) -> str:
    return hashlib.sha1(f"{etype}:{_norm_name(name)}".encode()).hexdigest()[:16]


def _find_md_files(ocr_dir: Path) -> list[Path]:
    return sorted(p for p in ocr_dir.rglob("*.md") if not p.name.endswith("content_list.json"))


# ── 高价值块筛选 ────────────────────────────────────────────────────────────

def select_blocks(blocks, max_blocks: int) -> list[dict]:
    """
    筛选值得送 LLM 抽取的块：
    • 全部 TABLE 块（指标与模型对比的主要载体）
    • 含百分比数字的 TEXT 块（实验结果的文本描述）
    每个块附带其最近的上层标题作为语境。
    """
    selected: list[dict] = []
    current_header = ""
    for block in blocks:
        if block.block_type == BlockType.HEADER:
            current_header = block.content.lstrip("#").strip()
            continue
        if block.block_type == BlockType.TABLE:
            selected.append({"header": current_header, "content": block.content, "why": "table"})
        elif block.block_type == BlockType.TEXT and _PCT_RE.search(block.content):
            selected.append({"header": current_header, "content": block.content, "why": "metric_text"})

    # 控制成本：表格优先，其次按文本长度降序（信息密度代理）
    selected.sort(key=lambda b: (b["why"] != "table", -len(b["content"])))
    return selected[:max_blocks]


# ── LLM 抽取 ────────────────────────────────────────────────────────────────

class GraphExtractor:
    """调用 DashScope LLM 对单个块做结构化抽取。"""

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def extract_block(self, paper: str, header: str, content: str) -> dict:
        prompt = _EXTRACT_PROMPT.format(header=header or "（无标题）", paper=paper, content=content)
        resp = self._client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_body={"enable_thinking": False},
        )
        text = (resp.choices[0].message.content or "").strip()
        return json.loads(text)


# ── 图谱构建与合并 ──────────────────────────────────────────────────────────

class GraphBuilder:
    """内存图谱：实体去重合并 + 关系校验，落盘 SQLite 与 JSON。"""

    def __init__(self) -> None:
        self.entities: dict[str, dict] = {}      # eid -> entity
        self.relations: dict[str, dict] = {}     # rid -> relation
        self._alias_index: dict[str, str] = {}   # norm(alias) -> eid

    def add_entity(self, name: str, etype: str, paper: str, aliases: list[str] | None = None) -> str:
        if etype not in ENTITY_TYPES or not name.strip():
            return ""
        key = _norm_name(name)
        if key in self._alias_index:
            eid = self._alias_index[key]
            self.entities[eid]["papers"].add(paper)
            return eid
        eid = _entity_id(etype, name)
        self.entities[eid] = {
            "id": eid, "name": name.strip(), "type": etype,
            "aliases": list(aliases or []), "papers": {paper},
        }
        self._alias_index[key] = eid
        for alias in aliases or []:
            self._alias_index.setdefault(_norm_name(alias), eid)
        return eid

    def add_relation(
        self, head_id: str, rtype: str, tail_id: str, paper: str,
        metric: str = "", value: str = "", evidence: str = "", header: str = "",
    ) -> None:
        if rtype not in RELATION_TYPES or not head_id or not tail_id or head_id == tail_id:
            return
        rid = hashlib.sha1(f"{head_id}:{rtype}:{tail_id}:{metric}:{value}".encode()).hexdigest()[:16]
        if rid in self.relations:
            return
        self.relations[rid] = {
            "id": rid, "head": head_id, "type": rtype, "tail": tail_id,
            "metric": metric, "value": value, "evidence": evidence,
            "header": header, "paper": paper,
        }

    # ── 落盘 ─────────────────────────────────────────────────────────────

    def save(self) -> None:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)

        entities = [
            {**e, "papers": sorted(e["papers"])} for e in self.entities.values()
        ]
        relations = list(self.relations.values())
        JSON_PATH.write_text(
            json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        conn = sqlite3.connect(DB_PATH)
        conn.executescript(
            """
            DROP TABLE IF EXISTS entities;
            DROP TABLE IF EXISTS relations;
            CREATE TABLE entities (
                id TEXT PRIMARY KEY, name TEXT, type TEXT,
                aliases TEXT, papers TEXT
            );
            CREATE TABLE relations (
                id TEXT PRIMARY KEY, head TEXT, type TEXT, tail TEXT,
                metric TEXT, value TEXT, evidence TEXT, header TEXT, paper TEXT,
                FOREIGN KEY(head) REFERENCES entities(id),
                FOREIGN KEY(tail) REFERENCES entities(id)
            );
            """
        )
        conn.executemany(
            "INSERT INTO entities VALUES (?, ?, ?, ?, ?)",
            [(e["id"], e["name"], e["type"], ", ".join(e["aliases"]), ", ".join(e["papers"])) for e in entities],
        )
        conn.executemany(
            "INSERT INTO relations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r["id"], r["head"], r["type"], r["tail"], r["metric"], r["value"],
              r["evidence"], r["header"], r["paper"]) for r in relations],
        )
        conn.commit()
        conn.close()
        logger.info(f"图谱已保存：{len(entities)} 实体 / {len(relations)} 关系 → {DB_PATH}")


# ── 子命令：extract ─────────────────────────────────────────────────────────

def cmd_extract(args) -> None:
    md_files = _find_md_files(Path(args.ocr_dir))
    if args.paper:
        md_files = [f for f in md_files if args.paper.lower() in f.parent.parent.name.lower()]
    if not md_files:
        logger.error("未找到任何待处理的 OCR Markdown 文件")
        return

    parser = TextParser()
    plan: list[tuple[str, list[dict]]] = []
    for md in md_files:
        paper = md.parent.parent.name
        blocks = parser.parse(md)
        selected = select_blocks(blocks, args.max_blocks)
        plan.append((paper, selected))
        logger.info(f"{paper}: {len(blocks)} blocks → 选中 {len(selected)} 个高价值块")

    total_chars = sum(len(b["content"]) for _, sel in plan for b in sel)
    logger.info(f"抽取计划：{sum(len(s) for _, s in plan)} 个块，约 {total_chars} 字符")

    if args.dry_run:
        for paper, selected in plan:
            print(f"\n=== {paper} ===")
            for b in selected:
                print(f"  [{b['why']}] {b['header'][:50]} | {len(b['content'])} chars | {b['content'][:80].replace(chr(10), ' ')}…")
        return

    extractor = GraphExtractor()
    builder = GraphBuilder()

    for paper, selected in plan:
        for i, block in enumerate(selected, start=1):
            logger.info(f"[{paper}] 抽取块 {i}/{len(selected)}（{block['why']}，{len(block['content'])} 字符）")
            try:
                data = extractor.extract_block(paper, block["header"], block["content"])
            except Exception as exc:
                logger.warning(f"块抽取失败，跳过：{exc}")
                continue

            # 先建本块实体索引（name -> eid），再校验关系端点
            local_ids: dict[str, str] = {}
            for ent in data.get("entities", []):
                eid = builder.add_entity(
                    ent.get("name", ""), ent.get("type", ""), paper, ent.get("aliases")
                )
                if eid:
                    local_ids[_norm_name(ent["name"])] = eid

            for rel in data.get("relations", []):
                head_id = local_ids.get(_norm_name(rel.get("head", "")), "")
                tail_id = local_ids.get(_norm_name(rel.get("tail", "")), "")
                builder.add_relation(
                    head_id, rel.get("type", ""), tail_id, paper,
                    metric=rel.get("metric", ""), value=rel.get("value", ""),
                    evidence=rel.get("evidence", "")[:300], header=block["header"],
                )

        # 每篇论文处理完即落盘，支持中途失败后保留已有成果
        builder.save()


# ── 子命令：stats ───────────────────────────────────────────────────────────

def cmd_stats(_args) -> None:
    if not DB_PATH.exists():
        logger.error(f"图谱不存在，请先运行 extract：{DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    print("=" * 60)
    print("实体统计（按类型）")
    print("=" * 60)
    for etype, n in conn.execute("SELECT type, COUNT(*) FROM entities GROUP BY type ORDER BY 2 DESC"):
        print(f"  {etype:<10} {n}")
    print("\n关系统计（按类型）")
    for rtype, n in conn.execute("SELECT type, COUNT(*) FROM relations GROUP BY type ORDER BY 2 DESC"):
        print(f"  {rtype:<15} {n}")
    print("\n带指标数值的关系（evaluated_on / outperforms）")
    rows = conn.execute(
        """
        SELECT h.name, r.type, t.name, r.metric, r.value, r.paper
        FROM relations r JOIN entities h ON r.head = h.id JOIN entities t ON r.tail = t.id
        WHERE r.value != '' ORDER BY h.name LIMIT 30
        """
    ).fetchall()
    for h, rt, t, metric, value, paper in rows:
        print(f"  {h} --{rt}--> {t}  [{metric}={value}]  ({paper[:40]})")
    conn.close()


# ── 子命令：ask（GraphRAG-lite） ────────────────────────────────────────────

def _triples_for_question(question: str, limit: int = 60) -> list[str]:
    """关键词召回：实体名出现在问题中 → 拉取其相关三元组。"""
    conn = sqlite3.connect(DB_PATH)
    entities = conn.execute("SELECT id, name, type FROM entities").fetchall()
    hit_ids: set[str] = set()
    q_lower = question.lower()
    for eid, name, _etype in entities:
        # 词边界匹配，避免 'MobileNetV3' 子串误命中 'LeNet' 等短实体
        pattern = rf"(?<![a-z0-9]){re.escape(_norm_name(name))}(?![a-z0-9])"
        if re.search(pattern, q_lower):
            hit_ids.add(eid)

    triples: list[str] = []
    if hit_ids:
        placeholders = ",".join("?" for _ in hit_ids)
        rows = conn.execute(
            f"""
            SELECT h.name, r.type, t.name, r.metric, r.value, r.paper
            FROM relations r JOIN entities h ON r.head = h.id JOIN entities t ON r.tail = t.id
            WHERE r.head IN ({placeholders}) OR r.tail IN ({placeholders})
            LIMIT {limit}
            """,
            list(hit_ids) + list(hit_ids),
        ).fetchall()
        for h, rt, t, metric, value, paper in rows:
            attr = f" [{metric}={value}]" if value else ""
            triples.append(f"({h}) --{rt}{attr}--> ({t})  来源: {paper}")
    conn.close()
    return triples


def cmd_ask(args) -> None:
    if not DB_PATH.exists():
        logger.error(f"图谱不存在，请先运行 extract：{DB_PATH}")
        return
    triples = _triples_for_question(args.question)
    if not triples:
        print("问题中未命中任何图谱实体，无法基于图谱回答。")
        return

    print(f"命中 {len(triples)} 条三元组：")
    for t in triples[:12]:
        print(f"  {t}")

    client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.dashscope_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": _ASK_PROMPT.format(
            triples="\n".join(triples), question=args.question)}],
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )
    print("\n【图谱问答】")
    print(resp.choices[0].message.content)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="农业 AI 研究图谱 PoC")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="从 OCR Markdown 抽取实体关系")
    p_extract.add_argument("--ocr-dir", default=str(DEFAULT_OCR_DIR))
    p_extract.add_argument("--paper", default="", help="仅处理名称包含该关键词的论文")
    p_extract.add_argument("--max-blocks", type=int, default=16, help="每篇论文最多抽取的块数")
    p_extract.add_argument("--dry-run", action="store_true", help="只打印抽取计划，不调用 API")
    p_extract.set_defaults(func=cmd_extract)

    p_stats = sub.add_parser("stats", help="输出图谱统计")
    p_stats.set_defaults(func=cmd_stats)

    p_ask = sub.add_parser("ask", help="GraphRAG-lite 问答")
    p_ask.add_argument("question", help="自然语言问题（需包含图谱中的实体名）")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
