"""
题库自检：确认每道题「标注的来源里真的有答案」。

为什么需要这个脚本
------------------
写题最容易犯的错不是写错问题，而是**标注了一个语料里根本不存在的来源**。
这种错误不会报错，只会在评测时表现为「方法 A 比方法 B 差」，
而真相是题目本身有问题 —— 这是最难归因的一类假结论。

检查两件事：
  1. 正文覆盖：关键词是否落在「正文块」而不只是目录/页眉。
     PDF 目录页会把所有章节标题聚在一起，含关键词但不含答案，
     是检索评测里典型的**高分噪声**。
  2. 来源归属：expected_source 里的正文命中数，是否显著多于其它文档。
     若其它文档也大量命中，说明这题不适合做「来源判定」。

用法：
    D:\\anaconda\\python.exe scripts/check_questions.py
    D:\\anaconda\\python.exe scripts/check_questions.py --questions eval/questions_v2.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import VECTOR_STORE_DIR  # noqa: E402


def source_name(doc) -> str:
    return Path(doc.get("metadata", {}).get("source", "?")).name


def is_body(text: str) -> bool:
    """
    判断是不是「正文块」。

    两个廉价但有效的信号：
      · PDF 目录行有大量点引导符（"...."），正文几乎没有；
      · 正文块通常有足够长度（切分是 384 token，正常正文远长于此）。
    这不是完美判据，但足以把「目录污染」这类明显噪声排掉。
    """
    if "...." in text:
        return False
    return len(text.strip()) >= 150


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(PROJECT_ROOT / "eval" / "questions_v2.json"))
    ap.add_argument("--store-dir", default=str(VECTOR_STORE_DIR))
    args = ap.parse_args()

    docs_path = Path(args.store_dir) / "documents.json"
    raw = json.loads(docs_path.read_text(encoding="utf-8"))
    docs = raw["documents"] if isinstance(raw, dict) else raw

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))["questions"]

    print(f"{'id':5s} {'type':12s} {'正文命中(按来源)':38s} 结论")
    print("-" * 100)
    problems = []

    for q in questions:
        qid = q["id"]
        qtype = q.get("type", "single_fact")
        keywords = q.get("expected_keywords", [])
        expected = q.get("expected_source")
        required = q.get("required_sources", [])

        if qtype == "no_answer":
            # 无答案题反过来查：语料里**不应该**有这些词
            terms = q.get("absent_terms") or []
            hits = 0
            for d in docs:
                low = d["page_content"].lower()
                if any(t.lower() in low for t in terms):
                    hits += 1
            status = "OK（语料确实没有）" if hits == 0 else f"⚠ {hits} 块命中，需换题"
            if hits:
                problems.append(qid)
            print(f"{qid:5s} {qtype:12s} {'-':38s} {status}")
            continue

        if qtype == "multi_hop" and not keywords:
            # 多跳题没有单一关键词：它的正确性取决于「两份文档是否都真的相关」，
            # 这一点只能靠实际检索验证（见 eval/run_eval.py 的 required_sources 覆盖率），
            # 静态检查在这里无法给结论，宁可显式标出来，也不假装通过。
            print(f"{qid:5s} {qtype:12s} {'-':38s} 跳过（多跳题需检索实测）")
            continue

        body_by_src = Counter()
        for d in docs:
            text = d["page_content"]
            if any(k.lower() in text.lower() for k in keywords) and is_body(text):
                body_by_src[source_name(d)] += 1

        desc = " ".join(f"{s.replace('.pdf','')}={n}" for s, n in body_by_src.most_common(3))
        if not body_by_src:
            status = "✗ 正文无命中（只有目录/页眉命中）"
            problems.append(qid)
        elif expected:
            n_exp = body_by_src.get(expected, 0)
            n_other = sum(v for s, v in body_by_src.items() if s != expected)
            if n_exp == 0:
                status = f"✗ 标注来源 {expected} 无命中"
                problems.append(qid)
            elif n_exp < n_other:
                status = f"⚠ 标注来源({n_exp}) 少于其它({n_other})，来源判定不可靠"
            else:
                status = f"OK 标注来源 {n_exp} 块"
        else:
            status = "OK（多跳题，来源覆盖需检索实测）"

        print(f"{qid:5s} {qtype:12s} {desc:38s} {status}")

    print("-" * 100)
    print(f"共 {len(questions)} 题，问题题号：{problems or '无'}")


if __name__ == "__main__":
    main()
