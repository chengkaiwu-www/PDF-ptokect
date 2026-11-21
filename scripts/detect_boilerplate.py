"""诊断语料里的「高分噪声」：目录页与页眉页脚。

只读脚本，不修改任何数据、不重建索引。回答三个问题：

  1. 有多少页长得像目录页？（点引导符 / 行尾页码 / 短行密集）
  2. 哪些行在多页反复出现？（页眉页脚、书名推广语）
  3. 这些噪声折合多少字符、多少 chunk，占全库多大比例？

为什么值得单独测：目录页把全书章节标题聚在一起，关键词全中却不含答案，
向量检索和 LLM 判定都会把它当成相关块 → 污染 Precision 的分子。
但「看起来像噪声」不等于「真的是噪声」，所以先出清单再人工抽查。

用法：
  python scripts/detect_boilerplate.py
  python scripts/detect_boilerplate.py --source pumpkin-book   # 只看一本
  python scripts/detect_boilerplate.py --show 30              # 多打几行
  python scripts/detect_boilerplate.py --no-chunks            # 跳过切分（更快）
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import PDF_DIR  # noqa: E402
from src.boilerplate import (  # noqa: E402
    REPEAT_MIN_PAGES,
    REPEAT_RATIO,
    is_toc_page,
    repeated_lines_by_source,
    toc_entry_count,
)
from src.document_loader import load_pdf_documents  # noqa: E402
from src.text_splitter import split_documents  # noqa: E402

# 点引导符：只用于诊断展示（「有多少处省略号」），不参与判定
DOT_LEADER = re.compile(r"(?:\.{3,}|…{2,}|·{3,}|-{4,}|_{4,}|—{4,})")
# 仅「行尾页码」：用作次选候选，只打印不判定 —— 公式页、代码页、表格页
# 都会有大量以数字结尾的短行（第一版就是被这一条误导，误判了 35.9% 的字符）
TRAIL_NUM = re.compile(r"(?:\s|^)(?:[ivxlcdm]{1,6}|\d{1,3})\s*$", re.I)

MIN_TOC_LINES = 8          # 与 src/boilerplate.py 的 TOC_MIN_LINES 保持一致
TRAIL_HIT = 6              # 次选候选：行尾页码条数
SHORT_RATIO = 0.6          # 次选候选：短行占比


def toc_features(text: str) -> Optional[Dict[str, Any]]:
    """给一页算目录页特征；页太短直接返回 None（不可能是有意义的目录）。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < MIN_TOC_LINES:
        return None
    entries = toc_entry_count(text)
    dots = len(DOT_LEADER.findall(text))
    trail = sum(1 for ln in lines if TRAIL_NUM.search(ln))
    short = sum(1 for ln in lines if len(ln) <= 30)
    # 主判据直接复用 src/boilerplate.py —— 诊断说什么，切分器就按什么删
    return {
        "n_lines": len(lines),
        "entries": entries,
        "dots": dots,
        "trail": trail,
        "short_ratio": round(short / len(lines), 2),
        "is_toc": is_toc_page(text),
        "maybe_toc": (not is_toc_page(text))
        and trail >= TRAIL_HIT
        and short / len(lines) >= SHORT_RATIO,
    }


def repeated_lines(pages: List[Any]) -> tuple:
    """找出跨页重复出现的行（页眉页脚 / 每页都印的推广语）。"""
    reps = repeated_lines_by_source(pages)
    threshold = max(REPEAT_MIN_PAGES, int(len(pages) * REPEAT_RATIO))
    # 单文档调用时 repeated_lines_by_source 只会返回一个 key
    lines: set = set()
    for v in reps.values():
        lines |= v
    ordered = sorted(lines, key=lambda ln: -sum(1 for d in pages if ln in d.page_content))
    return [(ln, sum(1 for d in pages if ln in d.page_content)) for ln in ordered], threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="只诊断文件名包含该子串的 PDF")
    ap.add_argument("--show", type=int, default=20, help="每类最多打印多少条")
    ap.add_argument("--no-chunks", action="store_true", help="跳过切分统计（更快）")
    args = ap.parse_args()

    all_pages = load_pdf_documents(PDF_DIR)
    if args.source:
        all_pages = [d for d in all_pages if args.source in d.metadata.get("source", "")]
    if not all_pages:
        print("没加载到任何页，检查 data/pdfs 目录")
        return

    # 按文档分组：跨页重复行必须在「同一本书内」统计，
    # 否则不同书的共同套话会被误判成页眉页脚。
    by_src: Dict[str, List[Any]] = defaultdict(list)
    for doc in all_pages:
        by_src[doc.metadata.get("source", "?")].append(doc)

    print("=" * 78)
    print(f"语料：{len(by_src)} 个文档 / {len(all_pages)} 页 / "
          f"{sum(len(d.page_content) for d in all_pages)} 字符")
    print("=" * 78)

    # ---------- 一、目录页候选 ----------
    toc_pages = []
    maybe_pages = []
    print("\n一、目录页候选（主判据：≥3 条「点引导符 + 页码」结尾的行）")
    for src, docs in by_src.items():
        hits, maybes = [], []
        for doc in docs:
            f = toc_features(doc.page_content)
            if not f:
                continue
            f["page_label"] = doc.metadata.get("page_label")
            f["chars"] = len(doc.page_content)
            f["head"] = [ln.strip() for ln in doc.page_content.splitlines() if ln.strip()][:3]
            if f["is_toc"]:
                hits.append(f)
            elif f["maybe_toc"]:
                maybes.append(f)
        toc_pages.extend((src, h) for h in hits)
        maybe_pages.extend((src, h) for h in maybes)
        flag = "  <-- 有目录" if hits else ""
        print(f"  {src}: 目录页 {len(hits)} / {len(docs)}，次选 {len(maybes)}{flag}")
        for h in hits[: args.show]:
            print(f"      p{h['page_label']:<4} 目录条目={h['entries']:<3} 行数={h['n_lines']:<3} "
                  f"{h['chars']:>5} 字符")
            for ln in h["head"]:
                print(f"          | {ln[:70]}")

    # ---------- 二、页眉页脚候选 ----------
    print("\n二、跨页重复行（同一本书内，出现在 ≥30% 页且 ≥3 页）")
    for src, docs in by_src.items():
        reps, thr = repeated_lines(docs)
        print(f"  {src}: 阈值 {thr} 页，命中 {len(reps)} 行")
        for ln, c in reps[: args.show]:
            print(f"      [{c:>3}/{len(docs)}] {ln[:70]}")

    # ---------- 三、折合规模 ----------
    toc_set = {(src, h["page_label"]) for src, h in toc_pages}
    toc_chars = sum(h["chars"] for _, h in toc_pages)
    all_chars = sum(len(d.page_content) for d in all_pages)
    print(f"\n三、目录页规模：{len(toc_pages)} 页 / {toc_chars} 字符 "
          f"（占全库 {toc_chars / max(all_chars, 1):.1%}）")
    if maybe_pages:
        m_chars = sum(h["chars"] for _, h in maybe_pages)
        print(f"    次选候选（仅行尾页码，需人工看，暂不判定）：{len(maybe_pages)} 页 / "
              f"{m_chars} 字符（占 {m_chars / max(all_chars, 1):.1%}）")

    if not args.no_chunks:
        chunks = split_documents(all_pages)
        toc_chunks = [
            c for c in chunks
            if (c.metadata.get("source", "?"), str(c.metadata.get("page_label"))) in toc_set
        ]
        print(f"    折合 chunk：{len(toc_chunks)} / {len(chunks)} "
              f"（占 {len(toc_chunks) / max(len(chunks), 1):.1%}）")
        print(f"    —— 这些块会把 Precision@k 的分子抬高，是「假阳性」的来源")

    # ---------- 四、给下一步的建议阈值 ----------
    print("\n四、若要加过滤，规则示例（需先人工抽查上面清单确认无误伤）")
    print("    · 目录页：整页有 ≥3 条「点引导符+页码」结尾的行 → 跳过整页")
    print("    · 页眉页脚：同一本书内出现在 ≥30% 页的行 → 从块中删除该行")
    print("    · 只删行不删块：页眉页脚往往与正文同块，删块会连带丢掉正文")
    print("    · 次选候选（只有行尾页码、没有点引导符）不要自动过滤 ——")
    print("      公式页/代码页/表格页同样满足这一条，第一版判据就因此误伤 35.9% 的字符")


if __name__ == "__main__":
    import contextlib

    # Windows 下 PowerShell 会把 UTF-8 输出按 GBK 解码弄成乱码，
    # 所以提供 --out 让 Python 自己按 UTF-8 写文件，绕开控制台编码。
    _argv = sys.argv
    _out = None
    if "--out" in _argv:
        _i = _argv.index("--out")
        _out = _argv[_i + 1]
        del _argv[_i : _i + 2]
    if _out:
        with open(_out, "w", encoding="utf-8") as _f, contextlib.redirect_stdout(_f):
            main()
        print(f"已写入 {_out}")
    else:
        main()
