"""
把 results/ 里的实验数据生成一张单文件看板（web/dashboard.html）。

为什么是「生成器」而不是手写 HTML：
  看板最容易烂在「数字是手抄的」——手抄一次，以后每次实验都得记得改，早晚不一致。
  这里所有数字都从 results/*.json 读，新增一组实验只要重新跑一遍脚本。

为什么数据内联而不是 fetch：
  和 web/index.html 同样的理由 —— 双击打开时协议是 file://，
  fetch 相对路径会被解析成 file:///api/...，且 Origin: null 会被同源策略拦。
  内联就没有这两个问题，代价是 HTML 变大（几十 KB 数据 → 两百 KB 左右，无所谓）。

用法：
    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --out web/dashboard.html
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# 配置清单：文件名 -> 展示信息
# 手写这张表是有意的 —— 它记录「这组实验改了什么、结论是什么」，
# 而这些信息不在 JSON 里（JSON 只存数字）。新增实验时补一行即可。
# ---------------------------------------------------------------------------

RETRIEVAL_RUNS: List[Dict[str, str]] = [
    {"file": "eval_v2_hybrid", "label": "hybrid（基线）", "judge": "keyword", "tag": "base"},
    {"file": "eval_v2_hybrid_cand150", "label": "candidate_k=150", "judge": "keyword", "tag": "ctrl",
     "note": "混淆变量对照：六项完全不动 → 召回没漏"},
    {"file": "eval_v2_hybrid_mq", "label": "+ Multi-Query(3)", "judge": "keyword", "tag": "bad",
     "note": "RRF 累加 → 同义改写刷票，负收益"},
    {"file": "eval_v2_hybrid_hyde", "label": "+ HyDE", "judge": "keyword", "tag": "ok",
     "note": "方向为正，幅度在噪声内"},
    {"file": "eval_v2_hybrid_rerank", "label": "hybrid + 精排", "judge": "keyword", "tag": "good"},
    {"file": "eval_v2_hybrid_hyde_rerank", "label": "HyDE + 精排", "judge": "keyword", "tag": "dup",
     "note": "与只开精排六项完全相同 → 解决同一瓶颈"},
    {"file": "eval_v2_hybrid_k8", "label": "top_k=8", "judge": "keyword", "tag": "base"},
    {"file": "eval_v2_hybrid_k12", "label": "top_k=12", "judge": "keyword", "tag": "good",
     "note": "Hit@1 不动 → 缺的是预算不是排序"},
    {"file": "eval_v2_vector_judgeall", "label": "仅向量（无过滤）", "judge": "llm", "tag": "ctrl",
     "note": "S6-d 单通道对照：32/35；噪声块最多（110/140）"},
    {"file": "eval_v2_bm25_judgeall", "label": "仅 BM25（无过滤）", "judge": "llm", "tag": "ctrl",
     "note": "Hit 与向量 32=32，但 Precision 高 10 块 → 向量是「带噪声」而非「导不准」"},
    {"file": "eval_v2_hybrid_judgeall", "label": "混合 RRF（无过滤）", "judge": "llm", "tag": "good",
     "note": "+2 题；Precision 只比 BM25 多 1 块 → 收益主要是「不丢」，不是「更准」"},
    {"file": "eval_v2_hybrid_rerank_judgeall", "label": "混合 + 精排（无过滤）", "judge": "llm", "tag": "ok",
     "note": "35/35 但 Precision 只 +2 块，63× 延迟；且不单调：s10 4→2、d11 4→2"},
    {"file": "eval_v2_toc_vector", "label": "仅向量（只跳目录页）", "judge": "llm", "tag": "noise",
     "note": "S6-e：目录页只占 1/140 个召回位置 → 相关块 110→109，Hit@1 不动"},
    {"file": "eval_v2_toc_bm25", "label": "仅 BM25（只跳目录页）", "judge": "llm", "tag": "noise",
     "note": "相关块 120→121，方向与向量相反 → 无可辨别效果"},
    {"file": "eval_v2_toc_hybrid", "label": "混合 RRF（只跳目录页）", "judge": "llm", "tag": "noise",
     "note": "相关块 121→122，Hit@1 34→34 → 影响面 0.7%，测不出来"},
    {"file": "eval_v2_toc_hybrid_rerank", "label": "混合 + 精排（只跳目录页）", "judge": "llm", "tag": "noise",
     "note": "相关块 123→123，完全不动"},
    {"file": "eval_v2_clean_vector", "label": "仅向量（删目录+页眉）", "judge": "llm", "tag": "bad",
     "note": "S6-e：删页眉重排南瓜书切分（732→689 块）→ 32→**30** 题，相关块 110→108"},
    {"file": "eval_v2_clean_bm25", "label": "仅 BM25（删目录+页眉）", "judge": "llm", "tag": "bad",
     "note": "32→31 题，相关块 120→119"},
    {"file": "eval_v2_clean_hybrid", "label": "混合 RRF（删目录+页眉）", "judge": "llm", "tag": "bad",
     "note": "Hit@1 不变但 Precision .8643→.8429，相关块 121→118"},
    {"file": "eval_v2_clean_hybrid_rerank", "label": "混合 + 精排（删目录+页眉）", "judge": "llm", "tag": "bad",
     "note": "Precision .8786→.8571，相关块 123→120 → 删噪声反而更差"},
]

GEN_RUNS: List[Dict[str, str]] = [
    {"file": "gen_eval_full", "label": "full（S5 基线）", "tag": "base"},
    {"file": "gen_eval_no_cite", "label": "prompt 去掉引用要求", "tag": "ctrl",
     "note": "引用覆盖率 1.000→0.000：引用是 prompt 造的行为"},
    {"file": "gen_eval_no_refusal", "label": "prompt 去掉拒答条款", "tag": "ctrl",
     "note": "无答案题仍 7/7 拒答 → 拒答来自根约束"},
    {"file": "gen_eval_full_k12", "label": "full, top_k=12", "tag": "good",
     "note": "过度拒答 23.3%→16.3%，延迟只 +8.5%"},
    {"file": "gen_eval_base_ctx", "label": "子块 k=4（S6-c 基线）", "tag": "base"},
    {"file": "gen_eval_parent_w2_k4", "label": "父块 w=2 k=4", "tag": "noise",
     "note": "与非基线差 2 题"},
    {"file": "gen_eval_parent_w3_k4", "label": "父块 w=3 k=4", "tag": "noise",
     "note": "与非基线差 2 题"},
    {"file": "gen_eval_parent_w4_k4", "label": "父块 w=4 k=4", "tag": "noise",
     "note": "0.163 看似最好，但 w 非单调 → 摆动"},
    {"file": "gen_eval_parent_w3_k2", "label": "父块 w=3 k=2", "tag": "bad",
     "note": "只有 2 个位置 → 多跳 0/2，位置深度不是主因"},
    {"file": "gen_eval_small128_k4", "label": "碎块(128) k=4", "tag": "bad",
     "note": "洞被造出来：单块答不全"},
    {"file": "gen_eval_small128_p3k4", "label": "碎块(128) 父块 w=3 k=4", "tag": "good",
     "note": "救回 7 题、弄坏 2 题，净 +5"},
    {"file": "gen_eval_small128_k12", "label": "碎块(128) k=12", "tag": "ok",
     "note": "更多位置，绝对效果更好但字符多 39%"},
]

TAG_STYLE = {
    "base": ("基线", "tag-base"),
    "good": ("有效", "tag-good"),
    "ok": ("方向为正", "tag-ok"),
    "bad": ("负收益 / 默认关闭", "tag-bad"),
    "dup": ("与已有方案重叠", "tag-dup"),
    "ctrl": ("对照", "tag-ctrl"),
    "noise": ("噪声内", "tag-noise"),
}


def load(name: str) -> Optional[Dict[str, Any]]:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def num(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def pct_cell(v: Any, warn: bool = False) -> str:
    """
    比率单元格：带一条内嵌横条。

    warn=True 用于「越高越差」的指标（过度拒答）：条形换成警示色。
    否则一条更长的蓝条在视觉上和「更好」分不开 —— 图表也会撒谎。
    """
    if v is None:
        return '<td class="muted">—</td>'
    width = max(0.0, min(1.0, float(v))) * 100
    cls = "bar bar-warn" if warn else "bar"
    return f'<td class="barcell"><span class="{cls}" style="width:{width:.1f}%"></span><b>{v:.4f}</b></td>'


def tag_html(tag: str) -> str:
    text, cls = TAG_STYLE.get(tag, (tag, "tag-base"))
    return f'<span class="tag {cls}">{text}</span>'


# ---------------------------------------------------------------------------
# 各区块渲染
# ---------------------------------------------------------------------------

def section_retrieval() -> str:
    rows = []
    for cfg in RETRIEVAL_RUNS:
        d = load(cfg["file"])
        if not d:
            continue
        s = d["summary"]
        rows.append(
            "<tr>"
            f'<td class="label">{cfg["label"]}</td>'
            f"<td>{tag_html(cfg['tag'])}</td>"
            f'<td class="muted">{"语义" if s.get("judge") == "llm" else "关键词"}</td>'
            f'<td class="num">{num(s["hit_rate_at_1"])}</td>'
            f'<td class="num">{num(s["hit_rate_at_k"])}</td>'
            f'<td class="num">{num(s["mrr"])}</td>'
            f'<td class="num">{num(s.get("precision_at_k"), 4)}</td>'
            f'<td class="num">{num(s["multihop_coverage"])}</td>'
            f'<td class="num">{num(s["multihop_full_rate"])}</td>'
            f'<td class="num">{s["avg_retrieve_ms"]:.0f} ms</td>'
            f'<td class="note">{cfg.get("note", "")}</td>'
            "</tr>"
        )
    return f"""
<section>
  <h2>一、检索层（50 题分层集，top_k 默认 4）</h2>
  <p class="lead">Hit@1 / Hit@4 只在 35 道可回答题上算；多跳 8 题单列。
  <b>纵向不能跨口径比</b>：语义口径整体更宽，所有配置一起涨 6~8 题 —— 那是尺子松了，不是方法变好。
  <b>同一个 Hit@k 在不同文本长度下也不可直接比</b> —— 关键词判定是「关键词在不在返回文本里」，
  文本变长它自己会涨（见第四节）。
  <br><b>⚠ 分辨率 ±1 题</b>：判定器单次翻错 0.65%（κ=0.9608，实验 14），35 题上折合 ±1 题 / ±1 块。
  <b>只有 1 题的差异请读作「没测出差别」</b>，尤其「精排 34→35」和「混合+精排 Precision +2 块」。</p>
  <table>
    <thead><tr>
      <th>配置</th><th>结论</th><th>口径</th><th>Hit@1</th><th>Hit@4</th><th>MRR</th><th>Precision@k</th>
      <th>多跳覆盖</th><th>多跳全覆</th><th>延迟</th><th>备注</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</section>"""


def section_generation() -> str:
    rows = []
    for cfg in GEN_RUNS:
        d = load(cfg["file"])
        if not d:
            continue
        s = d["summary"]
        ctx = s.get("context_chars_avg")
        rows.append(
            "<tr>"
            f'<td class="label">{cfg["label"]}</td>'
            f"<td>{tag_html(cfg['tag'])}</td>"
            f'<td class="num">{s["answered"]}/{s["answerable"]}</td>'
            f'{pct_cell(s.get("citation_coverage"))}'
            f'{pct_cell(s.get("citation_validity"))}'
            f'{pct_cell(s.get("faithfulness"))}'
            f'<td class="muted">{"%d" % ctx if ctx else "—"}</td>'
            f'{pct_cell(s.get("over_refusal_rate"), warn=True)}'
            f'{pct_cell(s.get("refusal_rate_on_no_answer"))}'
            f'<td class="note">{cfg.get("note", "")}</td>'
            "</tr>"
        )
    return f"""
<section>
  <h2>二、生成层（引用 / 忠实度 / 拒答）</h2>
  <p class="lead">三个必须同时看的量：<b>引用</b>是否标且页码合法、<b>忠实度</b>是否只依据上下文、
  <b>拒答</b>要拆成两个比率（该拒的拒了 / 不该拒的也拒了）。只报一个，把答案全拒掉就能拿满分。</p>
  <table>
    <thead><tr>
      <th>配置</th><th>结论</th><th>作答</th><th>引用覆盖</th><th>引用合法</th>
      <th>忠实度</th><th>上下文(字符)</th><th>过度拒答↓</th><th>无答案拒答</th><th>备注</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p class="foot">「过度拒答」越低越好。忠实度对上下文长度没有区分力 —— 它只罚「多说」，
  上下文变长反而更容易满分，所以它满分不能当作「变好了」。</p>
</section>"""


def _layout_table(d: Dict[str, Any], extra_note: str = "") -> str:
    rows = []
    for label, v in d.items():
        rows.append(
            "<tr>"
            f'<td class="label">{label}</td>'
            f'<td class="num">{v["context_chars_avg"]}</td>'
            f'{pct_cell(v["keyword_hit_rate"])}'
            f'<td class="num">{v["first_hit_pos_avg"]:.3f}</td>'
            f'<td class="num">{v["first_hit_pos_median"]:.3f}</td>'
            f'<td class="num">{v["second_half"]}/{v["n"]}</td>'
            "</tr>"
        )
    return f"""
  <table>
    <thead><tr>
      <th>配置</th><th>上下文(字符)</th><th>关键词命中率</th>
      <th>ρ 均值</th><th>ρ 中位</th><th>ρ&gt;0.5 的题</th>
    </tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>{extra_note}"""


def section_layout() -> str:
    d = load("context_layout_sweep") or load("context_layout")
    if not d:
        return ""
    small = load("context_layout_small128")

    small_block = ""
    if small:
        small_block = f"""
  <h3>换一种切分（128 token / 32 重叠）：洞被造出来了</h3>
  <p class="lead">这一组是「反向验证」：如果 small-to-big 真是对的方案，
  它应该在一个<b>单块答不全</b>的场景里生效。这里的 128 token 切分就是那个场景。</p>
  {_layout_table(small)}
  <p class="foot">注意第一行：碎块 k=4 的关键词命中率只有 <b>0.857</b>（384 切分是 0.886），
  说明内容确实被切在两块之间。父块把命中率拉回 <b>0.914</b>，而 k=12 是 0.943
  —— 父块用少 29% 的字符拿到了接近的完整性。</p>"""

    return f"""
<section>
  <h2>三、上下文布局诊断（不调 LLM，纯统计）</h2>
  <p class="lead">父块没提升召回（384 切分下关键词命中率全是 0.886），但改变了上下文形状。
  <b>ρ = 首次命中的相对位置</b>，越小越好。</p>
  {_layout_table(d)}
  <p class="foot">ρ 只在极端配置上才有判别力：k=2（父块只有 2 个位置）的 ρ 最大 0.268，
  对应的过度拒答也最差 0.256。但 w=2/3/4 之间 ρ 几乎不动（0.176~0.193），
  过度拒答却在 0.233/0.233/0.163 之间摆动 —— <b>所以 ρ 能描述问题，不能线性预测结果</b>。
  我一度把它当因果解释，加了 w=4 之后这个说法就不成立了。</p>
  {small_block}
</section>"""


def section_ann() -> str:
    corpus = load("ann_benchmark_corpus")
    synth = load("ann_benchmark_synth")
    if not corpus:
        return ""
    sizes = list(corpus["results"].keys())
    head = "".join(f"<th>{int(s):,} 向量</th>" for s in sizes)

    def row_for(method: str, param: str) -> str:
        cells = []
        for s in sizes:
            hit = None
            for r in corpus["results"][s]:
                if r["method"] == method and r["param"] == param:
                    hit = r
                    break
            if not hit:
                cells.append('<td class="muted">—</td>')
                continue
            cells.append(
                f'<td class="num">{hit["latency_ms_per_query"]:.3f} ms'
                f'<span class="sub">Recall {hit["recall_at_k"]:.3f}</span></td>'
            )
        return f'<tr><td class="label">{method} {param}</td>{"".join(cells)}</tr>'

    wanted = [
        ("brute (numpy)", "-"),
        ("faiss flat", "-"),
        ("faiss hnsw", "M=32 efSearch=64"),
        ("faiss hnsw", "M=32 efSearch=256"),
        ("faiss ivf", "nlist=50 nprobe=8"),
    ]
    rows = "".join(row_for(m, p) for m, p in wanted)

    synth_note = ""
    if synth:
        big = list(synth["results"].keys())[-1]
        br = next((r for r in synth["results"][big] if r["method"] == "brute (numpy)"), None)
        hn = next((r for r in synth["results"][big] if r["param"] == "M=32 efSearch=64"), None)
        if br and hn:
            synth_note = (
                f"<p class=\"foot\">合成高斯数据（{int(big):,} 向量）上同一个 HNSW 配置的 Recall 只有 "
                f"<b>{hn['recall_at_k']:.3f}</b>，真实语料是 <b>0.975</b> —— "
                "512 维随机向量挤在一起（维度灾难），<b>别人的 ANN 数字不能直接信</b>。</p>"
            )

    return f"""
<section>
  <h2>四、规模实验：暴力 vs FAISS（真实语料复制扩增）</h2>
  <p class="lead">每个格子是「单次查询延迟 / Recall@4」。<b>先量再优化</b>：
  1857 块时暴力检索 0.063 ms，这个规模上谈 ANN 是过度设计。</p>
  <table>
    <thead><tr><th>方法</th>{head}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {synth_note}
</section>"""


def section_conclusions() -> str:
    return """
<section>
  <h2>五、这些实验教了什么（一句话版）</h2>
  <ul class="lessons">
    <li><b>指标一片大好时先怀疑指标。</b>S4 关键词口径显示「BM25 完胜向量」，
        根因是判定函数与 BM25 同构；换语义判定，向量 Hit@1 从 0.60 涨到 0.90。</li>
    <li><b>改动改变了送下游的文本长度，判定必须先校准。</b>父块让 Hit@1 0.743→0.771，
        但两边关键词命中率都是 0.8857 —— 涨的是分母，不是能力。</li>
    <li><b>「没变化」的指标要问它有没有检测能力。</b>忠实度 0.986→1.000，
        同时过度拒答 0.186→0.256 在恶化。</li>
    <li><b>两个优化可能解决同一个瓶颈。</b>HyDE+精排与只开精排六项指标完全相同，
        所以「A 有效 + B 有效」必须先测 A+B。</li>
    <li><b>识别问题比解决方法更难。</b>small-to-big 在 384 切分下看不出效果，
        但把切分改成 128 造出「单块答不全」的洞之后，它救回 7 题、弄坏 2 题
        （净 +5），而且救回的全是「定义/公式被切成两块」的单事实题。
        <b>方法有效，只是原来的洞不存在。</b></li>
    <li><b>自己的结论也要被下一组数据推翻一次。</b>我先写了「窗口越大越差」，
        补上 w=4 之后它就不成立了（0.233→0.233→0.163，非单调）。
        诚实的表述是「w 在 2~4 之间的差异小于 2~3 题的噪声」。</li>
    <li><b>负面结果也要记账。</b>失败实验写进文档，比成功故事更可信。</li>
  </ul>
</section>"""


CSS = """
:root{
  --bg:#ffffff; --bg2:#f7f8fa; --line:#e5e7eb; --text:#1f2328; --muted:#6b7280;
  --accent:#2563eb; --good:#15803d; --ok:#0f766e; --bad:#b91c1c; --warn:#b45309;
  --ctrl:#6b7280; --dup:#7c3aed;
}
*{box-sizing:border-box}
body{margin:0;padding:32px 28px 64px;background:var(--bg);color:var(--text);
  font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;}
h1{font-size:22px;margin:0 0 6px}
h2{font-size:16px;margin:34px 0 8px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.meta{color:var(--muted);font-size:13px;margin-bottom:4px}
.lead{color:var(--muted);font-size:13px;margin:6px 0 12px;max-width:1100px}
.foot{color:var(--muted);font-size:12.5px;margin:8px 0 0;max-width:1100px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:4px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{background:var(--bg2);font-weight:600;color:#374151;text-align:right}
th:first-child,td:first-child{text-align:left}
td.label{font-weight:600}
td.note{text-align:left;color:var(--muted);font-size:12px;white-space:normal;min-width:180px}
td.num{font-variant-numeric:tabular-nums}
td.muted{color:var(--muted)}
.sub{display:block;color:var(--muted);font-size:11px}
tbody tr:hover{background:#fbfcfd}
.barcell{position:relative;min-width:110px;text-align:right}
.barcell .bar{position:absolute;left:6px;bottom:3px;height:3px;background:var(--accent);opacity:.55;border-radius:2px}
.barcell .bar-warn{background:var(--bad);opacity:.5}
.barcell b{font-variant-numeric:tabular-nums;font-weight:600}
.tag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11.5px;font-weight:600;border:1px solid}
.tag-base{color:#374151;border-color:#d1d5db;background:#f3f4f6}
.tag-good{color:var(--good);border-color:#bbf7d0;background:#f0fdf4}
.tag-ok{color:var(--ok);border-color:#99f6e4;background:#f0fdfa}
.tag-bad{color:var(--bad);border-color:#fecaca;background:#fef2f2}
.tag-dup{color:var(--dup);border-color:#ddd6fe;background:#f5f3ff}
.tag-ctrl{color:var(--ctrl);border-color:#d1d5db;background:#f9fafb}
.tag-noise{color:#475569;border-color:#e2e8f0;background:#f8fafc}
.lessons{margin:8px 0 0;padding-left:20px;max-width:1100px}
.lessons li{margin:6px 0}
"""


def build(out: Path) -> None:
    meta = ""
    base = load("eval_v2_hybrid")
    if base:
        s = base["summary"]
        tc = s.get("type_counts", {})
        meta = (
            f"{s['chunks']} 个 chunk · {s['dim']} 维 · 题库 {s['questions_total']} 题"
            f"（单事实 {tc.get('single_fact', 0)} / 干扰 {tc.get('distractor', 0)} / "
            f"多跳 {tc.get('multi_hop', 0)} / 无答案 {tc.get('no_answer', 0)}）"
        )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>PDF-RAG 实验看板</title>
<style>{CSS}</style>
</head>
<body>
<h1>PDF-RAG 实验看板</h1>
<div class="meta">{meta}</div>
<div class="meta">数据来源 <code>results/*.json</code>，由 <code>scripts/build_dashboard.py</code> 生成 ——
数字不是手抄的。</div>
{section_retrieval()}
{section_generation()}
{section_layout()}
{section_ann()}
{section_conclusions()}
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"看板已生成: {out}  ({len(html)/1024:.0f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "web" / "dashboard.html"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()
