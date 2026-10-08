# PDF RAG System

一个用于**学习 RAG（Retrieval-Augmented Generation）全链路**的 PDF 问答系统。

从 PDF 加载 → 文本切分 → 向量化 → 向量检索 → 调用 LLM 生成答案，每一步都拆成独立模块，
便于逐环节替换、做对照实验。目标不是「能跑」，而是**能说清每个环节为什么这么设计**。

## 目录结构

```
PDF-ptokect/
├── config/
│   └── config.py            # 唯一配置源：模型、切分参数、top_k、路径
├── src/
│   ├── document_loader.py   # 1. 加载 PDF（pypdfium2 / PDFium 引擎）
│   ├── text_splitter.py     # 2. 文本切分（RecursiveCharacterTextSplitter，支持 token 计数）
│   ├── embedding.py         # 3. 向量化（sentence-transformers）
│   ├── vector_store.py      # 4. 向量库（自研 JSON 持久化 + 暴力余弦检索）
│   └── rag_pipeline.py      # 5. 检索 + 组装 prompt + 调 LLM
├── eval/
│   ├── questions.json       # 固定问题集（20 题，每题带命中关键词）
│   └── run_eval.py          # 检索评测脚本：Hit@1 / Hit@k / MRR
├── scripts/
│   └── download_corpus.py   # 一键获取测试语料 PDF
├── utils/
│   └── logger.py            # 统一日志
├── models/                  # 本地 embedding 模型（不入库，见「环境准备」）
├── data/
│   ├── pdfs/                # 待索引的 PDF（第三方语料不入库，见「环境准备」）
│   └── vector_store/        # 向量索引持久化（index.json，派生产物，不入库）
├── main.py                  # 入口：初始化 / 查询 / 交互
├── simple_rag.py            # 早期教学版（单文件，伪向量，保留作对照）
├── create_test_pdf.py       # 生成测试用 PDF
├── requirements.txt         # 核心依赖
├── requirements_torch_cuda.txt  # 含 CUDA 版 torch 的完整依赖快照
└── .env.example             # 环境变量模板
```

## 环境准备

1. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

   若需要 GPU 版 PyTorch（推荐，向量化显著更快），改用：

   ```bash
   pip install -r requirements_torch_cuda.txt
   ```

2. 配置环境变量：复制 `.env.example` 为 `.env`，填入 DeepSeek API Key。

   ```
   DEEPSEEK_API_KEY=sk-xxxxxxxx
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_API_BASE=https://api.deepseek.com
   LOG_LEVEL=INFO
   ```

   > `.env` 已被 `.gitignore` 忽略，**不要提交到仓库**。

3. 准备 embedding 模型（首次运行需要）：

   默认使用中文模型 `BAAI/bge-small-zh-v1.5`，放在 `models/bge-small-zh-v1.5/`。

   ```bash
   python -c "
   import os
   os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
   from huggingface_hub import snapshot_download
   snapshot_download(
       repo_id='BAAI/bge-small-zh-v1.5',
       local_dir='models/bge-small-zh-v1.5',
       allow_patterns=['*.json', '*.txt', '*.safetensors'],
   )
   "
   ```

   > 国内网络无法直连 `huggingface.co`，需走镜像站 `hf-mirror.com`（`config.py` 已默认设置）。
   > 用 `allow_patterns` 跳过冗余的 `pytorch_model.bin`，下载量从 183MB 降到 92MB。
   > 也可直接指定 HuggingFace 模型名，例如 `EMBEDDING_MODEL=BAAI/bge-m3`（会自动走镜像下载）。

4. 准备语料（不入库，需自行下载）：

   ```bash
   python scripts/download_corpus.py            # GitHub 直连
   python scripts/download_corpus.py --mirror   # 走 ghproxy 镜像（直连慢时用）
   ```

   会拉取 3 篇中文 PDF（约 36MB，合计 719 页）：

   | 文件 | 来源 | 特点 |
   | --- | --- | --- |
   | `llm-universe.pdf` | Datawhale《动手学大模型应用开发》 | 与大模型应用 / RAG 直接相关 |
   | `pumpkin-book.pdf` | Datawhale《南瓜书：机器学习公式详解》 | XeLaTeX 排版，公式密集 |
   | `ml-notes-ng.pdf` | 吴恩达机器学习课程中文笔记 | Word 排版 |

   也可以直接把自己的 PDF 放进 `data/pdfs/`。

## 使用方法

```bash
# 单次提问
python main.py --query "什么是 RAG？"

# 重建索引后提问（更换 embedding 模型或切分参数后必须执行）
python main.py --recreate --query "什么是 RAG？"

# 进入交互模式
python main.py --chat

# 检索质量评测（Hit@1 / Hit@k / MRR）
python eval/run_eval.py --detail --save /tmp/eval.json
```

未配置 `DEEPSEEK_API_KEY` 时，程序仍可运行**检索部分**，只是不调用 LLM 生成答案。

> 本机推荐解释器：`D:\anaconda\python.exe`（依赖齐全且 GPU 可用）。

### 可调参数（环境变量覆盖，便于做对照实验）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `EMBEDDING_MODEL` | `models/bge-small-zh-v1.5` | 可填 HuggingFace 模型名 |
| `CHUNK_UNIT` | `token` | `token` 按模型 token 数切分；`char` 按字符数 |
| `CHUNK_SIZE` | `384` | 切分块大小（单位见上） |
| `CHUNK_OVERLAP` | `80` | 相邻块重叠大小 |
| `CHUNK_NORMALIZE` | `0` | 是否把 PDF 物理行合并为语义段落（实测无收益，默认关） |
| `RETRIEVAL_TOP_K` | `4` | 召回条数 |

## 设计说明

- **为什么自研向量库而不是直接用 Chroma/FAISS？**
  个人探索；当前数据量（1860 块）暴力余弦检索足够，也便于观察相似度计算过程。
  后续会接入成熟索引（见演进路线 S3）。
- **为什么用 pypdfium2 抽 PDF，而不是 langchain 默认的 PyPDFLoader？**
  实测中文 PDF 抽取失败，见实验 3。
- **为什么切分按 token 而不是按字符？**
  embedding 模型的输入上限是按 token 算的，按字符切会让中文块超限被静默截断，见实验 4。
- **embedding 模型**：`BAAI/bge-small-zh-v1.5`（512 维，中文）。取代了最初的
  `all-MiniLM-L6-v2`——后者是纯英文模型，见实验 1。

## 评测方法（`eval/`）

`eval/questions.json` 是 20 道固定问题（覆盖 3 篇文档），每题带 `expected_keywords`。
`eval/run_eval.py` 对全部块做余弦排序，统计：

| 指标 | 含义 |
| --- | --- |
| Hit@1 | 排第一的块是否包含任意关键词 |
| Hit@k | 前 k 个块里是否有命中 |
| MRR | 首个命中块排名的倒数平均值（第 1 位 = 1.0） |

**这个指标的边界（必须说清楚）**：
- 关键词命中是 Hit Rate 的**廉价代理**，不是人工标注的「标准答案块」，会低估检索质量——
  模型可能召回了内容正确但用词不同的块。
- 关键词覆盖率事先统计过（避免出现太普遍的词导致指标失去区分度），但**题目之间难度并不均一**。
- 因此它**只能用于同一批题、同一套判定下的横向比较**，不能当作绝对分数或对外宣称「准确率」。

## 实验记录

> 原则：**一次只改一个变量**，其余条件完全相同，改完用同一组问题对比前后结果。

### 实验 1 · 英文 embedding → 中文 embedding（S1）

**背景**：项目最初中文提问经常答不出，第一反应容易去调 prompt，但无效。

**假设**：根因不在 LLM，而在**检索**——`all-MiniLM-L6-v2` 是纯英文模型（384 维），
编码中文时语义空间塌陷，几乎所有中文句子的向量都挤在一起，检索接近随机。

**方法**：单变量对照。同一份文档、同一套代码，只切换 `EMBEDDING_MODEL` 后重建索引。

| 问题 | `all-MiniLM-L6-v2`（384维） | `bge-small-zh-v1.5`（512维） |
| --- | --- | --- |
| RAG 有哪些好处？ | 召回 **1** 块，且不含答案 → LLM 拒答 | 召回 **2** 块 → **正确答出全部 4 条好处** |
| RAG 是怎么工作的？ | 召回 1 块，能答（答案恰好在第 1 块） | 召回 2 块，能答 |

**补充证据（文本层面，不依赖语料规模）**：
对同一组中文句子做两两余弦相似度，比较「相关句对」与「无关句对」的分数差：

| 句对 | `all-MiniLM-L6-v2` | `bge-small-zh-v1.5` |
| --- | --- | --- |
| 「RAG 有哪些好处？」 ↔「RAG 的优点是更新及时、减少幻觉」（相关） | 0.688 | 0.706 |
| 「RAG 有哪些好处？」 ↔「今天天气不错」（无关） | **0.545** | **0.303** |
| **区分度（相关 − 无关）** | **0.143** | **0.403** |

英文模型给毫不相关的中文句子打了 0.545 的高分——**这正是中文检索失效的直接证据**：
无关内容也能拿到不低的相似度，排序自然失去意义。换模型后区分度提升到约 2.8 倍。

**结论**：RAG 效果差时，**先怀疑检索，不要习惯性甩锅给 LLM**。

**代价与注意**：
- 向量维度 384 → 512，**必须重建索引**（`python main.py --recreate`），旧索引不兼容
- 模型体积 92MB，CPU 也能跑；本项目在 GPU（RTX 4060）上编码

### 实验 2 · 语料升级：从 1 篇英文测试文档到 3 篇中文书籍（S2）

原来只有 1 篇英文测试 PDF、**2 个 chunk**，任何指标都没有区分度（怎么排都是这两块）。
S2 起换成 3 篇中文书籍：**720 页 / 700KB 文本 / 1860 个 chunk**，评测才有统计意义。

同时把语料 PDF 加入 `.gitignore`（体积大 + 第三方版权），改为提供
`scripts/download_corpus.py` 一键获取，保证 clone 后能复现同一批实验数据。

### 实验 3 · PDF 解析器选型（S2）

**现象**：`pumpkin-book.pdf` 用原来的 `PyPDFLoader` 抽出来**全是乱码**，汉字占比 0.0%。

**排查**：用 4 个解析器抽同一份文件（196 页）：

| 解析器 | 抽取字符数 | 汉字占比 | 耗时 |
| --- | --- | --- | --- |
| `pypdf`（原 `PyPDFLoader` 的底层） | 209,834 | **0.0%（乱码）** | 8.2s |
| `pdfminer` | 330,183 | 27.4% | 33.7s |
| `pdfplumber` | 312,030 | 29.0% | 32.8s |
| **`pypdfium2`（现用）** | 328,715 | **27.6%** | **2.5s** |

**根因**：该 PDF 由 XeLaTeX（`xdvipdfmx`）生成，内嵌的字体子集缺少 ToUnicode 映射，
文字能不能抽出来取决于解析器的兜底策略。对比另两份：Word 排版的 `ml-notes-ng.pdf`
三种解析器都能抽（≈48%），说明**问题不在「中文 PDF」本身，而在「特定生成方式 × 特定解析器」的组合**。

**为什么这个坑危险**：抽取失败**不报错**，只是把乱码默默喂给 embedding，检索静默失效。
因此 loader 里加了空页告警（扫描版 PDF 会触发，提示需要 OCR）。

**结论**：文档加载层要当作有质量风险的一环来验证——**先抽样打印，再谈建模**。

### 实验 4 · 切分单位：字符 → token（S2，关键修正）

**问题**：`bge-small-zh-v1.5` 输入上限 **512 token**，而 `chunk_size=1000` 是**字符数**。
中文约 1 字 ≈ 1 token，英文约 3.5 字 ≈ 1 token——所以「1000 字符」对英文安全，对中文必然超限。

切分后加了自检，基线索引立刻报出来：

```
chunk token 分布: min=15 median=492 max=950 (embedding 模型上限 512)
530/1142 chunk(s) 超出 embedding 模型上限 512 tokens (最大 950)，超出部分会被静默截断丢弃
```

**46.4% 的块超限，最大 950 token**——意味着这些块的后半段根本没进向量库。

**方法**：单变量对照，只改切分单位（其余：同一解析器、同一模型、同一批语料、同一组 20 题）。

| 配置 | 块数 | chunk token（中位/最大） | 超限块数 | Hit@1 | Hit@4 | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| `char` 1000 / 200 | 1142 | 492 / **950** | **530（46.4%）** | 0.450 | 0.600 | 0.517 |
| **`token` 384 / 80** | 1860 | 357 / 386 | **0** | **0.600** | **0.700** | **0.642** |

**结论**：Hit@1 +0.150、Hit@4 +0.100、MRR +0.125。切分单位必须与 embedding 模型的计数方式对齐。

> 顺带一提：本次同时把分隔符从默认的 `["\n\n","\n"," ",""]` 换成了中文友好的
> `["\n\n","\n","。","！","？","；","：","，"," ",""]`（中文几乎不用空格，默认分隔符
> 会把切点落在任意位置）。这一项与解析器修复一起发生在基线测量之前，**未单独度量**，
> 严格来说这笔账还挂着——见「已知限制」。

### 实验 5 · 段落规整（失败，默认关闭）

**假设**：PDF 里的换行是排版产物（为了塞进页宽），不是语义边界，合并成段落后切分质量应该更好。

**方法**：单变量，在 `token 384 / 80` 基础上打开 `CHUNK_NORMALIZE`。

| 配置 | 块数 | Hit@1 | Hit@4 | MRR |
| --- | --- | --- | --- | --- |
| `token` 384 / 80（对照） | 1860 | **0.600** | **0.700** | **0.642** |
| + 段落规整（首版） | 1929 | 0.550 | 0.700 | 0.604 |
| + 段落规整（加行长度门槛后重试） | 1926 | 0.550 | 0.700 | 0.617 |

**两个版本都没有跑赢对照，指标反而下降。**

**原因分析**：合并物理行时，**小标题会被并进上一段**，等于亲手毁掉了 `\n\n` 这个
优先级最高的分隔符——本来「标题 / 正文」是天然的语义边界，合并后切点变得随机。
第二版加了「只合并较长正文行」的门槛，回升一点但仍不达标。

**结论**：这是个**负面结果**，但它很有价值——「看起来更干净」的文本预处理不一定更好，
**必须用指标说话**。代码和开关都保留（`CHUNK_NORMALIZE=1` 可复现），默认关闭。

## 已知限制 / 待办

- [ ] 中文友好分隔符的收益**尚未单独度量**（与解析器修复捆绑在基线之前，见实验 4 备注）
- [ ] 页眉页脚类**重复样板文本**未处理（如南瓜书每页都印的书名推广语会污染检索），
      可用「跨页高频行」规则过滤
- [ ] 检索实现粗糙：硬编码 `> 0.1` 阈值（会让召回数少于 top_k，且把「检索」和「过滤」混在一起）、
      每次查询重算文档范数、`argsort` 全排序
- [ ] 评测集只有 20 题、只覆盖检索层；生成层（答案正确性、忠实度）尚无指标
- [ ] 未接入成熟向量索引（Chroma / FAISS）
- [ ] `RAGPipeline` 的 `use_reranker` 参数已声明但未实现

## 演进路线

按「**一阶段只改一个变量**，改完用固定问题集对比、把指标写进文档」的方式推进：

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| S0 | 工程清理：去掉死代码与悬空配置，让依赖可信 | 已完成 |
| S1 | 替换为中文 embedding 模型 | 已完成 |
| S2 | 切分策略对照实验（中文 token 上限 / 解析器选型 / 段落规整） | 已完成 |
| S3 | 接入成熟向量索引 + 向量预归一化 | 待办 |
| S4 | 混合检索（BM25）+ RRF 融合 + rerank | 待办 |
| S5 | 扩充评测集（50 题）+ 生成层指标 | 待办 |
| S6 | Query 改写 / 父子块 / 服务化 | 待办 |
