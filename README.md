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
│   ├── document_loader.py   # 1. 加载 PDF（PyPDFLoader）
│   ├── text_splitter.py     # 2. 文本切分（RecursiveCharacterTextSplitter）
│   ├── embedding.py         # 3. 向量化（sentence-transformers）
│   ├── vector_store.py      # 4. 向量库（自研 JSON 持久化 + 暴力余弦检索）
│   └── rag_pipeline.py      # 5. 检索 + 组装 prompt + 调 LLM
├── utils/
│   └── logger.py            # 统一日志
├── models/                  # 本地 embedding 模型（不入库，见「环境准备」）
├── data/
│   ├── pdfs/                # 待索引的 PDF
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

## 使用方法

```bash
# 单次提问
python main.py --query "什么是 RAG？"

# 重建索引后提问（更换 embedding 模型或切分参数后必须执行）
python main.py --recreate --query "什么是 RAG？"

# 进入交互模式
python main.py --chat
```

未配置 `DEEPSEEK_API_KEY` 时，程序仍可运行**检索部分**，只是不调用 LLM 生成答案。

> 本机推荐解释器：`D:\anaconda\python.exe`（依赖齐全且 GPU 可用）。

## 设计说明

- **为什么自研向量库而不是直接用 Chroma/FAISS？**
  个人探索；当前数据量只有 2 个 chunk，暴力余弦检索足够，也便于观察相似度计算过程。
  后续会接入成熟索引（见下）。
- **为什么切分参数是 1000 / 200？**
  当前是经验值，**尚未做过对照实验**，属于待验证项（S2）。
- **embedding 模型**：`BAAI/bge-small-zh-v1.5`（512 维，中文）。取代了最初的
  `all-MiniLM-L6-v2`——后者是纯英文模型，见下方实验记录。

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

### 实验 2 · 已知但尚未处理的风险：chunk 长度 vs 模型输入上限

`bge-small-zh-v1.5` 的最大输入长度是 **512 token**。当前 `chunk_size=1000` 是**字符**数：

| 语料语言 | 约合 token | 是否超限 |
| --- | --- | --- |
| 英文（当前测试 PDF） | 983 字符 ≈ 281 token | 安全 |
| 中文（真实文档） | 1000 字符 ≈ **1000 token** | **超出，尾部被静默截断丢弃** |

即：当前测试文档是英文所以没问题，**一旦放入中文 PDF，每个 chunk 的后半段根本不会进入向量**。
这是 S2（切分策略）要解决的问题。

## 已知限制 / 待办

- [ ] 切分策略未做对照实验；`chunk_size` 对中文文档会超出模型 512 token 上限（见实验 2）
- [ ] 检索实现粗糙：硬编码 `> 0.1` 阈值、每次查询重算文档范数、`argsort` 全排序
- [ ] 无评测集，改动效果只能靠少量问题定性观察
- [ ] 未接入成熟向量索引（Chroma / FAISS）
- [ ] `RAGPipeline` 的 `use_reranker` 参数已声明但未实现

## 演进路线

按「**一阶段只改一个变量**，改完用固定问题集对比、把指标写进文档」的方式推进：

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| S0 | 工程清理：去掉死代码与悬空配置，让依赖可信 | 已完成 |
| S1 | 替换为中文 embedding 模型 | 已完成 |
| S2 | 切分策略对照实验（含中文 token 上限问题） | 进行中 |
| S3 | 接入成熟向量索引 + 向量预归一化 | 待办 |
| S4 | 混合检索（BM25）+ RRF 融合 + rerank | 待办 |
| S5 | 构建评测集与指标脚本（Hit Rate / MRR） | 待办 |
| S6 | Query 改写 / 父子块 / 服务化 | 待办 |
