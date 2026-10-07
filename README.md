# PDF RAG System

一个用于**学习 RAG（Retrieval-Augmented Generation）全链路**的 PDF 问答系统。

从 PDF 加载 → 文本切分 → 向量化 → 向量检索 → 调用 LLM 生成答案，每一步都拆成独立模块，
便于逐环节替换、做对照实验。目标不是"能跑"，而是**能说清每个环节为什么这么设计**。

## 目录结构

```
PDF-ptokect/
├── config/
│   └── config.py            # 路径、切分参数、LLM 配置、检索参数
├── src/
│   ├── document_loader.py   # 1. 加载 PDF（PyPDFLoader）
│   ├── text_splitter.py     # 2. 文本切分（RecursiveCharacterTextSplitter）
│   ├── embedding.py         # 3. 向量化（sentence-transformers）
│   ├── vector_store.py      # 4. 向量库（自研 JSON 持久化 + 暴力余弦检索）
│   └── rag_pipeline.py      # 5. 检索 + 组装 prompt + 调 LLM
├── utils/
│   └── logger.py            # 统一日志
├── data/
│   ├── pdfs/                # 待索引的 PDF
│   └── vector_store/        # 向量库持久化（index.json）
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

## 使用方法

```bash
# 单次提问
python main.py --query "什么是 RAG？"

# 重建向量库后提问
python main.py --recreate --query "什么是 RAG？"

# 进入交互模式
python main.py --chat
```

未配置 `DEEPSEEK_API_KEY` 时，程序仍可运行**检索部分**，只是不调用 LLM 生成答案。

## 设计说明

- **为什么自研向量库而不是直接用 Chroma/FAISS？**
  个人探索；当前数据量只有 2 个 chunk，暴力余弦检索足够，也便于观察相似度计算过程。
  后续会接入成熟索引（见下）。
- **为什么切分参数是 1000 / 200？**
  当前是经验值，**尚未做过对照实验**，属于待验证项。
- **embedding 模型**：当前为 `all-MiniLM-L6-v2`（英文模型），**对中文支持有限**，是已知待改进项。

## 已知限制 / 待办

- [ ] embedding 模型对中文不友好，检索中文问题效果差（计划替换为中文模型）
- [ ] 切分策略未做对照实验
- [ ] 检索结果中的页码为 0 基，展示时应 +1
- [ ] 无评测集，改动效果无法量化
- [ ] 未接入成熟向量索引（Chroma / FAISS）

## 演进路线

按"**一阶段只改一个变量**，改完用固定问题集对比、把指标写进文档"的方式推进：

| 阶段 | 目标 |
| --- | --- |
| S0 | 工程清理：去掉死代码与悬空配置，让依赖可信 |
| S1 | 替换为中文 embedding 模型（预期收益最大） |
| S2 | 切分策略对照实验 |
| S3 | 接入成熟向量索引 + 向量预归一化 |
| S4 | 混合检索（BM25）+ RRF 融合 + rerank |
| S5 | 构建评测集与指标脚本（Hit Rate / MRR） |
| S6 | Query 改写 / 父子块 / 服务化 |
