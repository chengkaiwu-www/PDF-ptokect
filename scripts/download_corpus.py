"""
下载测试语料（3 篇中文 PDF + 1 篇自带英文样例）。

语料 PDF 不入库（体积大、属第三方版权材料），所以把获取方式写成脚本，
保证任何人 clone 之后都能一键复现同一批实验数据。

用法：
  python scripts/download_corpus.py            # 直连下载
  python scripts/download_corpus.py --mirror   # 走 ghproxy 镜像（GitHub 慢或被限速时用）

注意：这三份都来自 GitHub 上的开源中文书籍发行版，均为公开可下载材料，
      仅用于本地检索实验，请勿再分发。
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"

CORPUS = [
    {
        # 《动手学大模型应用开发》——与大模型应用/RAG 直接相关
        "filename": "llm-universe.pdf",
        "url": "https://github.com/datawhalechina/llm-universe/releases/download/v1/v1.pdf",
        "note": "Datawhale 动手学大模型应用开发",
    },
    {
        # 《南瓜书：机器学习公式详解》——公式推导密集，可用于验证公式类问答
        "filename": "pumpkin-book.pdf",
        "url": "https://github.com/datawhalechina/pumpkin-book/releases/download/v2.0.0/pumpkin_book.pdf",
        "note": "Datawhale 南瓜书（机器学习公式详解）",
    },
    {
        # 吴恩达机器学习课程中文笔记——Word 排版的 PDF
        "filename": "ml-notes-ng.pdf",
        "url": (
            "https://raw.githubusercontent.com/fengdu78/Coursera-ML-AndrewNg-Notes/master/"
            "%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%AA%E4%BA%BA%E7%AC%94%E8%AE%B0"
            "%E5%AE%8C%E6%95%B4%E7%89%88v5.52-A4%E6%89%93%E5%8D%B0%E7%89%88.pdf"
        ),
        "note": "吴恩达机器学习课程个人笔记（中文）",
    },
]

MIRROR_PREFIX = "https://ghproxy.net/"


def download(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pdf-rag-project/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                print(f"\r    已下载 {total / 1024 / 1024:6.1f} MB", end="", flush=True)
        print()
    except Exception as e:
        print(f"\n    下载失败: {type(e).__name__}: {e}")
        tmp.unlink(missing_ok=True)
        return False

    # 校验：必须是合法 PDF（避免把一堆 HTML 错误页存成 .pdf 还浑然不觉）
    with open(tmp, "rb") as f:
        if f.read(5) != b"%PDF-":
            print("    文件头不是 %PDF-，判定为无效下载，已丢弃")
            tmp.unlink(missing_ok=True)
            return False
    tmp.replace(dest)
    print(f"    -> {dest.name}  {dest.stat().st_size / 1024 / 1024:.1f} MB")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="下载测试语料 PDF")
    parser.add_argument("--mirror", action="store_true",
                        help="通过 ghproxy 镜像下载（GitHub 直连慢时使用）")
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    print(f"语料目录: {PDF_DIR}")

    ok = failed = 0
    for item in CORPUS:
        dest = PDF_DIR / item["filename"]
        if dest.exists() and not args.force:
            print(f"\n[跳过] {item['filename']} 已存在（{dest.stat().st_size / 1024 / 1024:.1f} MB）")
            ok += 1
            continue
        url = (MIRROR_PREFIX + item["url"]) if args.mirror else item["url"]
        print(f"\n[下载] {item['filename']}  <- {item['note']}")
        if download(url, dest):
            ok += 1
        else:
            failed += 1

    print(f"\n完成: 成功 {ok} / 失败 {failed}")
    if failed:
        print("提示: 若下载失败，重试加 --mirror 参数走镜像站")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
