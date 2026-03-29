#!/usr/bin/env python3
"""
使用 pdfplumber 处理《医院导诊指南大全》PDF
"""

import pdfplumber
import json
import os
import re

PDF_PATH = "demo/data/《医院导诊指南大全》.pdf"
OUTPUT_DIR = "demo/data/pdf_output"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PDF 处理工具 (pdfplumber)")
print("=" * 60)
print(f"输入文件: {PDF_PATH}")
print(f"输出目录: {OUTPUT_DIR}")
print()

all_text = []
all_pages = []

with pdfplumber.open(PDF_PATH) as pdf:
    total_pages = len(pdf.pages)
    print(f"总页数: {total_pages}")

    for i, page in enumerate(pdf.pages):
        print(f"\r处理进度: {i + 1}/{total_pages}...", end="", flush=True)

        # 提取文本
        text = page.extract_text()
        if text:
            all_text.append(f"--- 第 {i + 1} 页 ---\n{text}")

        # 提取表格
        tables = page.extract_tables()
        if tables:
            for j, table in enumerate(tables):
                table_text = f"\n[表格 {j + 1}]\n"
                for row in table:
                    table_text += (
                        " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                    )
                all_text.append(f"--- 第 {i + 1} 页 [表格] ---\n{table_text}")

        all_pages.append({"page": i + 1, "text": text, "tables": tables})

print("\n")

# 1. 保存纯文本
txt_content = "\n\n".join(all_text)
txt_path = os.path.join(OUTPUT_DIR, "hospital_guide.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content)
print(f"✓ 纯文本已保存: {txt_path}")

# 2. 保存为 JSON
json_path = os.path.join(OUTPUT_DIR, "hospital_guide.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(all_pages, f, ensure_ascii=False, indent=2)
print(f"✓ JSON已保存: {json_path}")

# 3. 保存为 Markdown (模拟)
md_content = "# 医院导诊指南大全\n\n"
for page in all_pages:
    if page["text"]:
        md_content += f"\n## 第 {page['page']} 页\n\n"
        md_content += page["text"]
        md_content += "\n\n"

md_path = os.path.join(OUTPUT_DIR, "hospital_guide.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"✓ Markdown已保存: {md_path}")

# 统计信息
print("\n" + "=" * 60)
print("处理完成！输出文件:")
print("=" * 60)

for f in os.listdir(OUTPUT_DIR):
    fpath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  - {f} ({size:,} bytes)")

# 显示预览
print("\n--- 文本预览 (前500字符) ---")
print(txt_content[:500] + "..." if len(txt_content) > 500 else txt_content)
