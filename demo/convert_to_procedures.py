#!/usr/bin/env python3
"""
将 PDF 输出转换为结构化 JSON 格式
"""

import json
import re
import os

INPUT_FILE = "demo/data/pdf_output/hospital_guide.json"
OUTPUT_FILE = "demo/data/hospital_procedures_formatted.json"


def clean_text(text):
    """清理文本"""
    # 清理 LaTeX 公式残留
    text = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", text)
    text = re.sub(r"\$\\text\{([^}]+)\}", r"\1", text)
    return text.strip()


def extract_scene_name(text):
    """提取流程名称"""
    patterns = [
        r"(流程|指南)\n(.+?)(?:\n|流程步骤|适用科室)",
        r"(.+?)流程\n",
        r"(.+?)检查流程",
        r"(.+?)流程\n",
    ]

    # 先找标题
    lines = text.split("\n")
    for i, line in enumerate(lines[:5]):
        line = line.strip()
        if "流程" in line and len(line) < 20:
            # 合并下一行（如果是换行的情况）
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (
                    next_line
                    and not next_line.startswith("流程")
                    and not next_line.startswith("适用")
                ):
                    return line + next_line

    # 尝试正则
    match = re.search(r"(.+?)(?:流程|指南)", text[:200])
    if match:
        name = match.group(1).strip()
        # 清理多余字符
        name = re.sub(r"\s+", "", name)
        if name and len(name) < 30:
            return name + "流程"

    return "未分类流程"


def extract_department(text):
    """提取科室"""
    match = re.search(r"适用科室.*?[：:]\s*\n(.+?)(?:\n|流程类型)", text, re.DOTALL)
    if match:
        dept = match.group(1).strip()
        # 清理
        dept = re.sub(r"[●•]", "", dept).strip()
        dept = re.sub(r"\s+", "", dept)
        return dept
    return ""


def extract_process_type(text):
    """提取流程类型"""
    match = re.search(r"流程类型.*?[：:]\s*(\w+)", text)
    if match:
        return match.group(1).strip()

    # 根据关键词推断
    text_lower = text.lower()
    if "挂号" in text or "预约" in text:
        return "registration"
    elif "急诊" in text:
        return "emergency"
    elif "住院" in text:
        return "inpatient"
    elif "缴费" in text:
        return "payment"
    elif "检查" in text or "化验" in text:
        return "exam"
    elif "取药" in text:
        return "pharmacy"
    else:
        return "guide"


def find_department_in_text(scene_name, text):
    """根据场景查找科室"""
    dept_map = {
        "普通门诊就诊": "所有一级临床科室（内科、外科、妇科、儿科、五官科等）",
        "专家门诊就诊": "所有设有主任医师、副主任医师职称的科室",
        "血常规/抽血": "检验科、采血中心",
        "尿常规/大便常规": "检验科",
        "心电图检查": "心电图室",
        "B超检查": "超声科",
        "CT检查": "放射科(CT室)",
        "核磁共振MRI": "放射科(MRI室)",
        "胃镜检查": "内镜中心",
        "肠镜检查": "内镜中心",
        "胸片/DR检查": "放射科",
        "夜间急诊": "急诊科",
        "急诊输液": "急诊输液室",
        "急诊留观": "急诊留观室",
        "急诊收费与取药": "急诊收费处/急诊药房",
        "住院办理": "住院处",
        "出院办理": "出院结算处",
        "处方开药": "门诊药房",
        "缴费流程": "门诊收费处",
        "智能导诊机使用": "门诊大厅导诊台",
        "预约挂号(线上&线下)": "所有门诊科室",
        "候诊流程说明": "所有门诊诊区",
    }

    for key, dept in dept_map.items():
        if key in scene_name:
            return dept

    return extract_department(text)


def process_page(page_data, page_num):
    """处理单页数据"""
    text = clean_text(page_data.get("text", ""))
    if not text:
        return []

    procedures = []

    # 检查是否是流程的开始
    # 查找"流程步骤"或"一、流程步骤"
    sections = re.split(r"\n(?=[一二三四五六七八九十]、)", text)

    for section in sections:
        # 提取场景名称
        scene_name = extract_scene_name(section)
        if scene_name == "未分类流程":
            continue

        # 提取科室
        department = find_department_in_text(scene_name, section)

        # 提取流程类型
        process_type = extract_process_type(section)

        # 生成 ID
        proc_id = f"proc_{len(procedures) + 1:03d}"

        procedures.append(
            {
                "id": proc_id,
                "hospital": "医院导诊指南大全",
                "scene": scene_name,
                "department": department,
                "process_type": process_type,
                "raw_text": section.strip(),
                "source_file": "《医院导诊指南大全》.pdf",
                "page_range": f"P{page_num}",
            }
        )

    return procedures


def main():
    print("=" * 60)
    print("PDF 转结构化 JSON 工具")
    print("=" * 60)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print()

    # 读取输入文件
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    print(f"共 {len(pages)} 页")

    all_procedures = []
    proc_counter = 0

    for page_data in pages:
        page_num = page_data.get("page", 0)
        text = page_data.get("text", "")

        # 清理文本
        text = clean_text(text)

        if not text:
            continue

        # 提取适用科室
        dept_match = re.search(
            r"适用科室.*?[：:]\s*\n(.+?)(?:\n|流程类型)", text, re.DOTALL
        )
        department = ""
        if dept_match:
            department = re.sub(r"[●•]\s*", "", dept_match.group(1)).strip()
            department = re.sub(r"\s+", "", department)

        # 提取流程类型
        type_match = re.search(r"流程类型.*?[：:]\s*(\w+)", text)
        process_type = type_match.group(1) if type_match else "guide"

        # 提取场景名称（从标题行）
        lines = text.split("\n")
        scene_name = ""

        # 查找流程名称
        for i, line in enumerate(lines[:8]):
            line = line.strip()
            # 跳过空行和太长的行
            if not line or len(line) > 50:
                continue
            # 找包含"流程"但不包含"步骤"的行
            if "流程" in line and "步骤" not in line and "说明" not in line:
                scene_name = re.sub(r"\s+", "", line)
                break
            # 找检查名称
            if any(kw in line for kw in ["检查", "办理", "指南", "流程"]):
                scene_name = re.sub(r"\s+", "", line)
                break

        if not scene_name:
            # 尝试从开头提取
            for line in lines[:5]:
                line = re.sub(r"\s+", "", line)
                if (
                    line
                    and len(line) < 30
                    and not line.startswith("●")
                    and not line.startswith("第")
                ):
                    scene_name = line
                    break

        if not scene_name:
            scene_name = f"第{page_num}页流程"

        # 构建条目
        proc_counter += 1
        procedure = {
            "id": f"proc_{proc_counter:03d}",
            "hospital": "医院导诊指南大全",
            "scene": scene_name,
            "department": department,
            "process_type": process_type,
            "raw_text": text.strip(),
            "source_file": "《医院导诊指南大全》.pdf",
            "page_range": f"P{page_num}",
        }

        all_procedures.append(procedure)

    # 保存输出
    output = {
        "hospital": "医院导诊指南大全",
        "version": "V1.0",
        "source": "《医院导诊指南大全》.pdf",
        "total_procedures": len(all_procedures),
        "procedures": all_procedures,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 已生成 {len(all_procedures)} 条流程记录")
    print(f"✓ 保存到: {OUTPUT_FILE}")

    # 显示预览
    print("\n--- 预览 (前3条) ---")
    for proc in all_procedures[:3]:
        print(f"\nID: {proc['id']}")
        print(f"场景: {proc['scene']}")
        print(f"科室: {proc['department']}")
        print(f"类型: {proc['process_type']}")
        print(f"页码: {proc['page_range']}")
        print(f"文本: {proc['raw_text'][:100]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
