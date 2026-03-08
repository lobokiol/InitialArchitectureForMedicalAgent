#!/usr/bin/env python3
"""
一键生成 500 条医疗槽位标注数据（非交互版本）
流程：
1. 使用 Qwen3-Max API 生成高质量数据
2. 自动验证和划分数据集
3. 生成统计报告
"""

import os
import sys
import json
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from generate_qwen_data import QwenDataGenerator
from prepare_data import DataPreparator, generate_synthetic_data


def main():
    """主执行流程"""
    print("\n" + "=" * 70)
    print(" " * 20 + "医疗槽位数据生成工具")
    print("=" * 70)
    
    # 第 1 步：检查环境
    print("\n【步骤 1/5】检查环境配置...")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("\n将使用模板合成方式生成数据...")
        use_api = False
    else:
        print(f"✓ API Key 已配置")
        use_api = True
    
    # 第 2 步：选择生成方式
    print("\n【步骤 2/5】选择数据生成方式...")
    
    if use_api:
        # 使用 API 生成
        print("  → 使用 Qwen3-Max API 生成（推荐，质量高）\n")
        
        print("【步骤 3/5】调用 Qwen3-Max API 生成数据...")
        generator = QwenDataGenerator(model="qwen-max")
        
        categories = [
            "common",      # 常见症状 - 200 条
            "emergency",   # 急诊症状 - 100 条
            "chronic",     # 慢性症状 - 100 条
            "pediatric",   # 儿科症状 - 50 条
            "geriatric"    # 老年症状 - 50 条
        ]
        
        output_file = "data/raw/qwen_generated.json"
        Path(output_file).parent.mkdir(exist_ok=True)
        
        try:
            all_data = generator.generate_batch(
                n_samples=500,
                categories=categories,
                output_file=output_file,
                save_interval=50
            )
            
            if len(all_data) == 0:
                print("⚠️  API 调用失败，切换到模板合成方式")
                use_api = False
            else:
                print(f"\n✓ 成功生成 {len(all_data)} 条数据")
        except Exception as e:
            print(f"⚠️  API 调用异常：{e}")
            print("切换到模板合成方式")
            use_api = False
    
    if not use_api:
        # 使用模板合成
        print("  → 使用模板生成合成数据（快速，无需 API）\n")
        print("【步骤 3/5】使用模板生成合成数据...")
        all_data = generate_synthetic_data(500)
        print(f"✓ 成功生成 {len(all_data)} 条合成数据")
        
        # 保存原始数据
        raw_dir = Path("data/raw")
        raw_dir.mkdir(exist_ok=True)
        with open(raw_dir / "synthetic.json", 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 原始数据已保存到 data/raw/synthetic.json")
    
    # 第 4 步：数据验证与处理
    print("\n【步骤 4/5】数据验证与处理...")
    preparator = DataPreparator()
    
    # 验证数据质量
    is_valid, errors = preparator.validate_data(all_data)
    if is_valid:
        print("✓ 数据质量验证通过")
    else:
        print(f"⚠  发现 {len(errors)} 个质量问题:")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    
    # 生成统计信息
    stats = preparator.generate_statistics(all_data)
    print(f"\n数据统计:")
    print(f"  • 总样本数：{stats['total_samples']}")
    print(f"  • 平均文本长度：{stats['avg_text_length']:.1f} 字")
    print(f"  • 平均每样本槽位数：{stats['avg_slots_per_sample']:.2f}")
    
    # 第 5 步：划分数据集
    print("\n【步骤 5/5】划分训练集、验证集、测试集...")
    train_data, val_data, test_data = preparator.split_data(
        all_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 保存划分后的数据
    preparator.save_splits(train_data, val_data, test_data)
    
    # 保存统计信息
    metadata_dir = Path("data/metadata")
    metadata_dir.mkdir(exist_ok=True)
    preparator.save_data(stats, metadata_dir / "statistics.json")
    print(f"✓ 统计信息已保存到 data/metadata/statistics.json")
    
    # 打印最终报告
    print("\n" + "=" * 70)
    print(" " * 25 + "生成完成报告")
    print("=" * 70)
    print(f"\n✅ 成功生成 {len(all_data)} 条医疗槽位标注数据")
    print(f"\n数据集划分:")
    print(f"  📊 训练集：{len(train_data)} 条 (70%)")
    print(f"  🔍 验证集：{len(val_data)} 条 (15%)")
    print(f"  🧪 测试集：{len(test_data)} 条 (15%)")
    
    print(f"\n文件位置:")
    print(f"  📁 原始数据：data/raw/")
    print(f"  📁 处理后数据：data/processed/")
    print(f"  📁 统计信息：data/metadata/")
    
    print(f"\n下一步操作:")
    print(f"  1. 检查 data/processed/train.json 前 10 条数据")
    print(f"  2. 如有需要，手动修正标注错误")
    print(f"  3. 开始 LoRA 模型微调训练")
    
    print("\n" + "=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
