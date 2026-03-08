"""
槽位提取模块测试
"""

import sys
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from slot_extraction_lora.extractor import SlotExtractor
from slot_extraction_lora.config import LoRAConfig
from slot_extraction_lora.dataset import MEDICAL_SLOTS, TRAINING_EXAMPLES


def test_extractor():
    """测试槽位提取器"""
    print("=" * 50)
    print("测试槽位提取器")
    print("=" * 50)
    
    extractor = SlotExtractor()
    
    # 测试用例
    test_cases = [
        "头痛两天了，伴有恶心",
        "发烧三天，体温 38.5 度",
        "肚子疼，可能是吃坏东西了"
    ]
    
    for text in test_cases:
        print(f"\n输入：{text}")
        slots = extractor.extract(text)
        print("提取结果:")
        print(extractor.format_slots(slots))
        print(f"验证通过：{extractor.validate_slots(slots)}")


def test_config():
    """测试配置"""
    print("\n" + "=" * 50)
    print("测试 LoRA 配置")
    print("=" * 50)
    
    config = LoRAConfig()
    print(f"\nLoRA 配置:")
    for key, value in config.to_dict().items():
        print(f"- {key}: {value}")


def test_dataset():
    """测试数据集"""
    print("\n" + "=" * 50)
    print("测试医疗槽位定义")
    print("=" * 50)
    
    print("\n槽位列表:")
    for slot, info in MEDICAL_SLOTS.items():
        required = "必需" if info.get("required") else "可选"
        print(f"- {slot} ({required}): {info['description']}")
    
    print("\n训练示例:")
    for i, example in enumerate(TRAINING_EXAMPLES[:2], 1):
        print(f"\n示例 {i}:")
        print(f"  文本：{example['text']}")
        print(f"  槽位：{example['slots']}")


if __name__ == "__main__":
    test_extractor()
    test_config()
    test_dataset()
    
    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
