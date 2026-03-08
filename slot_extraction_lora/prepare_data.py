"""
数据准备与处理工具
用于生成、划分、增强训练/验证/测试数据
支持模板合成和 LLM API 生成两种方式
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import os


class DataPreparator:
    """数据准备器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def load_data(self, filepath: str) -> List[Dict]:
        """加载 JSON 格式数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_data(self, data: List[Dict], filepath: str):
        """保存 JSON 格式数据"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def split_data(
        self, 
        data: List[Dict], 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        划分训练集、验证集、测试集
        
        Args:
            data: 原始数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            shuffle: 是否打乱数据
            
        Returns:
            (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        if shuffle:
            random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]
        
        print(f"数据划分完成:")
        print(f"  训练集：{len(train_data)} 条 ({len(train_data)/n*100:.1f}%)")
        print(f"  验证集：{len(val_data)} 条 ({len(val_data)/n*100:.1f}%)")
        print(f"  测试集：{len(test_data)} 条 ({len(test_data)/n*100:.1f}%)")
        
        return train_data, val_data, test_data
    
    def save_splits(
        self, 
        train_data: List[Dict], 
        val_data: List[Dict], 
        test_data: List[Dict]
    ):
        """保存划分后的数据集"""
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        self.save_data(train_data, processed_dir / "train.json")
        self.save_data(val_data, processed_dir / "val.json")
        self.save_data(test_data, processed_dir / "test.json")
        
        print(f"\n数据已保存到 {processed_dir}/")
    
    def generate_statistics(self, data: List[Dict]) -> Dict:
        """生成数据统计信息"""
        stats = {
            "total_samples": len(data),
            "slot_statistics": {},
            "slot_combinations": {},
            "avg_text_length": 0,
            "avg_slots_per_sample": 0
        }
        
        total_slots = 0
        slot_counts = {}
        text_lengths = []
        
        for sample in data:
            text = sample.get("text", "")
            slots = sample.get("slots", {})
            
            # 统计文本长度
            text_lengths.append(len(text))
            
            # 统计槽位出现次数
            total_slots += len(slots)
            for slot_name in slots.keys():
                slot_counts[slot_name] = slot_counts.get(slot_name, 0) + 1
            
            # 统计槽位组合
            slot_combo = tuple(sorted(slots.keys()))
            stats["slot_combinations"][str(slot_combo)] = \
                stats["slot_combinations"].get(str(slot_combo), 0) + 1
        
        stats["slot_statistics"] = slot_counts
        stats["avg_text_length"] = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        stats["avg_slots_per_sample"] = total_slots / len(data) if data else 0
        
        return stats
    
    def validate_data(self, data: List[Dict]) -> Tuple[bool, List[str]]:
        """
        验证数据质量
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        required_slots = ["symptom"]
        
        for i, sample in enumerate(data):
            # 检查必需字段
            if "text" not in sample:
                errors.append(f"样本 {i}: 缺少 text 字段")
            if "slots" not in sample:
                errors.append(f"样本 {i}: 缺少 slots 字段")
                continue
            
            slots = sample["slots"]
            
            # 检查必需槽位
            for slot in required_slots:
                if slot not in slots:
                    errors.append(f"样本 {i}: 缺少必需槽位 '{slot}'")
            
            # 检查槽位值类型
            for slot_name, slot_value in slots.items():
                if slot_name == "accompanying_symptoms":
                    if not isinstance(slot_value, list):
                        errors.append(f"样本 {i}: 槽位 '{slot_name}' 应为列表类型")
                elif not isinstance(slot_value, str):
                    errors.append(f"样本 {i}: 槽位 '{slot_name}' 应为字符串类型")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def create_sample(
        self,
        text: str,
        slots: Dict,
        source: str = "manual",
        annotator: str = "unknown"
    ) -> Dict:
        """创建标准格式的样本"""
        return {
            "id": f"sample_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}",
            "text": text,
            "slots": slots,
            "metadata": {
                "source": source,
                "annotator": annotator,
                "timestamp": datetime.now().isoformat()
            }
        }


def generate_synthetic_data(n_samples: int = 100) -> List[Dict]:
    """
    生成合成数据（使用模板）
    
    Args:
        n_samples: 生成样本数量
        
    Returns:
        合成的数据列表
    """
    # 症状模板库
    symptoms = [
        "头痛", "发烧", "咳嗽", "腹痛", "恶心", "呕吐", 
        "腹泻", "头晕", "乏力", "胸闷", "心悸", "咽痛"
    ]
    
    durations = [
        "一天", "两天", "三天", "一周", "两周", "一个月"
    ]
    
    severities = ["轻度", "中度", "重度"]
    
    locations = [
        "头部", "腹部", "胸部", "背部", "全身", "喉咙"
    ]
    
    triggers = [
        "劳累", "受凉", "饮食不当", "压力大", "熬夜", "感冒"
    ]
    
    accompanying = [
        ["恶心"], ["呕吐"], ["乏力"], ["头晕"], 
        ["恶心", "呕吐"], ["乏力", "头晕"], ["食欲不振"]
    ]
    
    departments = [
        "内科", "外科", "儿科", "神经内科", "消化内科", 
        "呼吸内科", "急诊科"
    ]
    
    samples = []
    
    for _ in range(n_samples):
        # 随机选择槽位值
        symptom = random.choice(symptoms)
        duration = random.choice(durations)
        severity = random.choice(severities)
        location = random.choice(locations)
        trigger = random.choice(triggers)
        accomp = random.choice(accompanying)
        dept = random.choice(departments)
        
        # 构建文本模板
        templates = [
            f"我{symptom}{duration}了，有点{accomp[0] if accomp else '不舒服'}",
            f"{duration}前开始{symptom}，{severity}程度",
            f"{location}{symptom}，可能是{trigger}引起的",
            f"主要症状是{symptom}，持续{duration}，伴有{', '.join(accomp) if accomp else '其他不适'}",
        ]
        
        text = random.choice(templates)
        
        # 构建槽位
        slots = {"symptom": symptom}
        
        if random.random() > 0.3:
            slots["duration"] = duration
        if random.random() > 0.5:
            slots["severity"] = severity
        if random.random() > 0.4:
            slots["location"] = location
        if random.random() > 0.6:
            slots["trigger"] = trigger
        if accomp and random.random() > 0.5:
            slots["accompanying_symptoms"] = accomp
        if random.random() > 0.7:
            slots["department"] = dept
        
        samples.append({
            "text": text,
            "slots": slots
        })
    
    return samples


def main():
    """主函数：演示数据准备流程"""
    print("=" * 60)
    print("医疗槽位提取数据准备工具")
    print("=" * 60)
    
    # 初始化准备器
    preparator = DataPreparator()
    
    # 示例 1: 生成合成数据
    print("\n1. 生成合成数据...")
    synthetic_data = generate_synthetic_data(200)
    print(f"   生成 {len(synthetic_data)} 条合成数据")
    
    # 示例 2: 验证数据
    print("\n2. 验证数据质量...")
    is_valid, errors = preparator.validate_data(synthetic_data)
    if is_valid:
        print("   ✓ 数据验证通过")
    else:
        print(f"   ✗ 发现 {len(errors)} 个错误:")
        for err in errors[:5]:
            print(f"     - {err}")
    
    # 示例 3: 生成统计信息
    print("\n3. 生成统计信息...")
    stats = preparator.generate_statistics(synthetic_data)
    print(f"   总样本数：{stats['total_samples']}")
    print(f"   平均文本长度：{stats['avg_text_length']:.1f} 字")
    print(f"   平均每样本槽位数：{stats['avg_slots_per_sample']:.2f}")
    print(f"   槽位分布:")
    for slot, count in sorted(stats['slot_statistics'].items(), key=lambda x: x[1], reverse=True):
        print(f"     - {slot}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    # 示例 4: 划分数据集
    print("\n4. 划分数据集...")
    train_data, val_data, test_data = preparator.split_data(
        synthetic_data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 保存数据
    print("\n5. 保存数据...")
    preparator.save_splits(train_data, val_data, test_data)
    
    # 保存统计信息
    stats_path = preparator.data_dir / "metadata"
    stats_path.mkdir(exist_ok=True)
    preparator.save_data(stats, stats_path / "statistics.json")
    print(f"   统计信息已保存到 {stats_path}/statistics.json")
    
    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
