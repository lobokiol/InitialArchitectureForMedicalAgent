"""
数据处理工具
针对医疗槽位提取任务的数据加载与预处理
"""

import os
import json
import random
from datasets import Dataset
from typing import Dict, List, Union


def load_medical_data(data_dir: str = "./data") -> Dict[str, List[Dict]]:
    """
    加载医疗槽位提取数据集
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        包含训练、验证、测试集的字典
    """
    # 如果数据文件不存在，则创建示例数据
    os.makedirs(data_dir, exist_ok=True)
    full_data_path = os.path.join(data_dir, "medical_slot_data.json")
    
    if not os.path.exists(full_data_path):
        print("⚠️ 未找到数据文件，创建示例数据...")
        # 创建示例数据
        sample_data = []
        symptoms = ["头痛", "咳嗽", "发热", "胸闷", "腹痛", "恶心", "头晕", "关节痛"]
        departments = ["内科", "外科", "儿科", "妇科", "眼科", "耳鼻喉科", "皮肤科", "神经科"]
        durations = ["一天", "两天", "三天", "一周", "两周", "一个月"]
        
        for i in range(100):
            symptom = random.choice(symptoms)
            department = random.choice(departments)
            duration = random.choice(durations)
            
            # 生成训练样本
            text = f"我{duration}{symptom}了，应该挂{department}"
            sample_data.append({
                "text": text,
                "slots": {
                    "symptom": symptom,
                    "department": department,
                    "duration": duration
                }
            })
        
        # 再添加一些变体
        for i in range(100):
            symptom = random.choice(symptoms)
            department = random.choice(departments)
            duration = random.choice(durations)
            
            # 生成不同格式的训练样本
            templates = [
                f"患者{duration}前开始出现{symptom}症状，建议就诊科室：{department}",
                f"{duration}的{symptom}，去{department}看看",
                f"我{symptom}得厉害，{duration}了，应该去{department}",
                f"{duration}以来一直{symptom}，需要挂{department}",
                f"{symptom}伴{duration}，{department}专科治疗"
            ]
            
            text = random.choice(templates)
            sample_data.append({
                "text": text,
                "slots": {
                    "symptom": symptom,
                    "department": department,
                    "duration": duration
                }
            })
        
        # 按 70/15/15 分割数据
        random.shuffle(sample_data)
        train_size = int(0.7 * len(sample_data))
        val_size = int(0.15 * len(sample_data))
        
        train_data = sample_data[:train_size]
        val_data = sample_data[train_size:train_size + val_size]
        test_data = sample_data[train_size + val_size:]
        
        # 保存数据
        with open(full_data_path, 'w', encoding='utf-8') as f:
            json.dump({
                "train": train_data,
                "validation": val_data,
                "test": test_data
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 示例数据已创建: {full_data_path}")
        print(f"  训练集: {len(train_data)} 条")
        print(f"  验证集: {len(val_data)} 条")
        print(f"  测试集: {len(test_data)} 条")
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
    
    # 加载已有的数据
    with open(full_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    return {
        "train": raw_data["train"],
        "val": raw_data["validation"],
        "test": raw_data["test"]
    }


def prepare_training_samples(dataset: List[Dict], tokenizer) -> Dataset:
    """
    准备训练样本，将原始数据转换为模型可训练的格式
    
    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        
    Returns:
        可用于训练的 HuggingFace Dataset
    """
    formatted_texts = []
    
    for item in dataset:
        # 格式化输入文本，用于槽位提取任务
        text = item["text"]
        slots = item["slots"]
        
        # 构造指令微调格式的数据
        instruction = "请从下面的句子中提取症状、科室、持续时间等医疗槽位信息。"
        formatted_input = f"### 指令:\n{instruction}\n\n### 输入:\n{text}\n\n### 输出:\n"
        
        # 构造期望的输出
        slot_info = []
        for slot_name, slot_value in slots.items():
            slot_info.append(f"{slot_name}: {slot_value}")
        output = ", ".join(slot_info)
        
        # 完整的训练样本
        full_text = formatted_input + output + tokenizer.eos_token
        
        formatted_texts.append({
            "text": full_text,
            "input_text": formatted_input
        })
    
    # 创建 HuggingFace Dataset
    hf_dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_texts]})
    
    # 对文本进行编码
    def tokenize_function(examples):
        # 直接对文本进行分词，并启用填充
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",  # 启用填充以确保所有样本长度一致
            max_length=512,
            return_tensors=None  # 不在这里返回 tensors
        )
        
        # 设置 labels 为 input_ids（对于因果语言模型）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 应用分词函数
    tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset