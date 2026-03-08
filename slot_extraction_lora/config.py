"""
LoRA 配置与训练工具
"""

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class LoRAConfig:
    """LoRA 配置参数"""
    
    # 模型配置
    base_model: str = "qwen-turbo"
    lora_r: int = 8
    lora_alpha: int = 32
    target_modules: List[str] = None
    
    # 训练配置
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 3
    max_length: int = 512
    
    # 医疗领域特定配置
    medical_special_tokens: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj"]
        
        if self.medical_special_tokens is None:
            self.medical_special_tokens = [
                "<症状>", "<科室>", "<持续时间>", 
                "<严重程度>", "<部位>", "<诱发因素>"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "base_model": self.base_model,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "max_length": self.max_length,
            "medical_special_tokens": self.medical_special_tokens
        }


def get_medical_slot_template() -> str:
    """获取医疗槽位提取模板"""
    return """
### 指令
从患者描述中提取医疗问诊槽位信息

### 输入
{input_text}

### 输出
{output_format}

""".strip()


def prepare_training_data(samples: List[Dict]) -> List[Dict]:
    """
    准备训练数据
    
    Args:
        samples: 原始样本列表
        
    Returns:
        格式化后的训练数据
    """
    training_data = []
    
    for sample in samples:
        text = sample.get("text", "")
        slots = sample.get("slots", {})
        
        prompt = get_medical_slot_template().format(
            input_text=text,
            output_format=json.dumps(slots, ensure_ascii=False)
        )
        
        training_data.append({
            "prompt": prompt,
            "completion": json.dumps(slots, ensure_ascii=False)
        })
    
    return training_data


# 使用示例
if __name__ == "__main__":
    config = LoRAConfig()
    print("LoRA 配置:", config.to_dict())
    
    # 示例训练数据
    samples = [
        {
            "text": "头痛两天了，伴有恶心",
            "slots": {
                "symptom": "头痛",
                "duration": "两天",
                "accompanying_symptoms": ["恶心"]
            }
        }
    ]
    
    training_data = prepare_training_data(samples)
    print("\n训练数据示例:")
    print(training_data[0]["prompt"])
