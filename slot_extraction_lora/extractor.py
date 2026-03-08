"""
槽位提取器核心实现
基于 LLM + LoRA 的医疗领域槽位填充
"""

from typing import Dict, Any, Optional, List
import json


class SlotExtractor:
    """
    医疗导诊槽位提取器
    
    支持提取以下槽位：
    - symptom: 症状描述
    - department: 推荐科室
    - duration: 症状持续时间
    - severity: 严重程度
    - location: 症状部位
    - trigger: 诱发因素
    - accompanying_symptoms: 伴随症状
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化槽位提取器
        
        Args:
            model_path: LoRA 模型权重路径
            config: 配置参数
        """
        self.model_path = model_path
        self.config = config or self._default_config()
        self.model = None
        self._initialized = False
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            "model_name": "qwen-turbo",  # 基础模型
            "lora_r": 8,  # LoRA rank
            "lora_alpha": 32,  # LoRA alpha
            "target_modules": ["q_proj", "v_proj"],  # 目标模块
            "max_length": 512,  # 最大序列长度
            "temperature": 0.1,  # 生成温度
            "medical_slots": [
                "symptom",
                "department", 
                "duration",
                "severity",
                "location",
                "trigger",
                "accompanying_symptoms"
            ]
        }
    
    def load_model(self):
        """加载 LoRA 模型"""
        if self._initialized:
            return
        
        # TODO: 实现 LoRA 模型加载逻辑
        # 示例伪代码：
        # from peft import PeftModel
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # 
        # base_model = AutoModelForCausalLM.from_pretrained(self.config["model_name"])
        # self.model = PeftModel.from_pretrained(base_model, self.model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        
        self._initialized = True
        print(f"模型加载完成：{self.model_path or '使用默认配置'}")
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        提取医疗槽位
        
        Args:
            text: 用户输入文本
            
        Returns:
            槽位字典
        """
        if not self._initialized:
            self.load_model()
        
        # TODO: 实现实际的槽位提取逻辑
        # 当前为示例实现
        
        prompt = self._build_prompt(text)
        
        # 调用 LLM 进行槽位提取
        slots = self._predict_slots(prompt)
        
        return slots
    
    def _build_prompt(self, text: str) -> str:
        """构建提示词"""
        return f"""
作为医疗导诊助手，请从以下患者描述中提取关键信息：

患者描述：{text}

请提取以下槽位信息（JSON 格式）：
{{
    "symptom": "主要症状",
    "department": "推荐科室",
    "duration": "持续时间",
    "severity": "严重程度 (轻/中/重)",
    "location": "症状部位",
    "trigger": "诱发因素",
    "accompanying_symptoms": ["伴随症状列表"]
}}
""".strip()
    
    def _predict_slots(self, prompt: str) -> Dict[str, Any]:
        """预测槽位"""
        # TODO: 实现模型推理
        # 临时返回示例数据
        return {
            "symptom": "头痛",
            "department": "神经内科",
            "duration": "2 天",
            "severity": "中度",
            "location": "头部",
            "trigger": "劳累",
            "accompanying_symptoms": ["恶心", "乏力"]
        }
    
    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量提取槽位"""
        return [self.extract(text) for text in texts]
    
    def validate_slots(self, slots: Dict[str, Any]) -> bool:
        """验证槽位完整性"""
        required_slots = ["symptom"]
        return all(slot in slots for slot in required_slots)
    
    def format_slots(self, slots: Dict[str, Any]) -> str:
        """格式化槽位输出"""
        formatted = []
        for key, value in slots.items():
            if isinstance(value, list):
                formatted.append(f"{key}: {', '.join(value)}")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)


# 使用示例
if __name__ == "__main__":
    extractor = SlotExtractor()
    
    # 单条提取
    text = "我头痛两天了，有点恶心"
    slots = extractor.extract(text)
    print(extractor.format_slots(slots))
    
    # 批量提取
    texts = [
        "发烧三天，喉咙痛",
        "肚子疼，拉肚子"
    ]
    results = extractor.extract_batch(texts)
    for result in results:
        print(result)
