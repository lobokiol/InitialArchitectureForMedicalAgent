"""
医疗槽位数据集定义
"""

from typing import List, Dict


# 医疗导诊槽位定义
MEDICAL_SLOTS = {
    "symptom": {
        "description": "主要症状",
        "type": "string",
        "required": True,
        "examples": ["头痛", "发烧", "咳嗽", "腹痛"]
    },
    "department": {
        "description": "推荐就诊科室",
        "type": "string",
        "required": False,
        "examples": ["内科", "外科", "儿科", "妇产科"]
    },
    "duration": {
        "description": "症状持续时间",
        "type": "string",
        "required": False,
        "examples": ["1 天", "一周", "一个月"]
    },
    "severity": {
        "description": "严重程度",
        "type": "string",
        "required": False,
        "enum": ["轻度", "中度", "重度"],
        "examples": ["轻度", "中度", "重度"]
    },
    "location": {
        "description": "症状部位",
        "type": "string",
        "required": False,
        "examples": ["头部", "腹部", "胸部", "背部"]
    },
    "trigger": {
        "description": "诱发因素",
        "type": "string",
        "required": False,
        "examples": ["劳累", "受凉", "饮食不当"]
    },
    "accompanying_symptoms": {
        "description": "伴随症状",
        "type": "array",
        "required": False,
        "examples": ["恶心", "呕吐", "乏力", "头晕"]
    }
}


def get_slot_schema() -> Dict:
    """获取槽位 schema"""
    return MEDICAL_SLOTS


def get_required_slots() -> List[str]:
    """获取必需槽位列表"""
    return [slot for slot, info in MEDICAL_SLOTS.items() if info.get("required")]


def validate_slot_value(slot_name: str, value) -> bool:
    """验证槽位值是否合法"""
    if slot_name not in MEDICAL_SLOTS:
        return False
    
    slot_info = MEDICAL_SLOTS[slot_name]
    
    # 检查枚举值
    if "enum" in slot_info:
        return value in slot_info["enum"]
    
    return True


# 示例训练数据
TRAINING_EXAMPLES = [
    {
        "text": "我头痛两天了，有点恶心",
        "slots": {
            "symptom": "头痛",
            "duration": "两天",
            "severity": "中度",
            "location": "头部",
            "accompanying_symptoms": ["恶心"]
        }
    },
    {
        "text": "发烧三天，体温 38.5 度，喉咙痛",
        "slots": {
            "symptom": "发烧",
            "duration": "三天",
            "severity": "中度",
            "location": "全身",
            "accompanying_symptoms": ["喉咙痛"]
        }
    },
    {
        "text": "肚子疼，拉肚子，可能是吃坏东西了",
        "slots": {
            "symptom": "腹痛",
            "location": "腹部",
            "trigger": "饮食不当",
            "accompanying_symptoms": ["腹泻"]
        }
    }
]


if __name__ == "__main__":
    print("医疗槽位定义:")
    for slot, info in MEDICAL_SLOTS.items():
        print(f"- {slot}: {info['description']} ({'必需' if info.get('required') else '可选'})")
    
    print("\n示例数据:")
    for example in TRAINING_EXAMPLES[:2]:
        print(f"\n输入：{example['text']}")
        print(f"槽位：{example['slots']}")
