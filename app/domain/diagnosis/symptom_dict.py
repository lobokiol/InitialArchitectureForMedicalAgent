"""
症状词典 - Layer 1: 基于 keywords 的快速匹配

用法:
    from app.domain.diagnosis.symptom_dict import SymptomDict, get_symptom_dict

    dict = get_symptom_dict()
    results = dict.match("我胃里翻江倒海，还发烧")
    # results: [{"keyword": "胃里翻江倒海", "symptom": "胃痛"}, {"keyword": "发烧", "symptom": "发热"}]
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class SymptomDict:
    """症状词典 - 基于 keywords 的快速匹配"""

    def __init__(self):
        self.symptom_map: Dict[str, str] = {}  # {keyword: symptom_name}
        self.symptom_keywords: Dict[str, List[str]] = {}  # {symptom_name: [keywords]}
        self._load_dict()

    def _load_dict(self):
        """从 symptoms.json 加载词典"""
        dict_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data/knowledge_graph/symptoms.json"
        )

        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data.get("症状列表", []):
                symptom_name = item.get("name", "")
                keywords = item.get("keywords", [])

                # 添加主名称作为关键词
                self.symptom_map[symptom_name] = symptom_name

                # 添加 keywords 映射
                for kw in keywords:
                    # 避免覆盖已有的映射（优先保留主名称）
                    if kw not in self.symptom_map:
                        self.symptom_map[kw] = symptom_name

                # 记录每个症状的关键词
                self.symptom_keywords[symptom_name] = [symptom_name] + keywords

            print(
                f"症状词典加载完成: {len(self.symptom_map)} 个关键词, {len(self.symptom_keywords)} 个症状"
            )

        except Exception as e:
            print(f"加载症状词典失败: {e}")
            self.symptom_map = {}
            self.symptom_keywords = {}

    def match(self, text: str) -> List[Dict[str, Any]]:
        """
        匹配文本中的症状关键词

        Args:
            text: 用户输入文本

        Returns:
            匹配结果列表: [{"keyword": "匹配到的词", "symptom": "症状名"}, ...]
        """
        results = []
        text_lower = text.lower()

        # 按关键词长度降序排序（优先匹配长词）
        sorted_keywords = sorted(self.symptom_map.keys(), key=len, reverse=True)

        for keyword in sorted_keywords:
            # 使用正则进行灵活匹配
            pattern = self._create_pattern(keyword)
            match = re.search(pattern, text_lower)

            if match:
                matched_text = match.group()
                symptom = self.symptom_map[keyword]

                # 避免重复添加同一症状
                if not any(r["symptom"] == symptom for r in results):
                    results.append(
                        {"keyword": matched_text, "symptom": symptom, "source": "dict"}
                    )

        return results

    def _create_pattern(self, keyword: str) -> re.Pattern:
        """创建匹配模式"""
        # 转义特殊字符
        escaped = re.escape(keyword)
        # 允许一定的灵活性
        pattern = f"{escaped}"
        return re.compile(pattern)

    def get_symptom_by_keyword(self, keyword: str) -> Optional[str]:
        """根据关键词获取症状名"""
        return self.symptom_map.get(keyword)

    def get_all_symptoms(self) -> List[str]:
        """获取所有症状列表"""
        return list(self.symptom_keywords.keys())

    def find_similar_symptoms(self, symptom: str, limit: int = 5) -> List[str]:
        """查找相似的症状（通过共享关键词）"""
        if symptom not in self.symptom_keywords:
            return []

        target_keywords = set(self.symptom_keywords[symptom])
        similar = []

        for other_symptom, keywords in self.symptom_keywords.items():
            if other_symptom == symptom:
                continue

            # 计算交集
            overlap = len(target_keywords & set(keywords))
            if overlap > 0:
                similar.append((other_symptom, overlap))

        # 按重叠数排序
        similar.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in similar[:limit]]


# 单例
_symptom_dict: Optional[SymptomDict] = None


def get_symptom_dict() -> SymptomDict:
    """获取症状词典单例"""
    global _symptom_dict
    if _symptom_dict is None:
        _symptom_dict = SymptomDict()
    return _symptom_dict


def match_symptoms_from_text(text: str) -> List[str]:
    """
    从文本中匹配症状（便捷函数）

    Args:
        text: 用户输入

    Returns:
        匹配到的症状列表
    """
    dict = get_symptom_dict()
    results = dict.match(text)
    return list(set(r["symptom"] for r in results))
