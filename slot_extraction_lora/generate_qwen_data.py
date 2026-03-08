"""
使用 Qwen3-Max API 生成高质量医疗槽位标注数据
通过 LLM 自动生成多样化的患者主诉和对应的槽位标注
"""

import json
import os
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai 库：pip install openai")
    exit(1)


class QwenDataGenerator:
    """Qwen API 数据生成器"""
    
    def __init__(self, api_key: str = None, model: str = "qwen-max"):
        """
        初始化 API 客户端
        
        Args:
            api_key: DashScope API Key，默认从环境变量读取
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或在代码中传入 API Key")
        
        # 配置阿里云 DashScope（兼容 OpenAI API）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        
    def generate_sample(self, category: str = "common") -> Dict:
        """
        生成单个样本
        
        Args:
            category: 数据类别 (common, emergency, chronic, pediatric, etc.)
            
        Returns:
            包含 text 和 slots 的字典
        """
        prompt = self._build_prompt(category)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗数据标注助手，擅长生成真实的患者主诉并提取准确的槽位信息。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                timeout=60
            )
            
            content = response.choices[0].message.content.strip()
            return self._parse_response(content)
            
        except Exception as e:
            print(f"API 调用失败：{e}")
            return None
    
    def _build_prompt(self, category: str) -> str:
        """构建提示词"""
        
        categories_info = {
            "common": "常见症状（发热、头痛、咳嗽、腹痛等）",
            "emergency": "急诊症状（胸痛、呼吸困难、剧烈疼痛等）",
            "chronic": "慢性症状（长期疼痛、乏力、消瘦等）",
            "pediatric": "儿科症状（儿童发热、哭闹、食欲不振等）",
            "geriatric": "老年症状（头晕、记忆力下降、关节疼痛等）"
        }
        
        target_category = categories_info.get(category, "常见症状")
        
        prompt = f"""请生成一个医疗问诊场景的患者主诉，要求：

1. 症状类型：{target_category}
2. 描述自然真实，符合患者口语表达习惯
3. 长度在 10-50 字之间
4. 包含以下槽位信息（JSON 格式）：

必需槽位：
- symptom: 主要症状（字符串）

可选槽位：
- department: 推荐科室（字符串）
- duration: 持续时间（字符串，如"两天"、"一周"）
- severity: 严重程度（"轻度"、"中度"、"重度"）
- location: 症状部位（字符串）
- trigger: 诱发因素（字符串）
- accompanying_symptoms: 伴随症状（数组）

输出格式示例：
{{
  "text": "我头痛两天了，有点恶心",
  "slots": {{
    "symptom": "头痛",
    "duration": "两天",
    "severity": "中度",
    "location": "头部",
    "accompanying_symptoms": ["恶心"]
  }}
}}

请直接返回 JSON 格式，不要有其他说明文字。"""
        
        return prompt
    
    def _parse_response(self, content: str) -> Dict:
        """解析 API 响应"""
        try:
            # 尝试直接解析 JSON
            data = json.loads(content)
            if "text" in data and "slots" in data:
                return data
        except:
            pass
        
        # 尝试提取 JSON 部分
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            try:
                json_str = content[start:end]
                data = json.loads(json_str)
                if "text" in data and "slots" in data:
                    return data
            except:
                pass
        
        print(f"无法解析响应：{content[:100]}...")
        return None
    
    def generate_batch(
        self, 
        n_samples: int = 100, 
        categories: List[str] = None,
        output_file: str = None,
        save_interval: int = 50
    ) -> List[Dict]:
        """
        批量生成数据
        
        Args:
            n_samples: 总样本数量
            categories: 类别列表，默认使用常见症状
            output_file: 输出文件路径（可选，用于保存中间结果）
            save_interval: 保存间隔
            
        Returns:
            生成的数据列表
        """
        if categories is None:
            categories = ["common"]
        
        samples_per_category = n_samples // len(categories)
        all_samples = []
        
        print(f"\n开始生成 {n_samples} 条数据...")
        print(f"类别分布：{categories}")
        print(f"每类约 {samples_per_category} 条\n")
        
        for i, category in enumerate(categories):
            print(f"[{i+1}/{len(categories)}] 生成类别：{category}")
            n_current = samples_per_category if i < len(categories) - 1 else (n_samples - len(all_samples))
            
            for j in range(n_current):
                sample = self.generate_sample(category)
                
                if sample:
                    # 添加元数据
                    sample["id"] = f"qwen_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(all_samples)}"
                    sample["metadata"] = {
                        "source": "qwen_api",
                        "model": self.model,
                        "category": category,
                        "annotator": "qwen-max",
                        "timestamp": datetime.now().isoformat()
                    }
                    all_samples.append(sample)
                    
                    # 进度显示
                    if (j + 1) % 10 == 0:
                        print(f"  已生成：{j + 1}/{n_current} 条")
                    
                    # 速率限制：避免请求过快
                    time.sleep(0.5)
                
                # 定期保存
                if output_file and (j + 1) % save_interval == 0:
                    self._save_progress(all_samples, output_file)
                    print(f"  已保存 {len(all_samples)} 条数据到 {output_file}")
        
        # 最终保存
        if output_file:
            self._save_progress(all_samples, output_file)
            print(f"\n所有数据已保存到：{output_file}")
        
        print(f"\n生成完成！共 {len(all_samples)} 条数据")
        return all_samples
    
    def _save_progress(self, data: List[Dict], filepath: str):
        """保存进度"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """主函数：生成 500 条标注数据"""
    print("=" * 60)
    print("使用 Qwen3-Max API 生成医疗槽位标注数据")
    print("=" * 60)
    
    # 检查 API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n错误：未找到 DASHSCOPE_API_KEY 环境变量")
        print("请设置环境变量或创建 .env 文件：")
        print("  export DASHSCOPE_API_KEY=your_api_key_here")
        return
    
    # 初始化生成器
    generator = QwenDataGenerator(model="qwen-max")
    
    # 定义数据类别（确保多样性）
    categories = [
        "common",      # 常见症状 - 200 条
        "emergency",   # 急诊症状 - 100 条
        "chronic",     # 慢性症状 - 100 条
        "pediatric",   # 儿科症状 - 50 条
        "geriatric"    # 老年症状 - 50 条
    ]
    
    # 生成数据
    output_file = "data/raw/qwen_generated.json"
    Path(output_file).parent.mkdir(exist_ok=True)
    
    all_data = generator.generate_batch(
        n_samples=500,
        categories=categories,
        output_file=output_file,
        save_interval=50
    )
    
    # 验证数据质量
    print("\n" + "=" * 60)
    print("数据质量验证")
    print("=" * 60)
    
    preparator = DataPreparator()
    is_valid, errors = preparator.validate_data(all_data)
    
    if is_valid:
        print("✓ 所有数据验证通过")
    else:
        print(f"✗ 发现 {len(errors)} 个错误:")
        for err in errors[:10]:
            print(f"  - {err}")
    
    # 生成统计信息
    stats = preparator.generate_statistics(all_data)
    print(f"\n数据统计:")
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  平均文本长度：{stats['avg_text_length']:.1f} 字")
    print(f"  平均每样本槽位数：{stats['avg_slots_per_sample']:.2f}")
    print(f"\n槽位分布:")
    for slot, count in sorted(stats['slot_statistics'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  - {slot}: {count} ({count/stats['total_samples']*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
