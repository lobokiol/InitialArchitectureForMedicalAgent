# 长文本病例矫正 Pipeline

## 一、场景分析

### 1.1 长文本病例特点

```
来源: 医院 HIS 系统导出
格式: 非结构化文本
长度: 2000-10000 字
时间跨度: 数年甚至数十年
编写者: 多位医生、多个科室
```

### 1.2 核心问题

| 问题 | 说明 | 示例 |
|------|------|------|
| **时间线混乱** | 多个手术散落在不同段落 | 2018年手术写在2020年记录之后 |
| **术语不统一** | 不同医生用词不同 | "阑尾炎" vs "急性阑尾炎" vs "急性单纯性阑尾炎" |
| **信息矛盾** | 前后描述不一致 | 一处写"无糖尿病史"，另一处写"2型糖尿病10年" |
| **冗余重复** | 多次记录相同信息 | 同一手术在不同医生记录中重复出现 |
| **关键信息淹没** | 重要病史被长文本淹没 | 药物过敏史混在大量文本中 |

### 1.3 解决思路

**分而治之 + 标准化 + 冲突仲裁**

1. 将长文本切分为小块
2. 逐块提取结构化信息
3. 统一医学术语
4. 检测并解决矛盾
5. 生成结构化摘要

---

## 二、Pipeline 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                    长文本病例矫正 Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

原始长文本病例
    ↓
【步骤1：分块切分】(Chunking)
├── 按时间/段落/医生切分
├── 每块 500-1000 token
└── 保留元数据（时间、医生、章节）
    ↓
【步骤2：信息提取】(Extraction)
├── 7B LLM 提取每块的关键信息
├── 输出结构化：手术/病史/用药/诊断
└── 保留原文引用（可追溯）
    ↓
【步骤3：时间线重建】(Timeline Reconstruction)
├── 按时间排序所有事件
├── 合并重复信息
└── 检测时间矛盾
    ↓
【步骤4：术语标准化】(Terminology Normalization)
├── 医学术语 → ICD-10 标准编码
├── 手术名称 → 标准手术术语
└── 药品名称 → 通用名
    ↓
【步骤5：冲突检测与解决】(Conflict Resolution)
├── 检测矛盾信息（如前后诊断不一致）
├── 72B LLM 仲裁（取最新/最权威）
└── 保留冲突标记供人工审核
    ↓
【步骤6：知识图谱校验】(KG Validation)
├── 验证医学术语是否在 KG 中
├── 补充缺失关联
└── 标记不确定信息
    ↓
【步骤7：摘要生成】(Summarization)
├── 72B LLM 生成结构化摘要
├── 按时间线组织
└── 输出标准病例格式
    ↓
【步骤8：人工审核】(Human Review)
├── 标记不确定/冲突信息
├── 医生确认关键信息
└── 最终归档
```

---

## 三、各步骤详细设计

### 3.1 步骤1：分块切分

```python
def chunk_medical_record(text: str) -> List[Chunk]:
    """
    按时间/医生/章节切分长文本
    """
    # 策略1: 按时间标记切分
    time_pattern = r"(\d{4}年\d{1,2}月\d{1,2}日?)"
    
    # 策略2: 按医生签名切分
    doctor_pattern = r"(主治医师[:：]\s*\w+|医师签名[:：]\s*\w+)"
    
    # 策略3: 按章节标题切分
    section_pattern = r"(病史|手术记录|用药记录|诊断意见)"
    
    # 返回带元数据的块
    return chunks
```

**输出示例：**
```json
[
  {
    "chunk_id": "C001",
    "text": "2018年5月，患者因右上腹疼痛入住普外科...",
    "metadata": {
      "date": "2018-05",
      "doctor": "张医生",
      "section": "手术记录"
    }
  }
]
```

### 3.2 步骤2：信息提取

```python
def extract_from_chunk(chunk: Chunk) -> MedicalEvent:
    """
    7B LLM 提取结构化信息
    """
    prompt = f"""
    从以下病例片段提取信息：
    {chunk.text}
    
    输出格式:
    {{
        "date": "2024-03-15",
        "event_type": "手术/诊断/用药/检查",
        "description": "胆囊切除术",
        "doctor": "张医生",
        "raw_text": "原文引用"
    }}
    """
```

### 3.3 步骤3：时间线重建

```python
def reconstruct_timeline(events: List[MedicalEvent]) -> Timeline:
    """
    按时间排序 + 合并重复 + 检测矛盾
    """
    # 1. 按时间排序
    sorted_events = sorted(events, key=lambda x: x.date)
    
    # 2. 合并重复（相同日期+相同事件类型）
    merged = merge_duplicates(sorted_events)
    
    # 3. 检测矛盾
    conflicts = detect_conflicts(merged)
    
    return Timeline(events=merged, conflicts=conflicts)
```

### 3.4 步骤4：术语标准化

```python
def normalize_terminology(event: MedicalEvent) -> MedicalEvent:
    """
    医学术语 → 标准术语
    """
    # 手术名称标准化
    event.procedure = neo4j_convert_procedure(event.procedure)
    
    # 诊断标准化 (ICD-10)
    event.diagnosis = neo4j_convert_diagnosis(event.diagnosis)
    
    # 药品通用名
    event.medication = normalize_drug_name(event.medication)
    
    return event
```

### 3.5 步骤5：冲突检测与解决

```python
def resolve_conflicts(timeline: Timeline) -> Timeline:
    """
    72B LLM 仲裁矛盾信息
    """
    for conflict in timeline.conflicts:
        prompt = f"""
        以下信息存在矛盾:
        版本A: {conflict.version_a}
        版本B: {conflict.version_b}
        
        请判断哪个更可信，并说明理由:
        - 时间更新者优先
        - 专科医生记录优先
        - 有检查报告支持的优先
        """
        resolution = llm_72b.resolve(prompt)
        conflict.resolved = resolution
```

### 3.6 步骤6：知识图谱校验

```python
def validate_with_kg(timeline: Timeline) -> Timeline:
    """
    Neo4j KG 校验
    """
    for event in timeline.events:
        # 验证诊断是否在 KG 中
        if not kg_exists("Diagnosis", event.diagnosis):
            event.confidence = 0.5  # 标记不确定
        
        # 验证手术是否在 KG 中
        if not kg_exists("Procedure", event.procedure):
            event.confidence = 0.5
    
    return timeline
```

### 3.7 步骤7：摘要生成

```python
def generate_summary(timeline: Timeline) -> str:
    """
    72B LLM 生成结构化摘要
    """
    prompt = f"""
    根据以下时间线生成结构化病例摘要：
    {timeline.to_json()}
    
    要求:
    1. 按时间顺序组织
    2. 突出关键信息（手术、过敏史、慢性病）
    3. 使用标准医学术语
    4. 格式清晰，便于医生快速阅读
    """
    return llm_72b.generate(prompt)
```

---

## 四、模型使用策略

| 步骤 | 模型 | 原因 |
|------|------|------|
| 分块切分 | 规则 | 不需要 LLM |
| 信息提取 | 7B | 简单提取任务 |
| 时间线重建 | 规则 | 排序/合并逻辑 |
| 术语标准化 | Neo4j + 7B | 向量匹配 + 兜底 |
| 冲突解决 | **72B** | 需要复杂推理判断 |
| 摘要生成 | **72B** | 需要高质量文本生成 |
| KG 校验 | Neo4j | 不需要 LLM |

---

## 五、输出格式

```json
{
  "patient_id": "P001",
  "timeline": [
    {
      "date": "2018-05-15",
      "event_type": "手术",
      "procedure": "胆囊切除术",
      "doctor": "张医生",
      "hospital": "XX医院",
      "raw_reference": "原文第3段"
    },
    {
      "date": "2020-08-20",
      "event_type": "手术",
      "procedure": "阑尾切除术",
      "doctor": "李医生",
      "raw_reference": "原文第7段"
    }
  ],
  "conflicts": [
    {
      "field": "诊断",
      "version_a": "2型糖尿病 (2022年)",
      "version_b": "糖尿病前期 (2023年)",
      "resolution": "采用2022年诊断，2023年可能为笔误",
      "confidence": 0.85
    }
  ],
  "summary": "患者有2次手术史...",
  "metadata": {
    "source_doctors": ["张医生", "李医生", "王医生"],
    "date_range": "2018-2024",
    "processing_time": "12s"
  }
}
```

---

## 六、技术实现要点

### 6.1 分块策略

| 策略 | 适用场景 | 优先级 |
|------|----------|--------|
| 按时间切分 | 有时间标记的病例 | 高 |
| 按医生切分 | 多医生协作记录 | 中 |
| 按章节切分 | 结构化病历 | 中 |
| 固定长度切分 | 无结构长文本 | 低 |

### 6.2 冲突解决规则

| 规则 | 说明 | 优先级 |
|------|------|--------|
| 时间更新者优先 | 新记录覆盖旧记录 | 1 |
| 专科医生优先 | 专科记录优先于全科 | 2 |
| 有检查报告支持 | 有客观证据的记录优先 | 3 |
| 72B LLM 仲裁 | 以上规则无法解决时 | 4 |

### 6.3 性能优化

| 优化点 | 方法 |
|--------|------|
| 并行提取 | 多块同时提取（ThreadPoolExecutor） |
| 缓存术语映射 | Redis 缓存术语标准化结果 |
| 增量处理 | 只处理新增/修改部分 |
| 异步生成 | 摘要生成异步执行 |

---

## 七、应用场景

| 场景 | 说明 |
|------|------|
| **转院病历整理** | 患者转院时整理多年病史 |
| **多学科会诊** | 多科室医生快速了解病史 |
| **科研数据清洗** | 从非结构化病历提取结构化数据 |
| **保险理赔** | 快速提取关键医疗信息 |

---

**文档版本**: 1.0  
**创建时间**: 2026年  
**适用项目**: 医疗数据处理
