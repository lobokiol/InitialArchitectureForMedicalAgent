# 生产级医院导诊 Agentic 助手

基于 FastAPI + LangGraph + Redis + Elasticsearch + Milvus + DashScope 的医院导诊问答与流程指引助手，同时提供命令行前端（rich CLI），可以作为生产级医疗导诊 / 医疗流程问答系统的参考实现。

后端通过 LangGraph 状态机编排多轮对话、症状问诊、流程检索与意图识别，前端则以 CLI 形式演示多会话聊天体验（类似 ChatGPT 的会话列表）。

---
web页面：

<img src="./demo/web页面展示.gif" width="50%" />
后端cli-debug：

<img src="./demo/demo.gif" width="50%" />

## 功能特性

- 医疗导诊对话
  - 支持面向「症状问诊」和「就医流程」的多轮对话。
  - **多轮问诊系统**：通过槽位填充逐步收集患者症状信息，输出结构化问诊表。
  - **语义症状匹配**：基于向量嵌入的语义对齐，将口语化症状描述映射至标准医学术语。
  - **四层症状提取架构**：
    - Layer 0: Neo4j CM3KG 向量语义匹配 (embedding 余弦相似度)
    - Layer 1: 症状词典 (317个keywords快速匹配) ⚠️ 已废弃，改用向量匹配
    - Layer 2: LLM抽取 (Qwen Turbo语义提取)
    - Layer 3: 合并到Slot (整合多源结果)
    - Layer 4: 知识图谱校验 (验证+扩展+消歧)
  - **KG + RAG 融合推理**：融合知识图谱与多路RAG检索的综合科室推荐。
  - **MCP 工具调度**：所有数据源通过 MCP (Model Context Protocol) 统一调度。
  - **危险信号检测**：实时检测胸痛、呼吸困难等危急症状，立即告警建议挂急诊。
  - 结合向量检索与流程文档检索，给出答案和建议。
- 知识图谱增强
  - **Neo4j图数据库**：存储 CM3KG 症状-科室映射、伴随症状、疾病关系
    - 3,108 个症状节点
    - 8,618 个疾病节点
    - 88 个科室节点
    - 32,876 条症状-疾病关系
  - **向量搜索**：症状语义匹配 (text-embedding-v2)
  - **两阶段检索**：向量搜索 + 图推理 (多跳查询)
  - **判别性症状**：动态生成追问问题，帮助区分不同科室
- Agentic 对话编排（LangGraph）
  - 使用 `AppState` 管理对话状态，基于 LangGraph 构建状态机。
  - **多Agent协作**：6个专业化Agent节点（意图识别、槽位填充、语义对齐、风险评估、追问生成、结束判断）。
  - 包含意图识别、RAG 检索、文档评估、Query 重写、答案生成等节点。
- 多会话管理（类似 ChatGPT）
  - 会话列表、创建会话、删除会话、切换当前会话。
  - 会话与用户元数据（名称、创建时间、最近活跃时间）存储在 Redis。
- 检索增强生成（RAG）
  - Elasticsearch：医院流程 / 制度等结构化文档检索（hospital_procedures 索引）。
  - Milvus：症状 / 医疗知识向量检索（medical_knowledge 集合）。
  - **混合检索增强**（milvus_rag 节点）：
    - 双路检索：ES (rag_es) + Milvus (medical_knowledge)
    - RRF (Reciprocal Rank Fusion) 融合排序
    - LLM (qwen3-rerank) Rerank 精排
  - DashScope Embedding + Chat 模型。
- MCP (Model Context Protocol) 工具调度
  - 所有数据源通过 MCP 统一调度
  - **MCP Server**: `app/mcp/patient_server.py`
  - **MCP Tools**:
    - Neo4j: `infer_department`, `semantic_match_symptoms`, `get_possible_diseases`
    - Milvus: `milvus_search`
    - Elasticsearch: `es_search`
    - PostgreSQL: `pg_get_patient_by_name`, `pg_get_patient_history`, `pg_search_patients`
    - 综合推理: `kg_rag_fusion`
- 命令行前端（rich CLI）
  - `cli.py` 提供交互式 CLI，支持斜杠命令和 Markdown 渲染。
  - 通过 REST API 与后端通信，可作为 Web 前端的参考。

---




### 多轮问诊示例

| 轮次 | 用户输入 | 系统回复 | 填入槽位 |
|------|---------|---------|---------|
| 1 | 我肚子疼 → normalize → 我腹痛 | 有没有恶心或腹胀等症状？ | chief_complaint: "我腹痛", symptoms: ["腹痛"], location: "腹部", accompanying_symptoms: ["恶心"] |
| 2 | 疼了3天了 | 疼痛程度如何？0-10分？ | duration |
| 3 | 大概7分疼 | 有没有什么情况下会加重或缓解？ | severity |
| 4 | 吃完饭更疼 | 以前有过类似症状吗？ | triggers |
| 5 | 还发烧，恶心 | 问诊完成，推荐消化内科 | accompaning_symptoms: ["发烧", "恶心"], medical_history |

**输出 JSON 问诊表**：
```json
{
  "chief_complaint": "我肚子疼",
  "symptoms": ["腹痛"],
  "duration": "3天",
  "severity": "6-7",
  "location": "腹部",
  "triggers": ["进食"],
  "accompanying_symptoms": ["发热", "恶心"],
  "medical_history": ["无"],
  "risk_signals": []
}
```
---

## 三、系统架构流程

核心对话工作流采用 **五层架构** 设计，以 MCP 为数据调度中心：

```mermaid
flowchart TB
    subgraph Input_Layer["1. 输入层"]
        A1["用户文字主诉"]
    end
    
    subgraph Perception_Layer["2. 感知层: 语义识别"]
        B1["LLM 症状实体提取"]
        B2["KG 语义对齐: 主诉→标准术语"]
    end
    
    subgraph MCP_Hub["MCP 数据调度中心"]
        MCP["<b>MCP Server</b>"]
        M1["CM3KG 医疗知识库"]
        M2["人物画像数据库"]
        M3["RAG 临床指南与病例库"]
    end
    
    subgraph Reasoning_Layer["3. 推理层: 知识驱动决策"]
        C["上下文融合"]
        D["推理引擎"]
        D1["KG 路径推理"]
        D2["RAG 语义检索"]
        E["置信度综合评估"]
    end
    
    subgraph Dialogue_Layer["4. 交互层: 动态问诊"]
        F["动态追问生成"]
        G["分诊决策生成"]
    end
    
    subgraph Output_Layer["5. 输出层"]
        H1["分诊建议卡片"]
        H2["病情预查小结"]
    end
    
    A1 --> B1
    B1 --> B2
    MCP <--> M1 & M2 & M3
    B2 --> C
    C <--> MCP
    C --> D
    D --> D1 & D2
    D1 --> E
    D2 --> E
    E -- "置信度 < 0.8" --> F
    F --> B1
    E -- "置信度 >= 0.8" --> G
    G --> H1 & H2
    B2 -.-> MCP
```

### 各层说明

| 层级 | 组件 | 功能 |
|------|------|------|
| **输入层** | 用户主诉 | 接收用户文字输入 |
| **感知层** | LLM 实体提取 | 提取症状/时长/程度等实体 |
| | KG 语义对齐 | 将口语映射至标准医学术语 (向量匹配) |
| **MCP 调度层** | MCP Server | 统一调度各数据源 |
| | Neo4j (CM3KG) | 症状-疾病-科室知识图谱 |
| | PostgreSQL | 患者画像/历史记录 |
| | ES + Milvus | 临床指南/病例库 RAG |
| **推理层** | 上下文融合 | 整合多源信息 |
| | KG 推理 | 症状→疑似疾病→建议科室 |
| | RAG 检索 | 相似病例语义匹配 |
| | 置信度评估 | 综合评分 (KG 0.6 + RAG 0.4) |
| **交互层** | 动态追问 | 置信度不足时补充提问 |
| | 分诊决策 | 生成最终科室推荐 |
| **输出层** | 分诊卡片 | 科室/风险/建议 |
| | 病情小结 | 结合画像的主诉总结 |

### LangGraph 节点映射

| 架构层 | LangGraph 节点 |
|--------|---------------|
| 输入层 | `trim_history` |
| 感知层 | `decision` → `slot_fill` |
| MCP 调度 | MCP Server (patient_server.py) |
| 推理层 | `diagnosis` → `kg_rag_fusion` |
| 交互层 | `question_gen` / `completion` |
| 输出层 | `answer_generate` |

## 项目结构


hospital_guidance_agent/
├── app/                        # 主应用代码
│   ├── main.py                 # FastAPI 入口
│   ├── api/                    # API 路由
│   │   └── routers/
│   │       ├── chat.py         # /chat 对话接口
│   │       ├── threads.py      # 会话管理
│   │       └── users.py        # 用户管理
│   ├── core/                  # 核心配置
│   │   ├── config.py           # 环境变量配置
│   │   ├── llm.py             # LLM/Embedding 封装
│   │   └── logging.py          # 日志配置
│   ├── domain/                 # 领域模型
│   │   ├── models.py           # AppState、IntentResult
│   │   └── diagnosis/           # 问诊系统
│   │       ├── slots.py        # 槽位定义
│   │       ├── filler.py       # 槽位填充 (向量语义匹配)
│   │       ├── risk.py         # 危险信号检测
│   │       └── questions.py    # 追问模板
│   ├── graph/                  # LangGraph 对话流
│   │   ├── builder.py          # 状态机构建
│   │   └── nodes/              # 节点实现
│   │       ├── decision.py     # 意图识别
│   │       ├── diagnosis.py    # 诊断推理 (KG+RAG)
│   │       ├── kg_rag_fusion.py # 综合推理模块
│   │       ├── question_gen.py # 追问生成
│   │       └── answer.py       # 答案生成
│   ├── infra/                  # 基础设施
│   │   ├── neo4j_client.py    # Neo4j (CM3KG 知识图谱)
│   │   ├── milvus_client.py   # Milvus (向量检索)
│   │   ├── es_client.py       # Elasticsearch
│   │   ├── redis_client.py    # Redis (会话存储)
│   │   └── postgres_client.py  # PostgreSQL (患者数据)
│   ├── mcp/                    # MCP 工具调度
│   │   ├── patient_server.py   # MCP Server 定义
│   │   └── client.py          # MCP Client 调用
│   └── tools/                  # 工具函数
│       └── knowledge_graph_tool.py
├── data/                       # 知识数据
│   └── knowledge_graph/
│       └── cm3kg/             # 医学知识图谱数据
├── demo/                       # 示例脚本
│   ├── es.py                  # ES 数据导入
│   ├── milvus.py              # Milvus 数据导入
│   └── ...
├── cli.py                      # 命令行前端
└── README.md


## 环境配置

### 基础设施（Docker）

| 服务 | 端口 | 用途 |
|------|------|------|
| Redis | 6379 | 会话存储/LangGraph Checkpoint |
| Elasticsearch | 9200 | 流程指南 RAG 检索 |
| Milvus | 19530 | 病历向量检索 |
| Neo4j | 7687 | CM3KG 知识图谱 |
| PostgreSQL | 5432 | 患者画像数据库 |

### 环境变量

```bash
# 必需
DASHSCOPE_API_KEY=your_api_key

# 可选（带默认值）
ES_URL=http://localhost:9200
MILVUS_URI=http://localhost:19530
REDIS_URI=redis://localhost:6379
POSTGRES_URI=postgresql://postgres:postgres@localhost:5432/hospital
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

### 快速启动

```bash
# 1. 启动所有基础设施
docker run -d --name redis -p 6379:6379 redis
docker run -d --name elasticsearch -p 9200:9200 -e discovery.type=single-node elasticsearch
docker run -d --name milvus -p 19530:19530 milvusdb/milvus
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres

# 2. 安装依赖
pip install -r requirements.txt

# 3. 导入数据
cd data/knowledge_graph && python import_cm3kg.py  # 知识图谱

# 4. 启动服务
uvicorn app.main:app --reload

# 5. 运行 CLI
python cli.py
```

---

## MCP 工具

所有数据源通过 MCP Server 统一调度：

| 数据源 | MCP 工具 | 功能 |
|--------|---------|------|
| **Neo4j** | `infer_department` | 症状→科室推理 |
| | `semantic_match_symptoms` | 向量语义匹配 |
| | `get_possible_diseases` | 查询可能疾病 |
| **Milvus** | `milvus_search` | 病历向量检索 |
| **ES** | `es_search` | 指南文档检索 |
| **PostgreSQL** | `pg_get_patient_*` | 患者画像查询 |
| **综合** | `kg_rag_fusion` | KG+RAG 融合推理 |

---

## 运行说明

### 数据导入

```bash
# Neo4j 知识图谱
cd data/knowledge_graph && python import_cm3kg.py

# ES 流程指南（可选）
cd demo && python es.py

# Milvus 病历库（可选）
cd demo && python milvus.py
```

### 启动服务

```bash
uvicorn app.main:app --reload
python cli.py
```

---

## API 简要说明

仅列出核心接口，详细字段可通过代码或自动文档（FastAPI Swagger）查看。

- `POST /chat`
  - 请求体：`{ user_id: string, thread_id?: string, message: string, password_verified?: boolean }`
  - 响应体（简化）：  
    - `user_id`: 用户 ID  
    - `thread_id`: 当前会话 ID  
    - `reply`: 助手回复文本（Markdown）  
    - `intent_result`: 意图识别结果（是否为症状/流程/混合等）  
    - `used_docs.medical` / `used_docs.process`: 本轮使用到的文档列表
    - `diagnosis`: 多轮问诊信息 ★新增★
      - `type`: 问诊阶段（in_progress / complete / emergency）
      - `completed`: 是否完成
      - `slots`: 已填充的槽位（JSON 问诊表）
      - `risk_signals`: 检测到的危险信号
      - `risk_level`: 风险等级（none / warning / critical）

- `GET /threads?user_id=...`
- `POST /threads`
- `DELETE /threads/{thread_id}?user_id=...`
- `GET /threads/current?user_id=...`
- `POST /threads/switch`

- `POST /users`
- `GET /users/{user_id}`

- `GET /healthz`

---

## 适用场景与扩展方向

- 医院导诊 / 分诊问答机器人。
- 医院内部流程、制度、规则的问答助手。
- 其他垂直领域（如保险、政务）的 Agentic RAG 助手参考实现。

可以进一步扩展的方向：

- 替换/增加更多 LLM 提供商或模型。
- 增加工具调用节点（如挂号、检查预约、费用查询）。
- 接入 Web 前端或小程序前端。
- 增强监控与日志分析，接入 APM / tracing。
---

## 说明

本项目主要用于展示「生产级医院导诊 Agentic 助手」的整体设计与实现思路，涉及的医学内容仅为技术演示示例，不构成任何医疗建议或诊断依据，请勿用于真实诊疗决策。

