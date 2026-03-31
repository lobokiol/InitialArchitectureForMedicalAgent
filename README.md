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
  - **方案A 症状提取架构**（LLM 提原词 → Neo4j 转标准术语 → 可追溯）：
    - **步骤1**: LLM 提取症状原词（保留用户原话，如"头壳痛"、"肚子疼"）
    - **步骤2**: Neo4j 向量语义匹配（原词 → 标准医学术语，如"头壳痛"→"头痛"）
    - **步骤3**: 槽位填充 + 知识图谱校验
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

核心对话工作流采用 **九步流水线** 设计，以 MCP 为数据调度中心：

```mermaid
graph TD
    %% 1. 输入预处理
    Start((用户输入)) --> Step1["<b>1. 历史裁剪 (trim_history)</b>"]
    Step1 --> Step2["<b>2. 意图识别 (Decision Node)</b><br/>7B INT4 分类: 导诊/流程/闲聊/危急"]
    
    %% 2. 非导诊分支 (快速路径)
    Step2 -- "闲聊 / 拒答" --> ExitNode["<b>结束节点 (Exit)</b><br/>礼貌回复并直接终止"]
    ExitNode --> End((结束))

    Step2 -- "触发危急值" --> Emergency["<b>急诊红色通道</b><br/>人工介入 / 紧急告警"]
    Emergency --> End

    Step2 -- "办事指南 / 流程咨询" --> Step_ES["<b>ES 指南搜索 (MCP)</b><br/>检索医院手册/政策/流程"]
    Step_ES --> Step9

    %% 3. 导诊分支 (深度路径) - 意图识别后才做症状提取
    Step2 -- "看病导诊" --> Step3["<b>3. LLM 提取症状原词</b><br/>保留用户原话, 不做标准化"]
    Step3 --> Step4["<b>4. Neo4j 俗语转换</b><br/>原词 → 标准医学术语<br/>(如'头壳痛'→'头痛')"]
    Step4 --> Step5["<b>5. 槽位填充 (Slot Table)</b><br/>症状/时长/部位/程度"]
    Step5 --> Step6{"<b>6. 槽位完整性检查</b>"}
    
    Step6 -- "信息不足" --> Clarify["<b>动态追问 (Clarify)</b>"]
    Clarify --> Step3
    
    Step6 -- "满足检索" --> Step7["<b>7. 混合检索模块 (MCP)</b><br/>Neo4j 推理 + Milvus 向量"]
    
    %% 4. 质量校验与输出
    Step7 --> Step8{"<b>8. 答案生产检查</b>"}
    
    Step8 -- "相关性低" --> Rewrite["<b>Query Rewrite</b>"]
    Rewrite --> Step7
    
    Step8 -- "校验通过" --> Step9["<b>9. 答案合成输出</b><br/>整合指南信息 / 导诊建议"]
    Step9 --> End

    %% MCP 协议中枢
    subgraph MCP_Infrastructure [MCP Protocol Layer]
        MCP_Server{<b>MCP Server / Router</b>}
        T_Neo4j[(Neo4j Tool<br/>俗语转换/科室推理)]
        T_ES[(ES Tool<br/>指南/规章搜索)]
        T_Milvus[(Milvus Tool<br/>医学知识检索)]
        T_HIS[HIS API<br/>挂号卡片]
    end

    %% MCP 调用链路
    Step4 <==> MCP_Server
    Step_ES <==> MCP_Server
    Step7 <==> MCP_Server
    Step9 <==> MCP_Server

    MCP_Server --- T_Neo4j
    MCP_Server --- T_ES
    MCP_Server --- T_Milvus
    MCP_Server --- T_HIS

    %% 样式美化
    style Step2 fill:#f96,stroke:#333,stroke-width:2px
    style Step3 fill:#bbf,stroke:#333
    style Step4 fill:#bfb,stroke:#333
    style ExitNode fill:#ddd,stroke:#999
    style Step_ES fill:#fff9c4,stroke:#fbc02d
    style Emergency fill:#f66,stroke:#fff,color:#fff
    style MCP_Server fill:#f1f,stroke:#fff,color:#fff
```

### 各步骤说明

| 步骤 | 节点 | 功能 |
|------|------|------|
| 1 | trim_history | 历史消息裁剪，控制上下文长度 |
| 2 | **decision** | **意图识别（7B INT4）：导诊/流程/闲聊/危急** |
| 3 | **LLM 提取原词** | **提取症状原词（保留用户原话，不做标准化）** |
| 4 | **Neo4j 俗语转换** | **原词 → 标准医学术语（如"头壳痛"→"头痛"）** |
| 5 | Slot Table | 槽位填充（症状、部位、时长、严重程度等） |
| 6 | 槽位检查 | 置信度 < 0.65 → 追问；≥ 0.65 → 检索 |
| 7 | 混合检索 | Neo4j 推理 + Milvus 向量 + ES 流程 |
| 8 | 答案检查 | 相关性低 → Rewrite；通过 → 输出 |
| 9 | 答案输出 | 科室推荐卡片 + 导诊建议 |

### LangGraph 节点映射

| 步骤 | LangGraph 节点 |
|------|---------------|
| 1 | `trim_history` |
| 2 | `decision` (意图识别) |
| 3-4 | `diagnosis` → `fill_slots` (LLM提原词 → Neo4j转换) |
| 5 | `slot_fill` |
| 6 | `completion` |
| 7 | `kg_rag_fusion` + `milvus_rag` |
| 8 | `check_docs` + `rewrite` |
| 9 | `answer_generate` |

### MCP 工具

| 工具 | 数据源 | 功能 |
|------|--------|------|
| Neo4j Tool | CM3KG 知识图谱 | 症状映射、科室推理 |
| ES Tool | 流程指南库 | 医院手册/政策搜索 |
| Milvus Tool | 医学知识库 | 症状向量检索 |
| HIS API | 医院信息系统 | 挂号卡片、医生排班 |





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

# 认证配置（微信登录 + JWT）
# 获取方式：微信公众平台/小程序后台 -> 开发管理 -> 开发设置
WECHAT_APP_ID=your_wechat_app_id
WECHAT_APP_SECRET=your_wechat_app_secret

# 生成方式：openssl rand -hex 32
JWT_SECRET_KEY=your-jwt-secret-key

# Token 过期时间（可选）
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### 快速启动

```bash
# 1. 启动所有基础设施（推荐使用 Docker Compose）
cd demo
docker-compose -f mcp_docker-compose.yaml up -d

# 或手动启动各服务
docker run -d --name redis -p 6379:6379 redis
docker run -d --name elasticsearch -p 9200:9200 -e discovery.type=single-node elasticsearch
docker run -d --name milvus -p 19530:19530 milvusdb/milvus
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
docker run -d --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres

# 2. 安装依赖
pip install -r requirements.txt

# 3. 导入数据
cd data/knowledge_graph && python import_cm3kg.py  # 知识图谱

# 4. 启动 MCP Server（SSE 模式，端口 8001）
python -m app.mcp.patient_server --sse &

# 5. 启动主服务
uvicorn app.main:app --reload

# 6. 运行 CLI
python cli.py
```

---

## MCP 工具

所有数据源通过 MCP Server 统一调度（SSE 模式）：

| 数据源 | MCP 工具 | 功能 |
|--------|---------|------|
| **Neo4j** | `infer_department` | 症状→科室推理 |
| | `semantic_match_symptoms` | 向量语义匹配 |
| | `get_possible_diseases` | 查询可能疾病 |
| **Milvus** | `milvus_search` | 病历向量检索 |
| **ES** | `es_search` | 指南文档检索 |
| **PostgreSQL** | `pg_get_patient_*` | 患者画像查询 |
| **综合** | `kg_rag_fusion` | KG+RAG 融合推理 |

### MCP Server 启动

```bash
# SSE 模式（长期运行，推荐）
python -m app.mcp.patient_server --sse

# Docker Compose（推荐）
cd demo && docker-compose -f mcp_docker-compose.yaml up -d
```

### MCP Client 使用

```python
from app.mcp.client import MCPClient

# SSE 模式（连接远程 Server）
client = MCPClient(use_sse=True, sse_url="http://localhost:8001")
result = client.call_tool("kg_rag_fusion", {"symptoms": ["头痛"], "top_k": 3})

# stdio 模式（自动 fork 子进程）
client = MCPClient(use_sse=False)
result = client.call_tool("kg_rag_fusion", {"symptoms": ["头痛"], "top_k": 3})
```

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

### 认证接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/auth/wechat/login` | POST | 微信小程序登录 |
| `/api/v1/auth/login` | POST | 手机号登录 |
| `/api/v1/auth/register` | POST | 用户注册 |
| `/api/v1/auth/refresh` | POST | 刷新 Access Token |
| `/api/v1/auth/me` | GET | 获取当前用户信息 |
| `/api/v1/auth/logout` | POST | 退出登录 |

#### 微信登录流程

```python
# 前端: wx.login() 获取 code
# 调用后端登录接口
POST /api/v1/auth/wechat/login
{ "code": "wx_code_from_login" }

# 响应
{
  "access_token": "eyJhbG...",
  "refresh_token": "eyJhbG...",
  "expires_in": 3600,
  "user": { "user_id": "wx_xxx", "nickname": "..." },
  "is_new_user": false
}

# 后续请求携带 Token
Authorization: Bearer <access_token>
```

### 聊天接口

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

