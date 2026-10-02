# 多 Agent 知识图谱构建系统 (Multi-Agent Knowledge Graph)

按照 `思路.md` 的设计，使用 **多 Agent 协作** 完成：数据采集 → 语义理解 →
实体/关系抽取 → Schema 设计 → 验证去重 → 图谱写入 → 向量索引(RAG)。
LLM 与 Embedding 端点直接复用项目 `.env` 中的配置。

## 架构

```
用户文本/文档
   │
   ▼
数据 Agent (采集/清洗/分块)
   ▼
理解 Agent + 抽取 Agent (实体 + 关系 + 事件, 调用 LLM)
   ▼
Schema Agent (节点/关系类型设计)
   ▼
验证 Agent (实体去重 / 孤儿关系 / 冲突检测)
   ▼
图谱写入 Agent ──► 图数据库 (NetworkX / FalkorDB / Neo4j)
   ▼
向量索引 (bge-m3) ──► RAG 问答 Agent
```

## 目录结构

```
kg_builder/
  config.py            读取 .env，集中配置
  clients.py           OpenAI 兼容的 LLM / Embedding 客户端
  schemas.py           Pydantic 数据模型
  prompts.py           各 Agent 提示词
  agents/              7 个 Agent + 基类
  graph_store/         NetworkX / FalkorDB / Neo4j 三种后端 + 工厂
  vector_store.py      轻量余弦向量库 (JSON 持久化)
  supervisor.py        编排多 Agent 流水线
  api.py               FastAPI 服务
frontend/index.html    vis-network 可视化前端
tests/                 单元测试 + 集成测试
samples/               示例文档
run.py                 服务入口
```

## 配置 (.env)

系统自动读取项目根目录的 `.env`，关键项：

| 配置 | 说明 |
|------|------|
| `LLM_BINDING_HOST` / `LLM_MODEL` | OpenAI 兼容聊天端点 (本机 `:9000` qwen) |
| `EMBEDDING_BINDING_HOST` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Embedding 端点 (本机 `:9001` bge-m3, 1024 维) |
| `KG_GRAPH_BACKEND` | 图存储后端：`networkx`(默认) / `falkordb` / `neo4j` |
| `KG_WEB_PORT` | Web 服务端口，默认 `9622` (避开 LightRAG 的 9621) |

> 图数据库：默认使用 **NetworkX**（无需外部服务、可直接序列化给前端）；
> 思路.md 推荐的 **FalkorDB** 已在 `:6379` 运行，设 `KG_GRAPH_BACKEND=falkordb` 即可启用；
> **Neo4j** (`:7687`/`:7474`) 也已实现并测试通过。

## 运行

```bash
pip install -r requirements.txt

# 启动 Web 服务（浏览器打开 http://<host>:9622/）
python run.py

# 或指定 FalkorDB 后端
KG_GRAPH_BACKEND=falkordb python run.py
```

前端操作：
1. 在文本框粘贴文档，点击「构建知识图谱」；或上传 `.txt/.md/.pdf`。
2. 右侧画布实时渲染实体（节点）与关系（有向边）。
3. 底部输入框提问，调用「问答」进行图谱增强的 RAG 回答。

API 摘要：
- `POST /api/build`  `{text, source}` → 构建并返回统计/校验报告
- `POST /api/build_file`  上传文件构建
- `GET  /api/graph`  返回前端渲染所需的图 JSON
- `POST /api/query`  `{question}` → 图谱 + 向量 RAG 回答
- `GET  /api/stats` / `POST /api/reset`

## 测试

```bash
# 单元测试（无需联网，使用 FakeLLM）
python -m pytest tests/test_agents.py tests/test_graph_store.py -q

# 集成测试（真实调用 LLM / Embedding / 图数据库）
RUN_INTEGRATION=1 python -m pytest tests/ -q
```

## 说明

- 所有 LLM 调用均面向 OpenAI 兼容接口，可无缝替换为 llama.cpp / vLLM / Ollama 等。
- 抽取 Agent 使用 JSON 模式并做容错解析（兼容非严格 JSON 输出）。
- 服务为常驻进程；更改代码后重启 `python run.py` 即可生效。
