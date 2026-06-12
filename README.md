# QASystemOnMedicalKG

> 以**疾病为中心**的医药领域知识图谱问答系统：规则问答 + LLM+RAG 双引擎，支持多路召回、重排序与答案润色

[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-3.x%2F4.x-blue)](https://neo4j.com/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-purple)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow)](LICENSE)
[![Stars](https://img.shields.io/github/stars/liuhuanyong/QASystemOnMedicalKG)](https://github.com/liuhuanyong/QASystemOnMedicalKG)

---

## ✨ 项目亮点

- **两代架构共存** — 经典规则问答（18 类问题，模板化回复）+ LLM+RAG 智能问答，按场景自由切换
- **以疾病为中心** — 围绕"疾病—症状—药品—食物—检查—科室—厂商"7 类实体构建图谱
- **多路召回 + 重排序** — Aho-Corasick 实体识别 × Neo4j 多跳 × ChromaDB 向量检索三路召回，再经 Reranker 精排
- **多 LLM 适配** — 统一 LLM 服务层（`unified_llm_service.py`）支持 **MiniMax + SiliconFlow（双模型协作）/ Ollama / Qwen7B / DeepSeek** 等
- **可观测性** — Gradio Web 界面内置彩色处理日志，实体识别、召回、生成、润色全链路可见
- **完整测试** — `tests/` 覆盖 Neo4j 连接、KG 状态、LLM 服务、Prompt 逻辑等关键路径

---

## 🏗️ 系统架构

```
                ┌────────────────────────────────────────────┐
                │            用户问句（中文）                 │
                └──────────────────┬─────────────────────────┘
                                   │
                ┌──────────────────▼──────────────────────┐
                │  实体识别层 (Aho-Corasick 多模式匹配)    │
                │  qa_traditional/question_classifier.py   │
                └──────────────────┬──────────────────────┘
                                   │ 实体 + 实体类型
                ┌──────────────────▼──────────────────────┐
                │           多路召回 (Hybrid)              │
                │  ┌──────────┬──────────┬────────────┐    │
                │  │ 图谱多跳 │ 向量检索 │   AC 实体  │    │
                │  │ Neo4j   │ ChromaDB │   链接     │    │
                │  └──────────┴──────────┴────────────┘    │
                │       qa_llm_enhanced/hybrid_retriever   │
                └──────────────────┬──────────────────────┘
                                   │ 候选三元组
                ┌──────────────────▼──────────────────────┐
                │            Reranker 精排                 │
                │         MiniMax-M2.7-highspeed           │
                └──────────────────┬──────────────────────┘
                                   │ Top-K 三元组
                ┌──────────────────▼──────────────────────┐
                │  Prompt 构建 (Role-Prompt + 三元组)     │
                └──────────────────┬──────────────────────┘
                                   │
                ┌──────────────────▼──────────────────────┐
                │     LLM 统一服务 (unified_llm_service)   │
                │  MiniMax + SiliconFlow 双模型协作 / 单模  │
                └──────────────────┬──────────────────────┘
                                   │ 原始答案
                ┌──────────────────▼──────────────────────┐
                │            答案润色 (polish)             │
                └──────────────────┬──────────────────────┘
                                   │
                ┌──────────────────▼──────────────────────┐
                │   Gradio Web 界面 / 终端交互              │
                └──────────────────────────────────────────┘
```

---

## 🗂️ 项目结构

```
QASystemOnMedicalKG/
│
├── qa_traditional/                    # 规则问答（旧架构，无需 LLM）
│   ├── chatbot_graph.py                #   终端交互入口
│   ├── question_classifier.py          #   Aho-Corasick 实体识别 + 18 类问句分类
│   ├── question_parser.py              #   实体 → Cypher 查询模板
│   └── question_classifer_change.py    #   分类器增强版
│
├── qa_llm_enhanced/                    # LLM 增强问答（新架构）
│   ├── chat_with_llm.py                #   KGRAG 核心类（图谱召回 + Prompt 构造 + 答案润色）
│   ├── llm_question_classifier.py      #   LLM 兜底分类器
│   ├── llm_question_parser.py          #   LLM Cypher 解析
│   ├── hybrid_retriever.py             #   多路召回（图谱 + 向量 + AC）
│   ├── reranker.py                     #   Reranker 精排
│   ├── chat_context.py                 #   多轮对话上下文管理
│   ├── chat_llm_context.py             #   LLM 上下文工具
│   └── chat_only_llm.py                #   纯 LLM 模式（无 KG）
│
├── kg_builder/                         # 知识图谱构建
│   ├── build_medicalgraph.py           #   数据入库（Neo4j）
│   ├── answer_search.py                #   Cypher 执行 + 答案模板拼装
│   ├── build_vector_index.py           #   从 Neo4j 导出三元组并构建 ChromaDB 索引
│   └── vector_retriever.py             #   ChromaDB 向量检索器
│
├── llm_services/                       # LLM 服务适配层
│   ├── unified_llm_service.py          #   ⭐ 统一入口（推荐使用：Flask :3001）
│   ├── ollama.py                       #   Ollama 本地模型
│   ├── SiliconFlow.py                  #   硅基流动
│   ├── qwen7b_server.py                #   Qwen-7B 本地服务
│   ├── qwen_server_fixed.py            #   Qwen 修复版
│   ├── remote_server_config.py         #   远程 LLM 配置文件
│   ├── minimax.py                      #   MiniMax
│   ├── llm_server.py / llm_server_change.py / llm_server_fix.py
│   └── ...
│
├── web_apps/                           # Gradio Web 界面
│   ├── rag_gradio_interface.py         #   ⭐ 完整版 RAG 界面（多路召回 + Reranker 开关 + 日志）
│   ├── app.py                          #   主入口
│   ├── deepseekapp.py                  #   DeepSeek 专用
│   ├── chatapp.py / chatapp_optimized.py / chat_app.py
│   └── ...
│
├── tests/                              # 测试套件
│   ├── test_connect.py                 #   Neo4j 连接测试
│   ├── test_kg_status.py               #   KG 状态检查
│   ├── test_llm_qa_system.py           #   LLM 问答系统端到端
│   ├── test_fixed_server.py            #   LLM 服务连通性
│   ├── test_prompt_logic.py            #   Prompt 逻辑
│   ├── test_minimax_api.py             #   MiniMax API 测试
│   ├── test_siliconflow_api.py         #   SiliconFlow API 测试
│   └── debug_qa_system.py              #   调试工具
│
├── utils/                              # 工具模块
│   ├── config.py                       #   统一配置管理（从 .env 加载）
│   ├── embedding_model.py              #   Embedding 模型懒加载
│   ├── redownload.py                   #   数据重下载
│   └── web_scraper_to_markdown.py      #   通用爬虫转 Markdown
│
├── prepare_data/                       # 离线数据准备
│   ├── data_spider.py                  #   医疗网站数据采集
│   ├── max_cut.py                      #   基于词典的最大正向/反向切分
│   └── build_data.py                   #   数据清洗与结构化
│
├── dict/                               # 领域词典
│   ├── disease.txt / drug.txt / food.txt
│   ├── symptom.txt / check.txt / department.txt
│   ├── producer.txt / deny.txt
│
├── data/
│   └── medical.json                    # 原始医疗数据（JSONL）
│
├── document/                           # 设计文档与图片
├── img/                                # 文档插图
├── requirements.txt
├── .env.example                        # 环境变量示例（需自行创建）
└── README.md
```

---

## 📊 知识图谱规模

### 实体类型（7 类 / 共 4.4 万）

| 实体类型 | 中文 | 数量 | 示例 |
|:---|:---:|---:|:---|
| Disease | 疾病 | 8,807 | 血栓闭塞性脉管炎 |
| Symptom | 症状 | 5,998 | 乳腺组织肥厚 |
| Drug | 药品 | 3,828 | 京万红痔疮膏 |
| Producer | 厂商 | 17,201 | 通药制药青霉素V钾片 |
| Food | 食物 | 4,870 | 番茄冲菜牛肉丸汤 |
| Check | 检查 | 3,353 | 支气管造影 |
| Department | 科室 | 54 | 烧伤科 |
| **合计** | — | **44,111** | — |

### 关系类型（10 类 / 共 30 万）

| 关系 | 中文 | 数量 |
|:---|:---:|---:|
| `recommand_drug` | 推荐药品 | 59,467 |
| `recommand_eat` | 推荐食谱 | 40,221 |
| `need_check` | 所需检查 | 39,422 |
| `no_eat` | 忌吃 | 22,247 |
| `do_eat` | 宜吃 | 22,238 |
| `drugs_of` | 药品在售 | 17,315 |
| `common_drug` | 常用药品 | 14,649 |
| `acompany_with` | 并发症 | 12,029 |
| `belongs_to` | 所属科室 | 8,844 |
| `has_symptom` | 疾病症状 | 5,998 |
| **合计** | — | **~294,149** |

### 疾病属性

`name / desc / cause / prevent / cure_lasttime / cure_way / cured_prob / easy_get / cure_department`

---

## 🚀 快速开始

### 1. 环境准备

- Python ≥ 3.8
- Neo4j 3.x 或 4.x
- （可选）至少一个 LLM API Key：MiniMax / SiliconFlow / Ollama / Qwen7B

```bash
git clone https://github.com/liuhuanyong/QASystemOnMedicalKG.git
cd QASystemOnMedicalKG

pip install py2neo==2021.2.4 pyahocorasick chromadb sentence-transformers \
            gradio flask requests python-dotenv
```

### 2. 配置环境变量

复制一份 `.env.example` 为 `.env`，按需填入：

```ini
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM（至少二选一；都填则启用"双模型协作模式"）
MINIMAX_API_KEY=
MINIMAX_GROUP_ID=
SILICONFLOW_API_KEY=
SILICONFLOW_MODEL_NAME=deepseek-ai/DeepSeek-V3

# Ollama（本地模型可选）
OLLAMA_API_URL=http://localhost:11434/api/generate
OLLAMA_MODEL_NAME=qwen3:0.6b
```

### 3. 启动 Neo4j

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  neo4j:3.5
```

### 4. 构建知识图谱 + 向量索引

```bash
# 4.1 入库（约需数小时，30 万关系）
python kg_builder/build_medicalgraph.py

# 4.2 构建向量索引（ChromaDB）
python kg_builder/build_vector_index.py
```

### 5. 启动 LLM 统一服务

```bash
python llm_services/unified_llm_service.py
# 服务运行在 http://127.0.0.1:3001
# 健康检查：GET  /health
# 问答端点：POST /generate
```

`unified_llm_service` 会自动检测 `.env` 中的 API Key：

| 场景 | 行为 |
|---|---|
| 同时配置 `MINIMAX_*` + `SILICONFLOW_*` | **双模型协作**：MiniMax 生成 → SiliconFlow 优化 |
| 仅 `SILICONFLOW_API_KEY` | 单模型 SiliconFlow（默认 DeepSeek-V3） |
| 仅 `MINIMAX_API_KEY` | 单模型 MiniMax |
| 都没配 | 启动报错，请在 `.env` 至少填一项 |

### 6. 启动 Web 界面

```bash
# 完整版：多路召回 + Reranker 开关 + 彩色处理日志（推荐）
python web_apps/rag_gradio_interface.py

# 或：DeepSeek 专用精简版
python web_apps/deepseekapp.py
```

浏览器访问 [http://127.0.0.1:7860](http://127.0.0.1:7860) 即可体验。

### 7. 不想要 LLM？用纯规则问答

```bash
python qa_traditional/chatbot_graph.py
# 直接在终端输入：乳腺癌的症状有哪些？
```

---

## 🤖 支持的 LLM 服务

| 服务 | 文件 | 适用场景 |
|---|---|---|
| **统一入口（双模型协作）** | [llm_services/unified_llm_service.py](llm_services/unified_llm_service.py) | 首选：自动 fallback、链路完整 |
| SiliconFlow（硅基流动） | [llm_services/SiliconFlow.py](llm_services/SiliconFlow.py) | 国内访问稳定，DeepSeek/Qwen 全系 |
| MiniMax | [llm_services/minimax.py](llm_services/minimax.py) | abab 系列 |
| Ollama（本地） | [llm_services/ollama.py](llm_services/ollama.py) | 离线/隐私 |
| Qwen-7B 本地 | [llm_services/qwen7b_server.py](llm_services/qwen7b_server.py) | GPU 部署 |
| DeepSeek 专用 | [llm_services/llm_server.py](llm_services/llm_server.py) + [web_apps/deepseekapp.py](web_apps/deepseekapp.py) | DeepSeek 全栈 |
| 远程服务器 | [llm_services/remote_server_config.py](llm_services/remote_server_config.py) | 自建/代理 |

> 💡 **接入新模型**？只需在 `unified_llm_service.py` 中添加一个 `call_xxx_api` 函数即可，无需改动上游 `qa_llm_enhanced` 模块。

---

## 🧠 18 类规则问答支持

| 问句类型 | 示例 |
|:---|:---|
| `disease_symptom` | 乳腺癌的症状有哪些？ |
| `symptom_disease` | 流鼻涕可能是什么病？ |
| `disease_cause` | 为什么会失眠？ |
| `disease_acompany` | 失眠有哪些并发症？ |
| `disease_not_food` | 失眠的人不要吃啥？ |
| `disease_do_food` | 耳鸣了吃点啥？ |
| `food_not_disease` | 什么病最好不要吃蜂蜜？ |
| `food_do_disease` | 鹅肉有什么好处？ |
| `disease_drug` | 肝病要吃啥药？ |
| `drug_disease` | 板蓝根颗粒能治啥病？ |
| `disease_check` | 脑膜炎怎么检查？ |
| `check_disease` | 全血细胞计数能查出啥？ |
| `disease_prevent` | 怎样才能预防肾虚？ |
| `disease_lasttime` | 感冒要多久才能好？ |
| `disease_cureway` | 高血压要怎么治？ |
| `disease_cureprob` | 白血病能治好吗？ |
| `disease_easyget` | 什么人容易得高血压？ |
| `disease_desc` | 糖尿病是什么病？ |

---

## 🧪 运行测试

```bash
# Neo4j 连接 + KG 状态
python tests/test_connect.py
python tests/test_kg_status.py

# LLM 端到端 + Prompt 逻辑
python tests/test_llm_qa_system.py
python tests/test_prompt_logic.py

# LLM API 连通性
python tests/test_minimax_api.py
python tests/test_siliconflow_api.py
```

---

## 🛠️ 技术栈

| 层 | 选型 |
|---|---|
| 图数据库 | Neo4j 3.x / 4.x |
| 向量数据库 | ChromaDB（ANN 检索） |
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2`（默认） |
| 实体识别 | Aho-Corasick 自动机（`pyahocorasick`） |
| Reranker | `MiniMax-M2.7-highspeed` |
| LLM 适配 | Flask + `requests`（统一中间层） |
| Web UI | Gradio 5.x |
| 图谱客户端 | py2neo 2021.2.4 |

---

## 📦 部署建议 / .gitignore

本仓库开发过程中可能产生以下非交付文件，建议在 `.gitignore` 中忽略：

```gitignore
# Python
__pycache__/
*.pyc
venv/
.env

# IDE
.idea/
.vscode/

# 数据与索引（体积大、可重建）
data/medical.json
vector_db/
neo4j_data/

# 项目内的临时爬取内容
web_content_markdown/

# 日志
*.log
```

> 这些文件不影响代码运行；按需生成，避免污染仓库。

---

## 🙏 致谢

- **刘焕勇** — 原项目作者 [@liuhuanyong](https://github.com/liuhuanyong)，本项目在其基础上做了大规模重构（模块化 + LLM 增强 + 多路召回 + Web 完善）
- [Neo4j](https://neo4j.com/) — 图数据库
- [ChromaDB](https://www.trychroma.com/) — 向量数据库
- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) — 高效多模式字符串匹配
- [Gradio](https://gradio.app/) — Web 界面
- [SiliconFlow](https://siliconflow.cn/) / [MiniMax](https://api.minimax.chat/) / [Ollama](https://ollama.com/) — LLM 服务支持

---

## 📝 License

[Apache License 2.0](LICENSE)

---

> **⚠️ 免责声明**：本系统仅作为研究与演示用途，所有医疗信息**不能替代专业医生的诊断与治疗建议**。如有健康问题，请咨询执业医师。
