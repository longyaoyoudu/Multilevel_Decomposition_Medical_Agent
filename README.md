# Multilevel Decomposition Medical Agent Based on LLM and Knowledge Graph — 医药领域知识图谱问答系统

> 从零构建以疾病为中心的医药知识图谱，支持规则问答 + LLM+RAG 智能问答双模式

[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-3.x-blue)](https://neo4j.com/)
[![Py2neo](https://img.shields.io/badge/py2neo-2021.2.4-orange)](https://py2neo.org/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-purple)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache--2.0-yellow)](LICENSE)

---

## 📌 项目概述

QASystemOnMedicalKG 是一个完整的**医药领域知识图谱问答系统**，包含两代技术架构：

1. **传统规则问答** — 基于 Aho-Corasick 多模式匹配实体识别 + Neo4j Cypher 查询，无需模型，纯规则驱动
2. **LLM+RAG 智能问答** — 以知识图谱为检索源，构建医疗知识三元组作为上下文，调用 LLM 生成更自然的答案

项目从垂直医疗网站抓取数据，自底向上构建以**疾病为中心**的知识图谱，实体规模约 **4.4 万**，关系规模约 **30 万**，可回答 **18 类**医疗问题。

---

## 🗂️ 项目结构

```
QASystemOnMedicalKG/
│
├── build_medicalgraph.py     # 知识图谱构建入库脚本
├── chatbot_graph.py          # 规则问答主程序（终端交互）
│
├── question_classifier.py    # 问句类型分类器（Aho-Corasick 实体识别）
├── question_parser.py        # 问句解析 → Cypher SQL 生成
├── answer_search.py          # Neo4j 执行查询 → 答案模板拼装
│
├── chat_with_llm.py          # LLM+RAG 问答核心类（KGRAG）
├── llm_server.py             # LLM API 调用封装
├── app.py                    # Gradio Web 界面（LLM+RAG 模式）
│
├── prepare_data/
│   ├── data_spider.py        # 医疗网站数据采集爬虫
│   ├── max_cut.py            # 基于词典的最大正向/反向切分
│   └── build_data.py         # 数据清洗与结构化
│
├── dict/                     # 领域词典（实体词 + 否定词）
│   ├── disease.txt           # 疾病词典
│   ├── drug.txt              # 药品词典
│   ├── food.txt              # 食物词典
│   ├── symptom.txt            # 症状词典
│   ├── check.txt             # 检查项目词典
│   ├── department.txt         # 科室词典
│   ├── producer.txt           # 药品厂商词典
│   └── deny.txt              # 否定词词典
│
├── data/
│   └── medical.json          # 原始医疗数据（JSONL 格式）
│
└── README.md
```

---

## 🧠 知识图谱规模

### 实体类型

| 实体类型 | 中文含义 | 数量 | 示例 |
|:---|:---:|---:|:---|
| Disease | 疾病 | 8,807 | 血栓闭塞性脉管炎 |
| Symptom | 疾病症状 | 5,998 | 乳腺组织肥厚 |
| Drug | 药品 | 3,828 | 京万红痔疮膏 |
| Producer | 在售药品 | 17,201 | 通药制药青霉素V钾片 |
| Food | 食物 | 4,870 | 番茄冲菜牛肉丸汤 |
| Check | 诊断检查项目 | 3,353 | 支气管造影 |
| Department | 医疗科目 | 54 | 烧伤科 |
| **Total** | **总计** | **44,111** | 约 4.4 万实体 |

### 关系类型

| 关系类型 | 中文含义 | 数量 |
|:---|:---:|---:|
| recommand_drug | 疾病推荐药品 | 59,467 |
| recommand_eat | 疾病推荐食谱 | 40,221 |
| need_check | 疾病所需检查 | 39,422 |
| acompany_with | 疾病并发症 | 12,029 |
| common_drug | 疾病常用药品 | 14,649 |
| do_eat | 疾病宜吃食物 | 22,238 |
| no_eat | 疾病忌吃食物 | 22,247 |
| has_symptom | 疾病症状 | 5,998 |
| belongs_to | 科室从属 | 8,844 |
| drugs_of | 药品在售 | 17,315 |
| **Total** | **总计** | **~294,149** | 约 30 万关系 |

### 疾病属性

| 属性 | 中文含义 |
|:---|:---|
| name | 疾病名称 |
| desc | 疾病简介 |
| cause | 疾病病因 |
| prevent | 预防措施 |
| cure_lasttime | 治疗周期 |
| cure_way | 治疗方式 |
| cured_prob | 治愈概率 |
| easy_get | 易感人群 |

---

## 🚀 快速开始

### 环境依赖

```bash
pip install py2neo==2021.2.4
pip install ahocorasick
pip install pyahocorasick
pip install gradio
pip install requests
```

### 1. 部署 Neo4j

```bash
# Docker 启动 Neo4j
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:3.5
```

> 修改 `answer_search.py` / `build_medicalgraph.py` 中的 Neo4j 密码为你的密码。

### 2. 构建知识图谱

```bash
python build_medicalgraph.py
```

> 数据导入约需数小时（30 万关系），耐心等待。

### 3. 规则问答（终端模式）

```bash
python chatbot_graph.py
```

示例对话：
```
用户: 乳腺癌的症状有哪些？
医药助手: 乳腺癌的症状包括：乳腺癌的远处转移；胸痛；乳头溢液；...
```

### 4. LLM+RAG 问答（Web 界面）

```bash
# 启动 LLM 服务（如 Ollama / Qwen / SiliconFlow）
# 修改 llm_server.py 中的 MODEL_URL

python app.py
# 访问 http://127.0.0.1:7860
```

---

## 🔧 技术架构

### 规则问答流程

```
用户问句
   ↓
question_classifier.py    ← Aho-Corasick 多模式匹配识别医疗实体
   ↓
问句类型分类（18 类）    ← 疑问词 + 实体类型组合判断
   ↓
question_parser.py        ← 生成 Cypher 查询语句
   ↓
answer_search.py          ← Neo4j 执行查询
   ↓
答案模板拼装              ← 回复模板格式化
```

**支持 18 类问答：**

| 问句类型 | 示例 |
|:---|:---|
| disease_symptom | 乳腺癌的症状有哪些？ |
| symptom_disease | 流鼻涕可能是什么病？ |
| disease_cause | 为什么会失眠？ |
| disease_acompany | 失眠有哪些并发症？ |
| disease_not_food | 失眠的人不要吃啥？ |
| disease_do_food | 耳鸣了吃点啥？ |
| food_not_disease | 什么病最好不要吃蜂蜜？ |
| food_do_disease | 鹅肉有什么好处？ |
| disease_drug | 肝病要吃啥药？ |
| drug_disease | 板蓝根颗粒能治啥病？ |
| disease_check | 脑膜炎怎么检查？ |
| check_disease | 全血细胞计数能查出啥？ |
| disease_prevent | 怎样才能预防肾虚？ |
| disease_lasttime | 感冒要多久才能好？ |
| disease_cureway | 高血压要怎么治？ |
| disease_cureprob | 白血病能治好吗？ |
| disease_easyget | 什么人容易得高血压？ |
| disease_desc | 糖尿病是什么病？ |

### LLM+RAG 流程

```
用户问句
   ↓
KGRAG.entity_linking()    ← 复用 question_classifier 实体识别
   ↓
KGRAG.recall_facts()      ← Neo4j 多跳查询，召回知识三元组
   ↓
KGRAG.build_enhanced_prompt()  ← 三元组格式化为上下文 Prompt
   ↓
ModelAPI.chat()            ← 调用 LLM 生成答案
   ↓
Gradio Web 界面展示
```

---

## 🔑 核心模块详解

### question_classifier.py — Aho-Corasick 实体识别

使用 Aho-Corasick 自动机实现**多模式字符串匹配**，一次性从问句中找出所有医疗实体：

```python
# 构建 AC 自动机
self.region_tree = self.build_actree(list(self.region_words))

# 匹配
for i in self.region_tree.iter(question):
    wd = i[1][1]  # 匹配到的词
    region_wds.append(wd)
```

- 词典加载：7 类实体词典（疾病/症状/药品/食物/检查/科室/厂商）
- 去噪：子串去除（"乳腺癌" 命中则去除 "癌"）
- 问句分类：疑问词集合 × 实体类型组合 → 18 类问题类型

### question_parser.py — Cypher SQL 生成

根据问题类型，将实体映射为 Cypher 图查询：

```python
# 疾病 → 症状
"MATCH (m:Disease)-[r:has_symptom]->(n:Symptom)
 where m.name = '乳腺癌' return m.name, r.name, n.name"

# 疾病 → 忌口
"MATCH (m:Disease)-[r:no_eat]->(n:Food)
 where m.name = '失眠' return m.name, r.name, n.name"
```

### answer_search.py — 答案模板拼装

将 Cypher 查询结果按问题类型填充回复模板：

```python
elif question_type == 'disease_symptom':
    desc = [i['n.name'] for i in answers]
    subject = answers[0]['m.name']
    final_answer = '{0}的症状包括：{1}'.format(
        subject, '；'.join(list(set(desc))[:20]))
```

### chat_with_llm.py — KGRAG 核心类

```python
class KGRAG:
    def recall_facts(self, rels, entity_type, entity_name, depth=1):
        # Neo4j 多跳查询，支持 1-3 跳深度
        sql = "MATCH p=(m:{entity_type})-[r*..{depth}]-(n)"
              " where m.name = '{entity_name}' return p"

    def build_enhanced_prompt(self, query, facts):
        # 三元组格式化为 <subject, relation, object>
        # 注入 Role-Prompt，要求先引证据再给答案
```

---

## 📊 典型问答效果

### 规则模式

```
用户: 乳腺癌的症状有哪些？
小勇: 乳腺癌的症状包括：乳腺癌的远处转移；胸痛；乳头溢液；...

用户: 失眠有哪些并发症？
小勇: 失眠的症状包括：心肾不交；神经性耳鸣；偏执狂；抑郁症；...

用户: 耳鸣了吃点啥？
小勇: 耳鸣宜食的食物包括有：南瓜子仁;鸡翅;芝麻;腰果
      推荐食谱包括有：紫菜芙蓉汤;羊肉汤面;可乐鸡翅;...

用户: 感冒要多久才能好？
小勇: 感冒治疗可能持续的周期为：7-14天
```

### LLM+RAG 模式

```
用户: 乳腺癌的症状有哪些？
智能医生: 基于知识库中的证据：
  <乳腺癌, 症状, 乳头溢液>
  <乳腺癌, 症状, 乳房肿块>
  <乳腺癌, 症状, 胸痛>
  乳腺癌的常见症状主要包括乳房肿块、乳头溢液、皮肤改变...
```

---

## 🙏 致谢

- **刘焕勇** — 原项目作者 [@GitHub](https://github.com/liuhuanyong)
- **Neo4j** — 图数据库 [@neo4j](https://neo4j.com/)
- **pyahocorasick** — 高效多模式字符串匹配 [@GitHub](https://github.com/WojciechMula/pyahocorasick)
- **Gradio** — 机器学习 Web 界面 [@GitHub](https://github.com/gradio-app/gradio)

---

## 📝 License

Apache License 2.0
