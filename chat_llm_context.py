# coding=utf-8
import os
from question_classifier import *
from question_parser import *
from llm_server import *
from build_medicalgraph import *
import re
from typing import List, Tuple, Dict, Set

entity_parser = QuestionClassifier()
kg = MedicalGraph()
model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")


class KGRAG():
    def __init__(self):
        # 中文映射字典
        self.cn_dict = {
            "name": "名称",
            "desc": "疾病简介",
            "cause": "疾病病因",
            "prevent": "预防措施",
            "cure_department": "治疗科室",
            "cure_lasttime": "治疗周期",
            "cure_way": "治疗方式",
            "cured_prob": "治愈概率",
            "easy_get": "易感人群",
            "belongs_to": "所属科室",
            "common_drug": "常用药品",
            "do_eat": "宜吃",
            "drugs_of": "生产药品",
            "need_check": "诊断检查",
            "no_eat": "忌吃",
            "recommand_drug": "好评药品",
            "recommand_eat": "推荐食谱",
            "has_symptom": "症状",
            "acompany_with": "并发症",
            "Check": "诊断检查项目",
            "Department": "医疗科目",
            "Disease": "疾病",
            "Drug": "药品",
            "Food": "食物",
            "Producer": "在售药品",
            "Symptom": "疾病症状"
        }

        # 实体-关系映射
        self.entity_rel_dict = {
            "check": ["name", 'need_check'],
            "department": ["name", 'belongs_to'],
            "disease": ["prevent", "cure_way", "name", "cure_lasttime", "cured_prob",
                        "cause", "cure_department", "desc", "easy_get", 'recommand_eat',
                        'no_eat', 'do_eat', "common_drug", 'drugs_of', 'recommand_drug',
                        'need_check', 'has_symptom', 'acompany_with', 'belongs_to'],
            "drug": ["name", "common_drug", 'drugs_of', 'recommand_drug'],
            "food": ["name"],
            "producer": ["name"],
            "symptom": ["name", 'has_symptom'],
        }

        # 上下文缓存
        self.context_cache = {
            "entities": set(),
            "last_relations": set(),
            "history": []
        }

    def entity_linking(self, context: str) -> Dict[str, List[str]]:
        """带上下文的实体识别"""
        # 合并当前问题和历史上下文
        combined_text = context if isinstance(context, str) else "\n".join(
            [f"用户: {h[0]}\n助手: {h[1]}" for h in context]
        )

        # 识别新实体
        new_entities = entity_parser.check_medical(combined_text)

        # 合并缓存中的实体
        if self.context_cache["entities"]:
            for entity in self.context_cache["entities"]:
                if entity not in new_entities:
                    new_entities[entity] = ['disease']  # 默认类型

        # 更新缓存
        self.context_cache["entities"].update(new_entities.keys())
        return new_entities

    def link_entity_rel(self, context: str, entity: str, entity_type: str) -> Set[str]:
        """带上下文的关系链接"""
        # 从上下文中提取问题部分
        if isinstance(context, list):  # 如果是对话历史
            last_query = context[-1][0] if context else ""
        else:  # 如果是字符串上下文
            queries = re.findall(r"用户: (.*?)(?:\n|$)", context)
            last_query = queries[-1] if queries else ""

        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type, [])]

        # 构建考虑上下文的提示
        prompt = (
            f"根据对话历史分析关系:\n"
            f"上下文: {context[:500]}\n\n"
            f"请判定问题『{last_query}』中关于『{entity}』需要获取哪些信息？\n"
            f"可选关系: {cate}\n"
            "只需返回关系名称列表，如: ['症状', '治疗方式']"
        )

        answer, _ = model.chat(query=prompt, history=[])
        cls_rel = set(re.findall(r"['\"](.*?)['\"]", answer))
        cls_rel = cls_rel.intersection(set(cate))

        # 合并上次使用的关系
        if self.context_cache["last_relations"]:
            cls_rel.update(self.context_cache["last_relations"])

        print(f"实体关系分析: {prompt[:200]}... → {cls_rel}")
        return cls_rel

    def recall_facts(self, relations: Set[str], entity_type: str, entity_name: str, depth: int = 1) -> List[str]:
        """知识图谱事实召回"""
        entity_mapping = {
            "check": "Check",
            "department": "Department",
            "disease": "Disease",
            "drug": "Drug",
            "food": "Food",
            "producer": "Producer",
            "symptom": "Symptom"
        }

        # 构建查询语句
        cypher = (
            f"MATCH p=(m:{entity_mapping[entity_type]})-[r*..{depth}]-(n) "
            f"WHERE m.name = '{entity_name}' "
            "RETURN p"
        )

        results = kg.g.run(cypher).data()
        triples = set()

        for res in results:
            p_data = res["p"]
            nodes = p_data.nodes
            rels = p_data.relationships

            # 处理节点属性
            for node in nodes:
                node_name = node["name"]
                for attr, value in node.items():
                    if attr == "name" or value == node_name:
                        continue
                    if self.cn_dict.get(attr, "") in relations:
                        triples.add(f"<{node_name}, {self.cn_dict[attr]}, {value}>")

            # 处理关系
            for rel in rels:
                rel_name = rel["name"]
                if rel_name in relations:
                    start = rel.start_node["name"]
                    end = rel.end_node["name"]
                    if start != end:
                        triples.add(f"<{start}, {rel_name}, {end}>")

        # 更新缓存
        self.context_cache["last_relations"] = relations
        return list(triples)[:20]  # 限制返回数量

    def format_prompt(self, context: str, facts: List[str]) -> str:
        """构建上下文感知的提示词"""
        # 提取对话历史
        if isinstance(context, list):  # 对话历史列表
            history_str = "\n".join([f"用户: {q}\n助手: {a}" for q, a in context])
            last_query = context[-1][0] if context else ""
        else:  # 字符串上下文
            history_str = context
            last_query = re.findall(r"用户: (.*?)(?:\n|$)", context)[-1] if "用户:" in context else context

        # 知识结构化
        knowledge_str = "\n".join([f"{i + 1}. {fact}" for i, fact in enumerate(facts[:15])]) if facts else "无相关知识"

        return f"""
# 医疗问答任务
请基于以下对话历史和医学知识，专业地回答用户问题。

## 对话历史
{history_str}

## 相关知识
{knowledge_str}

## 当前问题
{last_query}

## 回答要求
1. 严格基于知识回答，不编造信息
2. 保持专业性和准确性
3. 如有不确定，明确说明
4. 回答格式:
   <分析>
   - 相关知识点: ...
   - 推理过程: ...

   <结论>
   - 最终答案: ...
"""

    def chat(self, prompt: str) -> str:
        """与LLM交互的统一接口"""
        try:
            answer, _ = model.chat(query=prompt, history=self.context_cache["history"])

            # 更新对话历史
            if isinstance(prompt, list):  # 如果是对话历史
                self.context_cache["history"] = prompt
            elif "用户:" in prompt:  # 如果是字符串上下文
                queries = re.findall(r"用户: (.*?)(?:\n|$)", prompt)
                if queries:
                    self.context_cache["history"].append((queries[-1], answer))

            return answer
        except Exception as e:
            print(f"LLM调用错误: {str(e)}")
            return "系统处理出错，请稍后再试"

    def clear_context(self):
        """清除上下文缓存"""
        self.context_cache = {
            "entities": set(),
            "last_relations": set(),
            "history": []
        }


if __name__ == "__main__":
    chatbot = KGRAG()
    while True:
        query = input("用户:").strip()
        if query.lower() in ["exit", "quit"]:
            break

        # 模拟对话历史
        context = chatbot.context_cache["history"] + [(query, "")]

        # 实体识别
        entities = chatbot.entity_linking(context)
        print(f"识别实体: {entities}")

        # 知识检索
        facts = []
        for entity, types in entities.items():
            for e_type in types:
                rels = chatbot.link_entity_rel(context, entity, e_type)
                facts.extend(chatbot.recall_facts(rels, e_type, entity))

        # 生成回答
        prompt = chatbot.format_prompt(context, facts)
        answer = chatbot.chat(prompt)
        print(f"助手: {answer}")