# coding=utf-8
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_traditional.question_classifier import *
from qa_traditional.question_parser import *
from llm_services.llm_server import *
from kg_builder.build_medicalgraph import *
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
        if isinstance(context, list):
            last_query = context[-1][0] if context else ""
        else:
            queries = re.findall(r"用户: (.*?)(?:\n|$)", context)
            last_query = queries[-1] if queries else ""

        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type, [])]

        prompt = f"""【角色】你是一个精确的医疗关系分类器。
【任务】从给定选项中选择问题涉及的信息类型。
【上下文摘要】{str(context)[:300] if context else "无"}
【当前问题】{last_query}
【实体】{entity}
【可选信息类型】{cate}
【输出要求】
1. 只返回 JSON 格式，不要其他文字
2. 格式：{{"selected": ["类型1", "类型2"]}}
3. 如果不确定，返回 {{"selected": []}}"""

        answer, _ = model.chat(query=prompt, history=[])
        print(f"[关系分类] 实体: {entity}, 模型返回: {answer[:100] if answer else '空'}")
        
        try:
            import json
            result = json.loads(answer)
            cls_rel = set(result.get("selected", []))
        except:
            cls_rel = set(re.findall(r"['\"](.*?)['\"]", answer))
            if not cls_rel:
                cls_rel = set([i.strip() for i in re.split(r"[\[。、, ;'\]\"']", answer) if i.strip()])
        
        cls_rel = cls_rel.intersection(set(cate))

        if self.context_cache["last_relations"]:
            cls_rel.update(self.context_cache["last_relations"])

        print(f"[关系分类] 最终选中: {cls_rel}")
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
        if isinstance(context, list):
            history_str = "\n".join([f"用户: {q}\n助手: {a}" for q, a in context])
            last_query = context[-1][0] if context else ""
        else:
            history_str = context
            last_query = re.findall(r"用户: (.*?)(?:\n|$)", context)[-1] if "用户:" in context else context

        knowledge_str = "\n".join([f"- {fact}" for fact in facts[:15]]) if facts else "无相关知识"

        return f"""【角色】你是一位专业、严谨的医疗问答助手。
【核心规则】
1. 只基于提供的知识回答，不编造、不猜测
2. 如果知识中没有答案，回答："抱歉，知识库中没有相关信息。"
3. 回答简洁、准确、专业
4. 考虑对话上下文进行连贯回答

【对话历史】
{history_str}

【相关知识】
{knowledge_str}

【当前问题】{last_query}

【回答】"""

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