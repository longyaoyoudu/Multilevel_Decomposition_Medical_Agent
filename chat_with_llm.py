# ccoding = utf-8
import os
from question_classifier import *
from question_parser import *
from llm_server import *
from build_medicalgraph import *
import re

entity_parser = QuestionClassifier()

kg = MedicalGraph()
model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")

class KGRAG():
    def __init__(self):
        self.cn_dict = {
                "name":"名称",
                "desc":"疾病简介",
                "cause":"疾病病因",
                "prevent":"预防措施",
                "cure_department":"治疗科室",
                "cure_lasttime":"治疗周期",
                "cure_way":"治疗方式",
                "cured_prob":"治愈概率",
                "easy_get":"易感人群",
                "belongs_to":"所属科室",
                "common_drug":"常用药品",
                "do_eat":"宜吃",
                "drugs_of":"生产药品",
                "need_check":"诊断检查",
                "no_eat":"忌吃",
                "recommand_drug":"好评药品",
                "recommand_eat":"推荐食谱",
                "has_symptom":"症状",
                "acompany_with":"并发症",
                "Check":"诊断检查项目",
                "Department":"医疗科目",
                "Disease":"疾病",
                "Drug":"药品",
                "Food":"食物",
                "Producer":"在售药品",
                "Symptom":"疾病症状"
        }
        self.entity_rel_dict = {
            "check":["name", 'need_check'],
            "department":["name", 'belongs_to'],
            "disease":["prevent", "cure_way", "name", "cure_lasttime", "cured_prob", "cause", "cure_department", "desc", "easy_get", 'recommand_eat', 'no_eat', 'do_eat', "common_drug", 'drugs_of', 'recommand_drug', 'need_check', 'has_symptom', 'acompany_with', 'belongs_to'],
            "drug":["name", "common_drug", 'drugs_of', 'recommand_drug'],
            "food":["name"],
            "producer":["name"],
            "symptom":["name", 'has_symptom'],
        }
        return

    def entity_linking(self, query):
        return entity_parser.check_medical(query)

    def link_entity_rel(self, query, entity, entity_type):
        cate = [self.cn_dict.get(i) for i in self.entity_rel_dict.get(entity_type)]
        prompt = "请判定问题：{query}所提及的是{entity}的哪几个信息，请从{cate}中进行选择，并以列表形式返回。".format(query=query, entity=entity, cate=cate)
        answer, history = model.chat(query=prompt, history=[])
        cls_rel = set([i for i in re.split(r"[\[。、, ;'\]]", answer)]).intersection(set(cate))
        print([prompt, answer, cls_rel])
        return cls_rel

    def recall_facts(self, cls_rel, entity_type, entity_name, depth=1):
        entity_dict = {
            "check":"Check",
            "department":"Department",
            "disease":"Disease",
            "drug":"Drug",
            "food":"Food",
            "producer":"Producer",
            "symptom":"Symptom"
        }
        # "MATCH p=(m:Disease)-[r*..2]-(n) where m.name = '耳聋' return p "
        sql = "MATCH p=(m:{entity_type})-[r*..{depth}]-(n) where m.name = '{entity_name}' return p".format(depth=depth, entity_type=entity_dict.get(entity_type), entity_name=entity_name)
        print(sql)
        ress = kg.g.run(sql).data()
        triples = set()
        for res in ress:
            p_data = res["p"]
            nodes = p_data.nodes
            rels = p_data.relationships
            for node in nodes:
                node_name = node["name"]
                for k,v in node.items():
                    # print(k)
                    if v == node_name:
                        continue
                    if self.cn_dict[k] not in cls_rel:
                        continue
                    triples.add("<" + ','.join([str(node_name), str(self.cn_dict[k]), str(v)]) + ">")
            for rel in rels:
                if rel.start_node["name"] == rel.end_node["name"]:
                    continue
                # print(rel["name"])
                if rel["name"] not in cls_rel:
                    continue
                triples.add("<" + ','.join([str(rel.start_node["name"]), str(rel["name"]), str(rel.end_node["name"])]) + ">")
        print(len(triples), list(triples)[:3])
        return list(triples)


    def format_prompt(self, query, context):
        prompt = "这是一个关于医疗领域的问题。给定以下知识三元组集合，三元组形式为<subject, relation, object>，表示subject和object之间存在relation关系" \
                 "请先从这些三元组集合中找到能够支撑问题的部分，在这里叫做证据，并基于此回答问题。如果没有找到，那么直接回答没有找到证据，回答不知道，如果找到了，请先回答证据的内容，然后在给出最终答案" \
                 "知识三元组集合为：" + str(context) + "\n问题是：" + query + "\n请回答："
        return prompt

    # def chat(self, query):
    #     "{'耳聋': ['disease', 'symptom']}"
    #     print("step1: linking entity.....")
    #     entity_dict = self.entity_linking(query)
    #     depth = 1
    #     facts = list()
    #     answer = ""
    #     default = "抱歉，这个问题超出了我的知识范围，无法回答。"
    #     if not entity_dict:
    #         print("no entity founded...finished...")
    #         return default
    #     print("step2：recall kg facts....")
    #     for entity_name, types in entity_dict.items():
    #         for entity_type in types:
    #             rels = self.link_entity_rel(query, entity_name, entity_type)
    #             entity_triples = self.recall_facts(rels, entity_type, entity_name, depth)
    #             facts += entity_triples
    #     fact_prompt = self.format_prompt(query, facts)
    #     print("step3：generate answer...")
    #     answer = model.chat(query=fact_prompt, history=[])
    #     return answer

    # 动态调整召回深度
    def determine_recall_depth(self, query, entity_dict):
        """根据问题复杂度动态调整知识图谱查询深度"""
        # 简单问题：1跳关系
        if "症状" in query or "治疗" in query:
            return 1
        # 复杂问题：2跳关系
        elif "并发症" in query or "预防" in query:
            return 2
        # 探索性问题：3跳关系
        elif "原因" in query or "关联" in query:
            return 3
        return 1

    def build_enhanced_prompt(self, query, facts):
        """构建更有效的提示词模板"""
        if not facts:
            # 无知识时的备选方案
            return f"问题：{query}\n注意：知识库中未找到相关信息，请谨慎回答。"
        # 结构化展示知识
        knowledge_str = "\n".join([
            f"{idx + 1}. {fact}" for idx, fact in enumerate(facts)
        ])
        return f"""
    # 角色
    你是一位专业医疗问答助手，需要基于以下结构化知识回答问题

    # 知识库
    {knowledge_str}

    # 任务
    1. 严格根据知识库内容回答问题
    2. 如果知识不足，明确说明"根据现有知识无法回答"
    3. 避免编造信息

    # 问题
    {query}

    # 回答格式
    <思考过程>
    <最终答案>
    """




    def chat(self, query):
        print("【Step1】实体链接...")
        entity_dict = self.entity_linking(query)

        # 优化1：添加默认回答和日志
        default = "抱歉，这个问题超出了我的知识范围，无法回答。"
        if not entity_dict:
            print("⚠️ 未识别到相关医疗实体")
            return default

        print(f"✅ 识别到实体: {entity_dict}")

        # 优化2：动态调整召回深度
        depth = self.determine_recall_depth(query, entity_dict)
        print(f"📊 召回深度: {depth}")

        # 优化3：并行召回（可选）& 结果去重
        facts = set()
        print("【Step2】知识图谱召回...")
        for entity_name, types in entity_dict.items():
            for entity_type in types:
                rels = self.link_entity_rel(query, entity_name, entity_type)
                if not rels:
                    continue

                print(f"🔍 实体[{entity_name}-{entity_type}] 相关关系: {rels}")
                entity_triples = self.recall_facts(rels, entity_type, entity_name, depth)
                facts.update(entity_triples)  # 使用set自动去重

        # 优化4：结果截断（防止prompt过长）
        max_facts = 15
        facts_list = list(facts)
        if len(facts_list) > max_facts:
            print(f"⚠️ 三元组数量超过{max_facts}，进行截断")
            facts_list = facts_list[:max_facts]

        # 优化5：增强prompt工程
        print("【Step3】构建prompt...")
        fact_prompt = self.build_enhanced_prompt(query, facts_list)

        # 优化6：添加LLM调用保护
        print("【Step4】LLM生成答案...")
        try:
            answer, _ = model.chat(query=fact_prompt, history=[])
            return answer
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            return default

if __name__ == "__main__":
    chatbot = KGRAG()
    while 1:
        query = input("用户:").strip()
        answer = chatbot.chat(query)
        print("智能医生:", answer)
