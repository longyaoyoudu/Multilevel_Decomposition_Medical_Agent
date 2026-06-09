# ccoding = utf-8
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_traditional.question_classifier import *
from qa_traditional.question_parser import *
from llm_services.llm_server import *
from kg_builder.build_medicalgraph import *
from utils.config import Config
from qa_llm_enhanced.hybrid_retriever import HybridRetriever
from qa_llm_enhanced.reranker import Reranker
import re

entity_parser = QuestionClassifier()

kg = MedicalGraph()
model = ModelAPI(MODEL_URL=Config.LLM_SERVICE_URL)

class KGRAG():
    def __init__(self):
        # 多路召回器（延迟初始化）
        self.hybrid_retriever = None
        # Reranker重排序器（延迟初始化）
        self.reranker = None

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
        prompt = """【角色】你是一个精确的医疗关系分类器。
【任务】从给定选项中选择问题涉及的信息类型。
【问题】{query}
【实体】{entity}
【可选信息类型】{cate}
【输出要求】
1. 只返回 JSON 格式，不要其他文字
2. 格式：{{"selected": ["类型1", "类型2"]}}
3. 如果不确定，返回 {{"selected": []}}""".format(query=query, entity=entity, cate=cate)
        answer, history = model.chat(query=prompt, history=[])
        print(f"[关系分类] Prompt: {prompt[:200]}...")
        print(f"[关系分类] 模型返回: {answer}")
        try:
            import json
            result = json.loads(answer)
            cls_rel = set(result.get("selected", []))
        except:
            cls_rel = set([i.strip() for i in re.split(r"[\[。、, ;'\]\"']", answer) if i.strip()])
        cls_rel = cls_rel.intersection(set(cate))
        print(f"[关系分类] 最终选中: {cls_rel}")
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
        knowledge_str = "\n".join([f"- {fact}" for fact in context]) if context else "无相关知识"
        prompt = f"""【角色】你是一位专业、严谨的医疗问答助手。
【核心规则】
1. 只基于提供的知识回答，不编造、不猜测
2. 如果知识中没有答案，直接回答："抱歉，知识库中没有相关信息。"
3. 回答简洁准确，避免冗余

【知识】
{knowledge_str}

【问题】{query}

【回答】"""
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
            return f"""【角色】你是一位专业的医疗问答助手。
【限制】知识库未找到相关信息，请基于通用医学知识谨慎回答。
【要求】
1. 回答简洁、准确、专业
2. 明确说明："[信息来源：通用医学知识]"
3. 建议用户咨询专业医生

【问题】{query}

【回答】"""
        
        knowledge_str = "\n".join([f"{idx + 1}. {fact}" for idx, fact in enumerate(facts)])
        
        return f"""【角色】你是一位专业、严谨的医疗问答助手。
【核心规则】
1. 只基于知识库回答，禁止编造信息
2. 如果知识库没有答案，回答："抱歉，知识库中没有相关信息。"
3. 回答简洁、准确、专业，避免冗余

【知识库】
{knowledge_str}

【问题】{query}

【回答】"""

    def polish_answer(self, query, raw_answer):
        """
        使用LLM对原始回答进行润色优化

        Args:
            query: 用户问题
            raw_answer: 原始回答

        Returns:
            str: 润色后的回答
        """
        if not raw_answer or raw_answer.startswith("抱歉"):
            return raw_answer

        polish_prompt = f"""【角色】你是一位专业、温暖、耐心的医疗问答助手。
【任务】将以下回答润色得更自然、流畅、易读。
【原始回答】
{raw_answer}

【润色要求】
1. 保持医学专业性，确保信息准确
2. 语言自然流畅，像医生与患者对话
3. 可以适当添加过渡词使回答更连贯
4. 保留关键医学信息，可适当精简冗余
5. 如果原回答是列表形式，可以转为自然段落

【润色后的回答】"""

        try:
            polished, _ = model.chat(query=polish_prompt, history=[])
            print(f"[答案优化] 原始长度: {len(raw_answer)}, 优化后: {len(polished)}")
            return polished
        except Exception as e:
            print(f"[答案优化] 润色失败，使用原始回答: {str(e)}")
            return raw_answer

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

    def _get_hybrid_retriever(self):
        """延迟获取多路召回器"""
        if self.hybrid_retriever is None:
            self.hybrid_retriever = HybridRetriever()
        return self.hybrid_retriever

    def _get_reranker(self):
        """延迟获取Reranker"""
        if self.reranker is None:
            self.reranker = Reranker()
        return self.reranker

    def chat_v2(self, query):
        """
        基于多路召回+Reranker的新版对话流程

        Args:
            query: 用户问题

        Returns:
            str: LLM生成的答案
        """
        print("\n" + "=" * 60)
        print("[KGRAG.chat_v2] 新版对话流程 - 多路召回 + Reranker")
        print("=" * 60)

        default_answer = "抱歉，这个问题超出了我的知识范围，无法回答。"

        # Step 1: 多路召回
        print("\n【Step 1】多路召回...")
        try:
            hybrid = self._get_hybrid_retriever()
            candidates = hybrid.retrieve(query, use_vector=True, use_graph=True)
        except Exception as e:
            print(f"❌ 多路召回失败: {str(e)}")
            return default_answer

        if not candidates:
            print("⚠️ 多路召回未返回候选")
            return default_answer

        print(f"✅ 多路召回返回 {len(candidates)} 条候选")

        # Step 2: Reranker重排序
        print("\n【Step 2】Reranker重排序...")
        try:
            reranker = self._get_reranker()
            reranked = reranker.rerank(query, candidates)
        except Exception as e:
            print(f"❌ Reranker失败: {str(e)}")
            return default_answer

        if not reranked:
            print("⚠️ Reranker未返回结果")
            return default_answer

        print(f"✅ Reranker返回 {len(reranked)} 条精排结果")

        # Step 3: 构建Prompt
        print("\n【Step 3】构建Prompt...")
        facts = [r['triple'] for r in reranked]
        fact_prompt = self.build_enhanced_prompt(query, facts)

        # Step 4: LLM生成答案
        print("\n【Step 4】LLM生成答案...")
        try:
            answer, _ = model.chat(query=fact_prompt, history=[])
            print(f"✅ 答案生成成功，长度: {len(answer)} 字")
            return answer
        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")
            return default_answer

if __name__ == "__main__":
    print("=" * 60)
    print("医疗问答系统 - KGRAG")
    print("=" * 60)
    print("1. 旧版流程 (chat) - 单一图谱召回")
    print("2. 新版流程 (chat_v2) - 多路召回 + Reranker")
    print("=" * 60)

    chatbot = KGRAG()
    version = input("请选择对话流程 (1/2，默认2): ").strip() or "2"

    while 1:
        query = input("\n用户: ").strip()
        if not query:
            continue
        if query.lower() in ['quit', 'exit', '退出']:
            print("感谢使用！")
            break

        if version == "1":
            print("\n[使用旧版流程]")
            answer = chatbot.chat(query)
        else:
            print("\n[使用新版流程 - 多路召回 + Reranker]")
            answer = chatbot.chat_v2(query)

        print(f"智能医生: {answer}")
