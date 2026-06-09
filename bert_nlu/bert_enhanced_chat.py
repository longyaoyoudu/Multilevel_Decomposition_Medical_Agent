#!/usr/bin/env python3
# coding: utf-8
"""
BERT增强的双模型协作聊天系统
结合BERT的自然语言理解能力与现有的双模型协作模式

工作流程：
1. 预处理阶段：使用BERT进行意图识别和实体识别
2. 知识图谱查询阶段：使用识别到的实体和意图进行知识图谱查询
3. 答案生成阶段：使用双模型协作模式生成和优化答案
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
from utils.config import Config
from bert_nlu.config import BertConfig
from bert_nlu.bert_question_classifier import BertQuestionClassifier
from llm_services.unified_llm_service import (
    get_llm_provider,
    call_dual_model_api,
    call_siliconflow_api,
    call_minimax_api
)


class BertEnhancedChat:
    """
    BERT增强的双模型协作聊天系统
    
    特性：
    1. 使用BERT进行意图识别和实体识别
    2. 基于识别结果构建更精准的提示词
    3. 使用双模型协作模式生成和优化答案
    4. 支持知识图谱增强（可选）
    """
    
    def __init__(self, 
                 intent_model_path: Optional[str] = None,
                 entity_model_path: Optional[str] = None,
                 use_kg: bool = True):
        """
        初始化BERT增强的聊天系统
        
        Args:
            intent_model_path: 意图分类器模型路径
            entity_model_path: 实体识别器模型路径
            use_kg: 是否使用知识图谱增强
        """
        print("=" * 60)
        print("初始化BERT增强的双模型协作聊天系统")
        print("=" * 60)
        
        # 初始化BERT问题分类器
        print("\n[1/3] 初始化BERT NLU模块...")
        self.bert_classifier = BertQuestionClassifier(
            intent_model_path=intent_model_path,
            entity_model_path=entity_model_path
        )
        
        # 检查LLM提供商
        print("\n[2/3] 检查LLM配置...")
        self.llm_provider_info = get_llm_provider()
        self.llm_provider = self.llm_provider_info['provider']
        
        if self.llm_provider == 'none':
            print("⚠️  警告: 未配置任何LLM API Key")
            print("   请在 .env 文件中配置 SILICONFLOW_API_KEY 或 MINIMAX_API_KEY")
        elif self.llm_provider == 'dual_model':
            print("✅ 检测到双模型配置，已启用双模型协作模式")
            print("   工作模式: MiniMax 生成初步回答 + SiliconFlow 优化回答")
        elif self.llm_provider == 'siliconflow':
            print("✅ LLM 提供商: SiliconFlow (硅基流动)")
        elif self.llm_provider == 'minimax':
            print("✅ LLM 提供商: MiniMax")
        
        # 初始化知识图谱（可选）
        self.use_kg = use_kg
        self.kg = None
        self.kg_available = False
        
        if use_kg:
            print("\n[3/3] 初始化知识图谱...")
            try:
                from kg_builder.build_medicalgraph import MedicalGraph
                self.kg = MedicalGraph()
                self.kg_available = True
                print("✅ 知识图谱初始化成功")
            except Exception as e:
                print(f"⚠️  知识图谱初始化失败: {e}")
                print("   将使用纯LLM模式")
                self.use_kg = False
        
        # 意图类型到中文描述的映射
        self.intent_descriptions = {
            'disease_symptom': '查询疾病症状',
            'symptom_disease': '根据症状查疾病',
            'disease_cause': '查询疾病病因',
            'disease_acompany': '查询疾病并发症',
            'disease_do_food': '查询疾病宜吃食物',
            'disease_not_food': '查询疾病忌吃食物',
            'food_do_disease': '查询食物能治疗的疾病',
            'food_not_disease': '查询食物不能治疗的疾病',
            'disease_drug': '查询疾病常用药品',
            'drug_disease': '查询药品能治疗的疾病',
            'disease_check': '查询疾病需要做的检查',
            'check_disease': '查询检查能诊断的疾病',
            'disease_prevent': '查询疾病预防措施',
            'disease_lasttime': '查询疾病治疗周期',
            'disease_cureway': '查询疾病治疗方式',
            'disease_cureprob': '查询疾病治愈概率',
            'disease_easyget': '查询疾病易感人群',
            'disease_desc': '查询疾病简介',
            'others': '其他问题'
        }
        
        # 实体类型到中文描述的映射
        self.entity_type_descriptions = {
            'disease': '疾病',
            'symptom': '症状',
            'drug': '药品',
            'food': '食物',
            'check': '检查项目',
            'department': '科室',
            'producer': '药品厂商'
        }
        
        print("\n" + "=" * 60)
        print("BERT增强的双模型协作聊天系统初始化完成")
        print("=" * 60)
    
    def process_question(self, question: str) -> Dict:
        """
        处理用户问题的完整流程
        
        Args:
            question: 用户输入的问题
            
        Returns:
            包含处理结果的字典
        """
        print("\n" + "=" * 60)
        print(f"处理用户问题: {question}")
        print("=" * 60)
        
        # Step 1: 使用BERT进行意图识别和实体识别
        print("\n[Step 1] BERT自然语言理解...")
        nlu_result = self.bert_classifier.classify(question)
        
        entities = nlu_result.get('args', {})
        intents = nlu_result.get('question_types', [])
        
        print(f"  识别到的实体: {entities}")
        print(f"  识别到的意图: {intents}")
        
        # Step 2: 构建增强的提示词
        print("\n[Step 2] 构建增强提示词...")
        enhanced_prompt = self._build_enhanced_prompt(question, entities, intents)
        print(f"  增强提示词构建完成，长度: {len(enhanced_prompt)} 字符")
        
        # Step 3: 知识图谱查询（如果启用）
        kg_context = ""
        if self.use_kg and self.kg_available and entities:
            print("\n[Step 3] 知识图谱查询...")
            kg_context = self._query_knowledge_graph(entities, intents)
            if kg_context:
                print(f"  知识图谱查询成功，获取到 {len(kg_context)} 字符的上下文")
                # 将知识图谱上下文添加到提示词中
                enhanced_prompt = self._add_kg_context(enhanced_prompt, kg_context)
            else:
                print("  知识图谱未找到相关信息")
        else:
            print("\n[Step 3] 跳过知识图谱查询（未启用或无实体）")
        
        # Step 4: 使用双模型协作生成答案
        print("\n[Step 4] 双模型协作生成答案...")
        answer = self._generate_answer(enhanced_prompt, question)
        
        print("\n" + "=" * 60)
        print("处理完成")
        print("=" * 60)
        
        return {
            'original_question': question,
            'entities': entities,
            'intents': intents,
            'kg_context': kg_context,
            'answer': answer,
            'llm_provider': self.llm_provider
        }
    
    def _build_enhanced_prompt(self, question: str, entities: Dict, intents: List[str]) -> str:
        """
        基于BERT识别结果构建增强的提示词
        
        Args:
            question: 原始问题
            entities: 识别到的实体
            intents: 识别到的意图
            
        Returns:
            增强的提示词
        """
        # 构建实体描述
        entity_description = ""
        if entities:
            entity_description = "**识别到的医疗实体**：\n"
            for entity_text, entity_types in entities.items():
                type_descriptions = [self.entity_type_descriptions.get(t, t) for t in entity_types]
                entity_description += f"- {entity_text}: {', '.join(type_descriptions)}\n"
        
        # 构建意图描述
        intent_description = ""
        if intents:
            intent_description = "**识别到的用户意图**：\n"
            for intent in intents:
                intent_description += f"- {self.intent_descriptions.get(intent, intent)}\n"
        
        # 构建系统提示词
        system_prompt = """【角色】你是一位专业、严谨的医疗问答助手。

【核心规则】
1. 只基于提供的信息回答，不编造、不猜测
2. 如果信息中没有答案，直接回答："抱歉，我暂时无法回答这个问题。建议您咨询专业医生。"
3. 回答简洁、准确、专业，避免冗余
4. 对于医疗问题，务必提醒用户："以上建议仅供参考，如有不适请及时就医"

【回答要求】
- 使用清晰的结构组织回答
- 分点列出重要信息
- 语言通俗易懂，但保持专业性
- 避免使用过于专业的术语，必要时进行解释"""
        
        # 构建完整的提示词
        full_prompt = system_prompt
        
        if entity_description:
            full_prompt += "\n\n" + entity_description
        
        if intent_description:
            full_prompt += "\n" + intent_description
        
        # 添加用户问题
        full_prompt += f"\n\n【用户问题】\n{question}\n\n【回答】"
        
        return full_prompt
    
    def _query_knowledge_graph(self, entities: Dict, intents: List[str]) -> str:
        """
        查询知识图谱获取相关信息
        
        Args:
            entities: 识别到的实体
            intents: 识别到的意图
            
        Returns:
            知识图谱上下文信息
        """
        if not self.kg or not self.kg_available:
            return ""
        
        context_parts = []
        
        # 意图到知识图谱关系的映射
        intent_relation_map = {
            'disease_symptom': 'has_symptom',
            'disease_cause': 'cause',
            'disease_acompany': 'acompany_with',
            'disease_drug': ['common_drug', 'recommand_drug'],
            'disease_do_food': ['do_eat', 'recommand_eat'],
            'disease_not_food': 'no_eat',
            'disease_check': 'need_check',
            'disease_prevent': 'prevent',
            'disease_lasttime': 'cure_lasttime',
            'disease_cureway': 'cure_way',
            'disease_cureprob': 'cured_prob',
            'disease_easyget': 'easy_get',
            'disease_desc': 'desc',
            'symptom_disease': 'has_symptom',
            'drug_disease': ['common_drug', 'recommand_drug'],
            'check_disease': 'need_check',
        }
        
        # 遍历所有实体
        for entity_text, entity_types in entities.items():
            for entity_type in entity_types:
                # 根据意图查询知识图谱
                for intent in intents:
                    relations = intent_relation_map.get(intent)
                    if not relations:
                        continue
                    
                    if isinstance(relations, str):
                        relations = [relations]
                    
                    for relation in relations:
                        try:
                            # 构建Cypher查询
                            if entity_type == 'disease':
                                # 查询疾病的属性或关系
                                if relation in ['desc', 'cause', 'prevent', 'cure_lasttime', 'cure_way', 'cured_prob', 'easy_get']:
                                    # 属性查询
                                    cypher = f"""
                                    MATCH (m:Disease) 
                                    WHERE m.name = '{entity_text}' 
                                    RETURN m.name as name, m.{relation} as result
                                    """
                                else:
                                    # 关系查询
                                    cypher = f"""
                                    MATCH (m:Disease)-[r:{relation}]->(n) 
                                    WHERE m.name = '{entity_text}' 
                                    RETURN m.name as source, r.name as relation, n.name as target
                                    """
                            elif entity_type == 'symptom' and intent == 'symptom_disease':
                                # 根据症状查疾病
                                cypher = f"""
                                MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) 
                                WHERE n.name = '{entity_text}' 
                                RETURN m.name as disease, n.name as symptom
                                """
                            elif entity_type == 'drug' and intent == 'drug_disease':
                                # 根据药品查疾病
                                cypher = f"""
                                MATCH (m:Disease)-[r:common_drug|recommand_drug]->(n:Drug) 
                                WHERE n.name = '{entity_text}' 
                                RETURN m.name as disease, n.name as drug
                                """
                            elif entity_type == 'check' and intent == 'check_disease':
                                # 根据检查查疾病
                                cypher = f"""
                                MATCH (m:Disease)-[r:need_check]->(n:Check) 
                                WHERE n.name = '{entity_text}' 
                                RETURN m.name as disease, n.name as check
                                """
                            else:
                                continue
                            
                            # 执行查询
                            result = self.kg.g.run(cypher).data()
                            
                            if result:
                                for record in result:
                                    if 'result' in record and record['result']:
                                        context_parts.append(
                                            f"- {record['name']}的{self._get_relation_name(relation)}：{record['result']}"
                                        )
                                    elif 'target' in record:
                                        context_parts.append(
                                            f"- {record['source']} {self._get_relation_name(relation)} {record['target']}"
                                        )
                                    elif 'disease' in record:
                                        if 'symptom' in record:
                                            context_parts.append(
                                                f"- 有{record['symptom']}症状的疾病可能是：{record['disease']}"
                                            )
                                        elif 'drug' in record:
                                            context_parts.append(
                                                f"- {record['drug']}可以治疗：{record['disease']}"
                                            )
                                        elif 'check' in record:
                                            context_parts.append(
                                                f"- {record['check']}可以检查出：{record['disease']}"
                                            )
                        
                        except Exception as e:
                            print(f"    知识图谱查询失败: {e}")
                            continue
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _get_relation_name(self, relation: str) -> str:
        """
        获取关系的中文名称
        """
        relation_names = {
            'has_symptom': '的症状是',
            'acompany_with': '的并发症是',
            'common_drug': '的常用药品是',
            'recommand_drug': '的推荐药品是',
            'do_eat': '宜吃',
            'recommand_eat': '推荐吃',
            'no_eat': '忌吃',
            'need_check': '需要做的检查是',
            'belongs_to': '属于',
            'desc': '简介',
            'cause': '病因',
            'prevent': '预防措施',
            'cure_lasttime': '治疗周期',
            'cure_way': '治疗方式',
            'cured_prob': '治愈概率',
            'easy_get': '易感人群'
        }
        return relation_names.get(relation, relation)
    
    def _add_kg_context(self, prompt: str, kg_context: str) -> str:
        """
        将知识图谱上下文添加到提示词中
        
        Args:
            prompt: 原始提示词
            kg_context: 知识图谱上下文
            
        Returns:
            增强后的提示词
        """
        # 在系统提示词和用户问题之间插入知识图谱上下文
        kg_section = f"\n\n【知识图谱参考信息】\n{kg_context}"
        
        # 找到【用户问题】的位置并插入
        user_question_marker = "【用户问题】"
        if user_question_marker in prompt:
            parts = prompt.split(user_question_marker)
            return parts[0] + kg_section + "\n\n" + user_question_marker + parts[1]
        else:
            return prompt + kg_section
    
    def _generate_answer(self, prompt: str, original_question: str) -> str:
        """
        使用双模型协作生成答案
        
        Args:
            prompt: 增强的提示词
            original_question: 原始问题（用于日志）
            
        Returns:
            生成的答案
        """
        # 构建请求数据
        data = {
            "message": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        try:
            if self.llm_provider == 'dual_model':
                print("  使用双模型协作模式...")
                return call_dual_model_api(data)
            elif self.llm_provider == 'siliconflow':
                print("  使用SiliconFlow...")
                return call_siliconflow_api(data)
            elif self.llm_provider == 'minimax':
                print("  使用MiniMax...")
                return call_minimax_api(data)
            else:
                return "抱歉，未配置任何LLM服务，无法回答您的问题。"
        except Exception as e:
            print(f"  LLM调用失败: {e}")
            return "抱歉，生成回答时出现错误，请稍后再试。"
    
    def chat(self, question: str) -> str:
        """
        简单的聊天接口
        
        Args:
            question: 用户问题
            
        Returns:
            回答
        """
        result = self.process_question(question)
        return result['answer']
    
    def interactive_chat(self):
        """
        交互式聊天模式
        """
        print("\n" + "=" * 60)
        print("BERT增强的双模型协作聊天系统")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n用户: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', 'e']:
                    print("\n感谢使用，再见！")
                    break
                
                if not question:
                    continue
                
                answer = self.chat(question)
                print(f"\n智能医生: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"\n错误: {e}")


def main():
    """
    主函数：演示如何使用BERT增强的双模型协作聊天系统
    """
    print("=" * 60)
    print("BERT增强的双模型协作聊天系统演示")
    print("=" * 60)
    
    # 创建聊天系统实例
    chat_system = BertEnhancedChat(use_kg=False)  # 先禁用知识图谱，避免依赖问题
    
    # 测试问题
    test_questions = [
        "感冒有什么症状？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "怎么预防感冒？",
        "失眠是什么？",
    ]
    
    print("\n测试BERT增强的问答功能...")
    print("-" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        
        # 只使用BERT进行NLU分析（不调用LLM）
        nlu_result = chat_system.bert_classifier.classify(question)
        print(f"  实体: {nlu_result.get('args', {})}")
        print(f"  意图: {nlu_result.get('question_types', [])}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n使用方法：")
    print("-" * 60)
    print("1. 创建聊天系统实例：")
    print("   chat_system = BertEnhancedChat(use_kg=True)")
    print()
    print("2. 处理单个问题：")
    print("   result = chat_system.process_question('感冒有什么症状？')")
    print("   print(result['answer'])")
    print()
    print("3. 简单聊天：")
    print("   answer = chat_system.chat('感冒有什么症状？')")
    print()
    print("4. 交互式聊天：")
    print("   chat_system.interactive_chat()")


if __name__ == '__main__':
    main()
