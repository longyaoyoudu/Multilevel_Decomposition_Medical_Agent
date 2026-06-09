#!/usr/bin/env python3
# coding: utf-8
# File: llm_question_parser.py
# 基于 LLM 的问句解析器，使用零样本学习生成 Cypher 查询

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from llm_services.llm_server import ModelAPI


class LLMQuestionParser:
    """
    基于 LLM 的问句解析器
    直接根据用户问题生成 Cypher 查询语句
    """
    
    def __init__(self, model_url=None):
        # 知识图谱 schema 定义
        self.schema = {
            'nodes': {
                'Disease': '疾病',
                'Symptom': '症状',
                'Drug': '药品',
                'Food': '食物',
                'Check': '检查项目',
                'Department': '科室',
                'Producer': '药品厂商'
            },
            'relationships': {
                'has_symptom': {'from': 'Disease', 'to': 'Symptom', 'desc': '疾病有症状'},
                'acompany_with': {'from': 'Disease', 'to': 'Disease', 'desc': '疾病并发症'},
                'recommand_drug': {'from': 'Disease', 'to': 'Drug', 'desc': '疾病推荐药品'},
                'common_drug': {'from': 'Disease', 'to': 'Drug', 'desc': '疾病常用药品'},
                'recommand_eat': {'from': 'Disease', 'to': 'Food', 'desc': '疾病推荐食谱'},
                'do_eat': {'from': 'Disease', 'to': 'Food', 'desc': '疾病宜吃食物'},
                'no_eat': {'from': 'Disease', 'to': 'Food', 'desc': '疾病忌吃食物'},
                'need_check': {'from': 'Disease', 'to': 'Check', 'desc': '疾病所需检查'},
                'belongs_to': {'from': 'Disease', 'to': 'Department', 'desc': '科室从属'},
                'drugs_of': {'from': 'Drug', 'to': 'Producer', 'desc': '药品在售'}
            },
            'properties': {
                'Disease': [
                    {'name': 'name', 'desc': '疾病名称'},
                    {'name': 'desc', 'desc': '疾病简介'},
                    {'name': 'cause', 'desc': '疾病病因'},
                    {'name': 'prevent', 'desc': '预防措施'},
                    {'name': 'cure_lasttime', 'desc': '治疗周期'},
                    {'name': 'cure_way', 'desc': '治疗方式'},
                    {'name': 'cured_prob', 'desc': '治愈概率'},
                    {'name': 'easy_get', 'desc': '易感人群'}
                ]
            }
        }
        
        # 初始化 LLM 模型
        if model_url:
            self.model = ModelAPI(model_url)
        else:
            self.model = ModelAPI("http://127.0.0.1:11434/api/generate")
        
        # 构建提示模板
        self.cypher_prompt = self._build_cypher_prompt()
        
        print('LLM Question Parser init finished ......')
    
    def _build_cypher_prompt(self):
        """构建 Cypher 生成提示模板"""
        nodes_str = '\n'.join([f"- {k}: {v}" for k, v in self.schema['nodes'].items()])
        
        rels_str = '\n'.join([
            f"- {k}: {v['desc']} ({v['from']}->{v['to']})" 
            for k, v in self.schema['relationships'].items()
        ])
        
        props_str = '\n'.join([
            f"- {p['name']}: {p['desc']}" 
            for p in self.schema['properties']['Disease']
        ])
        
        prompt = f"""【角色】你是一个专业的 Cypher 查询生成器。
【任务】根据医疗问题，基于 schema 生成 Neo4j Cypher 查询。

【知识图谱 Schema】
节点类型：
{nodes_str}

关系类型：
{rels_str}

Disease 节点属性：
{props_str}

【核心示例】
1. 疾病→症状：MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) WHERE m.name='疾病名' RETURN m.name, n.name
2. 症状→疾病：MATCH (m:Disease)-[r:has_symptom]->(n:Symptom) WHERE n.name='症状名' RETURN m.name
3. 查询属性：MATCH (m:Disease) WHERE m.name='疾病名' RETURN m.属性名
4. 疾病→药品：MATCH (m:Disease)-[r:common_drug|recommand_drug]->(n:Drug) WHERE m.name='疾病名' RETURN n.name
5. 疾病→食物：MATCH (m:Disease)-[r:do_eat|recommand_eat|no_eat]->(n:Food) WHERE m.name='疾病名' RETURN n.name

【输出要求】
1. 只输出 JSON，不要其他文字
2. 格式：{{"cypher": "查询语句", "explanation": "简要解释", "confidence": 0.9}}
3. 无法生成时：cypher 为空，confidence 为 0
4. 节点标签和关系类型必须严格匹配 schema

【问题】
"""
        return prompt
    
    def parser_main(self, res_classify):
        """
        解析主函数，保持与原解析器相同的接口
        参数:
            res_classify: 分类器返回的结果，格式为:
            {
                'args': {实体1: [类型1, 类型2...], 实体2: [类型1...]},
                'question_types': [类型1, 类型2...]
            }
        返回:
            与原解析器相同格式的列表:
            [
                {'question_type': 'disease_symptom', 'sql': ['Cypher 查询语句']},
                ...
            ]
        """
        args = res_classify.get('args', {})
        question_types = res_classify.get('question_types', [])
        
        sqls = []
        
        # 如果有明确的问题类型，使用原解析器的逻辑
        if question_types:
            # 导入原解析器
            try:
                from qa_traditional.question_parser import QuestionPaser
                original_parser = QuestionPaser()
                return original_parser.parser_main(res_classify)
            except ImportError:
                pass
        
        # 如果没有明确的问题类型或原解析器不可用，使用 LLM 生成
        # 构建问题描述
        entities_str = ', '.join([f"{k}({','.join(v)})" for k, v in args.items()])
        question_desc = f"识别到的实体: {entities_str}"
        
        # 调用 LLM 生成 Cypher
        try:
            full_prompt = self.cypher_prompt + f"\n用户问题：{question_desc}"
            response, _ = self.model.chat(full_prompt, history=[])
            print(f"LLM Cypher Response: {response}")
            
            # 解析响应
            result = self._parse_llm_response(response)
            
            if result and result.get('cypher'):
                sqls.append({
                    'question_type': 'llm_generated',
                    'sql': [result['cypher']]
                })
        except Exception as e:
            print(f"LLM Parser Error: {e}")
        
        return sqls
    
    def generate_cypher_from_question(self, question):
        """
        直接从用户问题生成 Cypher 查询
        参数:
            question: 用户输入的问题
        返回:
            Cypher 查询语句
        """
        full_prompt = self.cypher_prompt + f"\n用户问题：{question}"
        
        try:
            response, _ = self.model.chat(full_prompt, history=[])
            print(f"LLM Cypher Response: {response}")
            
            result = self._parse_llm_response(response)
            
            if result and result.get('cypher'):
                return result['cypher']
        except Exception as e:
            print(f"LLM Cypher Generation Error: {e}")
        
        return None
    
    def _parse_llm_response(self, response):
        """解析 LLM 的响应，提取 JSON"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # 尝试从文本中提取 JSON
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 如果都失败了，返回 None
        return None


class HybridQuestionParser:
    """
    混合解析器：结合原解析器和 LLM 解析器
    优先使用原解析器处理常见问题，使用 LLM 处理复杂问题
    """
    
    def __init__(self, model_url=None):
        self.llm_parser = LLMQuestionParser(model_url)
        self.original_parser = None
        
        # 尝试导入原解析器
        try:
            from qa_traditional.question_parser import QuestionPaser
            self.original_parser = QuestionPaser()
        except ImportError:
            print("Warning: Original parser not available")
        
        print('Hybrid Question Parser init finished ......')
    
    def parser_main(self, res_classify):
        """
        解析主函数
        """
        # 首先尝试原解析器
        if self.original_parser:
            try:
                result = self.original_parser.parser_main(res_classify)
                if result:
                    return result
            except Exception as e:
                print(f"Original parser error: {e}")
        
        # 如果原解析器失败，尝试 LLM 解析器
        return self.llm_parser.parser_main(res_classify)


if __name__ == '__main__':
    # 测试示例
    test_questions = [
        "乳腺癌的症状有哪些？",
        "流鼻涕可能是什么病？",
        "为什么会失眠？",
        "失眠有哪些并发症？",
        "失眠的人不要吃啥？",
        "耳鸣了吃点啥？"
    ]
    
    print("=" * 50)
    print("Testing LLM Question Parser")
    print("=" * 50)
    
    try:
        parser = LLMQuestionParser(model_url="http://127.0.0.1:11434/api/generate")
        
        for question in test_questions[:3]:  # 只测试前3个
            print(f"\n问题: {question}")
            cypher = parser.generate_cypher_from_question(question)
            print(f"生成的 Cypher: {cypher}")
    except Exception as e:
        print(f"LLM Parser test failed: {e}")
        print("Please ensure LLM service is running.")
