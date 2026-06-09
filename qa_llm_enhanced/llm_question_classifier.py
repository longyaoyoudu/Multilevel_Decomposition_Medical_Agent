#!/usr/bin/env python3
# coding: utf-8
# File: llm_question_classifier.py
# 基于 LLM 的问句分类器，使用零样本学习实现实体识别和问句分类

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from llm_services.llm_server import ModelAPI


class LLMQuestionClassifier:
    def __init__(self, model_url=None):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        
        # 实体类型定义
        self.entity_types = {
            'disease': '疾病',
            'symptom': '症状',
            'drug': '药品',
            'food': '食物',
            'check': '检查项目',
            'department': '科室',
            'producer': '药品厂商'
        }
        
        # 问句类型定义
        self.question_types = {
            'disease_symptom': '疾病症状查询',
            'symptom_disease': '症状对应疾病查询',
            'disease_cause': '疾病原因查询',
            'disease_acompany': '疾病并发症查询',
            'disease_not_food': '疾病忌口查询',
            'disease_do_food': '疾病宜吃食物查询',
            'food_not_disease': '食物忌口疾病查询',
            'food_do_disease': '食物宜吃疾病查询',
            'disease_drug': '疾病用药查询',
            'drug_disease': '药品治疗疾病查询',
            'disease_check': '疾病检查项目查询',
            'check_disease': '检查项目对应疾病查询',
            'disease_prevent': '疾病预防查询',
            'disease_lasttime': '疾病治疗周期查询',
            'disease_cureway': '疾病治疗方式查询',
            'disease_cureprob': '疾病治愈概率查询',
            'disease_easyget': '疾病易感人群查询',
            'disease_desc': '疾病描述查询'
        }
        
        # 初始化 LLM 模型
        if model_url:
            self.model = ModelAPI(model_url)
        else:
            # 默认使用本地 Ollama 或其他服务
            self.model = ModelAPI("http://127.0.0.1:11434/api/generate")
        
        # 构建提示模板
        self.classification_prompt = self._build_classification_prompt()
        
        print('LLM Question Classifier init finished ......')
    
    def _build_classification_prompt(self):
        """构建分类提示模板"""
        entity_types_str = '\n'.join([f"- {k}: {v}" for k, v in self.entity_types.items()])
        question_types_str = '\n'.join([f"- {k}: {v}" for k, v in self.question_types.items()])
        
        prompt = f"""你是一个专业的医疗领域问答系统助手。请分析用户的问题，识别其中的医疗实体，并判断问题类型。

## 实体类型定义：
{entity_types_str}

## 问题类型定义：
{question_types_str}

## 输出格式要求：
请以严格的 JSON 格式输出，格式如下：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型"}},
        ...
    ],
    "question_types": ["问题类型1", "问题类型2", ...],
    "deny": false  // 是否包含否定词，如"不要"、"不能"等
}}

## 注意事项：
1. 只输出 JSON 格式，不要有其他解释性文字
2. 实体类型必须从上面定义的类型中选择
3. 问题类型必须从上面定义的类型中选择
4. 如果问题中包含"不要"、"不能"、"忌吃"等否定词，deny 设为 true
5. 如果无法识别实体或问题类型，entities 或 question_types 可以为空数组

现在请分析以下问题：
"""
        return prompt
    
    def classify(self, question):
        """
        分类主函数
        参数:
            question: 用户输入的问题
        返回:
            与原分类器相同格式的字典:
            {
                'args': {实体1: [类型1, 类型2...], 实体2: [类型1...]},
                'question_types': [类型1, 类型2...]
            }
        """
        data = {}
        
        # 构建完整提示
        full_prompt = self.classification_prompt + f"\n用户问题：{question}"
        
        # 调用 LLM
        try:
            response, _ = self.model.chat(full_prompt, history=[])
            print(f"LLM Response: {response}")
            
            # 解析 JSON 响应
            result = self._parse_llm_response(response)
            
            if not result:
                return {}
            
            # 转换为原分类器格式
            args = {}
            for entity in result.get('entities', []):
                name = entity.get('name', '')
                entity_type = entity.get('type', '')
                if name and entity_type:
                    if name not in args:
                        args[name] = []
                    if entity_type not in args[name]:
                        args[name].append(entity_type)
            
            # 处理否定词，调整问题类型
            question_types = result.get('question_types', [])
            deny = result.get('deny', False)
            
            # 如果有否定词，调整相关问题类型
            if deny:
                adjusted_types = []
                for qt in question_types:
                    if qt == 'disease_do_food':
                        adjusted_types.append('disease_not_food')
                    elif qt == 'food_do_disease':
                        adjusted_types.append('food_not_disease')
                    else:
                        adjusted_types.append(qt)
                question_types = adjusted_types
            
            data['args'] = args
            data['question_types'] = question_types
            
            return data
            
        except Exception as e:
            print(f"LLM Classification Error: {e}")
            return {}
    
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


class HybridQuestionClassifier:
    """
    混合分类器：结合 LLM 和原规则分类器
    优先使用 LLM，当 LLM 无法识别时回退到规则分类器
    """
    
    def __init__(self, model_url=None):
        self.llm_classifier = LLMQuestionClassifier(model_url)
        self.rule_classifier = None
        
        # 尝试导入原规则分类器
        try:
            from qa_traditional.question_classifier import QuestionClassifier
            self.rule_classifier = QuestionClassifier()
        except ImportError:
            print("Warning: Rule classifier not available")
        
        print('Hybrid Question Classifier init finished ......')
    
    def classify(self, question):
        """
        分类主函数：优先使用 LLM，失败时回退到规则分类器
        """
        # 首先尝试 LLM 分类
        result = self.llm_classifier.classify(question)
        
        # 如果 LLM 分类失败或结果为空，尝试规则分类器
        if not result or not result.get('args') or not result.get('question_types'):
            if self.rule_classifier:
                print("Falling back to rule classifier...")
                result = self.rule_classifier.classify(question)
        
        return result


if __name__ == '__main__':
    # 测试示例
    test_questions = [
        "乳腺癌的症状有哪些？",
        "流鼻涕可能是什么病？",
        "为什么会失眠？",
        "失眠有哪些并发症？",
        "失眠的人不要吃啥？",
        "耳鸣了吃点啥？",
        "什么病最好不要吃蜂蜜？",
        "鹅肉有什么好处？",
        "肝病要吃啥药？",
        "板蓝根颗粒能治啥病？",
        "脑膜炎怎么检查？",
        "全血细胞计数能查出啥？",
        "怎样才能预防肾虚？",
        "感冒要多久才能好？",
        "高血压要怎么治？",
        "白血病能治好吗？",
        "什么人容易得高血压？",
        "糖尿病是什么病？"
    ]
    
    # 测试 LLM 分类器
    print("=" * 50)
    print("Testing LLM Question Classifier")
    print("=" * 50)
    
    # 注意：这里需要根据实际情况修改 model_url
    # 如果没有可用的 LLM 服务，可以跳过测试
    try:
        classifier = LLMQuestionClassifier(model_url="http://127.0.0.1:11434/api/generate")
        
        for question in test_questions[:3]:  # 只测试前3个，节省时间
            print(f"\n问题: {question}")
            result = classifier.classify(question)
            print(f"分类结果: {result}")
    except Exception as e:
        print(f"LLM Classifier test failed: {e}")
        print("Please ensure LLM service is running or use HybridClassifier.")
