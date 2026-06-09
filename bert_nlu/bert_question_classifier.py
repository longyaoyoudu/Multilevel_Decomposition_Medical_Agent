#!/usr/bin/env python3
# coding: utf-8
"""
集成的BERT问题分类器
结合意图识别和实体识别，保持与现有系统兼容的接口
"""

import os
from typing import Dict, List, Optional, Tuple

from config import BertConfig
from intent_classifier import BertIntentClassifier
from entity_recognizer import BertEntityRecognizer


class BertQuestionClassifier:
    """
    基于BERT的问题分类器
    接口与原有的QuestionClassifier保持兼容，可以直接替换
    """
    
    def __init__(self, 
                 intent_model_path: Optional[str] = None,
                 entity_model_path: Optional[str] = None,
                 config: Optional[BertConfig] = None):
        """
        初始化BERT问题分类器
        
        Args:
            intent_model_path: 意图分类器模型路径，如果为None则使用未训练的模型
            entity_model_path: 实体识别器模型路径，如果为None则使用未训练的模型
            config: 配置对象
        """
        self.config = config or BertConfig()
        
        # 初始化意图分类器
        print("初始化意图分类器...")
        if intent_model_path and os.path.exists(intent_model_path):
            self.intent_classifier = BertIntentClassifier.load(intent_model_path)
        else:
            self.intent_classifier = BertIntentClassifier(config=config)
        
        # 初始化实体识别器
        print("初始化实体识别器...")
        if entity_model_path and os.path.exists(entity_model_path):
            self.entity_recognizer = BertEntityRecognizer.load(entity_model_path)
        else:
            self.entity_recognizer = BertEntityRecognizer(config=config)
        
        # 加载字典用于后备匹配（与原分类器保持兼容）
        self._load_backup_dictionaries()
        
        print("=" * 50)
        print("BERT问题分类器初始化完成")
        print("=" * 50)
    
    def _load_backup_dictionaries(self):
        """
        加载后备字典，当BERT模型未训练时使用规则匹配
        与原有的QuestionClassifier使用相同的字典
        """
        dict_dir = self.config.get_dict_dir()
        
        # 字典文件映射
        dict_files = {
            'disease': 'disease.txt',
            'symptom': 'symptom.txt',
            'drug': 'drug.txt',
            'food': 'food.txt',
            'check': 'check.txt',
            'department': 'department.txt',
            'producer': 'producer.txt'
        }
        
        self.dictionaries = {}
        self.all_entities = set()
        self.entity_type_dict = {}
        
        for entity_type, filename in dict_files.items():
            file_path = os.path.join(dict_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    entities = [line.strip() for line in f if line.strip()]
                self.dictionaries[entity_type] = entities
                self.all_entities.update(entities)
                
                for entity in entities:
                    if entity not in self.entity_type_dict:
                        self.entity_type_dict[entity] = []
                    self.entity_type_dict[entity].append(entity_type)
            else:
                print(f"警告: 字典文件不存在 {file_path}")
                self.dictionaries[entity_type] = []
        
        # 否定词
        deny_path = os.path.join(dict_dir, 'deny.txt')
        if os.path.exists(deny_path):
            with open(deny_path, 'r', encoding='utf-8') as f:
                self.deny_words = [line.strip() for line in f if line.strip()]
        else:
            self.deny_words = []
        
        # 疑问词模板（用于规则匹配）
        self.symptom_qwds = ['症状', '表征', '现象', '症候', '表现']
        self.cause_qwds = ['原因','成因', '为什么', '怎么会', '怎样才', '咋样才', '怎样会', '如何会', '为啥', '为何', '如何才会', '怎么才会', '会导致', '会造成']
        self.acompany_qwds = ['并发症', '并发', '一起发生', '一并发生', '一起出现', '一并出现', '一同发生', '一同出现', '伴随发生', '伴随', '共现']
        self.food_qwds = ['饮食', '饮用', '吃', '食', '伙食', '膳食', '喝', '菜' ,'忌口', '补品', '保健品', '食谱', '菜谱', '食用', '食物','补品']
        self.drug_qwds = ['药', '药品', '用药', '胶囊', '口服液', '炎片']
        self.prevent_qwds = ['预防', '防范', '抵制', '抵御', '防止','躲避','逃避','避开','免得','逃开','避开','避掉','躲开','躲掉','绕开',
                             '怎样才能不', '怎么才能不', '咋样才能不','咋才能不', '如何才能不',
                             '怎样才不', '怎么才不', '咋样才不','咋才不', '如何才不',
                             '怎样才可以不', '怎么才可以不', '咋样才可以不', '咋才可以不', '如何可以不',
                             '怎样才可不', '怎么才可不', '咋样才可不', '咋才可不', '如何可不']
        self.lasttime_qwds = ['周期', '多久', '多长时间', '多少时间', '几天', '几年', '多少天', '多少小时', '几个小时', '多少年']
        self.cureway_qwds = ['怎么治疗', '如何医治', '怎么医治', '怎么治', '怎么医', '如何治', '医治方式', '疗法', '咋治', '怎么办', '咋办', '咋治']
        self.cureprob_qwds = ['多大概率能治好', '多大几率能治好', '治好希望大么', '几率', '几成', '比例', '可能性', '能治', '可治', '可以治', '可以医']
        self.easyget_qwds = ['易感人群', '容易感染', '易发人群', '什么人', '哪些人', '感染', '染上', '得上']
        self.check_qwds = ['检查', '检查项目', '查出', '检查', '测出', '试出']
        self.belong_qwds = ['属于什么科', '属于', '什么科', '科室']
        self.cure_qwds = ['治疗什么', '治啥', '治疗啥', '医治啥', '治愈啥', '主治啥', '主治什么', '有什么用', '有何用', '用处', '用途',
                          '有什么好处', '有什么益处', '有何益处', '用来', '用来做啥', '用来作甚', '需要', '要']
    
    def classify(self, question: str) -> Dict:
        """
        分类主函数，接口与原有的QuestionClassifier保持兼容
        
        Args:
            question: 用户输入的问题
            
        Returns:
            与原分类器相同格式的字典:
            {
                'args': {实体1: [类型1, 类型2...], 实体2: [类型1...]},
                'question_types': [类型1, 类型2...]
            }
        """
        # 1. 实体识别
        entities = self._extract_entities(question)
        
        if not entities:
            # 没有识别到实体，返回空
            return {}
        
        # 2. 意图识别
        intent, confidence = self._classify_intent(question, entities)
        
        # 3. 构建返回结果
        result = {
            'args': entities,
            'question_types': [intent] if intent != 'others' else []
        }
        
        # 4. 补充处理：如果没有明确意图，但有疾病实体，默认返回疾病描述
        if not result['question_types']:
            types = []
            for entity_types in entities.values():
                types.extend(entity_types)
            
            if 'disease' in types:
                result['question_types'] = ['disease_desc']
            elif 'symptom' in types:
                result['question_types'] = ['symptom_disease']
        
        return result
    
    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """
        提取问题中的实体
        优先使用BERT模型，如果效果不好则使用规则匹配作为后备
        """
        # 首先尝试使用BERT实体识别器
        bert_entities = self.entity_recognizer.extract_entities(question)
        
        # 如果BERT识别到了实体，使用BERT的结果
        if bert_entities:
            return bert_entities
        
        # 否则使用规则匹配作为后备
        return self._rule_based_entity_extraction(question)
    
    def _rule_based_entity_extraction(self, question: str) -> Dict[str, List[str]]:
        """
        基于规则的实体提取（后备方案）
        使用字典匹配，与原有的QuestionClassifier逻辑类似
        """
        entities = {}
        
        # 最长匹配原则
        matched_entities = []
        
        for entity in self.all_entities:
            if entity in question:
                matched_entities.append(entity)
        
        # 去重和过滤子串
        # 例如：如果同时匹配到"感冒"和"流行性感冒"，只保留"流行性感冒"
        filtered_entities = []
        for entity1 in matched_entities:
            is_substring = False
            for entity2 in matched_entities:
                if entity1 != entity2 and entity1 in entity2:
                    is_substring = True
                    break
            if not is_substring:
                filtered_entities.append(entity1)
        
        # 构建结果
        for entity in filtered_entities:
            if entity in self.entity_type_dict:
                entities[entity] = self.entity_type_dict[entity]
        
        return entities
    
    def _classify_intent(self, question: str, entities: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        分类意图
        优先使用BERT模型，如果效果不好则使用规则匹配作为后备
        """
        # 首先尝试使用BERT意图分类器
        intent, confidence = self.intent_classifier.predict(question)
        
        # 如果置信度较高，使用BERT的结果
        if confidence > 0.5:
            return intent, confidence
        
        # 否则使用规则匹配作为后备
        return self._rule_based_intent_classification(question, entities)
    
    def _rule_based_intent_classification(self, question: str, entities: Dict[str, List[str]]) -> Tuple[str, float]:
        """
        基于规则的意图分类（后备方案）
        与原有的QuestionClassifier逻辑类似
        """
        # 收集实体类型
        types = []
        for entity_types in entities.values():
            types.extend(entity_types)
        types = list(set(types))
        
        question_type = 'others'
        question_types = []
        
        # 症状查询
        if self._check_words(self.symptom_qwds, question) and ('disease' in types):
            question_type = 'disease_symptom'
            question_types.append(question_type)
        
        if self._check_words(self.symptom_qwds, question) and ('symptom' in types):
            question_type = 'symptom_disease'
            question_types.append(question_type)
        
        # 原因查询
        if self._check_words(self.cause_qwds, question) and ('disease' in types):
            question_type = 'disease_cause'
            question_types.append(question_type)
        
        # 并发症查询
        if self._check_words(self.acompany_qwds, question) and ('disease' in types):
            question_type = 'disease_acompany'
            question_types.append(question_type)
        
        # 食物查询
        if self._check_words(self.food_qwds, question) and 'disease' in types:
            deny_status = self._check_words(self.deny_words, question)
            if deny_status:
                question_type = 'disease_not_food'
            else:
                question_type = 'disease_do_food'
            question_types.append(question_type)
        
        if self._check_words(self.food_qwds + self.cure_qwds, question) and 'food' in types:
            deny_status = self._check_words(self.deny_words, question)
            if deny_status:
                question_type = 'food_not_disease'
            else:
                question_type = 'food_do_disease'
            question_types.append(question_type)
        
        # 药品查询
        if self._check_words(self.drug_qwds, question) and 'disease' in types:
            question_type = 'disease_drug'
            question_types.append(question_type)
        
        if self._check_words(self.cure_qwds, question) and 'drug' in types:
            question_type = 'drug_disease'
            question_types.append(question_type)
        
        # 检查查询
        if self._check_words(self.check_qwds, question) and 'disease' in types:
            question_type = 'disease_check'
            question_types.append(question_type)
        
        if self._check_words(self.check_qwds + self.cure_qwds, question) and 'check' in types:
            question_type = 'check_disease'
            question_types.append(question_type)
        
        # 预防查询
        if self._check_words(self.prevent_qwds, question) and 'disease' in types:
            question_type = 'disease_prevent'
            question_types.append(question_type)
        
        # 治疗周期查询
        if self._check_words(self.lasttime_qwds, question) and 'disease' in types:
            question_type = 'disease_lasttime'
            question_types.append(question_type)
        
        # 治疗方式查询
        if self._check_words(self.cureway_qwds, question) and 'disease' in types:
            question_type = 'disease_cureway'
            question_types.append(question_type)
        
        # 治愈概率查询
        if self._check_words(self.cureprob_qwds, question) and 'disease' in types:
            question_type = 'disease_cureprob'
            question_types.append(question_type)
        
        # 易感人群查询
        if self._check_words(self.easyget_qwds, question) and 'disease' in types:
            question_type = 'disease_easyget'
            question_types.append(question_type)
        
        # 返回第一个匹配的意图
        if question_types:
            return question_types[0], 0.8  # 规则匹配的置信度设为0.8
        
        return 'others', 0.0
    
    def _check_words(self, words: List[str], sentence: str) -> bool:
        """
        检查句子中是否包含指定的词
        """
        for word in words:
            if word in sentence:
                return True
        return False
    
    def train_intent_classifier(self, train_data_path: str, val_data_path: str, save_dir: str):
        """
        训练意图分类器
        """
        return self.intent_classifier.train(train_data_path, val_data_path, save_dir)
    
    def train_entity_recognizer(self, train_data_path: str, val_data_path: str, save_dir: str):
        """
        训练实体识别器
        """
        return self.entity_recognizer.train(train_data_path, val_data_path, save_dir)
    
    def evaluate_intent_classifier(self, test_data_path: str):
        """
        评估意图分类器
        """
        return self.intent_classifier.evaluate(test_data_path)
    
    def evaluate_entity_recognizer(self, test_data_path: str):
        """
        评估实体识别器
        """
        return self.entity_recognizer.evaluate(test_data_path)


def main():
    """测试函数"""
    print("=" * 60)
    print("BERT问题分类器测试")
    print("=" * 60)
    
    # 创建分类器
    classifier = BertQuestionClassifier()
    
    # 测试问题
    test_questions = [
        "感冒有什么症状？",
        "头痛是什么病？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "怎么预防感冒？",
        "高血压要治疗多久？",
        "感冒怎么治疗？",
        "高血压能治好吗？",
        "什么人容易得糖尿病？",
        "高血压需要做什么检查？",
        "阿莫西林能治什么病？",
        "感冒是什么？",
    ]
    
    print("\n测试分类功能:")
    print("-" * 60)
    
    for question in test_questions:
        result = classifier.classify(question)
        print(f"\n问题: {question}")
        print(f"  实体: {result.get('args', {})}")
        print(f"  意图: {result.get('question_types', [])}")


if __name__ == '__main__':
    main()
