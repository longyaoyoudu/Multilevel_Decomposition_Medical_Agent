#!/usr/bin/env python3
# coding: utf-8
"""
BERT模型配置文件
"""

import os


class BertConfig:
    """BERT模型配置类"""
    
    # 基础配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型类型
    MODEL_NAME = "bert-base-chinese"  # 使用中文BERT模型
    MAX_SEQ_LENGTH = 128  # 最大序列长度
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10
    WARMUP_STEPS = 100
    WEIGHT_DECAY = 0.01
    
    # 实体类型定义（与现有字典对应）
    ENTITY_TYPES = [
        'O',  # 其他
        'B-disease', 'I-disease',  # 疾病
        'B-symptom', 'I-symptom',  # 症状
        'B-drug', 'I-drug',  # 药品
        'B-food', 'I-food',  # 食物
        'B-check', 'I-check',  # 检查
        'B-department', 'I-department',  # 科室
        'B-producer', 'I-producer'  # 药品厂商
    ]
    
    # 意图类型定义（与现有问题类型对应）
    INTENT_TYPES = [
        'disease_symptom',  # 查询疾病症状
        'symptom_disease',  # 根据症状查疾病
        'disease_cause',  # 查询疾病原因
        'disease_acompany',  # 查询疾病并发症
        'disease_do_food',  # 疾病宜吃食物
        'disease_not_food',  # 疾病忌吃食物
        'food_do_disease',  # 食物能治什么病
        'food_not_disease',  # 食物不能治什么病
        'disease_drug',  # 疾病常用药品
        'drug_disease',  # 药品能治什么病
        'disease_check',  # 疾病需要做什么检查
        'check_disease',  # 检查能查出什么病
        'disease_prevent',  # 疾病预防措施
        'disease_lasttime',  # 疾病治疗周期
        'disease_cureway',  # 疾病治疗方式
        'disease_cureprob',  # 疾病治愈概率
        'disease_easyget',  # 疾病易感人群
        'disease_desc',  # 疾病描述
        'others'  # 其他意图
    ]
    
    # 意图类型到ID的映射
    @classmethod
    def get_intent2id(cls):
        return {intent: idx for idx, intent in enumerate(cls.INTENT_TYPES)}
    
    @classmethod
    def get_id2intent(cls):
        return {idx: intent for idx, intent in enumerate(cls.INTENT_TYPES)}
    
    # 实体类型到ID的映射
    @classmethod
    def get_entity2id(cls):
        return {entity: idx for idx, entity in enumerate(cls.ENTITY_TYPES)}
    
    @classmethod
    def get_id2entity(cls):
        return {idx: entity for idx, entity in enumerate(cls.ENTITY_TYPES)}
    
    # 模型保存路径
    @classmethod
    def get_intent_model_path(cls):
        return os.path.join(cls.BASE_DIR, 'models', 'intent_classifier')
    
    @classmethod
    def get_entity_model_path(cls):
        return os.path.join(cls.BASE_DIR, 'models', 'entity_recognizer')
    
    @classmethod
    def get_data_dir(cls):
        return os.path.join(cls.BASE_DIR, 'bert_nlu', 'data')
    
    # 字典路径（复用现有字典）
    @classmethod
    def get_dict_dir(cls):
        return os.path.join(cls.BASE_DIR, 'dict')
    
    # 问题模板
    QUESTION_TEMPLATES = {
        'disease_symptom': [
            '{disease}有什么症状？',
            '{disease}的症状是什么？',
            '{disease}会有什么表现？',
            '{disease}的表征有哪些？',
            '{disease}有什么现象？',
            '得了{disease}会怎么样？',
        ],
        'symptom_disease': [
            '{symptom}是什么病？',
            '{symptom}可能是什么病？',
            '有{symptom}的症状是什么病？',
            '{symptom}是怎么回事？',
        ],
        'disease_cause': [
            '{disease}是怎么引起的？',
            '为什么会得{disease}？',
            '{disease}的原因是什么？',
            '怎么会得{disease}？',
        ],
        'disease_acompany': [
            '{disease}有什么并发症？',
            '{disease}会伴随什么疾病？',
            '{disease}会并发什么病？',
        ],
        'disease_do_food': [
            '{disease}吃什么好？',
            '{disease}应该吃什么？',
            '{disease}宜吃什么食物？',
            '{disease}的食谱是什么？',
        ],
        'disease_not_food': [
            '{disease}不能吃什么？',
            '{disease}忌吃什么？',
            '{disease}不应该吃什么？',
            '得了{disease}要忌口什么？',
        ],
        'disease_drug': [
            '{disease}吃什么药？',
            '{disease}用什么药？',
            '{disease}的常用药有哪些？',
        ],
        'drug_disease': [
            '{drug}能治什么病？',
            '{drug}的作用是什么？',
            '{drug}用来治疗什么？',
        ],
        'disease_check': [
            '{disease}需要做什么检查？',
            '{disease}要检查什么？',
            '{disease}的检查项目有哪些？',
        ],
        'check_disease': [
            '{check}能查出什么病？',
            '{check}是检查什么的？',
            '{check}能诊断什么病？',
        ],
        'disease_prevent': [
            '怎么预防{disease}？',
            '{disease}怎么预防？',
            '如何避免得{disease}？',
        ],
        'disease_lasttime': [
            '{disease}要治疗多久？',
            '{disease}的治疗周期是多久？',
            '{disease}多长时间能好？',
        ],
        'disease_cureway': [
            '{disease}怎么治疗？',
            '{disease}如何治疗？',
            '{disease}的治疗方法有哪些？',
        ],
        'disease_cureprob': [
            '{disease}能治好吗？',
            '{disease}治愈的概率是多少？',
            '{disease}治好的几率有多大？',
        ],
        'disease_easyget': [
            '什么人容易得{disease}？',
            '{disease}的易感人群是什么？',
            '哪些人容易得{disease}？',
        ],
        'disease_desc': [
            '{disease}是什么？',
            '介绍一下{disease}',
            '{disease}的简介是什么？',
        ]
    }
