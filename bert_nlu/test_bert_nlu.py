#!/usr/bin/env python3
# coding: utf-8
"""
BERT自然语言理解模块测试脚本
测试意图识别、实体识别和问题分类功能
"""

import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert_nlu.config import BertConfig
from bert_nlu.data_preparation import DataPreparator
from bert_nlu.intent_classifier import BertIntentClassifier
from bert_nlu.entity_recognizer import BertEntityRecognizer
from bert_nlu.bert_question_classifier import BertQuestionClassifier


def test_data_preparation():
    """
    测试数据准备功能
    """
    print("=" * 60)
    print("测试1: 数据准备功能")
    print("=" * 60)
    
    try:
        preparator = DataPreparator()
        
        # 准备数据（每个意图生成50个样本，用于快速测试）
        stats = preparator.prepare_all_data(num_samples_per_intent=50)
        
        print("\n数据准备成功！")
        print("数据统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"数据准备失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intent_classifier():
    """
    测试意图分类器
    """
    print("\n" + "=" * 60)
    print("测试2: 意图分类器")
    print("=" * 60)
    
    try:
        # 创建分类器（使用未训练的模型进行测试）
        classifier = BertIntentClassifier()
        
        # 测试预测
        test_texts = [
            "感冒有什么症状？",
            "头痛是什么病？",
            "高血压吃什么药？",
            "糖尿病不能吃什么？",
            "怎么预防感冒？",
            "你好",
        ]
        
        print("\n测试预测功能（未训练模型）:")
        for text in test_texts:
            intent, confidence = classifier.predict(text)
            print(f"  文本: {text}")
            print(f"  预测意图: {intent}, 置信度: {confidence:.4f}")
            print()
        
        print("意图分类器测试完成！")
        return True
    except Exception as e:
        print(f"意图分类器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entity_recognizer():
    """
    测试实体识别器
    """
    print("\n" + "=" * 60)
    print("测试3: 实体识别器")
    print("=" * 60)
    
    try:
        # 创建识别器（使用未训练的模型进行测试）
        recognizer = BertEntityRecognizer()
        
        # 测试预测
        test_texts = [
            "感冒有什么症状？",
            "头痛是什么病？",
            "高血压吃什么药？",
            "糖尿病不能吃什么？",
            "阿莫西林能治什么病？",
            "血常规能查出什么病？",
        ]
        
        print("\n测试预测功能（未训练模型）:")
        for text in test_texts:
            entities = recognizer.predict(text)
            print(f"  文本: {text}")
            print(f"  识别到的实体:")
            for entity in entities:
                print(f"    - {entity['text']} ({entity['type']})")
            
            # 测试按类型分组
            grouped = recognizer.extract_entities(text)
            print(f"  按类型分组: {grouped}")
            print()
        
        print("实体识别器测试完成！")
        return True
    except Exception as e:
        print(f"实体识别器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bert_question_classifier():
    """
    测试集成的BERT问题分类器
    """
    print("\n" + "=" * 60)
    print("测试4: 集成的BERT问题分类器")
    print("=" * 60)
    
    try:
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
            "失眠有什么并发症？",
            "为什么会失眠？",
            "失眠的人不要吃啥？",
        ]
        
        print("\n测试分类功能（使用规则匹配作为后备）:")
        print("-" * 60)
        
        for question in test_questions:
            result = classifier.classify(question)
            print(f"\n问题: {question}")
            print(f"  识别到的实体: {result.get('args', {})}")
            print(f"  识别到的意图: {result.get('question_types', [])}")
        
        print("\n集成的BERT问题分类器测试完成！")
        return True
    except Exception as e:
        print(f"集成的BERT问题分类器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_original_classifier():
    """
    与原有的QuestionClassifier进行对比测试
    """
    print("\n" + "=" * 60)
    print("测试5: 与原有分类器对比")
    print("=" * 60)
    
    try:
        # 尝试导入原有的分类器
        from qa_traditional.question_classifier import QuestionClassifier
        
        # 创建BERT分类器
        bert_classifier = BertQuestionClassifier()
        
        # 创建原有分类器
        original_classifier = QuestionClassifier()
        
        # 测试问题
        test_questions = [
            "感冒有什么症状？",
            "头痛是什么病？",
            "高血压吃什么药？",
            "糖尿病不能吃什么？",
            "怎么预防感冒？",
            "失眠有什么并发症？",
            "为什么会失眠？",
        ]
        
        print("\n对比测试结果:")
        print("-" * 60)
        
        for question in test_questions:
            print(f"\n问题: {question}")
            
            # BERT分类器结果
            bert_result = bert_classifier.classify(question)
            print(f"  BERT分类器:")
            print(f"    实体: {bert_result.get('args', {})}")
            print(f"    意图: {bert_result.get('question_types', [])}")
            
            # 原有分类器结果
            original_result = original_classifier.classify(question)
            print(f"  原有分类器:")
            print(f"    实体: {original_result.get('args', {})}")
            print(f"    意图: {original_result.get('question_types', [])}")
        
        print("\n对比测试完成！")
        return True
    except ImportError as e:
        print(f"无法导入原有分类器: {e}")
        print("跳过对比测试")
        return True
    except Exception as e:
        print(f"对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主测试函数
    """
    print("=" * 60)
    print("BERT自然语言理解模块测试")
    print("=" * 60)
    print(f"配置:")
    print(f"  模型: {BertConfig.MODEL_NAME}")
    print(f"  最大序列长度: {BertConfig.MAX_SEQ_LENGTH}")
    print(f"  意图类型数: {len(BertConfig.INTENT_TYPES)}")
    print(f"  实体类型数: {len(BertConfig.ENTITY_TYPES)}")
    print("=" * 60)
    
    # 测试结果
    results = {}
    
    # 运行所有测试
    results['数据准备'] = test_data_preparation()
    results['意图分类器'] = test_intent_classifier()
    results['实体识别器'] = test_entity_recognizer()
    results['集成分类器'] = test_bert_question_classifier()
    results['对比测试'] = compare_with_original_classifier()
    
    # 打印测试结果汇总
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息。")
    print("=" * 60)
    
    # 打印使用说明
    print("\n使用说明:")
    print("-" * 60)
    print("1. 准备训练数据:")
    print("   python bert_nlu/data_preparation.py")
    print()
    print("2. 训练意图分类器:")
    print("   可在代码中调用 intent_classifier.train() 方法")
    print()
    print("3. 训练实体识别器:")
    print("   可在代码中调用 entity_recognizer.train() 方法")
    print()
    print("4. 使用集成分类器（与原有系统兼容）:")
    print("   from bert_nlu import BertQuestionClassifier")
    print("   classifier = BertQuestionClassifier()")
    print("   result = classifier.classify('感冒有什么症状？')")
    print("=" * 60)


if __name__ == '__main__':
    main()
