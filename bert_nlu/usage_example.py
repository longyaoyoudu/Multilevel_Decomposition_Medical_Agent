#!/usr/bin/env python3
# coding: utf-8
"""
BERT增强的双模型协作系统使用示例
展示如何将BERT意图识别和实体识别与双模型协作模式结合使用
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bert_nlu.config import BertConfig
from bert_nlu.bert_question_classifier import BertQuestionClassifier
from bert_nlu.bert_enhanced_chat import BertEnhancedChat


def example_1_basic_nlu():
    """
    示例1：基本的BERT自然语言理解（意图识别+实体识别）
    """
    print("=" * 60)
    print("示例1：基本的BERT自然语言理解")
    print("=" * 60)
    
    # 创建BERT问题分类器
    print("\n初始化BERT问题分类器...")
    classifier = BertQuestionClassifier()
    
    # 测试问题
    test_questions = [
        "感冒有什么症状？",
        "头痛是什么病？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "怎么预防感冒？",
        "失眠有什么并发症？",
        "为什么会失眠？",
        "阿莫西林能治什么病？",
        "血常规能查出什么病？",
        "什么人容易得糖尿病？",
    ]
    
    print("\n测试意图识别和实体识别:")
    print("-" * 60)
    
    for question in test_questions:
        result = classifier.classify(question)
        entities = result.get('args', {})
        intents = result.get('question_types', [])
        
        print(f"\n问题: {question}")
        print(f"  识别到的实体: {entities}")
        print(f"  识别到的意图: {intents}")
    
    return classifier


def example_2_intent_classifier_details():
    """
    示例2：详细展示意图分类器的使用
    """
    print("\n" + "=" * 60)
    print("示例2：意图分类器详细使用")
    print("=" * 60)
    
    from bert_nlu.intent_classifier import BertIntentClassifier
    
    print("\n初始化意图分类器...")
    classifier = BertIntentClassifier()
    
    # 测试预测
    test_texts = [
        "感冒有什么症状？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "怎么预防感冒？",
        "你好，我想咨询一些健康问题",
    ]
    
    print("\n测试意图预测:")
    print("-" * 60)
    
    for text in test_texts:
        intent, confidence = classifier.predict(text)
        print(f"\n文本: {text}")
        print(f"  预测意图: {intent}")
        print(f"  置信度: {confidence:.4f}")
    
    return classifier


def example_3_entity_recognizer_details():
    """
    示例3：详细展示实体识别器的使用
    """
    print("\n" + "=" * 60)
    print("示例3：实体识别器详细使用")
    print("=" * 60)
    
    from bert_nlu.entity_recognizer import BertEntityRecognizer
    
    print("\n初始化实体识别器...")
    recognizer = BertEntityRecognizer()
    
    # 测试预测
    test_texts = [
        "感冒有什么症状？",
        "头痛、发烧是什么病？",
        "高血压和糖尿病患者应该吃什么药？",
        "阿莫西林和头孢能治什么病？",
        "血常规和尿常规能查出什么病？",
    ]
    
    print("\n测试实体识别:")
    print("-" * 60)
    
    for text in test_texts:
        entities = recognizer.predict(text)
        grouped = recognizer.extract_entities(text)
        
        print(f"\n文本: {text}")
        print(f"  识别到的实体（详细）:")
        for entity in entities:
            print(f"    - 文本: {entity['text']}, 类型: {entity['type']}, "
                  f"位置: ({entity['start']}, {entity['end']})")
        print(f"  按类型分组: {grouped}")
    
    return recognizer


def example_4_data_preparation():
    """
    示例4：展示如何准备训练数据
    """
    print("\n" + "=" * 60)
    print("示例4：训练数据准备")
    print("=" * 60)
    
    from bert_nlu.data_preparation import DataPreparator
    
    print("\n初始化数据准备器...")
    preparator = DataPreparator()
    
    # 准备少量数据用于演示
    print("\n准备训练数据（每个意图生成20个样本）...")
    stats = preparator.prepare_all_data(num_samples_per_intent=20)
    
    print("\n数据统计:")
    print("-" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n数据保存位置: {preparator.data_dir}")
    
    return preparator


def example_5_bert_enhanced_chat():
    """
    示例5：BERT增强的双模型协作聊天系统
    """
    print("\n" + "=" * 60)
    print("示例5：BERT增强的双模型协作聊天系统")
    print("=" * 60)
    
    print("\n初始化BERT增强的聊天系统...")
    print("注意：如果配置了LLM API Key，将启用双模型协作模式")
    
    # 创建聊天系统实例（禁用知识图谱以避免依赖问题）
    chat_system = BertEnhancedChat(use_kg=False)
    
    # 测试问题
    test_questions = [
        "感冒有什么症状？",
        "高血压吃什么药？",
        "糖尿病不能吃什么？",
        "怎么预防感冒？",
    ]
    
    print("\n测试BERT NLU分析（不调用LLM）:")
    print("-" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        
        # 使用BERT进行NLU分析
        nlu_result = chat_system.bert_classifier.classify(question)
        entities = nlu_result.get('args', {})
        intents = nlu_result.get('question_types', [])
        
        print(f"  实体: {entities}")
        print(f"  意图: {intents}")
        
        # 构建增强的提示词
        enhanced_prompt = chat_system._build_enhanced_prompt(question, entities, intents)
        print(f"  增强提示词预览: {enhanced_prompt[:200]}...")
    
    print("\n" + "=" * 60)
    print("示例5完成")
    print("=" * 60)
    print("\n如果配置了LLM API Key，可以使用以下方法：")
    print("1. chat_system.process_question('感冒有什么症状？') - 完整处理流程")
    print("2. chat_system.chat('感冒有什么症状？') - 简单聊天接口")
    print("3. chat_system.interactive_chat() - 交互式聊天模式")
    
    return chat_system


def main():
    """
    主函数：运行所有示例
    """
    print("=" * 60)
    print("BERT增强的双模型协作系统使用示例")
    print("=" * 60)
    
    print("\n" + "=" * 60)
    print("配置信息")
    print("=" * 60)
    print(f"  BERT模型: {BertConfig.MODEL_NAME}")
    print(f"  最大序列长度: {BertConfig.MAX_SEQ_LENGTH}")
    print(f"  意图类型数: {len(BertConfig.INTENT_TYPES)}")
    print(f"  实体类型数: {len(BertConfig.ENTITY_TYPES)}")
    print("=" * 60)
    
    # 运行示例
    print("\n" + "#" * 60)
    print("运行示例1：基本的BERT自然语言理解")
    print("#" * 60)
    example_1_basic_nlu()
    
    print("\n" + "#" * 60)
    print("运行示例2：意图分类器详细使用")
    print("#" * 60)
    example_2_intent_classifier_details()
    
    print("\n" + "#" * 60)
    print("运行示例3：实体识别器详细使用")
    print("#" * 60)
    example_3_entity_recognizer_details()
    
    print("\n" + "#" * 60)
    print("运行示例4：训练数据准备")
    print("#" * 60)
    example_4_data_preparation()
    
    print("\n" + "#" * 60)
    print("运行示例5：BERT增强的双模型协作聊天系统")
    print("#" * 60)
    example_5_bert_enhanced_chat()
    
    # 打印总结
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    print("\n快速使用指南：")
    print("-" * 60)
    print("""
1. 基本NLU（意图识别+实体识别）：
   from bert_nlu import BertQuestionClassifier
   classifier = BertQuestionClassifier()
   result = classifier.classify('感冒有什么症状？')
   # 输出: {'args': {'感冒': ['disease']}, 'question_types': ['disease_symptom']}

2. BERT增强的双模型协作聊天：
   from bert_nlu import BertEnhancedChat
   chat_system = BertEnhancedChat(use_kg=True)
   answer = chat_system.chat('感冒有什么症状？')
   # 需要配置LLM API Key才能生成答案

3. 准备训练数据：
   from bert_nlu import DataPreparator
   preparator = DataPreparator()
   preparator.prepare_all_data(num_samples_per_intent=200)

4. 训练意图分类器：
   from bert_nlu import BertIntentClassifier
   classifier = BertIntentClassifier()
   classifier.train(
       train_data_path='bert_nlu/data/intent/train.json',
       val_data_path='bert_nlu/data/intent/val.json',
       save_dir='models/intent_classifier'
   )

5. 训练实体识别器：
   from bert_nlu import BertEntityRecognizer
   recognizer = BertEntityRecognizer()
   recognizer.train(
       train_data_path='bert_nlu/data/ner/train.json',
       val_data_path='bert_nlu/data/ner/val.json',
       save_dir='models/entity_recognizer'
   )
""")
    print("-" * 60)
    print("更多详细信息请查看各模块的源代码和注释。")


if __name__ == '__main__':
    main()
