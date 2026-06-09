#!/usr/bin/env python3
# coding: utf-8

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_llm_enhanced.chat_with_llm import KGRAG
import re

def test_prompt_logic():
    print("=== 测试Prompt构建逻辑 ===")
    
    chatbot = KGRAG()
    query = "高血压患者应该选择哪些降压药？"
    
    print(f"问题: {query}")
    
    # 1. 测试实体链接
    print("\n1. 实体链接测试...")
    entity_dict = chatbot.entity_linking(query)
    print(f"实体识别结果: {entity_dict}")
    
    # 2. 测试关系识别（模拟LLM失败的情况）
    print("\n2. 关系识别测试...")
    entity_name = "高血压"
    entity_type = "disease"
    
    # 手动构建应该识别的正确关系
    correct_rels = ["常用药品", "好评药品"]
    print(f"问题应该识别的关系: {correct_rels}")
    
    # 3. 测试知识图谱召回
    print("\n3. 知识图谱召回测试...")
    facts = chatbot.recall_facts(correct_rels, entity_type, entity_name, depth=1)
    print(f"召回的三元组数量: {len(facts)}")
    if facts:
        print("前5个三元组:")
        for fact in facts[:5]:
            print(f"  {fact}")
    
    # 4. 测试prompt构建
    print("\n4. Prompt构建测试...")
    prompt = chatbot.build_enhanced_prompt(query, facts)
    print("构建的prompt:")
    print("-" * 50)
    print(prompt)
    print("-" * 50)
    
    # 5. 分析问题
    print("\n5. 问题分析...")
    print("当前系统的主要问题:")
    print("1. LLM服务器连接失败 - 无法进行实体关系识别")
    print("2. 实体关系识别依赖LLM - 当LLM不可用时整个流程中断")
    print("3. 缺乏降级方案 - 没有备用关系识别机制")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_prompt_logic()