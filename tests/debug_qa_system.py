#!/usr/bin/env python3
# coding: utf-8

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qa_traditional.question_classifier import QuestionClassifier
from kg_builder.build_medicalgraph import MedicalGraph
from py2neo import Graph

def debug_qa_system():
    print("=== 医疗问答系统调试分析 ===")
    
    # 1. 测试实体识别
    print("\n1. 测试实体识别...")
    entity_parser = QuestionClassifier()
    query = "高血压患者应该选择哪些降压药？"
    entity_dict = entity_parser.check_medical(query)
    print(f"问题: {query}")
    print(f"实体识别结果: {entity_dict}")
    
    # 2. 测试知识图谱连接
    print("\n2. 测试知识图谱连接...")
    try:
        kg = MedicalGraph()
        print("知识图谱连接成功")
        
        # 检查高血压相关数据
        print("\n3. 检查高血压相关数据...")
        
        # 检查高血压节点
        hypertension_node = kg.g.run('MATCH (d:Disease) WHERE d.name = "高血压" RETURN d').data()
        print(f"高血压节点: {len(hypertension_node)} 个")
        
        # 检查高血压的常用药品
        common_drugs = kg.g.run('MATCH (d:Disease)-[r:common_drug]->(drug:Drug) WHERE d.name = "高血压" RETURN drug.name').data()
        print(f"高血压常用药品: {common_drugs}")
        
        # 检查高血压的好评药品
        recommand_drugs = kg.g.run('MATCH (d:Disease)-[r:recommand_drug]->(drug:Drug) WHERE d.name = "高血压" RETURN drug.name').data()
        print(f"高血压好评药品: {recommand_drugs}")
        
        # 检查所有与高血压相关的药品
        all_drugs = kg.g.run('MATCH (d:Disease)-[r]->(drug:Drug) WHERE d.name = "高血压" RETURN type(r), drug.name').data()
        print(f"所有与高血压相关的药品关系: {all_drugs}")
        
        # 检查知识图谱查询
        print("\n4. 测试知识图谱查询...")
        sql = "MATCH p=(m:Disease)-[r*..1]-(n) where m.name = '高血压' return p"
        print(f"查询语句: {sql}")
        results = kg.g.run(sql).data()
        print(f"查询结果数量: {len(results)}")
        
        if results:
            print("查询结果示例:")
            for i, res in enumerate(results[:2]):  # 只显示前2个结果
                print(f"结果 {i+1}: {res}")
        
    except Exception as e:
        print(f"知识图谱连接失败: {str(e)}")
    
    # 3. 分析问题类型
    print("\n5. 分析问题类型...")
    print("问题涉及的信息类型:")
    print("- 疾病名称: 高血压")
    print("- 药品选择: 降压药")
    print("- 治疗方式: 药物治疗")
    
    print("\n=== 调试完成 ===")

if __name__ == "__main__":
    debug_qa_system()