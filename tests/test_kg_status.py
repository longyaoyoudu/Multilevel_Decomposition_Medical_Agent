#!/usr/bin/env python3
# coding: utf-8

from py2neo import Graph
import sys

def test_kg_status():
    try:
        # 连接Neo4j
        g = Graph("bolt://localhost:7687", auth=("neo4j", "978626572"))
        print("Neo4j连接成功")
        
        # 检查高血压数据
        hypertension_result = g.run('MATCH (d:Disease) WHERE d.name = "高血压" RETURN d').data()
        print(f"高血压查询结果: {hypertension_result}")
        
        # 检查所有疾病数量
        all_diseases = g.run('MATCH (d:Disease) RETURN d.name').data()
        print(f"数据库中疾病总数: {len(all_diseases)}")
        
        # 显示部分疾病名称
        disease_names = [d['d.name'] for d in all_diseases[:10]]
        print(f"前10个疾病: {disease_names}")
        
        # 检查是否有数据
        if len(all_diseases) == 0:
            print("知识图谱中没有疾病数据")
            return False
        elif not hypertension_result:
            print("知识图谱中没有高血压数据")
            return False
        else:
            print("知识图谱数据正常")
            return True
            
    except Exception as e:
        print(f"Neo4j连接失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_kg_status()