#!/usr/bin/env python3
# coding: utf-8

import requests
import json
import time

def test_fixed_server():
    """测试修复后的服务器"""
    
    url = "http://127.0.0.1:3001/generate"
    
    # 测试用例
    test_cases = [
        {
            "name": "简单问候",
            "data": {
                "message": [{"role": "user", "content": "你好，请简单回答"}],
                "max_tokens": 100,
                "temperature": 0.7
            }
        },
        {
            "name": "医疗问题",
            "data": {
                "message": [{"role": "user", "content": "高血压患者应该选择哪些降压药？"}],
                "max_tokens": 200,
                "temperature": 0.7
            }
        },
        {
            "name": "关系识别问题",
            "data": {
                "message": [{"role": "user", "content": "请判定问题：高血压患者应该选择哪些降压药？所提及的是高血压的哪几个信息，请从['预防措施', '治疗方式', '名称', '治疗周期', '治愈概率', '疾病病因', '治疗科室', '疾病简介', '易感人群', '推荐食谱', '忌吃', '宜吃', '常用药品', '生产药品', '好评药品', '诊断检查', '症状', '并发症', '所属科室']中进行选择，并以列表形式返回。"}],
                "max_tokens": 300,
                "temperature": 0.7
            }
        }
    ]
    
    print("=== 测试修复后的服务器 ===")
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        print(f"问题: {test_case['data']['message'][0]['content'][:50]}...")
        
        try:
            start_time = time.time()
            response = requests.post(url, json=test_case['data'], timeout=30)
            elapsed_time = time.time() - start_time
            
            print(f"响应时间: {elapsed_time:.2f}秒")
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"状态: {result.get('status')}")
                print(f"输出: {result.get('output', [''])[0]}")
                
                if result.get('status') == 'success' and result.get('output', [''])[0]:
                    print("✅ 测试成功")
                else:
                    print("❌ 测试失败 - 空响应")
            else:
                print(f"❌ HTTP错误: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
        except requests.exceptions.ConnectionError:
            print("❌ 连接错误")
        except Exception as e:
            print(f"❌ 异常: {str(e)}")
    
    # 测试健康检查
    print("\n=== 健康检查 ===")
    try:
        health_url = "http://127.0.0.1:3001/health"
        response = requests.get(health_url, timeout=5)
        print(f"健康检查状态: {response.status_code}")
        print(f"健康检查结果: {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {str(e)}")

def test_original_system():
    """测试原始系统是否能正常工作"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("\n=== 测试原始医疗问答系统 ===")
    
    try:
        from qa_llm_enhanced.chat_with_llm import KGRAG
        
        chatbot = KGRAG()
        query = "高血压患者应该选择哪些降压药？"
        
        print(f"测试问题: {query}")
        answer = chatbot.chat(query)
        print(f"系统回答: {answer}")
        
    except Exception as e:
        print(f"系统测试失败: {str(e)}")

if __name__ == "__main__":
    # 测试修复后的服务器
    test_fixed_server()
    
    # 测试原始系统
    test_original_system()