# coding: utf-8
"""
Minimax API 测试脚本
用于测试Minimax API服务的连接和功能
"""
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from utils.config import Config, config


def test_minimax_direct():
    """直接测试Minimax API连接"""
    print("=" * 60)
    print("测试1: 直接调用Minimax API")
    print("=" * 60)
    
    minimax_config = Config.get_minimax_config()
    gen_params = Config.get_generation_params()
    
    api_key = minimax_config['api_key']
    group_id = minimax_config['group_id']
    api_url = minimax_config['api_url']
    model_name = minimax_config['model_name']
    
    if not api_key:
        print("❌ 错误: MINIMAX_API_KEY 未设置")
        print("请创建 .env 文件并配置 API Key")
        return False
    
    print(f"API Key 已配置: {'是' if api_key else '否'}")
    print(f"Group ID 已配置: {'是' if group_id else '否'}")
    print(f"模型: {model_name}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {}
    if group_id:
        params["GroupId"] = group_id
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ],
        "max_tokens": gen_params['max_tokens'] // 4,
        "temperature": gen_params['temperature'],
        "stream": False
    }
    
    print(f"\n请求URL: {api_url}")
    print(f"测试问题: 你好，请介绍一下你自己")
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            params=params,
            json=payload,
            timeout=60
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API请求失败: {response.text}")
            return False
        
        result = response.json()
        print(f"\n响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if "choices" in result and len(result["choices"]) > 0:
            reply = result["choices"][0]["message"]["content"]
            print(f"\n✅ API响应成功!")
            print(f"模型回复: {reply}")
            return True
        else:
            print(f"❌ 无法解析响应")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_minimax_flask_service():
    """测试Minimax Flask服务（需要先启动minimax.py）"""
    print("\n" + "=" * 60)
    print("测试2: 测试Minimax Flask服务接口")
    print("=" * 60)
    print("注意: 此测试需要先在另一个终端启动: python llm_services/minimax.py")
    
    test_url = Config.LLM_SERVICE_URL
    health_url = f"http://127.0.0.1:{Config.FLASK_PORT}/health"
    
    try:
        health_resp = requests.get(health_url, timeout=5)
        if health_resp.status_code == 200:
            print("✅ Flask服务运行中")
            print(f"健康检查响应: {health_resp.json()}")
        else:
            print("⚠️  Flask服务未运行或健康检查失败")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到Flask服务 (http://127.0.0.1:{Config.FLASK_PORT})")
        print("请先在另一个终端运行: python llm_services/minimax.py")
        return False
    
    payload = {
        "message": [
            {"role": "user", "content": "糖尿病患者应该注意什么饮食？"}
        ],
        "max_tokens": Config.MAX_TOKENS // 2,
        "temperature": Config.TEMPERATURE,
        "history": []
    }
    
    print(f"\n测试问题: 糖尿病患者应该注意什么饮食？")
    
    try:
        response = requests.post(
            test_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120
        )
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.text}")
            return False
        
        result = response.json()
        print(f"\n完整响应: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get("status") == "success" and result.get("output"):
            reply = result["output"][0]
            print(f"\n✅ Flask服务响应成功!")
            print(f"模型回复: {reply[:300]}..." if len(reply) > 300 else f"模型回复: {reply}")
            return True
        else:
            print(f"❌ 服务返回错误: {result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_medical_question():
    """测试医疗相关问题"""
    print("\n" + "=" * 60)
    print("测试3: 测试医疗问题回答")
    print("=" * 60)
    
    minimax_config = Config.get_minimax_config()
    gen_params = Config.get_generation_params()
    
    api_key = minimax_config['api_key']
    group_id = minimax_config['group_id']
    api_url = minimax_config['api_url']
    model_name = minimax_config['model_name']
    
    if not api_key:
        print("❌ 错误: MINIMAX_API_KEY 未设置")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {}
    if group_id:
        params["GroupId"] = group_id
    
    medical_questions = [
        "高血压的症状有哪些？",
        "感冒患者应该吃什么食物？",
        "糖尿病有哪些并发症？"
    ]
    
    system_prompt = """你是一位专业的医疗问答助手。
请基于你的医学知识回答问题。
回答要简洁明了，不超过3句话。
如果涉及诊断建议，请建议用户咨询专业医生。"""
    
    for question in medical_questions:
        print(f"\n问题: {question}")
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "max_tokens": gen_params['max_tokens'] // 2,
            "temperature": gen_params['temperature'],
            "stream": False
        }
        
        try:
            response = requests.post(
                api_url,
                headers=headers,
                params=params,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    reply = result["choices"][0]["message"]["content"]
                    print(f"回答: {reply}")
                else:
                    print(f"⚠️  无法解析响应")
            else:
                print(f"⚠️  请求失败: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️  请求异常: {str(e)}")
    
    return True


def main():
    print("=" * 60)
    print("Minimax API 测试工具")
    print("=" * 60)
    
    print("\n当前配置摘要:")
    Config.print_config_summary()
    
    test1_passed = test_minimax_direct()
    
    if test1_passed:
        print("\n" + "=" * 60)
        print("🎉 直接API测试通过!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  直接API测试失败")
        print("=" * 60)
        print("\n可能的原因:")
        print("1. API Key 未配置或无效")
        print("2. 网络连接问题")
        print("3. API 服务暂时不可用")
        return
    
    test2_passed = test_minimax_flask_service()
    
    if test2_passed:
        print("\n" + "=" * 60)
        print("🎉 Flask服务测试通过!")
        print("=" * 60)
    
    test_medical_question()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print("\n使用说明:")
    print("1. 配置API Key: 创建项目根目录下的 .env 文件")
    print("2. 启动服务: python llm_services/minimax.py")
    print("3. 使用问答系统: python qa_llm_enhanced/chat_with_llm.py")


if __name__ == "__main__":
    main()
