#!/usr/bin/env python3
# coding: utf-8
"""
测试 SiliconFlow API 连接
"""

import os
import sys
import json
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config


def test_api_connection():
    """
    测试 API 连接
    """
    config = Config.get_siliconflow_config()
    
    api_key = config.get('api_key')
    api_url = config.get('api_url', 'https://api.siliconflow.cn/v1/chat/completions')
    model_name = config.get('model_name', 'deepseek-ai/DeepSeek-V3')
    
    print("=" * 60)
    print("SiliconFlow API 测试")
    print("=" * 60)
    print(f"API Key: {'已配置' if api_key else '未配置'}")
    print(f"API URL: {api_url}")
    print(f"模型名称: {model_name}")
    print("=" * 60)
    
    if not api_key:
        print("❌ 错误: API Key 未配置")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 测试的模型列表
    test_models = [
        model_name,  # 用户配置的模型
        "deepseek-ai/DeepSeek-V3",  # 推荐的模型
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
    ]
    
    # 去重
    test_models = list(dict.fromkeys(test_models))
    
    for test_model in test_models:
        print(f"\n{'='*60}")
        print(f"测试模型: {test_model}")
        print(f"{'='*60}")
        
        payload = {
            "model": test_model,
            "messages": [
                {"role": "user", "content": "你好，请简单介绍一下自己"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            print(f"发送请求到: {api_url}")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"\n✅ 模型 {test_model} 可用!")
                    print(f"模型回答: {content[:100]}...")
                    return test_model
                else:
                    print(f"⚠️ 响应格式异常")
            elif response.status_code == 404:
                print(f"❌ 模型 {test_model} 不存在 (404)")
                try:
                    error_info = response.json()
                    print(f"错误详情: {json.dumps(error_info, ensure_ascii=False, indent=2)}")
                except:
                    print(f"错误响应: {response.text}")
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                try:
                    error_info = response.json()
                    print(f"错误详情: {json.dumps(error_info, ensure_ascii=False, indent=2)}")
                except:
                    print(f"错误响应: {response.text}")
                    
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
        except requests.exceptions.ConnectionError as e:
            print(f"❌ 连接错误: {str(e)}")
        except Exception as e:
            print(f"❌ 发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("所有测试模型都不可用，请检查:")
    print("1. API Key 是否正确")
    print("2. 网络连接是否正常")
    print("3. 账户余额是否充足")
    print("=" * 60)
    
    return None


def get_available_models():
    """
    获取可用模型列表（如果 API 支持）
    """
    config = Config.get_siliconflow_config()
    api_key = config.get('api_key')
    
    if not api_key:
        print("API Key 未配置，无法获取模型列表")
        return []
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 尝试获取模型列表
    models_url = "https://api.siliconflow.cn/v1/models"
    
    try:
        print(f"\n尝试获取模型列表: {models_url}")
        response = requests.get(
            models_url,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"模型列表: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return result
        else:
            print(f"获取模型列表失败，状态码: {response.status_code}")
            print(f"响应: {response.text}")
    except Exception as e:
        print(f"获取模型列表时出错: {str(e)}")
    
    return []


if __name__ == "__main__":
    # 首先测试 API 连接
    working_model = test_api_connection()
    
    if working_model:
        print(f"\n🎉 建议将 .env 文件中的 SILICONFLOW_MODEL_NAME 改为: {working_model}")
    
    # 尝试获取模型列表
    get_available_models()
