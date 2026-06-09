#!/usr/bin/env python3
# coding: utf-8
"""
统一的 LLM 服务中间层
支持 SiliconFlow 和 MiniMax 两种 API，自动读取 .env 配置
工作模式：
1. 双模型协作模式（默认）：MiniMax 生成初步回答，SiliconFlow 对回答进行优化
2. 单模型模式：优先使用 SiliconFlow，如果未配置则使用 MiniMax
"""

import os
import sys
import json
from flask import Flask, request, jsonify
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

app = Flask(__name__)


def get_llm_provider():
    """
    确定使用哪个 LLM 提供商
    优先级：双模型协作 > SiliconFlow > MiniMax
    双模型协作：当 SiliconFlow 和 MiniMax 都配置时，使用 MiniMax 生成初步回答，SiliconFlow 优化回答
    """
    siliconflow_config = Config.get_siliconflow_config()
    minimax_config = Config.get_minimax_config()
    
    has_siliconflow = siliconflow_config.get('api_key')
    has_minimax = minimax_config.get('api_key')
    
    if has_siliconflow and has_minimax:
        return {
            'provider': 'dual_model',
            'siliconflow_config': siliconflow_config,
            'minimax_config': minimax_config
        }
    elif has_siliconflow:
        return {
            'provider': 'siliconflow',
            'config': siliconflow_config
        }
    elif has_minimax:
        return {
            'provider': 'minimax',
            'config': minimax_config
        }
    else:
        return {
            'provider': 'none',
            'config': None
        }


def call_siliconflow_api(data):
    """
    调用 SiliconFlow API
    """
    config = Config.get_siliconflow_config()
    gen_params = Config.get_generation_params()
    
    api_key = config.get('api_key')
    api_url = config.get('api_url', 'https://api.siliconflow.cn/v1/chat/completions')
    model_name = config.get('model_name', 'deepseek-ai/DeepSeek-V3')
    
    if not api_key:
        raise ValueError("SiliconFlow API Key 未配置，请在 .env 文件中设置 SILICONFLOW_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = data.get("message", [])
    
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": data.get("max_tokens", gen_params.get('max_tokens', 2048)),
        "temperature": data.get("temperature", gen_params.get('temperature', 0.7)),
        "top_p": data.get("top_p", gen_params.get('top_p', 0.9)),
        "stream": False
    }
    
    print(f"[SiliconFlow] 调用 API，模型: {model_name}")
    print(f"[SiliconFlow] 请求 URL: {api_url}")
    
    response = requests.post(
        api_url,
        headers=headers,
        json=payload,
        timeout=120
    )
    
    print(f"[SiliconFlow] 响应状态码: {response.status_code}")
    
    if response.status_code != 200:
        print(f"[SiliconFlow] 错误响应: {response.text}")
        raise Exception(f"SiliconFlow API 错误: {response.status_code} - {response.text}")
    
    result = response.json()
    
    print(f"[SiliconFlow] 原始响应: {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}")
    
    if "choices" in result and len(result["choices"]) > 0:
        message = result["choices"][0].get("message", {})
        
        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")
        
        print(f"[SiliconFlow] content: '{content[:100] if content else '(empty)'}'")
        print(f"[SiliconFlow] reasoning_content: '{reasoning_content[:100] if reasoning_content else '(empty)'}'")
        
        if content and content.strip():
            return content
        elif reasoning_content and reasoning_content.strip():
            return reasoning_content
        else:
            print(f"[SiliconFlow] 警告: content 和 reasoning_content 都为空")
            return "模型返回了空响应，请尝试重新提问或更换模型。"
    else:
        raise Exception(f"无法解析 SiliconFlow 响应: {result}")


def call_minimax_api(data, config=None):
    """
    调用 MiniMax API
    """
    if config is None:
        config = Config.get_minimax_config()
    gen_params = Config.get_generation_params()
    
    api_key = config.get('api_key')
    group_id = config.get('group_id')
    api_url = config.get('api_url', 'https://api.minimaxi.chat/v1/text/chatcompletion_v2')
    model_name = config.get('model_name', 'abab6.5s-chat')
    
    if not api_key:
        raise ValueError("MiniMax API Key 未配置，请在 .env 文件中设置 MINIMAX_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {}
    if group_id:
        params["GroupId"] = group_id
    
    messages = data.get("message", [])
    
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    
    system_prompt = data.get("system", "")
    if system_prompt:
        formatted_messages.insert(0, {
            "role": "system",
            "content": system_prompt
        })
    
    payload = {
        "model": model_name,
        "messages": formatted_messages,
        "max_tokens": data.get("max_tokens", gen_params.get('max_tokens', 2048)),
        "temperature": data.get("temperature", gen_params.get('temperature', 0.7)),
        "top_p": data.get("top_p", gen_params.get('top_p', 0.9)),
        "stream": False
    }
    
    print(f"[MiniMax] 调用 API，模型: {model_name}")
    print(f"[MiniMax] 请求 URL: {api_url}")
    
    response = requests.post(
        api_url,
        headers=headers,
        params=params,
        json=payload,
        timeout=120
    )
    
    print(f"[MiniMax] 响应状态码: {response.status_code}")
    
    if response.status_code != 200:
        print(f"[MiniMax] 错误响应: {response.text}")
        raise Exception(f"MiniMax API 错误: {response.status_code} - {response.text}")
    
    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    elif "reply" in result:
        return result["reply"]
    else:
        raise Exception(f"无法解析 MiniMax 响应: {result}")


def call_dual_model_api(data):
    """
    双模型协作 API 调用
    1. 首先使用 MiniMax 生成初步回答
    2. 然后使用 SiliconFlow 对回答进行优化
    """
    print("=" * 60)
    print("[双模型协作模式] 开始执行")
    print("=" * 60)
    
    # 第一步：使用 MiniMax 生成初步回答
    print("\n[步骤 1/2] 使用 MiniMax 生成初步回答...")
    minimax_config = Config.get_minimax_config()
    
    try:
        minimax_answer = call_minimax_api(data, minimax_config)
        print(f"[MiniMax] 初步回答生成成功，长度: {len(minimax_answer)} 字符")
        print(f"[MiniMax] 回答预览: {minimax_answer[:200] if len(minimax_answer) > 200 else minimax_answer}")
    except Exception as e:
        print(f"[MiniMax] 生成失败: {str(e)}")
        print("尝试单独使用 SiliconFlow...")
        return call_siliconflow_api(data)
    
    # 第二步：使用 SiliconFlow 对回答进行优化
    print("\n[步骤 2/2] 使用 SiliconFlow 优化回答...")
    
    # 构建优化提示词
    original_question = ""
    if data.get("message") and len(data["message"]) > 0:
        original_question = data["message"][0].get("content", "")
    
    optimization_prompt = f"""请你作为专业的医疗问答优化专家，对以下医疗问题的回答进行优化。

**原始问题**：
{original_question}

**初步回答（需要优化）**：
{minimax_answer}

**优化要求**：
1. 准确性：确保所有医疗信息准确无误，符合医学常识
2. 完整性：补充遗漏的重要信息，让回答更加全面
3. 清晰度：优化语言表达，让回答更加清晰易懂
4. 结构：使用清晰的结构组织回答，如分点、使用小标题等
5. 专业性：保持专业、严谨的语气，但也要通俗易懂
6. 如果原始回答已经很好，可以适当润色后返回

请直接提供优化后的完整回答，不要说明优化过程。"""

    # 构建 SiliconFlow 的请求数据
    siliconflow_data = {
        "message": [
            {
                "role": "user",
                "content": optimization_prompt
            }
        ],
        "max_tokens": data.get("max_tokens", 4096),
        "temperature": data.get("temperature", 0.7),
        "top_p": data.get("top_p", 0.9)
    }
    
    try:
        optimized_answer = call_siliconflow_api(siliconflow_data)
        print(f"[SiliconFlow] 优化完成，长度: {len(optimized_answer)} 字符")
        print(f"[SiliconFlow] 优化后回答预览: {optimized_answer[:200] if len(optimized_answer) > 200 else optimized_answer}")
        
        # 检查优化后的回答是否为空或无效，如果是则返回 MiniMax 的原始回答
        if not optimized_answer or len(optimized_answer.strip()) < 10:
            print("[警告] SiliconFlow 优化后的回答无效，返回 MiniMax 的原始回答")
            return minimax_answer
        
        print("\n[双模型协作] 执行完成，返回优化后的回答")
        print("=" * 60)
        
        return optimized_answer
        
    except Exception as e:
        print(f"[SiliconFlow] 优化失败: {str(e)}")
        print("返回 MiniMax 的原始回答作为备选")
        print("=" * 60)
        return minimax_answer


@app.route("/generate", methods=["POST"])
def generate():
    """
    API 端点：处理生成请求
    与项目现有接口兼容
    """
    try:
        data = request.get_json()
        print("=" * 60)
        print("收到新的请求")
        print("请求数据:", json.dumps(data, ensure_ascii=False, indent=2)[:500])
        
        provider_info = get_llm_provider()
        provider = provider_info['provider']
        
        if provider == 'none':
            return jsonify({
                "error": "未配置任何 LLM API Key，请在 .env 文件中配置 SILICONFLOW_API_KEY 或 MINIMAX_API_KEY",
                "status": "error"
            }), 500
        
        print(f"使用 LLM 提供商: {provider}")
        
        try:
            if provider == 'dual_model':
                generated_text = call_dual_model_api(data)
            elif provider == 'siliconflow':
                generated_text = call_siliconflow_api(data)
            else:
                generated_text = call_minimax_api(data)
        except Exception as api_error:
            print(f"API 调用失败: {str(api_error)}")
            return jsonify({
                "error": f"LLM API 调用失败: {str(api_error)}",
                "status": "error"
            }), 500
        
        user_input = data["message"][0]["content"] if data.get("message") else ""
        input_history = data.get("history", [])
        
        updated_history = input_history + [[user_input, generated_text]]
        
        print("生成的回答:", generated_text[:200] if len(generated_text) > 200 else generated_text)
        print("=" * 60)
        
        return jsonify({
            "output": [generated_text],
            "history": updated_history,
            "status": "success"
        })
        
    except Exception as e:
        print(f"处理请求时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


@app.route("/health", methods=["GET"])
def health_check():
    """健康检查端点"""
    provider_info = get_llm_provider()
    provider = provider_info['provider']
    
    status_info = {
        "status": "ok",
        "service": "unified-llm-service",
        "port": Config.FLASK_PORT
    }
    
    if provider == 'dual_model':
        siliconflow_config = Config.get_siliconflow_config()
        minimax_config = Config.get_minimax_config()
        status_info["llm_provider"] = "Dual_Model"
        status_info["mode"] = "MiniMax生成 + SiliconFlow优化"
        status_info["minimax_model"] = minimax_config.get('model_name')
        status_info["siliconflow_model"] = siliconflow_config.get('model_name')
        status_info["minimax_api_key_configured"] = bool(minimax_config.get('api_key'))
        status_info["siliconflow_api_key_configured"] = bool(siliconflow_config.get('api_key'))
    elif provider == 'siliconflow':
        config = Config.get_siliconflow_config()
        status_info["llm_provider"] = "SiliconFlow"
        status_info["model"] = config.get('model_name')
        status_info["api_key_configured"] = bool(config.get('api_key'))
    elif provider == 'minimax':
        config = Config.get_minimax_config()
        status_info["llm_provider"] = "MiniMax"
        status_info["model"] = config.get('model_name')
        status_info["api_key_configured"] = bool(config.get('api_key'))
    else:
        status_info["llm_provider"] = "None"
        status_info["api_key_configured"] = False
    
    return jsonify(status_info)


if __name__ == '__main__':
    print("=" * 60)
    print("统一 LLM 服务启动中...")
    print("=" * 60)
    
    provider_info = get_llm_provider()
    provider = provider_info['provider']
    
    if provider == 'dual_model':
        siliconflow_config = Config.get_siliconflow_config()
        minimax_config = Config.get_minimax_config()
        print("✅ 检测到双模型配置，已启用双模型协作模式")
        print("=" * 60)
        print("工作模式: MiniMax 生成初步回答 + SiliconFlow 优化回答")
        print("-" * 60)
        print("[MiniMax 配置]")
        print(f"  模型: {minimax_config.get('model_name')}")
        print(f"  API Key 已配置: {'是' if minimax_config.get('api_key') else '否'}")
        print("-" * 60)
        print("[SiliconFlow 配置]")
        print(f"  模型: {siliconflow_config.get('model_name')}")
        print(f"  API Key 已配置: {'是' if siliconflow_config.get('api_key') else '否'}")
        print("-" * 60)
        print("工作流程:")
        print("  1. 用户问题 → MiniMax → 初步回答")
        print("  2. 初步回答 → SiliconFlow → 优化后的回答")
        print("  3. 返回优化后的回答给用户")
    elif provider == 'siliconflow':
        config = Config.get_siliconflow_config()
        print("LLM 提供商: SiliconFlow (硅基流动)")
        print(f"模型: {config.get('model_name')}")
        print(f"API Key 已配置: {'是' if config.get('api_key') else '否'}")
    elif provider == 'minimax':
        config = Config.get_minimax_config()
        print("LLM 提供商: MiniMax")
        print(f"模型: {config.get('model_name')}")
        print(f"API Key 已配置: {'是' if config.get('api_key') else '否'}")
    else:
        print("警告: 未配置任何 LLM API Key")
        print("请在 .env 文件中配置以下任一选项:")
        print("  1. 同时配置 SILICONFLOW_API_KEY 和 MINIMAX_API_KEY（启用双模型协作模式）")
        print("  2. 仅配置 SILICONFLOW_API_KEY")
        print("  3. 仅配置 MINIMAX_API_KEY")
    
    print(f"服务端口: {Config.FLASK_PORT}")
    print("=" * 60)
    print(f"服务地址: http://127.0.0.1:{Config.FLASK_PORT}/generate")
    print(f"健康检查: http://127.0.0.1:{Config.FLASK_PORT}/health")
    print("=" * 60)
    
    app.run(port=Config.FLASK_PORT, debug=False, host='127.0.0.1')
