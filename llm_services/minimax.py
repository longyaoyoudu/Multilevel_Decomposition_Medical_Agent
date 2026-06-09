# coding: utf-8
import os
import sys
import json
from flask import Flask, request, jsonify
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

app = Flask(__name__)


def predict_minimax(data):
    """调用Minimax API进行模型推理"""
    minimax_config = Config.get_minimax_config()
    gen_params = Config.get_generation_params()
    
    api_key = minimax_config['api_key']
    group_id = minimax_config['group_id']
    api_url = minimax_config['api_url']
    model_name = minimax_config['model_name']
    
    if not api_key:
        raise ValueError("MINIMAX_API_KEY 未设置，请在.env文件中配置")
    
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
        "max_tokens": data.get("max_tokens", gen_params['max_tokens']),
        "temperature": data.get("temperature", gen_params['temperature']),
        "top_p": data.get("top_p", gen_params['top_p']),
        "stream": False
    }
    
    print(f"调用Minimax API，模型: {model_name}")
    print(f"请求URL: {api_url}")
    print(f"参数: {params}")
    print(f"请求体: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    response = requests.post(
        api_url,
        headers=headers,
        params=params,
        json=payload,
        timeout=120
    )
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    if response.status_code != 200:
        raise Exception(f"Minimax API 错误: {response.status_code} - {response.text}")
    
    result = response.json()
    
    if "base_resp" in result and result["base_resp"].get("status_code") != 0 and result["base_resp"].get("status_code") != 200:
        status_code = result["base_resp"].get("status_code")
        status_msg = result["base_resp"].get("status_msg", "未知错误")
        error_msg = f"MiniMax API 业务错误: [{status_code}] {status_msg}"
        if status_code == 2061:
            error_msg += "\n提示: 你的API套餐不支持当前模型，请在.env中切换到支持的模型（如 abab6.5s-chat）"
        raise Exception(error_msg)
    
    if "choices" in result and result["choices"] is not None and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    elif "reply" in result:
        return result["reply"]
    else:
        raise Exception(f"无法解析MiniMax响应: {result}")


@app.route("/generate", methods=["POST"])
def generate():
    """
    API 端点：处理生成请求
    与项目现有接口兼容
    """
    try:
        data = request.get_json()
        print("收到请求数据:", data)
        
        minimax_config = Config.get_minimax_config()
        if not minimax_config['api_key']:
            return jsonify({
                "error": "MINIMAX_API_KEY 未配置，请在.env文件中设置",
                "status": "error"
            }), 500
        
        generated_text = predict_minimax(data)
        
        user_input = data["message"][0]["content"] if data.get("message") else ""
        input_history = data.get("history", [])
        
        updated_history = input_history + [[user_input, generated_text]]
        
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
    minimax_config = Config.get_minimax_config()
    return jsonify({
        "status": "ok",
        "service": "minimax-api",
        "model": minimax_config['model_name'],
        "api_key_configured": bool(minimax_config['api_key'])
    })


if __name__ == '__main__':
    minimax_config = Config.get_minimax_config()
    
    print("=" * 60)
    print("Minimax API 服务启动中...")
    print(f"模型: {minimax_config['model_name']}")
    print(f"API Key 已配置: {'是' if minimax_config['api_key'] else '否'}")
    print(f"Group ID 已配置: {'是' if minimax_config['group_id'] else '否'}")
    print(f"服务端口: {Config.FLASK_PORT}")
    print("=" * 60)
    
    if not minimax_config['api_key']:
        print("\n⚠️  警告: MINIMAX_API_KEY 未设置")
        print("请创建 .env 文件并添加配置:")
        print("  MINIMAX_API_KEY=your_api_key_here")
        print()
    
    app.run(port=Config.FLASK_PORT, debug=False, host='127.0.0.1')
