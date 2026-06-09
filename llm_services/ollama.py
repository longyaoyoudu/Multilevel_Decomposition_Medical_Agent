# coding: utf-8
import os
import json
from flask import Flask, request, jsonify
import requests  # 用于调用本地 Ollama API

app = Flask(__name__)

# Ollama 配置
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Ollama 默认 API 地址
MODEL_NAME = "qwen3:0.6b"  # 替换为您使用的 Ollama 模型名称


def predict_ollama(data):
    """
    调用本地 Ollama API 进行模型推理
    """
    # 构造 Ollama API 请求体
    payload = {
        "model": MODEL_NAME,
        "prompt": data["message"][0]["content"],
        "stream": False,  # 非流式响应
        "options": {
            "temperature": data.get("temperature", 0.8),
            "top_p": data.get("top_p", 0.9),
            "top_k": data.get("top_k", 40),
            "num_predict": data.get("max_tokens", 512),
            "repeat_penalty": data.get("repetition_penalty", 1.1),
            "seed": data.get("seed", 42),
            "stop": data.get("stop", []),
        }
    }

    # 添加系统提示（如果提供）
    if "system" in data:
        payload["system"] = data["system"]

    # 添加对话历史（如果提供）
    if "history" in data:
        payload["context"] = data["history"]

    # 发送请求到本地 Ollama 服务
    response = requests.post(
        OLLAMA_API_URL,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120  # 超时时间设置为120秒
    )

    # 检查响应状态
    if response.status_code != 200:
        raise Exception(f"Ollama API 错误: {response.status_code} - {response.text}")

    # 解析 Ollama 响应
    ollama_response = response.json()

    # 返回生成的文本
    return ollama_response["response"]


@app.route("/generate", methods=["POST"])
def generate():
    """
    API 端点：处理生成请求
    """
    try:
        # 解析 JSON 请求数据
        data = request.get_json()
        print("收到请求数据:", data)

        # 调用 Ollama 模型
        generated_text = predict_ollama(data)

        # 获取用户输入和当前历史
        user_input = data["message"][0]["content"]
        input_history = data.get("history", [])

        # 更新历史记录（如果需要）
        updated_history = input_history + [[user_input, generated_text]]

        # 返回响应
        return jsonify({
            "output": [generated_text],
            "history": updated_history,
            "status": "success"
        })

    except Exception as e:
        # 错误处理
        print(f"处理请求时出错: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


if __name__ == '__main__':
    # 启动 Flask 应用
    app.run(port=3001, debug=False, host='0.0.0.0')