import os
import json
from flask import Flask, request, jsonify
import requests  # 新增requests库用于API调用

app = Flask(__name__)

# 硅基流动API配置（需替换为你的实际信息）
SILICONFLOW_API_KEY = "sk-ryjmxpcquxrpuxuotxevpksbgezsvbdmdieeopkingsmmuis"  # 替换为你的API密钥
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动API端点
MODEL_NAME = "deepseek-ai/DeepSeek-V3"  # 或硅基平台上的其他模型名称


def predict_siliconflow(data):
    """调用硅基流动API进行模型推理"""
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": data["message"],
        "max_tokens": data.get("max_tokens", 512),
        "temperature": data.get("temperature", 0.8),
        "top_p": data.get("top_p", 0.9),
        "repetition_penalty": data.get("repetition_penalty", 1.1),
        "stream": False
    }

    response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload)
    response.raise_for_status()  # 如果请求失败会抛出异常

    result = response.json()
    return result["choices"][0]["message"]["content"]


@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        print("Received request data:", data)

        # 构造硅基API请求
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "messages": data["message"],
            "temperature": 0.7
        }

        # 调用API
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        # 1. 获取当前输入和历史
        user_input = data["message"][0]["content"]
        input_history = data.get("history", [])

        # 2. 调用模型生成回复（伪代码）
        model_response = response.json()["choices"][0]["message"]["content"]  # 实际替换为模型调用

        # 3. 更新历史记录
        updated_history = input_history + [[user_input, model_response]]

        return jsonify({
            "output": [model_response],
            "history": updated_history
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=3001, debug=False, host='127.0.0.1')