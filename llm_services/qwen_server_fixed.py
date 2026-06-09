#!/usr/bin/env python3
# coding = utf-8
import os
import torch
# If you want to limit visible GPUs, keep this; otherwise remove or change it.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Choose device dynamically so the script can run on CPU or GPU depending on availability.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import json
from flask import Flask, request, jsonify
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# If you are using ModelScope mirroring for transformers downloads, keep the endpoint; otherwise this can be omitted.
os.environ['MODELSCOPE_ENDPOINT'] = 'https://mirror.sjtu.edu.cn'

# Path to the fine-tuned local checkpoint (replace with the actual path accessible at runtime).
# Make sure the runtime (where you run this script) can access this path.
model_path = r"/data/AIfusion/Tony2016Edu/Ruilong_Jin/Program/qwen_program/medical_output/Qwen3-32B/checkpoint-1084"

# Load tokenizer and model from the local checkpoint. trust_remote_code=True is often
# required for Qwen-style models that ship custom model/tokenizer classes.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# Move model to device
model = model.to(device)

def predict_model_fixed(data):
    """修复版本的模型预测函数"""
    # 提取消息内容
    if "message" in data and isinstance(data["message"], list):
        text = data["message"][0]["content"]
    else:
        text = data.get("message", "")
    
    # 设置默认生成参数
    max_new_tokens = data.get("max_tokens", 512)
    top_k = data.get("top_k", 50)
    top_p = data.get("top_p", 0.9)
    temperature = data.get("temperature", 0.7)
    repetition_penalty = data.get("repetition_penalty", 1.1)
    num_beams = data.get("num_beams", 1)
    
    print(f"生成参数: max_tokens={max_new_tokens}, temperature={temperature}")
    
    try:
        # 编码输入
        inputs = tokenizer(text, return_tensors='pt').to(device)
        
        # 生成响应
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            top_k=top_k, 
            top_p=top_p, 
            temperature=temperature, 
            repetition_penalty=repetition_penalty, 
            num_beams=num_beams,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 解码响应
        response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        print(f"生成的响应: {response}")
        return response
        
    except Exception as e:
        print(f"模型生成错误: {str(e)}")
        return f"模型生成错误: {str(e)}"

app = Flask(import_name=__name__)

@app.route("/generate", methods=["POST", "GET"])
def generate():
    """生成接口 - 兼容多种请求格式"""
    try:
        data = json.loads(request.data)
        print(f"收到请求数据: {data}")
        
        # 检查必需字段
        if not data or ("message" not in data and "query" not in data):
            return jsonify({"output":[""], "status":"error", "error": "缺少message或query字段"})
        
        # 调用修复后的预测函数
        res = predict_model_fixed(data)
        label = "success"
        
    except Exception as e:
        print(f"接口错误: {str(e)}")
        res = ""
        label = "error"
        
    return jsonify({"output":[res], "status":label})

@app.route("/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return jsonify({"status": "healthy", "model_loaded": True})

@app.route("/test", methods=["POST"])
def test_generation():
    """测试生成接口"""
    test_data = {
        "message": [{"role": "user", "content": "你好，请简单介绍一下自己"}],
        "max_tokens": 100,
        "temperature": 0.7
    }
    try:
        response = predict_model_fixed(test_data)
        return jsonify({"output": [response], "status": "success"})
    except Exception as e:
        return jsonify({"output": [""], "status": "error", "error": str(e)})

if __name__ == '__main__':
    print("启动修复后的Qwen服务器...")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")
    app.run(port=3001, debug=False, host='0.0.0.0')