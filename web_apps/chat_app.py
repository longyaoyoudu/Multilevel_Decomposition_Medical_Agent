#!/usr/bin/env python3
# coding: utf-8
"""
医疗问答系统前端应用
提供简约风格的聊天界面，支持双模型协作
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config

app = Flask(__name__)

LLM_SERVICE_URL = Config.LLM_SERVICE_URL

@app.route('/')
def index():
    """首页路由"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """聊天API接口"""
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        history = data.get('history', [])
        
        if not user_message:
            return jsonify({
                "error": "消息内容不能为空",
                "status": "error"
            }), 400
        
        messages = []
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'assistant':
                role = 'assistant'
            messages.append({
                "role": role,
                "content": content
            })
        
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        payload = {
            "message": messages,
            "history": []
        }
        
        try:
            response = requests.post(
                LLM_SERVICE_URL,
                json=payload,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return jsonify(result)
            else:
                return jsonify({
                    "error": f"LLM服务错误: {response.status_code}",
                    "status": "error"
                }), 500
                
        except requests.exceptions.Timeout:
            return jsonify({
                "error": "请求超时，请稍后重试（双模型协作可能需要较长时间）",
                "status": "error"
            }), 504
        except requests.exceptions.ConnectionError:
            return jsonify({
                "error": f"无法连接到LLM服务，请确保服务已启动: {LLM_SERVICE_URL}",
                "status": "error"
            }), 502
        except Exception as e:
            return jsonify({
                "error": f"请求失败: {str(e)}",
                "status": "error"
            }), 500
            
    except Exception as e:
        return jsonify({
            "error": f"服务器错误: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    try:
        response = requests.get(
            f"http://127.0.0.1:{Config.FLASK_PORT}/health",
            timeout=5
        )
        
        if response.status_code == 200:
            llm_status = response.json()
            return jsonify({
                "status": "ok",
                "service": "chat-app",
                "llm_service": llm_status
            })
        else:
            return jsonify({
                "status": "warning",
                "service": "chat-app",
                "llm_service": "unavailable"
            })
    except:
        return jsonify({
            "status": "warning",
            "service": "chat-app",
            "llm_service": "unavailable"
        })

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 医疗问答系统前端应用")
    print("=" * 60)
    print(f"LLM服务地址: {LLM_SERVICE_URL}")
    print(f"前端端口: 7861")
    print("=" * 60)
    print(f"访问地址: http://127.0.0.1:7861/")
    print("=" * 60)
    print("提示: 请确保先启动LLM服务:")
    print(f"  python llm_services/unified_llm_service.py")
    print("=" * 60)
    
    app.run(
        port=7861,
        debug=False,
        host='127.0.0.1'
    )
