#!/usr/bin/env python3
# coding: utf-8

import requests
import json
from typing import List, Dict, Any

class RemoteServerConfig:
    """远程服务器配置管理器"""
    
    def __init__(self):
        self.servers = {
            "local": {
                "url": "http://127.0.0.1:3001/generate",
                "name": "本地服务器",
                "type": "local"
            },
            "remote_10": {
                "url": "http://10.139.21.11:3001/generate", 
                "name": "远程服务器(10.139.21.11)",
                "type": "remote"
            }
        }
        self.current_server = None
        self.test_servers()
    
    def test_servers(self):
        """测试所有服务器连接"""
        print("正在测试服务器连接...")
        for server_id, server_info in self.servers.items():
            if self.test_server(server_info["url"]):
                print(f"[SUCCESS] {server_info['name']}: {server_info['url']}")
                if not self.current_server:
                    self.current_server = server_id
            else:
                print(f"[FAILED] {server_info['name']}: {server_info['url']}")
    
    def test_server(self, url: str) -> bool:
        """测试单个服务器连接"""
        try:
            test_data = {
                "message": [{"role": "user", "content": "你好"}],
                "history": []
            }
            response = requests.post(url, json=test_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get("status") == "success" and result.get("output") != [""]
        except Exception as e:
            print(f"测试服务器 {url} 失败: {str(e)}")
        return False
    
    def get_current_url(self) -> str:
        """获取当前服务器URL"""
        if self.current_server:
            return self.servers[self.current_server]["url"]
        return None
    
    def switch_server(self, server_id: str) -> bool:
        """切换服务器"""
        if server_id in self.servers and self.test_server(self.servers[server_id]["url"]):
            self.current_server = server_id
            print(f"已切换到: {self.servers[server_id]['name']}")
            return True
        return False

class RemoteModelAPI:
    """远程模型API客户端"""
    
    def __init__(self):
        self.server_config = RemoteServerConfig()
        self.current_url = self.server_config.get_current_url()
        
    def send_request(self, message: str, history: List = None):
        """发送请求到远程服务器"""
        if not self.current_url:
            return "抱歉，没有可用的LLM服务器。", []
            
        data = json.dumps({
            "message": [{"role": "user", "content": message}],
            "history": history or []
        })
        
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(
                self.current_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success" and result.get("output"):
                    return result["output"][0], result.get("history", history or [])
                else:
                    return f"服务器返回错误: {result.get('status', 'unknown')}", []
            else:
                return f"HTTP错误: {response.status_code}", []
                
        except requests.exceptions.Timeout:
            return "请求超时，请稍后重试。", []
        except requests.exceptions.ConnectionError:
            return "连接失败，请检查网络连接。", []
        except Exception as e:
            return f"请求异常: {str(e)}", []
    
    def chat(self, query: str, history: List = None):
        """聊天接口"""
        history = history or []
        response, new_history = self.send_request(query, history)
        return response, new_history

def test_remote_connection():
    """测试远程连接"""
    print("=== 远程服务器连接测试 ===")
    
    # 创建远程API客户端
    remote_api = RemoteModelAPI()
    
    if not remote_api.current_url:
        print("[ERROR] 没有可用的服务器")
        return None
    
    print(f"[SUCCESS] 当前使用服务器: {remote_api.server_config.servers[remote_api.server_config.current_server]['name']}")
    print(f"[INFO] 服务器地址: {remote_api.current_url}")
    
    # 测试连接
    test_queries = [
        "你好，请简单介绍一下自己",
        "高血压患者应该选择哪些降压药？",
        "糖尿病的症状有哪些？"
    ]
    
    for query in test_queries:
        print(f"\n测试问题: {query}")
        response, _ = remote_api.chat(query)
        print(f"服务器响应: {response}")
    
    return remote_api

def update_system_config():
    """更新系统配置以使用远程服务器"""
    
    # 更新 chat_with_llm.py 中的模型配置
    config_update = """
# 在 chat_with_llm.py 中更新模型配置
# 将原来的:
# model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")

# 更新为:
from remote_server_config import RemoteModelAPI
model = RemoteModelAPI()
"""
    print("系统配置更新建议:")
    print(config_update)
    
    return config_update

if __name__ == "__main__":
    # 测试远程连接
    api_client = test_remote_connection()
    
    if api_client:
        print("\n=== 远程连接测试成功 ===")
        print("✅ 系统现在可以使用远程LLM服务器")
        print("📋 配置更新建议:")
        update_system_config()
    else:
        print("\n=== 远程连接测试失败 ===")
        print("[ERROR] 请检查:")
        print("1. 服务器是否正在运行")
        print("2. 网络连接是否正常")
        print("3. 防火墙设置")
        print("4. 服务器地址是否正确")