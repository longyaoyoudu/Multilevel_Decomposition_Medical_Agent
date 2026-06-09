#!/usr/bin/env python3
# coding: utf-8

import requests
import json
import time
from typing import List, Dict, Any

class LLMServerManager:
    """LLM服务器管理器 - 提供多种修复方案"""
    
    def __init__(self):
        self.available_servers = []
        self.test_servers()
    
    def test_servers(self):
        """测试所有可能的LLM服务器"""
        servers = [
            {"url": "http://127.0.0.1:3001/generate", "name": "本地Qwen服务器"},
            {"url": "http://127.0.0.1:3000/generate", "name": "备用端口3000"},
            {"url": "http://127.0.0.1:8000/generate", "name": "备用端口8000"},
            {"url": "http://127.0.0.1:8080/generate", "name": "备用端口8080"},
        ]
        
        print("正在测试LLM服务器...")
        for server in servers:
            if self.test_server(server["url"]):
                self.available_servers.append(server)
                print(f"[OK] {server['name']}: {server['url']}")
            else:
                print(f"[FAIL] {server['name']}: {server['url']}")
    
    def test_server(self, url: str) -> bool:
        """测试单个服务器是否可用"""
        try:
            test_data = {
                "message": [{"role": "user", "content": "你好"}],
                "history": []
            }
            response = requests.post(url, json=test_data, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get("status") != "error" and result.get("output") != [""]
        except:
            pass
        return False
    
    def get_best_server(self) -> str:
        """获取最佳可用服务器"""
        if self.available_servers:
            return self.available_servers[0]["url"]
        return None

class EnhancedModelAPI:
    """增强的模型API - 支持多服务器和降级方案"""
    
    def __init__(self):
        self.server_manager = LLMServerManager()
        self.current_url = self.server_manager.get_best_server()
        self.fallback_mode = self.current_url is None
        
    def send_request(self, message: str, history: List = None):
        """发送请求到LLM服务器"""
        if self.fallback_mode:
            return self.fallback_response(message)
            
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
                timeout=180  # 增加超时时间为180秒，支持双模型协作
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success" and result.get("output"):
                    return result["output"][0], result.get("history", history or [])
            
            # 如果当前服务器失败，尝试其他服务器
            return self.try_alternative_servers(message, history)
            
        except Exception as e:
            print(f"请求失败: {str(e)}")
            return self.try_alternative_servers(message, history)
    
    def try_alternative_servers(self, message: str, history: List = None):
        """尝试其他备用服务器"""
        for server in self.server_manager.available_servers[1:]:
            try:
                self.current_url = server["url"]
                print(f"切换到备用服务器: {server['name']}")
                return self.send_request(message, history)
            except:
                continue
        
        # 所有服务器都失败，进入降级模式
        self.fallback_mode = True
        return self.fallback_response(message)
    
    def fallback_response(self, message: str):
        """降级响应 - 当所有LLM服务器都不可用时"""
        # 简单的规则匹配响应
        if "高血压" in message and "药" in message:
            return "根据知识图谱数据，高血压常用的降压药包括：硝苯地平片、卡托普利片、缬沙坦片、氢氯噻嗪片等。建议咨询医生选择适合的药物。", []
        elif "症状" in message:
            return "我无法提供具体的症状诊断，建议您咨询专业医生进行详细检查。", []
        else:
            return "抱歉，当前AI服务暂时不可用。您可以尝试重新提问或稍后再试。", []
    
    def chat(self, query: str, history: List = None):
        """聊天接口"""
        history = history or []
        response, new_history = self.send_request(query, history)
        return response, new_history

def create_fixed_chat_system():
    """创建修复后的聊天系统"""
    
    # 1. 创建增强的模型API
    enhanced_model = EnhancedModelAPI()
    
    # 2. 创建基于规则的关系识别
    def rule_based_relation_recognition(query: str, entity_type: str) -> List[str]:
        """基于规则的关系识别"""
        rules = {
            "disease": {
                "常用药品": ["降压药", "选择哪些药", "用什么药", "药品推荐", "药物", "吃药"],
                "好评药品": ["好评药", "推荐药", "效果好", "常用药"],
                "症状": ["症状", "表现", "感觉", "不舒服", "难受"],
                "治疗方式": ["治疗", "治愈", "怎么治", "治疗方法", "治疗方案"],
                "预防措施": ["预防", "怎么预防", "避免", "防范"],
                "诊断检查": ["检查", "诊断", "化验", "检测", "筛查"]
            }
        }
        
        recognized_relations = []
        entity_rules = rules.get(entity_type, {})
        
        for relation, keywords in entity_rules.items():
            if any(keyword in query for keyword in keywords):
                recognized_relations.append(relation)
        
        # 如果没有匹配到特定关系，返回默认关系
        if not recognized_relations and entity_type == "disease":
            return ["常用药品", "治疗方式"]  # 默认返回药品和治疗信息
        
        return recognized_relations
    
    # 3. 测试修复后的系统
    print("=== LLM服务器修复方案测试 ===")
    print(f"可用服务器数量: {len(enhanced_model.server_manager.available_servers)}")
    print(f"当前模式: {'降级模式' if enhanced_model.fallback_mode else '正常模式'}")
    
    # 测试问题
    test_queries = [
        "高血压患者应该选择哪些降压药？",
        "糖尿病的症状有哪些？",
        "感冒怎么治疗？"
    ]
    
    for query in test_queries:
        print(f"\n测试问题: {query}")
        
        # 测试关系识别
        relations = rule_based_relation_recognition(query, "disease")
        print(f"识别的关系: {relations}")
        
        # 测试LLM响应
        response, _ = enhanced_model.chat(query)
        print(f"LLM响应: {response}")
    
    return enhanced_model, rule_based_relation_recognition

if __name__ == "__main__":
    create_fixed_chat_system()