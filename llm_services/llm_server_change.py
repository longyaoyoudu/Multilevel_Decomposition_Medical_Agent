#!/usr/bin/env python3
# coding: utf-8
import json
import time
import requests
from typing import Optional, List, Dict, Tuple, Any


class RobustModelAPI:
    def __init__(self, model_url: str, max_retries: int = 3, timeout: int = 15):
        """
        增强健壮性的模型API客户端

        参数:
            model_url: 模型服务端点URL
            max_retries: 最大重试次数 (默认3次)
            timeout: 请求超时时间(秒) (默认15秒)
        """
        self.url = model_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def validate_response(self, response_data: Dict) -> bool:
        """
        验证服务端响应是否有效

        参数:
            response_data: 解析后的JSON响应

        返回:
            bool: 响应是否有效
        """
        # 检查必须字段是否存在
        if "status" not in response_data:
            return False

        # 成功情况
        if response_data["status"] == "success":
            return "output" in response_data and isinstance(response_data["output"], list)

        # 错误情况
        if response_data["status"] == "error":
            print(f"服务端返回错误: {response_data.get('message', '无错误详情')}")
            return False

        return False

    def send_request(
            self,
            prompt: str,
            history: Optional[List] = None,
            retry_count: int = 0
    ) -> Tuple[str, List]:
        """
        发送请求到模型API (带自动重试)

        参数:
            prompt: 用户输入提示
            history: 对话历史
            retry_count: 当前重试次数

        返回:
            tuple: (response_text, new_history)
        """
        history = history or []
        payload = {
            "message": [{"role": "user", "content": prompt}],
            "history": history
        }

        try:
            response = self.session.post(
                self.url,
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.timeout
            )
            response.raise_for_status()

            response_data = response.json()

            # 调试日志
            print(f"Request: {payload['message'][0]['content'][:50]}...")
            print(f"Response: {json.dumps(response_data, ensure_ascii=False)[:200]}...")

            # 验证响应有效性
            if not self.validate_response(response_data):
                raise ValueError("Invalid server response structure")

            # 处理成功响应
            if response_data["status"] == "success":
                return (
                    response_data["output"][0] if response_data["output"] else "",
                    response_data.get("history", history)
                )

            # 处理错误响应
            return "", history

        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {retry_count + 1}): {str(e)}")
            if retry_count < self.max_retries - 1:
                wait_time = min(2 ** retry_count, 5)  # 指数退避，最多等待5秒
                time.sleep(wait_time)
                return self.send_request(prompt, history, retry_count + 1)
            return "服务暂时不可用，请稍后再试", []

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Response parsing failed: {str(e)}")
            if retry_count < self.max_retries - 1:
                return self.send_request(prompt, history, retry_count + 1)
            return "服务响应格式错误", []

    def chat(self, query: str, history: Optional[List] = None) -> Tuple[str, List]:
        """
        用户友好的聊天接口

        参数:
            query: 用户查询
            history: 对话历史

        返回:
            tuple: (response_text, new_history)
        """
        # 简化prompt工程
        simplified_prompt = f"""
        用户问题：{query}
        请以专业医疗助手的身份回答，要求：
        1. 基于医学知识回答
        2. 不知道时明确说明
        3. 回答格式：
        <思考过程>
        <最终答案>
        """

        response, new_history = self.send_request(simplified_prompt, history)

        # 处理空响应
        if not response.strip():
            return "未能获取有效回答，请尝试简化您的问题", new_history

        return response, new_history


if __name__ == '__main__':
    # 使用示例
    api = RobustModelAPI("http://127.0.0.1:3001/generate")

    # 测试查询
    test_queries = [
        "感冒有什么症状？",
        "高血压应该怎么预防？",
        "糖尿病患者的饮食建议"
    ]

    history = []
    for query in test_queries:
        print(f"\n用户: {query}")
        response, history = api.chat(query, history)
        print(f"助手: {response}")