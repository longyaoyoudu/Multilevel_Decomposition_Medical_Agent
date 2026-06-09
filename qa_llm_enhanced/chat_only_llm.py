# coding = utf-8
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from qa_traditional.question_classifier import *
from llm_services.llm_server import *


class MedicalAssistant():
    def __init__(self):
        self.model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")
        self.entity_parser = QuestionClassifier()
        self.fallback_responses = [
            "这个问题我需要查阅更多资料才能回答",
            "目前我还没有掌握这方面的信息",
            "关于这个问题，我建议咨询专业医生",
            "我还在学习中，暂时无法回答这个问题"
        ]
        return

    def build_medical_prompt(self, query):
        """构建医疗问答提示词"""
        return f"""【角色】你是一位专业的医疗问答助手。
【核心规则】
1. 基于你的医学知识回答问题
2. 如果不确定或超出知识范围，回答："根据现有知识无法回答此问题。"
3. 禁止编造信息
4. 始终建议用户咨询专业医生获取诊断和治疗建议
5. 回答简洁准确，不超过3句话

【问题】{query}

【回答】"""

    def handle_api_error(self, retries=3, delay=1):
        """处理API错误的重试机制"""
        for attempt in range(retries):
            print(f"⚠️ API错误，尝试重试 ({attempt + 1}/{retries})...")
            time.sleep(delay)  # 等待一段时间再重试
            try:
                # 发送一个简单的测试查询检查API是否恢复
                test_response, _ = self.model.chat(query="你好", history=[])
                if test_response.strip():
                    return True  # API恢复正常
            except:
                continue
        return False  # 所有重试都失败

    def chat(self, query):
        # 构建专业的医疗提示词
        prompt = self.build_medical_prompt(query)

        try:
            # 直接使用LLM生成答案
            answer, _ = self.model.chat(query=prompt, history=[])

            # 检查返回结果是否有效
            if not answer or answer.strip() == "":
                # 返回空结果时的处理
                return "抱歉，我没有获取到有效的回答。请尝试换一种方式提问或稍后再试。"

            return answer

        except Exception as e:
            print(f"❌ LLM调用失败: {str(e)}")

            # 尝试处理API错误
            if self.handle_api_error():
                # API恢复后重新尝试
                try:
                    answer, _ = self.model.chat(query=prompt, history=[])
                    return answer
                except:
                    pass  # 如果仍然失败，则使用备用响应

            # 提供友好的备用响应
            import random
            return f"⚠️ 服务暂时不可用。{random.choice(self.fallback_responses)}"


if __name__ == "__main__":
    assistant = MedicalAssistant()
    print("医疗问答助手已启动，输入'退出'结束对话")
    while True:
        query = input("用户:").strip()
        if query.lower() in ["退出", "exit", "quit"]:
            print("对话结束，再见！")
            break

        answer = assistant.chat(query)
        print("智能医生:", answer)