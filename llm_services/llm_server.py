# coding = utf-8
import os
import re
from tqdm import tqdm
import requests
import json
import time


# class ModelAPI():
#     def __init__(self, MODEL_URL):
#         self.url = MODEL_URL
#         return
#
#     def send_request(self, message, history):
#         data = json.dumps({"message":message, "history":history})
#         headers = {'Content-Type': 'application/json'}
#         try:
#             res = requests.post(self.url, data=data, headers=headers)
#             print(res)
#             predict = json.loads(res.text)["output"][0]
#             history = json.loads(res.text)["history"]
#             return predict, history
#         except Exception as e:
#             print("request error", e)
#             return "", []
#
#     ## 防止并不稳定，需要多次访问
#     def chat(self, query, history=[]):
#         message = [{"role": "user", "content": query}]
#         count = 0
#         response = ''
#         history = []
#         while count <=10:
#             try:
#                 count +=1
#                 response, history = self.send_request(message, history)
#                 if response:
#                     return response, history
#             except Exception as e:
#                 print('Exception:', e)
#                 time.sleep(1)
#         return response, history

class ModelAPI():
    def __init__(self, MODEL_URL):
        self.url = MODEL_URL
        self.timeout = 180  # 增加超时时间为180秒，支持双模型协作（MiniMax + SiliconFlow）
        return

    def send_request(self, message, history):
        data = json.dumps({
            "message": message,
            "history": history or []  # 确保总是发送有效的历史记录
        })
        headers = {'Content-Type': 'application/json'}
        try:
            res = requests.post(
                self.url,
                data=data,
                headers=headers,
                timeout=self.timeout
            )
            print("Raw response:", res.text)  # 调试用：打印原始响应

            response_data = json.loads(res.text)

            # 更健壮的响应解析
            predict = response_data.get("output", [""])[0]  # 获取output，默认为[""]

            # 关键修改：使用get()方法提供默认值
            new_history = response_data.get("history", history)  # 如果缺少history字段，使用传入的history

            return predict, new_history

        except Exception as e:
            print(f"Request error: {str(e)}")
            if hasattr(e, 'response'):  # 如果有响应内容，打印出来
                print(f"Error response: {e.response.text}")
            return "", history  # 返回原始history以便重试

    def chat(self, query, history=None):
        history = history or []  # 处理None情况
        message = [{"role": "user", "content": query}]

        for attempt in range(3):  # 减少重试次数为3次
            try:
                response, new_history = self.send_request(message, history)
                if response:  # 如果有有效响应
                    return response, new_history
            except Exception as e:
                print(f'Attempt {attempt + 1} failed:', e)
                time.sleep(min(2 ** attempt, 4))  # 指数退避策略

        return "", history  # 最终失败时返回最后已知的历史

if __name__ == '__main__':
    model = ModelAPI(MODEL_URL="http://127.0.0.1:3001/generate")
    res= model.chat(query="你叫啥", history=[])
    print(res)
