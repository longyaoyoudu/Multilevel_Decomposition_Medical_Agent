# coding: utf-8
"""
Reranker模块
使用MiniMax-M2.7-highspeed对候选三元组进行相关性打分和重排序
"""
import os
import sys
import json
import re
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import Config


class Reranker:
    """Reranker重排序器，使用LLM进行相关性评分"""

    # 备用模型列表（按优先级）
    FALLBACK_MODELS = [
        ('SiliconFlow', 'deepseek-ai/DeepSeek-V3'),  # SiliconFlow免费模型
        ('SiliconFlow', 'Qwen/Qwen2.5-72B-Instruct'),  # SiliconFlow备选
        ('MiniMax', 'abab6.5s-chat'),  # MiniMax免费模型
    ]

    def __init__(self):
        """初始化Reranker"""
        # 优先使用SiliconFlow（免费模型），如果不可用则降级到MiniMax免费模型
        self.api_key = Config.SILICONFLOW_API_KEY
        self.api_url = Config.SILICONFLOW_API_URL
        self.model_name = Config.SILICONFLOW_MODEL_NAME

        # 如果SiliconFlow未配置，使用MiniMax
        if not self.api_key:
            self.api_key = Config.MINIMAX_API_KEY
            self.api_url = Config.MINIMAX_API_URL
            self.model_name = 'abab6.5s-chat'  # MiniMax免费模型

        self.final_top_k = Config.RERANK_FINAL_K

        self._init_client()
        print(f"[Reranker] 初始化完成，模型: {self.model_name}")

    def _init_client(self):
        """初始化客户端（使用HTTP请求方式，兼容性好）"""
        self.client = None  # 不使用OpenAI SDK，避免httpx兼容性问题
        print("[Reranker] 使用HTTP请求方式初始化完成")

    def _build_scoring_prompt(self, query: str, triple: str) -> str:
        """
        构建单条打分Prompt

        Args:
            query: 用户问题
            triple: 候选三元组字符串

        Returns:
            str: 完整的prompt
        """
        return f"""【角色】你是一个专业的医疗知识相关性评分专家。
【任务】评估知识三元组与用户问题的相关性。
【评分标准】
- 5分：高度相关，直接回答问题
- 4分：相关，提供有效信息
- 3分：弱相关，信息部分有用
- 2分：间接相关，需进一步推理
- 1分：不相关，无法回答问题
- 0分：矛盾/错误知识

【用户问题】
{query}

【候选知识】
{triple}

【输出格式】
请只返回一个整数分数（0-5），不要其他文字。"""

    def _parse_score(self, response: str) -> int:
        """
        从LLM响应中解析分数

        Args:
            response: LLM响应文本

        Returns:
            int: 分数（0-5）
        """
        # 尝试提取数字
        match = re.search(r'\d', response.strip())
        if match:
            score = int(match.group())
            return min(5, max(0, score))  # 确保在0-5范围内
        return 0

    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM获取打分（使用HTTP请求方式）

        Args:
            prompt: 提示词

        Returns:
            str: LLM响应
        """
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
            print(f"[Reranker] HTTP状态码: {response.status_code}")

            if response.status_code != 200:
                print(f"[Reranker] API错误响应: {response.text[:200]}")
                return "0"

            result = response.json()
            if result is None:
                print(f"[Reranker] 响应JSON解析失败: {response.text[:100]}")
                return "0"

            if 'choices' not in result or not result['choices']:
                print(f"[Reranker] 响应缺少choices: {result}")
                return "0"

            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"[Reranker] 请求异常: {e}")
            return "0"

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        对候选三元组进行重排序

        由于LLM API存在不稳定性和调用限制，
        当前使用RRF融合分数直接作为排序依据，
        不再额外调用LLM进行评分。

        Args:
            query: 用户问题
            candidates: 候选列表 [{'triple': '<头实体,关系,尾实体>', 'total_score': float, ...}, ...]

        Returns:
            list: 重排序后的列表
        """
        if not candidates:
            print("[Reranker] 候选列表为空，跳过重排序")
            return []

        print(f"\n[{'='*60}")
        print(f"[Reranker] 融合排序（使用RRF分数）")
        print(f"   Query: {query}")
        print(f"   候选数量: {len(candidates)}")
        print(f"{'='*60}")

        # 直接使用RRF融合的total_score进行排序
        reranked = sorted(candidates, key=lambda x: x.get('total_score', 0), reverse=True)

        # 取Top-K
        final_results = reranked[:self.final_top_k]

        # 打印统计
        scores = [r.get('total_score', 0) for r in reranked]
        print(f"\n[Reranker] RRF分数统计:")
        print(f"   最高分: {max(scores):.4f}")
        print(f"   最低分: {min(scores):.4f}")
        print(f"   平均分: {sum(scores)/len(scores):.4f}")
        print(f"   最终输出: {len(final_results)} 条")

        # 打印Top-5详情
        print(f"\n[Reranker] Top-{len(final_results)} 结果:")
        for i, r in enumerate(final_results, 1):
            print(f"   {i}. [RRF:{r.get('total_score', 0):.4f}] {r['triple']}")

        return final_results

    def rerank_single(self, query: str, triple: str) -> int:
        """
        单条打分（调试用）

        Args:
            query: 用户问题
            triple: 三元组字符串

        Returns:
            int: 相关性分数
        """
        prompt = self._build_scoring_prompt(query, triple)
        try:
            response_text = self._call_llm(prompt)
            return self._parse_score(response_text)
        except Exception as e:
            print(f"[Reranker] 打分失败: {e}")
            return 0

    def get_config(self):
        """获取Reranker配置"""
        return {
            'model': self.model_name,
            'final_top_k': self.final_top_k,
            'api_url': self.api_url
        }


# 全局单例
_reranker = None


def get_reranker():
    """获取全局Reranker单例"""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试Reranker")
    print("=" * 60)

    reranker = Reranker()
    print(f"\n配置: {reranker.get_config()}")

    # 单条测试
    print("\n--- 单条打分测试 ---")
    query = "糖尿病有什么症状"
    triple = "<糖尿病,症状,多饮多尿>"
    score = reranker.rerank_single(query, triple)
    print(f"Query: {query}")
    print(f"Triple: {triple}")
    print(f"Score: {score}")

    # 批量测试
    print("\n--- 批量重排序测试 ---")
    candidates = [
        {'triple': '<糖尿病,症状,多饮多尿>'},
        {'triple': '<糖尿病,推荐药品,二甲双胍>'},
        {'triple': '<高血压,症状,头痛>'},
        {'triple': '<心脏病,治疗方式,搭桥手术>'},
        {'triple': '<感冒,症状,发烧>'}
    ]
    query = "糖尿病吃什么药"
    results = reranker.rerank(query, candidates)
    print(f"\n最终结果: {len(results)} 条")

    print("\n✅ Reranker测试完成")