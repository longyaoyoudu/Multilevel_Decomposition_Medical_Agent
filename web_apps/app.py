import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from qa_llm_enhanced.chat_with_llm import KGRAG, model
import time
import json
import re

# 初始化问答系统
chatbot = KGRAG()

# 全局开关
USE_MULTI_CHANNEL = {"value": True}  # 是否使用多路召回
USE_RERANKER = {"value": True}  # 是否使用Reranker


# 颜色编码的日志格式化
def color_log(text, color="black"):
    """为不同级别的日志添加颜色"""
    color_map = {
        "error": "#ff4d4d",
        "warning": "#ff9900",
        "success": "#33cc33",
        "info": "#3399ff"
    }
    color_code = color_map.get(color, "black")
    return f"<span style='color: {color_code};'>{text}</span>"


# 处理用户查询的核心函数
def process_query(query, history=None, use_multi_channel=True, use_reranker=True):
    """处理用户查询并返回响应和日志

    Args:
        query: 用户问题
        history: 对话历史（暂未使用）
        use_multi_channel: 是否使用多路召回
        use_reranker: 是否使用Reranker
    """
    start_time = time.time()
    logs = []

    # 初始化日志
    logs.append(color_log(f"用户查询: {query}", "info"))
    logs.append(color_log(f"配置: 多路召回={use_multi_channel}, Reranker={use_reranker}", "info"))

    try:
        if use_multi_channel and use_reranker:
            # 新版流程：多路召回 + Reranker
            return process_query_v2(query, logs, start_time)
        else:
            # 旧版流程：单一图谱召回
            return process_query_v1(query, logs, start_time)

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        return "系统处理出错，请稍后再试", "\n".join(logs)


def process_query_v1(query, logs, start_time):
    """旧版流程：单一图谱召回"""
    logs.append(color_log("【流程选择】旧版流程 - 单一图谱召回", "info"))

    try:
        # Step 1: 实体链接
        logs.append(color_log("【Step 1】实体链接...", "info"))
        entity_dict = chatbot.entity_linking(query)
        logs.append(color_log(f"✅ 识别到实体: {entity_dict}", "success"))

        if not entity_dict:
            logs.append(color_log("⚠️ 未识别到相关医疗实体", "warning"))
            return "抱歉，这个问题超出了我的知识范围，无法回答。", "\n".join(logs)

        # Step 2: 知识图谱召回
        depth = 1
        logs.append(color_log(f"【Step 2】知识图谱召回 (深度={depth})...", "info"))
        facts = set()

        for entity_name, types in entity_dict.items():
            for entity_type in types:
                logs.append(color_log(f"🔍 处理实体: {entity_name}({entity_type})", "info"))
                rels = chatbot.link_entity_rel(query, entity_name, entity_type)
                logs.append(color_log(f"  相关关系: {rels}", "success"))

                entity_triples = chatbot.recall_facts(rels, entity_type, entity_name, depth)
                facts.update(entity_triples)
                logs.append(color_log(f"  召回三元组: {len(entity_triples)}条", "success"))

        max_facts = 15
        facts_list = list(facts)
        if len(facts_list) > max_facts:
            logs.append(color_log(f"⚠️ 三元组数量超过{max_facts}，进行截断", "warning"))
            facts_list = facts_list[:max_facts]

        # Step 3: 构建Prompt
        logs.append(color_log("【Step 3】构建Prompt...", "info"))
        fact_prompt = chatbot.format_prompt(query, facts_list)

        # Step 4: LLM生成答案
        logs.append(color_log("【Step 4】LLM生成答案...", "info"))
        answer = chatbot.chat(query)
        logs.append(color_log(f"✅ 生成答案成功", "success"))

        elapsed = time.time() - start_time
        logs.append(color_log(f"⏱️ 总处理时间: {elapsed:.2f}秒", "info"))

        return answer, "\n".join(logs)

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        return "系统处理出错，请稍后再试", "\n".join(logs)


def process_query_v2(query, logs, start_time):
    """新版流程：多路召回 + Reranker"""
    logs.append(color_log("【流程选择】新版流程 - 多路召回 + Reranker", "info"))

    try:
        # Step 1: 多路召回
        logs.append(color_log("【Step 1】多路召回...", "info"))
        hybrid = chatbot._get_hybrid_retriever()
        candidates = hybrid.retrieve(query, use_vector=True, use_graph=True)
        logs.append(color_log(f"✅ 多路召回返回 {len(candidates)} 条候选", "success"))

        if not candidates:
            logs.append(color_log("⚠️ 多路召回未返回候选", "warning"))
            return "抱歉，这个问题超出了我的知识范围，无法回答。", "\n".join(logs)

        # Step 2: Reranker重排序
        logs.append(color_log("【Step 2】Reranker重排序...", "info"))
        reranker = chatbot._get_reranker()
        reranked = reranker.rerank(query, candidates)
        logs.append(color_log(f"✅ Reranker返回 {len(reranked)} 条精排结果", "success"))

        if not reranked:
            logs.append(color_log("⚠️ Reranker未返回结果", "warning"))
            return "抱歉，这个问题超出了我的知识范围，无法回答。", "\n".join(logs)

        # Step 3: 构建Prompt (使用精排结果)
        logs.append(color_log("【Step 3】构建Prompt...", "info"))
        facts = [r['triple'] for r in reranked]
        fact_prompt = chatbot.build_enhanced_prompt(query, facts)

        # Step 4: 直接调用LLM生成答案 (不重复走召回流程)
        logs.append(color_log("【Step 4】LLM生成答案...", "info"))
        answer, _ = model.chat(query=fact_prompt, history=[])
        logs.append(color_log(f"✅ 生成答案成功", "success"))

        # Step 5: 答案润色优化
        logs.append(color_log("【Step 5】答案润色优化...", "info"))
        answer = chatbot.polish_answer(query, answer)
        logs.append(color_log(f"✅ 答案润色完成", "success"))

        elapsed = time.time() - start_time
        logs.append(color_log(f"⏱️ 总处理时间: {elapsed:.2f}秒", "info"))

        return answer, "\n".join(logs)

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        return "系统处理出错，请稍后再试", "\n".join(logs)


# 创建Gradio界面
with gr.Blocks(title="基于LLM和知识图谱的私人医生智能体", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 基于LLM和知识图谱的私人医生智能体")
    gr.Markdown("基于知识图谱与大语言模型的私人医生智能体，可查询疾病、症状、治疗方案等信息")

    with gr.Row():
        with gr.Column(scale=7):
            input_query = gr.Textbox(
                label="请输入相关问题",
                placeholder="例如：乳腺癌的症状有哪些？",
                lines=3
            )

            with gr.Row():
                submit_btn = gr.Button("提交查询", variant="primary")
                clear_btn = gr.Button("清除内容")

            # 多路召回和Reranker开关
            with gr.Row():
                use_multi_channel = gr.Checkbox(
                    label="启用多路召回",
                    value=True,
                    info="启用后使用 Aho-Corasick + 图谱多跳 + 向量检索 三路召回"
                )
                use_reranker = gr.Checkbox(
                    label="启用Reranker",
                    value=True,
                    info="启用后使用 MiniMax-M2.7-highspeed 进行重排序"
                )

        with gr.Column(scale=3):
            gr.Markdown("### 示例问题")
            gr.Examples(
                examples=[
                    "糖尿病应该吃什么食物？",
                    "什么人容易得高血压？",
                    "乳腺癌的症状有哪些？",
                    "感冒要多久才能好？"
                ],
                inputs=input_query
            )

    with gr.Row():
        with gr.Column():
            output_answer = gr.Textbox(
                label="系统回答",
                interactive=False,
                lines=8
            )

        # with gr.Column():
        #     process_logs = gr.HTML(
        #         label="处理日志",
        #         value="<div style='height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'></div>"
        #     )
        # 添加折叠面板
        with gr.Accordion("查看处理日志", open=False):  # open=False表示默认折叠
            process_logs = gr.HTML(
                label="处理日志",
                value="<div style='height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'></div>"
            )

    # 事件处理
    submit_btn.click(
        fn=process_query,
        inputs=[input_query, gr.State(), use_multi_channel, use_reranker],
        outputs=[output_answer, process_logs]
    )

    clear_btn.click(
        fn=lambda: ("", "",
                    "<div style='height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 5px;'></div>"),
        inputs=[],
        outputs=[input_query, output_answer, process_logs]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        # favicon_path="medical_icon.png"  # 可选的医疗图标
    )