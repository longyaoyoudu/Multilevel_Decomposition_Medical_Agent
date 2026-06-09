import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from qa_llm_enhanced.chat_llm_context import KGRAG
import time
import json

# 初始化问答系统
chatbot = KGRAG()


# 颜色编码的日志格式化
def color_log(text, color="black"):
    """为不同级别的日志添加颜色"""
    color_map = {
        "error": "#ff4d4d",
        "warning": "#ff9900",
        "success": "#33cc33",
        "info": "#3399ff",
        "system": "#666666"
    }
    color_code = color_map.get(color, "black")
    return f"<span style='color: {color_code};'>{text}</span>"


# 处理用户查询的核心函数 - 支持完整对话历史上下文
def process_query(query, history):
    """处理用户查询并返回响应和日志，支持完整对话上下文"""
    start_time = time.time()
    logs = []

    # 初始化日志
    logs.append(color_log(f"用户查询: {query}", "info"))

    try:
        # 构建完整的对话上下文
        context = ""
        if history:
            context = "\n".join([f"用户: {h[0]}\n助手: {h[1]}" for h in history])
            context += f"\n用户: {query}"
            logs.append(color_log(f"对话历史上下文:\n{context}", "info"))
        else:
            context = f"用户: {query}"

        # Step 1: 实体链接 - 使用完整上下文
        logs.append(color_log("【步骤1】实体识别...", "system"))
        entity_dict = chatbot.entity_linking(context)
        logs.append(color_log(f"✅ 识别到实体: {json.dumps(entity_dict, ensure_ascii=False)}", "success"))

        if not entity_dict:
            logs.append(color_log("⚠️ 未识别到相关实体", "warning"))
            answer = "抱歉，这个问题超出了我的知识范围。"
            new_history = history + [(query, answer)]
            chat_html = update_chat_display(new_history)
            return answer, "\n".join(logs), new_history, chat_html

        # Step 2: 知识图谱召回 - 使用完整上下文
        depth = 1
        logs.append(color_log(f"【步骤2】知识检索 (深度={depth})...", "system"))
        facts = set()

        for entity_name, types in entity_dict.items():
            for entity_type in types:
                # 关系链接 - 使用完整上下文
                logs.append(color_log(f"🔍 处理实体: {entity_name}({entity_type})", "info"))
                rels = chatbot.link_entity_rel(context, entity_name, entity_type)
                logs.append(color_log(f"  相关关系: {', '.join(rels)}", "success"))

                # 事实召回
                entity_triples = chatbot.recall_facts(rels, entity_type, entity_name, depth)
                facts.update(entity_triples)
                logs.append(color_log(f"  检索到三元组: {len(entity_triples)}条", "success"))

        # 结果截断
        max_facts = 15
        facts_list = list(facts)
        if len(facts_list) > max_facts:
            logs.append(color_log(f"⚠️ 三元组数量超过{max_facts}，进行截断", "warning"))
            facts_list = facts_list[:max_facts]

        # Step 3: 构建Prompt - 包含完整对话历史
        logs.append(color_log("【步骤3】构建提示词...", "system"))
        fact_prompt = chatbot.format_prompt(context, facts_list)
        logs.append(color_log(f"提示词内容:\n{fact_prompt}", "info"))

        # Step 4: LLM生成答案
        logs.append(color_log("【步骤4】生成回答...", "system"))
        answer = chatbot.chat(fact_prompt)
        logs.append(color_log(f"✅ 回答生成成功", "success"))

        # 计算处理时间
        elapsed = time.time() - start_time
        logs.append(color_log(f"⏱️ 总处理时间: {elapsed:.2f}秒", "info"))

        # 更新对话历史
        new_history = history + [(query, answer)]
        chat_html = update_chat_display(new_history)
        return answer, "\n".join(logs), new_history, chat_html

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        error_msg = "系统处理出错，请稍后再试"
        new_history = history + [(query, error_msg)]
        chat_html = update_chat_display(new_history)
        return error_msg, "\n".join(logs), new_history, chat_html


# 浅色主题CSS
light_theme_css = """
:root {
    --primary: #0a84ff;
    --primary-dark: #0066cc;
    --background: #f5f7fa;
    --surface: #ffffff;
    --surface-light: #f0f2f5;
    --text-primary: #1a1a1a;
    --text-secondary: #666666;
    --accent: #ff9500;
    --success: #30d158;
    --warning: #ff9500;
    --error: #ff453a;
    --border: #e0e0e0;
}

/* 聊天界面样式 */
.chat-container {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.chat-history {
    flex: 1;
    overflow-y: auto;
    margin-bottom: 15px;
    padding: 10px;
    background: var(--surface);
    border-radius: 10px;
    border: 1px solid var(--border);
    max-height: 500px;
}

.user-message {
    background-color: #e3f2fd;
    border-radius: 18px;
    padding: 10px 15px;
    margin: 8px 0;
    max-width: 85%;
    align-self: flex-end;
}

.assistant-message {
    background-color: #f5f5f5;
    border-radius: 18px;
    padding: 10px 15px;
    margin: 8px 0;
    max-width: 85%;
    align-self: flex-start;
}

.log-panel {
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 15px;
    max-height: 300px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.9rem;
}

/* 其他CSS规则保持不变... */
"""

# 创建聊天界面
with gr.Blocks(
        title="基于LLM和知识图谱的多层级拆解医疗智能体",
        css=light_theme_css + """
        /* 添加示例问题悬停效果 */
        .example:hover {
            background-color: var(--surface-light);
            cursor: pointer;
        }
        .header {
            background: linear-gradient(135deg, #0a84ff 0%, #0066cc 100%);
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            color: white;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        """,
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
        )
) as demo:
    # 存储对话历史的状态变量
    chat_history = gr.State([])

    # 页头
    with gr.Row(equal_height=True, elem_classes="header"):
        gr.Markdown("""
        <div style="display: flex; align-items: center;">
            <span class="header-icon">🔍</span>
            <div style="margin-left: 15px;">
                <h1 style="margin: 0; font-size: 28px; color: white;">医疗智能体</h1>
                <p style="margin: 0; font-size: 16px; color: rgba(255,255,255,0.8);">基于LLM和知识图谱的多层级拆解医疗智能体</p>
            </div>
        </div>
        """)

    # 主内容区
    with gr.Row():
        with gr.Column(scale=2):
            # 左侧示例问题面板
            gr.Markdown("""<div class="section-title">示例问题</div>""")
            with gr.Column(elem_classes="examples-container"):
                # 创建隐藏的输入框用于示例
                example_input = gr.Textbox(visible=False)

                # 示例问题分组
                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">疾病症状</div>""")
                gr.Examples(
                    examples=[
                        ["糖尿病的早期症状有哪些？"],
                        ["高血压的常见表现是什么？"],
                        ["乳腺癌有哪些典型症状？"]
                    ],
                    inputs=[example_input],
                    label="",
                    elem_id="symptoms_examples"
                )

                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">治疗方案</div>""")
                gr.Examples(
                    examples=[
                        ["糖尿病的一线治疗方案是什么？"],
                        ["高血压患者应该选择哪些降压药？"],
                        ["乳腺癌的靶向治疗有哪些？"]
                    ],
                    inputs=[example_input],
                    label="",
                    elem_id="treatment_examples"
                )

                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">药物咨询</div>""")
                gr.Examples(
                    examples=[
                        ["阿司匹林的主要副作用有哪些？"],
                        ["二甲双胍不能和什么药一起吃？"],
                        ["他汀类药物需要注意什么？"]
                    ],
                    inputs=[example_input],
                    label="",
                    elem_id="medication_examples"
                )

        with gr.Column(scale=5):
            # 聊天区域
            with gr.Column(elem_classes="chat-container"):
                # 聊天历史显示区域
                chat_display = gr.HTML(
                    value="<div class='chat-history'><div style='text-align: center; padding: 20px; color: var(--text-secondary);'>对话历史将显示在这里</div></div>",
                    elem_classes="chat-history",
                    label="对话历史"
                )

                # 输入区域
                with gr.Row():
                    input_query = gr.Textbox(
                        placeholder="输入您的问题...",
                        lines=1,
                        elem_classes="gr-box",
                        elem_id="main_input",
                        scale=5
                    )
                    submit_btn = gr.Button("发送", variant="primary", elem_classes="gr-button-primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("清除对话", variant="secondary", elem_classes="gr-button-secondary")

            # 处理日志区域
            with gr.Accordion("处理日志", open=False):
                process_logs = gr.HTML(
                    label="系统日志",
                    value="<div class='log-panel'>系统准备就绪，日志将显示在这里...</div>"
                )

    # 页脚
    gr.Markdown("""
    <div class="footer">
        <p>医疗智能体 v1.0 | 基于知识图谱与大语言模型 | 仅供信息参考</p>
        <p>医疗问题请咨询专业医生</p>
    </div>
    """)


    # 更新聊天显示
    def update_chat_display(history):
        """生成聊天历史HTML"""
        chat_html = "<div class='chat-history'>"
        for i, (user_msg, assistant_msg) in enumerate(history):
            # 添加消息索引用于上下文参考
            chat_html += f"""
            <div class="user-message">
                <strong>您 ({i + 1}):</strong> {user_msg}
            </div>
            <div class="assistant-message">
                <strong>助手 ({i + 1}):</strong> {assistant_msg}
            </div>
            """
        if not history:
            chat_html += "<div style='text-align: center; padding: 20px; color: var(--text-secondary);'>对话历史将显示在这里</div>"
        chat_html += "</div>"
        return chat_html


    # 清除对话历史（包括输入框）
    def clear_chat():
        """清除所有对话历史和输入框内容"""
        # 调用聊天机器人的清除上下文方法（如果存在）
        if hasattr(chatbot, 'clear_context'):
            chatbot.clear_context()

        return (
            [],  # 清空对话历史
            "<div class='log-panel'>系统准备就绪，日志将显示在这里...</div>",  # 重置日志
            "<div class='chat-history'><div style='text-align: center; padding: 20px; color: var(--text-secondary);'>对话历史已清除</div></div>",
            # 重置聊天显示
            ""  # 清空输入框
        )


    # 示例问题点击后更新输入框
    def update_input_from_example(text):
        return text


    # 示例问题点击事件
    example_input.change(
        fn=update_input_from_example,
        inputs=[example_input],
        outputs=[input_query]
    )

    # 提交按钮点击事件 - 处理对话历史
    submit_btn.click(
        fn=process_query,
        inputs=[input_query, chat_history],
        outputs=[input_query, process_logs, chat_history, chat_display]
    )

    # 输入框回车事件
    input_query.submit(
        fn=process_query,
        inputs=[input_query, chat_history],
        outputs=[input_query, process_logs, chat_history, chat_display]
    )

    # 清除按钮点击事件 - 现在也清除输入框
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat_history, process_logs, chat_display, input_query]  # 增加输入框作为输出
    )

    # 注入JavaScript代码
    js_code = """
    function() {
        // 为所有示例项添加点击事件
        document.querySelectorAll('.example').forEach(item => {
            item.addEventListener('click', function() {
                const question = this.textContent.trim();
                const inputElement = document.getElementById('main_input');
                if (inputElement) {
                    // 填充输入框
                    inputElement.value = question;

                    // 触发输入事件确保Gradio检测到变化
                    const event = new Event('input', { bubbles: true });
                    inputElement.dispatchEvent(event);

                    // 自动提交
                    setTimeout(() => {
                        const submitBtn = document.querySelector('.gr-button-primary');
                        if (submitBtn) {
                            submitBtn.click();
                        }
                    }, 100);
                }
            });
        });

        // 添加键盘快捷键支持：Ctrl+Enter 提交
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                const submitBtn = document.querySelector('.gr-button-primary');
                if (submitBtn) {
                    submitBtn.click();
                }
            }
        });
    }
    """

    # 使用blocks._js属性注入JavaScript
    demo._js = js_code

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )