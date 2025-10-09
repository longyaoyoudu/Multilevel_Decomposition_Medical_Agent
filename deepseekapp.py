import gradio as gr
from chat_with_llm import KGRAG
import time

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


# 处理用户查询的核心函数
def process_query(query, history=None):
    """处理用户查询并返回响应和日志"""
    start_time = time.time()
    logs = []

    # 初始化日志
    logs.append(color_log(f"用户查询: {query}", "info"))

    try:
        # Step 1: 实体链接
        logs.append(color_log("【步骤1】实体识别...", "system"))
        entity_dict = chatbot.entity_linking(query)
        logs.append(color_log(f"✅ 识别到实体: {entity_dict}", "success"))

        if not entity_dict:
            logs.append(color_log("⚠️ 未识别到相关实体", "warning"))
            return "抱歉，这个问题超出了我的知识范围。", "\n".join(logs)

        # Step 2: 知识图谱召回
        depth = 1
        logs.append(color_log(f"【步骤2】知识检索 (深度={depth})...", "system"))
        facts = set()

        for entity_name, types in entity_dict.items():
            for entity_type in types:
                # 关系链接
                logs.append(color_log(f"🔍 处理实体: {entity_name}({entity_type})", "info"))
                rels = chatbot.link_entity_rel(query, entity_name, entity_type)
                logs.append(color_log(f"  相关关系: {rels}", "success"))

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

        # Step 3: 构建Prompt
        logs.append(color_log("【步骤3】构建提示词...", "system"))
        fact_prompt = chatbot.format_prompt(query, facts_list)
        logs.append(color_log(f"提示词内容:\n{fact_prompt}", "info"))

        # Step 4: LLM生成答案
        logs.append(color_log("【步骤4】生成回答...", "system"))
        answer = chatbot.chat(query)
        logs.append(color_log(f"✅ 回答生成成功", "success"))

        # 计算处理时间
        elapsed = time.time() - start_time
        logs.append(color_log(f"⏱️ 总处理时间: {elapsed:.2f}秒", "info"))

        return answer, "\n".join(logs)

    except Exception as e:
        logs.append(color_log(f"❌ 处理失败: {str(e)}", "error"))
        return "系统处理出错，请稍后再试", "\n".join(logs)


# 浅色主题CSS - 重新添加定义
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

/* 其他CSS规则保持不变... */
"""

# 创建Gradio界面
with gr.Blocks(
        title="基于LLM和知识图谱的私人医生智能体",
        css=light_theme_css + """
        /* 添加示例问题悬停效果 */
        .example-item:hover {
            background-color: var(--surface-light);
            cursor: pointer;
        }
        """,
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
        )
) as demo:
    # 页头
    with gr.Row(equal_height=True, elem_classes="header"):
        gr.Markdown("""
        <div style="display: flex; align-items: center;">
            <span class="header-icon">🔍</span>
            <div>
                <h1 style="margin: 0; font-size: 28px;">医疗智能体</h1>
                <p style="margin: 0; color: var(--text-secondary); font-size: 16px;">基于LLM和知识图谱的私人医生智能体</p>
            </div>
        </div>
        """)

    # 主内容区
    with gr.Row():
        with gr.Column(scale=2):
            # 左侧示例问题面板
            gr.Markdown("""<div class="section-title">示例问题</div>""")
            with gr.Column(elem_classes="examples-container"):
                # 示例问题分组
                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">疾病症状</div>""")
                # 创建隐藏的输入框用于示例
                hidden_input = gr.Textbox(visible=False)
                gr.Examples(
                    examples=[
                        "糖尿病的早期症状有哪些？",
                        "高血压的常见表现是什么？",
                        "乳腺癌有哪些典型症状？"
                    ],
                    inputs=[hidden_input],
                    label="",
                    elem_id="symptoms_examples"
                )

                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">治疗方案</div>""")
                gr.Examples(
                    examples=[
                        "糖尿病的一线治疗方案是什么？",
                        "高血压患者应该选择哪些降压药？",
                        "乳腺癌的靶向治疗有哪些？"
                    ],
                    inputs=[hidden_input],
                    label="",
                    elem_id="treatment_examples"
                )

                gr.Markdown(
                    """<div style="color: var(--text-secondary); font-size: 0.9rem; margin: 12px 0 8px 0;">药物咨询</div>""")
                gr.Examples(
                    examples=[
                        "阿司匹林的主要副作用有哪些？",
                        "二甲双胍不能和什么药一起吃？",
                        "他汀类药物需要注意什么？"
                    ],
                    inputs=[hidden_input],
                    label="",
                    elem_id="medication_examples"
                )

        with gr.Column(scale=5):
            # 聊天区域
            # 定义可见的输入框
            input_query = gr.Textbox(
                label="输入您的问题",
                placeholder="例如：糖尿病的早期症状有哪些？",
                lines=3,
                elem_classes="gr-box",
                elem_id="main_input"
            )

            with gr.Row():
                submit_btn = gr.Button("提交查询", variant="primary", elem_classes="gr-button-primary")
                clear_btn = gr.Button("清除", variant="secondary", elem_classes="gr-button-secondary")

            output_answer = gr.Textbox(
                label="助手回答",
                interactive=False,
                lines=8,
                elem_classes="gr-box"
            )

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


    # 事件处理
    def process_query_with_loading(query):
        """带加载状态的查询处理"""
        if not query.strip():
            return "", "<div class='log-panel'>请输入有效问题</div>"

        # 显示加载状态
        yield "正在处理中... <span class='loading-spinner'></span>", "<div class='log-panel'>系统处理中，请稍候...</div>"

        # 实际处理查询
        answer, logs = process_query(query)
        yield answer, logs


    # 当隐藏输入框的值改变时，更新可见输入框
    def update_main_input(text):
        return text


    # 隐藏输入框变化事件
    hidden_input.change(
        fn=update_main_input,
        inputs=[hidden_input],
        outputs=[input_query]
    )

    # 提交按钮点击事件
    submit_btn.click(
        fn=process_query_with_loading,
        inputs=[input_query],
        outputs=[output_answer, process_logs]
    )

    # 清除按钮点击事件
    clear_btn.click(
        fn=lambda: ("", "", "<div class='log-panel'>系统准备就绪，日志将显示在这里...</div>"),
        inputs=[],
        outputs=[input_query, output_answer, process_logs]
    )

    # 注入JavaScript代码 - 修改为处理示例点击并填充输入框
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

                    // 自动提交（可选）
                    setTimeout(() => {
                        const submitBtn = document.querySelector('.gr-button-primary');
                        if (submitBtn) {
                            submitBtn.click();
                        }
                    }, 500);
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